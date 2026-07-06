"""Immersed / fictitious-domain linear-elastic FEM on a background grid.

Finite-cell style: trilinear hex elements on the SAME GridSpec the kernel's
projections use; the material field is the kernel's occupancy projection
(make_occupancy), so stiffness integrals reuse exactly the smoothed-indicator
sampling the mass projection uses — no boundary mesh anywhere. Dirichlet
support is a volumetric penalty over a tagged region's occupancy (springs to
ground); loads are body forces over a tagged region's occupancy. Small-strain
isotropic elasticity; ersatz void stiffness E_min keeps the operator SPD.

Solved matrix-free with Jacobi-preconditioned CG inside
jax.lax.custom_linear_solve, so gradients come from the implicit adjoint
(one extra solve), not from unrolling CG.
"""
import jax
import jax.numpy as jnp
import numpy as np

# local node order: index a = di + 2*dj + 4*dk
_CORNERS = np.array([(di, dj, dk) for dk in (0, 1) for dj in (0, 1)
                     for di in (0, 1)])
_CORNERS = _CORNERS[np.lexsort((_CORNERS[:, 0], _CORNERS[:, 1], _CORNERS[:, 2]))]


def hex_ke(dx, nu):
    """24x24 stiffness of a hx*hy*hz trilinear brick for E = 1 (2^3 Gauss)."""
    hx, hy, hz = dx
    lam = nu / ((1 + nu) * (1 - 2 * nu))
    mu = 1.0 / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[3:, 3:] = np.eye(3) * mu
    signs = 2.0 * _CORNERS - 1.0                     # xi_a, eta_a, zeta_a
    g = 1.0 / np.sqrt(3.0)
    Ke = np.zeros((24, 24))
    detJ = hx * hy * hz / 8.0
    for gx in (-g, g):
        for gy in (-g, g):
            for gz in (-g, g):
                dN = np.stack([
                    signs[:, 0] * (1 + gy * signs[:, 1]) * (1 + gz * signs[:, 2]) / 8 * (2 / hx),
                    signs[:, 1] * (1 + gx * signs[:, 0]) * (1 + gz * signs[:, 2]) / 8 * (2 / hy),
                    signs[:, 2] * (1 + gx * signs[:, 0]) * (1 + gy * signs[:, 1]) / 8 * (2 / hz),
                ], axis=1)                            # (8, 3) physical grads
                B = np.zeros((6, 24))
                for a in range(8):
                    bx, by, bz = dN[a]
                    B[0, 3 * a] = bx
                    B[1, 3 * a + 1] = by
                    B[2, 3 * a + 2] = bz
                    B[3, 3 * a], B[3, 3 * a + 1] = by, bx
                    B[4, 3 * a + 1], B[4, 3 * a + 2] = bz, by
                    B[5, 3 * a], B[5, 3 * a + 2] = bz, bx
                Ke += B.T @ D @ B * detJ
    return Ke


def build_edof(shape):
    """(n_elems, 24) dof indices; element order == GridSpec cell order."""
    nx, ny, nz = shape

    def node_id(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                             indexing="ij")
    cells = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)  # (ne, 3)
    nodes = np.stack([node_id(cells[:, 0] + di, cells[:, 1] + dj,
                              cells[:, 2] + dk)
                      for di, dj, dk in _CORNERS], axis=1)          # (ne, 8)
    edof = (3 * nodes[:, :, None] + np.arange(3)[None, None, :]).reshape(-1, 24)
    return edof.astype(np.int32)


class ImmersedFEM:
    def __init__(self, grid, nu=0.3, e_solid=5.0e9, e_min_ratio=1.0e-3,
                 support_k=2.0e4, cg_tol=1.0e-12, cg_maxiter=30000):
        self.grid = grid
        self.ndof = 3 * int(np.prod(np.array(grid.shape) + 1))
        self.Ke = jnp.asarray(hex_ke(grid.dx, nu))
        self.Ke_diag = jnp.asarray(np.diag(hex_ke(grid.dx, nu)))
        self.edof = jnp.asarray(build_edof(grid.shape))
        self.e_solid = e_solid
        self.e_min = e_min_ratio * e_solid
        # Per-node ground-spring of the volumetric Dirichlet penalty, in
        # N/m per unit occupancy. CALIBRATE against the member stiffness
        # scale (here arms ~1e6 N/m; ~500 clamp nodes x 2e4 = 1e7, a ~10x
        # stiff clamp). A dimensional beta*E*dx choice (first attempt) gave
        # ~1e9/node — 5 orders too stiff — and the smoothed indicator's
        # exponential tail under the hubs then short-circuited the load path
        # to ground, flipping the soft-path gradient's sign. Measured.
        self.k_sup = support_k
        self.cg_tol, self.cg_maxiter = cg_tol, cg_maxiter

    def body_force(self, occ_load, total_force):
        """Body-force vector: total_force (3,) spread over the load region in
        proportion to its occupancy (fixed total load, theta-independent)."""
        share = occ_load / jnp.sum(occ_load)          # per element
        fe = share[:, None, None] / 8.0 * jnp.asarray(total_force)[None, None, :]
        f = jnp.zeros(self.ndof).at[self.edof.reshape(-1)].add(
            jnp.broadcast_to(fe, (occ_load.shape[0], 8, 3)).reshape(-1))
        return f

    def solve(self, occ_struct, occ_support, f_ext):
        """Returns (compliance, u). SPD system, matrix-free preconditioned CG
        under custom_linear_solve (adjoint = one more solve)."""
        E_e = self.e_min + (self.e_solid - self.e_min) * occ_struct
        diag_sup = jnp.zeros(self.ndof).at[self.edof.reshape(-1)].add(
            jnp.repeat(self.k_sup * occ_support / 8.0, 24))

        def matvec(u):
            ue = u[self.edof]
            fe = E_e[:, None] * (ue @ self.Ke)
            f = jnp.zeros(self.ndof, u.dtype).at[self.edof.reshape(-1)].add(
                fe.reshape(-1))
            return f + diag_sup * u

        diag = (jnp.zeros(self.ndof).at[self.edof.reshape(-1)].add(
            (E_e[:, None] * self.Ke_diag[None, :]).reshape(-1)) + diag_sup)
        minv = 1.0 / diag

        def cg_solve(mv, b):
            u, _ = jax.scipy.sparse.linalg.cg(
                mv, b, tol=self.cg_tol, atol=0.0, maxiter=self.cg_maxiter,
                M=lambda x: minv * x)
            return u

        u = jax.lax.custom_linear_solve(matvec, f_ext, solve=cg_solve,
                                        symmetric=True)
        return jnp.dot(f_ext, u), u

    def strain_energy_density(self, occ_struct, u):
        """Per-element strain energy density (stress proxy for display)."""
        E_e = self.e_min + (self.e_solid - self.e_min) * occ_struct
        ue = u[self.edof]
        return 0.5 * E_e * jnp.einsum("ei,ij,ej->e", ue, self.Ke, ue) / self.grid.dV

    def residual_norm(self, occ_struct, occ_support, f_ext, u):
        """||K u - f|| / ||f|| — solver honesty check for the report."""
        E_e = self.e_min + (self.e_solid - self.e_min) * occ_struct
        diag_sup = jnp.zeros(self.ndof).at[self.edof.reshape(-1)].add(
            jnp.repeat(self.k_sup * occ_support / 8.0, 24))
        ue = u[self.edof]
        fe = E_e[:, None] * (ue @ self.Ke)
        Ku = jnp.zeros(self.ndof).at[self.edof.reshape(-1)].add(
            fe.reshape(-1)) + diag_sup * u
        return jnp.linalg.norm(Ku - f_ext) / jnp.linalg.norm(f_ext)
