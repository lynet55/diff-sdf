"""Projections: smoothed-occupancy integrals on a fixed grid.

Mass properties are integrals of smoothed indicators over precedence-composed
regions (never a hardened segmentation), so per-component mass is
differentiable *through boundary ownership* — the load-bearing property of
the whole kernel. Meshing/FlexiCubes and boundary-integral shape derivatives
are deferred; this carries the prototype.

Value accurate, gradient smooth (opt-in): with accurate=True, the returned
absolute quantities are produced by a hardened (tau -> 0) supersampled
occupancy — raw-union sign for solid, argmin of the precedence-composed
fields for ownership, exactly the hard_labels semantics — while the gradient
stays on the soft partition-of-unity path via a stop-gradient straddle:
O = O_accurate + (O_soft - stop_gradient(O_soft)). The value equals
O_accurate (a theta-staircase, intentionally non-differentiable); the
gradient equals d(O_soft)/dtheta exactly. Default accurate=False is the
pure-soft path, byte-for-byte unchanged.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .compose import (Assembly, make_region_fields, make_background_field,
                      pou_weights)
from .contract import FieldContract, DIFFERENTIABILITY_NOTE
from .dag import _nid, metric_clean
from .evaluate import eval_node
from .reparam import constrain

# Default deterministic sub-voxel jitter of the cell-center sample positions,
# as a fraction of the voxel edge per axis. A grid symmetric about the same
# planes as the geometry places quadrature nodes exactly on primitive-SDF
# max/abs tie sets (box face diagonals, x = +-y planes), where AD subgradients
# disagree with central finite differences; the kink sets are measure-zero in
# space but a symmetric grid samples them with nonzero measure. The fractions
# are asymmetric (pairwise sums/differences well away from 0 and 1) so shifted
# axes cannot re-align with each other, scale with dx, and are deterministic
# so repeated calls and self-referential computations match exactly.
GRID_JITTER = (0.317, 0.113, 0.437)

# Background-bridge bandwidth for mate-declaring assemblies, in units of tau:
# the smooth union fills a flush seam by k*ln2, so 5*tau puts the residual
# void weight at the mating plane at e^{-5 ln 2} / (2 + e^{-5 ln 2}) ~ 0.015.
BG_BRIDGE_TAUS = 5.0


@dataclass(frozen=True)
class GridSpec:
    lo: tuple
    hi: tuple
    shape: tuple
    tau: float = None  # PoU temperature; defaults to 1.5 * coarsest voxel edge
    jitter: tuple = GRID_JITTER  # sub-voxel sample offset, fraction of dx

    def __post_init__(self):
        if self.tau is None:
            object.__setattr__(self, "tau", 1.5 * max(self.dx))

    @property
    def dx(self):
        return tuple((h - l) / n for l, h, n in zip(self.lo, self.hi, self.shape))

    @property
    def dV(self):
        return float(np.prod(self.dx))

    def points(self, supersample=1):
        """Cell centers (N, 3), row-major over shape, jittered off tie planes.

        supersample=s refines each cell into s^3 subcells and returns their
        (jittered) centers — same deterministic jitter rule at the fine dx,
        so accurate-path quadrature stays consistent with the coarse grid.
        """
        s = int(supersample)
        axes = [np.linspace(l + d / (2 * s), h - d / (2 * s), n * s) + j * d / s
                for l, h, n, d, j in zip(self.lo, self.hi, self.shape,
                                         self.dx, self.jitter)]
        g = np.meshgrid(*axes, indexing="ij")
        return np.stack([a.ravel() for a in g], axis=-1)


def make_hard_ownership(asm: Assembly):
    """Hardened tau->0 segmentation kernel (shared with topology.hard_labels):
    owner(theta, points) -> (owner ids via argmin of the composed fields,
    solid mask via the sign of the raw component union). Precedence
    composition redistributes ownership but does not remove material, so the
    solid mask comes from the raw union; its smooth-subtract erosion must not
    open spurious background at interfaces."""
    region_fields = make_region_fields(asm)
    roots = [_nid(c.root) for c in asm.components]

    def ownership(theta, points):
        q = constrain(theta, asm.graph.positive_mask)
        d = jnp.stack([eval_node(asm.graph, r, q, points) for r in roots])
        phi = region_fields(theta, points)
        owner = jnp.argmin(phi, axis=0)
        solid = jnp.min(d, axis=0) <= 0.0
        return owner, solid

    return ownership


def make_mass_properties(asm: Assembly, grid: GridSpec,
                         accurate: bool = False, supersample: int = 3):
    """Returns props(theta) -> dict with per-component mass, total mass, com,
    total inertia tensor (about the com), and per-region volumes.

    accurate=False (default): pure soft smoothed-occupancy integrals — value
    and gradient are the same differentiable function (existing semantics).

    accurate=True: values come from hardened supersampled occupancy (see
    module docstring); gradients stay exactly the soft-path gradients via the
    stop-gradient straddle. Values are then theta-staircases: do not finite-
    difference them, and do not expect value == its own gradient's potential.
    """
    region_fields = make_region_fields(asm)
    pts = jnp.asarray(grid.points())
    rho = asm.densities
    dV, tau = grid.dV, grid.tau
    background = make_background_field(asm) if asm.mates else None

    def integrals(w_or_occ, points, dvol):
        volume = jnp.sum(w_or_occ, axis=1) * dvol
        component_mass = rho * volume
        rho_x = jnp.sum(rho[:, None] * w_or_occ, axis=0)
        total_mass = jnp.sum(rho_x) * dvol
        com = (rho_x @ points) * dvol / total_mass
        r = points - com
        r2 = jnp.sum(r * r, axis=-1)
        inertia = dvol * (jnp.eye(3) * jnp.sum(rho_x * r2)
                          - r.T @ (rho_x[:, None] * r))
        return {
            "component_mass": component_mass,
            "component_volume": volume,
            "total_mass": total_mass,
            "com": com,
            "inertia": inertia,
        }

    def soft_props(theta):
        phi = region_fields(theta, pts)          # (n_regions, N)
        phi_bg = background(theta, pts, BG_BRIDGE_TAUS * tau) \
            if background is not None else None
        w, _ = pou_weights(phi, tau, phi_bg)     # (n_regions, N)
        return integrals(w, pts, dV)

    if not accurate:
        return soft_props

    ownership = make_hard_ownership(asm)
    pts_ss = jnp.asarray(grid.points(supersample))
    dV_ss = dV / supersample ** 3
    ids = jnp.arange(len(asm.components))

    def hard_props(theta):
        owner, solid = ownership(theta, pts_ss)
        occ = ((owner[None, :] == ids[:, None]) & solid[None, :]
               ).astype(pts_ss.dtype)
        return integrals(occ, pts_ss, dV_ss)

    def props(theta):
        s = soft_props(theta)
        a = jax.tree.map(jax.lax.stop_gradient, hard_props(theta))
        return {key: a[key] + (s[key] - jax.lax.stop_gradient(s[key]))
                for key in s}

    return props


def make_contract(asm: Assembly, grid: GridSpec, topology_stamp: int,
                  accurate: bool = False, supersample: int = 3) -> FieldContract:
    clean = all(metric_clean(asm.graph, c.root) for c in asm.components)
    if accurate:
        absolute = (f"hardened supersampled occupancy (tau->0 raw-union sign "
                    f"+ argmin ownership, supersample={int(supersample)}; "
                    f"stop-gradient straddle: values are theta-staircases, "
                    f"gradients are the soft-path gradients)")
    else:
        absolute = ("soft smoothed-occupancy integrals (value and gradient "
                    "are the same differentiable function)")
    mates = tuple(sorted(tuple(sorted(m)) for m in asm.mate_set))
    return FieldContract(
        smoothing_tau=grid.tau,
        boolean_bandwidth=asm.k_compose,
        compose_bandwidth=asm.k_compose,
        differentiability=DIFFERENTIABILITY_NOTE,
        metric_clean=clean,
        topology_stamp=topology_stamp,
        absolute_value_path=absolute,
        sharpened_mates=mates,
    )
