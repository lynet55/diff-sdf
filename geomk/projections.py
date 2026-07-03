"""Projections: smoothed-occupancy integrals on a fixed grid.

Mass properties are integrals of smoothed indicators over precedence-composed
regions (never a hardened segmentation), so per-component mass is
differentiable *through boundary ownership* — the load-bearing property of
the whole kernel. Meshing/FlexiCubes and boundary-integral shape derivatives
are deferred; this carries the prototype.
"""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .compose import Assembly, make_region_fields, pou_weights
from .contract import FieldContract, DIFFERENTIABILITY_NOTE
from .dag import metric_clean


@dataclass(frozen=True)
class GridSpec:
    lo: tuple
    hi: tuple
    shape: tuple
    tau: float = None  # PoU temperature; defaults to 1.5 * coarsest voxel edge

    def __post_init__(self):
        if self.tau is None:
            object.__setattr__(self, "tau", 1.5 * max(self.dx))

    @property
    def dx(self):
        return tuple((h - l) / n for l, h, n in zip(self.lo, self.hi, self.shape))

    @property
    def dV(self):
        return float(np.prod(self.dx))

    def points(self):
        """Cell centers, (N, 3), row-major over shape."""
        axes = [np.linspace(l + d / 2, h - d / 2, n)
                for l, h, n, d in zip(self.lo, self.hi, self.shape, self.dx)]
        g = np.meshgrid(*axes, indexing="ij")
        return np.stack([a.ravel() for a in g], axis=-1)


def make_mass_properties(asm: Assembly, grid: GridSpec):
    """Returns props(theta) -> dict with per-component mass, total mass, com,
    total inertia tensor (about the com), and per-region soft volumes."""
    region_fields = make_region_fields(asm)
    pts = jnp.asarray(grid.points())
    rho = asm.densities
    dV, tau = grid.dV, grid.tau

    def props(theta):
        phi = region_fields(theta, pts)          # (n_regions, N)
        w, _ = pou_weights(phi, tau)             # (n_regions, N)
        volume = jnp.sum(w, axis=1) * dV
        component_mass = rho * volume
        rho_x = jnp.sum(rho[:, None] * w, axis=0)  # (N,) smoothed density field
        total_mass = jnp.sum(rho_x) * dV
        com = (rho_x @ pts) * dV / total_mass
        r = pts - com
        r2 = jnp.sum(r * r, axis=-1)
        inertia = dV * (jnp.eye(3) * jnp.sum(rho_x * r2)
                        - r.T @ (rho_x[:, None] * r))
        return {
            "component_mass": component_mass,
            "component_volume": volume,
            "total_mass": total_mass,
            "com": com,
            "inertia": inertia,
        }

    return props


def make_contract(asm: Assembly, grid: GridSpec, topology_stamp: int) -> FieldContract:
    clean = all(metric_clean(asm.graph, c.root) for c in asm.components)
    return FieldContract(
        smoothing_tau=grid.tau,
        boolean_bandwidth=asm.k_compose,
        compose_bandwidth=asm.k_compose,
        differentiability=DIFFERENTIABILITY_NOTE,
        metric_clean=clean,
        topology_stamp=topology_stamp,
    )
