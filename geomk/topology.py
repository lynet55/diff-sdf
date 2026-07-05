"""Topology version stamp (CLAUDE.md "do now").

Region identity/count is not stable under theta. The stamp is a plain integer
bumped whenever the hard (tau -> 0) segmentation changes its connectivity
signature; consumers caching meshes/region counts/connectivity compare stamps.
Intentionally non-differentiable — it is a cache-invalidation signal, not a
loss term.
"""
import jax.numpy as jnp
import numpy as np
from scipy import ndimage

from .compose import Assembly
from .projections import GridSpec, make_hard_ownership


def hard_labels(asm: Assembly, theta, grid: GridSpec):
    """tau -> 0 projection: voxel -> owning region id, -1 for background.

    Solid occupancy comes from the *raw* union of the component fields:
    precedence composition redistributes ownership but does not remove
    material, whereas its smooth-subtract erodes thin lower-precedence
    regions by O(k) near an interface — enough to open a spurious sub-voxel
    background gap that makes hard connectivity flicker under theta.
    Ownership among solid voxels is argmin of the composed fields.
    (Kernel shared with the accurate-value mass-properties path.)
    """
    owner, solid = make_hard_ownership(asm)(theta, jnp.asarray(grid.points()))
    labels = np.where(np.asarray(solid), np.asarray(owner), -1)
    return labels.reshape(grid.shape)


def topology_signature(asm: Assembly, theta, grid: GridSpec):
    """(total solid CC count, per-region CC counts) on the hardened grid."""
    labels = hard_labels(asm, theta, grid)
    _, n_solid = ndimage.label(labels >= 0)
    per_region = tuple(int(ndimage.label(labels == i)[1])
                       for i in range(len(asm.components)))
    return (int(n_solid),) + per_region


class TopologyTracker:
    def __init__(self):
        self.stamp = 0
        self._last = None

    def update(self, signature):
        if self._last is not None and signature != self._last:
            self.stamp += 1
        self._last = signature
        return self.stamp
