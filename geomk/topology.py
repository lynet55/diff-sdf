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

from .compose import Assembly, make_region_fields
from .dag import _nid
from .evaluate import eval_node
from .projections import GridSpec
from .reparam import constrain


def hard_labels(asm: Assembly, theta, grid: GridSpec):
    """tau -> 0 projection: voxel -> owning region id, -1 for background.

    Solid occupancy comes from the *raw* union of the component fields:
    precedence composition redistributes ownership but does not remove
    material, whereas its smooth-subtract erodes thin lower-precedence
    regions by O(k) near an interface — enough to open a spurious sub-voxel
    background gap that makes hard connectivity flicker under theta.
    Ownership among solid voxels is argmin of the composed fields.
    """
    pts = grid.points()
    phi = np.asarray(make_region_fields(asm)(theta, pts))
    q = constrain(theta, asm.graph.positive_mask)
    d = np.stack([np.asarray(eval_node(asm.graph, _nid(c.root), q, jnp.asarray(pts)))
                  for c in asm.components])
    labels = np.argmin(phi, axis=0)
    labels[d.min(axis=0) > 0.0] = -1
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
