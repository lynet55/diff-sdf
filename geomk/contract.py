"""The contract every consumer relies on (CLAUDE.md).

The field is the contract, but the contract is thicker than a callable: each
projection publishes what is differentiable and at what bandwidth, whether
distances are trustworthy, and a topology stamp for cache invalidation.
Interface identity is a reserved slot — first-class (region_i, region_j)
objects are deferred until a sim (CFD/FSI) needs them.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class FieldContract:
    # Differentiability spec
    smoothing_tau: float          # PoU temperature; occupancy = smoothed Heaviside
    boolean_bandwidth: float      # log-sum-exp k of smooth booleans
    compose_bandwidth: float      # precedence-composition bandwidth
    differentiability: str        # where the kinks are

    # Metric status (invariant 3)
    metric_clean: bool

    # Topology stamp: region identity/count is NOT stable under theta.
    # Any consumer caching a mesh, region count or connectivity must compare.
    topology_stamp: int

    # Reserved: named (region_i, region_j) interface objects. Deferred.
    interface_identity: Optional[object] = None


DIFFERENTIABILITY_NOTE = (
    "All projections are C^inf in theta except: box/capsule primitives have "
    "measure-zero C^0 kinks (face diagonals, cap junctions); mass properties "
    "are integrals of smoothed indicators on a fixed grid, never of a "
    "hardened segmentation; occupancy is a smoothed Heaviside(-phi) with "
    "bandwidth tau tied to the voxel size. The topology stamp is "
    "intentionally non-differentiable (it is a cache-invalidation signal)."
)
