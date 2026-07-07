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

    # How absolute physical quantities (mass, volume, com, inertia) are
    # produced. Default: the soft smoothed-occupancy path (value == gradient
    # function). With the accurate straddle: hardened supersampled occupancy
    # for values (theta-staircase, do not finite-difference), soft-path
    # gradients (stop-gradient straddle).
    absolute_value_path: str = ("soft smoothed-occupancy integrals (value "
                                "and gradient are the same differentiable "
                                "function)")

    # Declared-mate interface pairs (component indices) whose composition is
    # sharpened to the exact max and whose background is bridged; empty means
    # composition is everywhere the smooth log-sum-exp path.
    sharpened_mates: tuple = ()

    # --- obligations surfaced by the second-consumer (FEM) probe ------------

    # What the smoothed occupancy's bandwidth is calibrated FOR. The tau-halo
    # and exponential tails average out under linear integrands but act as
    # parasitic material to nonlinear / large-coefficient consumers.
    occupancy_semantics: str = (
        "partition-of-unity occupancy with bandwidth tau is calibrated for "
        "LINEAR integrands (mass, volume, inertia): halo and tail errors "
        "cancel under summation. Nonlinear or large-coefficient consumers "
        "(compliance, penalty boundary conditions) must size coefficients "
        "against member scales, use mode='indicator' for region predicates, "
        "and heed the resolution diagnostics.")

    # Scope of the accurate-value stop-gradient straddle.
    straddle_scope: str = (
        "exact for (near-)linear integrands; under nonlinear functionals the "
        "straddled gradient (soft sensitivities at the hard state's "
        "Jacobian) can oppose the true trend — such consumers optimize with "
        "fidelity='soft' and report accurate values at anchors.")

    # The measured resolution trust rule (probe A), machine-checkable via
    # make_resolution_diagnostics.
    resolution_rule: str = (
        "features are trustworthy at thickness >= 2*tau (saturation "
        "diagnostic >= 0.88); below the floor, soft values inflate toward "
        "the 2*tau*ln2 recovered-thickness floor and hardened responses "
        "become hypersensitive.")


DIFFERENTIABILITY_NOTE = (
    "All projections are C^inf in theta except: box/capsule primitives have "
    "measure-zero C^0 kinks (face diagonals, cap junctions); mass properties "
    "are integrals of smoothed indicators on a fixed grid, never of a "
    "hardened segmentation; occupancy is a smoothed Heaviside(-phi) with "
    "bandwidth tau tied to the voxel size. The topology stamp is "
    "intentionally non-differentiable (it is a cache-invalidation signal)."
)
