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


def _resolve_fidelity(fidelity, accurate):
    """Explicit consumer-facing fidelity spelling (second-consumer probe):
    'soft'      — value == gradient function; the right OPTIMIZATION surface
                  for nonlinear functionals of the field (immersed FEM
                  compliance), whose straddled gradient can oppose the true
                  trend. Report accurate values at anchors separately.
    'straddle'  — hardened supersampled values, soft-path gradients; exact
                  for (near-)linear integrands (mass/inertia/COM consumers
                  such as rigid-body plants). Values are theta-staircases:
                  never finite-difference them.
    None falls back to the boolean `accurate` flag (compatibility)."""
    if fidelity is None:
        return bool(accurate)
    if fidelity not in ("soft", "straddle"):
        raise ValueError(f"fidelity must be 'soft' or 'straddle', got {fidelity!r}")
    return fidelity == "straddle"


def make_mass_properties(asm: Assembly, grid: GridSpec,
                         accurate: bool = False, supersample: int = 3,
                         fidelity: str = None):
    """Returns props(theta) -> dict with per-component mass, total mass, com,
    total inertia tensor (about the com), and per-region volumes.

    accurate=False (default): pure soft smoothed-occupancy integrals — value
    and gradient are the same differentiable function (existing semantics).

    accurate=True: values come from hardened supersampled occupancy (see
    module docstring); gradients stay exactly the soft-path gradients via the
    stop-gradient straddle. Values are then theta-staircases: do not finite-
    difference them, and do not expect value == its own gradient's potential.

    fidelity ('soft' | 'straddle') is the preferred explicit spelling of the
    same choice — see _resolve_fidelity for which consumer class needs which.
    """
    accurate = _resolve_fidelity(fidelity, accurate)
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


def make_occupancy(asm: Assembly, grid: GridSpec, components=None,
                   accurate: bool = False, supersample: int = 3,
                   mode: str = "partition", soft_supersample: int = 1,
                   fidelity: str = None):
    """Per-cell occupancy of a subset of components, as a field.

    fidelity ('soft' | 'straddle') is the preferred explicit spelling of the
    accurate flag — see _resolve_fidelity for which consumer class needs
    which path.

    mode="partition" (default): the components' share of the partition of
    unity — matter. Shares compete in the softmax, sum to <= 1 across all
    components, and are the right integrand for material fields (mass,
    stiffness). mode="indicator": smoothed membership predicate of the
    selected regions, sigmoid(-phi/tau) of the precedence-composed field —
    the right object for boundary-condition predicates (clamped region,
    loaded region), because a predicate must not be eroded when another
    component's halo competes for the same cells. Indicator occupancies of
    overlapping selections double-count; they are predicates, not matter.
    (Second-consumer finding: rigid-body needed neither; FEM needed both.)

    soft_supersample: quadrature order of the SOFT path — s^3 subcell samples
    averaged per cell (same deterministic jittered lattice the accurate path
    uses). Default 1 is the mass projection's midpoint rule. Note from the
    second-consumer probe: refining soft quadrature does NOT repair the
    stop-gradient straddle's direction error under nonlinear consumers (the
    straddle applies soft sensitivities at the hard state's Jacobian, exact
    for linear integrands, direction-unreliable for K^-1-type functionals —
    measured in bench/probe_f); such consumers should optimize on the pure
    soft path and report accurate values at anchors.

    The same disciplined sampling as make_mass_properties — same GridSpec
    cells and jitter, same PoU weights (mate-aware background), same hardened
    supersampled ownership and stop-gradient straddle — but exposed *before*
    integration, for volumetric consumers that need the indicator itself
    (immersed/fictitious-domain FEM uses it as the material field). Added for
    the second-consumer probe: rigid-body consumed only integrated moments;
    this is the one thing a field-consuming simulator needed that mass
    properties did not expose.

    components: iterable of component indices (default: all). Selection is by
    index; mapping intent tags to indices is the consumer's business — intent
    stays inert inside the kernel (invariant 6).

    Returns occ(theta) -> (N,) cell occupancies in grid cell order.
    accurate=True: values are hardened supersampled per-cell volume fractions
    (quantized to 1/supersample^3 — a theta-staircase, do not finite-
    difference); gradients are the soft partition-of-unity gradients via the
    stop-gradient straddle. accurate=False: pure soft path, value == gradient
    function.
    """
    accurate = _resolve_fidelity(fidelity, accurate)
    if mode not in ("partition", "indicator"):
        raise ValueError(f"unknown occupancy mode {mode!r}")
    sel = np.arange(len(asm.components)) if components is None \
        else np.asarray(list(components), dtype=int)
    region_fields = make_region_fields(asm)
    background = make_background_field(asm) if asm.mates else None
    ss = int(soft_supersample)
    pts = jnp.asarray(grid.points(ss))
    tau = grid.tau
    sel_j = jnp.asarray(sel)

    def _cell_mean(vals):
        if ss == 1:
            return vals
        n0, n1, n2 = grid.shape
        return vals.reshape(n0, ss, n1, ss, n2, ss).mean(
            axis=(1, 3, 5)).reshape(-1)

    def _phi_min(theta, points):
        phi = region_fields(theta, points)[sel_j]
        return jnp.min(phi, axis=0) if sel.size > 1 else phi[0]

    def soft_occ(theta):
        if mode == "indicator":
            return _cell_mean(jax.nn.sigmoid(-_phi_min(theta, pts) / tau))
        phi = region_fields(theta, pts)
        phi_bg = background(theta, pts, BG_BRIDGE_TAUS * tau) \
            if background is not None else None
        w, _ = pou_weights(phi, tau, phi_bg)
        return _cell_mean(jnp.sum(w[sel_j], axis=0))

    if not accurate:
        return soft_occ

    ownership = make_hard_ownership(asm)
    s = int(supersample)
    pts_ss = jnp.asarray(grid.points(s))
    n0, n1, n2 = grid.shape

    def hard_occ(theta):
        if mode == "indicator":
            inside = (_phi_min(theta, pts_ss) <= 0.0).astype(pts_ss.dtype)
        else:
            owner, solid = ownership(theta, pts_ss)
            inside = (jnp.any(owner[None, :] == sel_j[:, None], axis=0)
                      & solid).astype(pts_ss.dtype)
        # subsample index along each axis is minor within its coarse cell
        cells = inside.reshape(n0, s, n1, s, n2, s)
        return cells.mean(axis=(1, 3, 5)).reshape(-1)

    def occ(theta):
        soft = soft_occ(theta)
        hard = jax.lax.stop_gradient(hard_occ(theta))
        return hard + (soft - jax.lax.stop_gradient(soft))

    return occ


def make_surface_measure(asm: Assembly, grid: GridSpec):
    """Surface quantities WITHOUT a boundary mesh (co-area formula).

    The solid occupancy 1 - w_bg is a smoothed indicator whose spatial
    gradient concentrates in a tau-band around the EXTERIOR surface — the
    partition of unity sums to ~1 across internal component interfaces, so
    those contribute (almost) nothing: |grad(1 - w_bg)| dV is wetted-surface
    measure for free, on the same grid/jitter/PoU discipline as every other
    projection. Third consumer class probed: moments (rigid body), fields
    (FEM), and now surfaces (aero proxies) — added so a drag-type objective
    can be tried BEFORE promoting the deferred FlexiCubes / interface-
    identity items; whether this measure suffices is the probe's question.

    Returns surf(theta, direction=None) -> dict:
      'wetted_area'    : ∫ |grad_x occ| dV — total exterior area (soft;
                         carries the same tau-band bias family as volumes:
                         O(tau * mean curvature) + thin-feature inflation —
                         heed the resolution diagnostics);
      'projected_area' : only if direction is given — (3,) or (k, 3) rows —
                         ∫ max(0, grad_x occ · d̂) dV per direction, the
                         front-facing silhouette integral (counts overlapping
                         front-facing patches once each; exact projected area
                         for convex bodies, an upper bound otherwise — the
                         standard drag-reference-area proxy); scalar for one
                         direction, (k,) for several (one gradient pass).
                         Measured (probe G, quadrotor): non-occluding views
                         carry only the tau-band bias (+9% at tau=0.011,
                         shrinking with tau), but occluding views over-count
                         structurally (+53% along the axis where hubs hide
                         hubs) — occlusion is NON-LOCAL and no local surface
                         measure can fix it; an objective needing occlusion-
                         corrected areas is the genuine trigger for the
                         deferred mesh/interface-identity work;
      'surface_density': (N,) per-cell |grad_x occ| for visualization.

    Per-component splits are deliberately NOT offered: attributing shared
    surface to components is interface identity, which stays deferred.
    Differentiable in theta end-to-end (grad_x is computed by autodiff at
    the sample points; theta-gradients flow through it as second-order).
    """
    region_fields = make_region_fields(asm)
    background = make_background_field(asm) if asm.mates else None
    pts = jnp.asarray(grid.points())
    tau = grid.tau

    def occupancy_at(theta, point):
        """Solid occupancy 1 - w_bg at a single point (3,) — scalar."""
        phi = region_fields(theta, point)                    # (n_regions,)
        bg = -background(theta, point, BG_BRIDGE_TAUS * tau) / tau \
            if background is not None else 0.0
        logits = jnp.concatenate([-phi / tau, jnp.atleast_1d(bg)])
        m = jnp.max(logits)
        e = jnp.exp(logits - m)
        return 1.0 - e[-1] / jnp.sum(e)

    def surf(theta, direction=None):
        g = jax.vmap(jax.grad(lambda p: occupancy_at(theta, p)))(pts)
        density = jnp.linalg.norm(g, axis=-1)
        out = {
            "wetted_area": jnp.sum(density) * grid.dV,
            "surface_density": density,
        }
        if direction is not None:
            d = jnp.atleast_2d(jnp.asarray(direction, dtype=g.dtype))
            d = d / jnp.linalg.norm(d, axis=-1, keepdims=True)
            proj = jnp.sum(jnp.maximum(g @ d.T, 0.0), axis=0) * grid.dV
            out["projected_area"] = proj[0] if jnp.ndim(direction) == 1 \
                else proj
        return out

    return surf


# Saturation threshold of the resolution rule: at a slab's midplane the soft
# partition weight is ~sigmoid(t/tau), so probe A's "trustworthy at t >= 2*tau"
# maps to saturation >= sigmoid(2) ~ 0.88.
TRUST_SATURATION = 0.88


def make_resolution_diagnostics(asm: Assembly, grid: GridSpec,
                                supersample: int = 3,
                                include_inflation: bool = True):
    """Differentiable resolution diagnostics (CLAUDE.md: report validity as
    differentiable diagnostics; never pretend to be a feasible-set oracle).

    Probe A's measured rule: a feature of half-thickness t is trustworthy at
    t >= 2*tau (recovered-thickness error <= 6%); below that, soft values
    inflate toward the 2*tau*ln2 floor and hardened responses become
    hypersensitive (probe E: 25x compliance change from a 5e-5 m move; probe
    F: an optimizer walked an arm radius through the floor unchecked). This
    projection turns the rule into a per-component signal:

      saturation: (n_components,) max soft PoU weight over cells. A resolved
          solid saturates to ~1; a component whose thinnest dimension is
          2*tau reads ~TRUST_SATURATION (0.88). Differentiable in theta
          (subgradient at argmax ties), so an optimizer can subscribe, e.g.
          penalty = softplus((TRUST_SATURATION - saturation) / 0.02).
      inflation: (n_components,) soft volume / hardened supersampled volume.
          A COMPARATIVE signal: even resolved bodies carry an edge/corner
          halo baseline of 1 + O(tau^2 * edge length / V) (measured ~1.17
          for a chunky box at tau = 0.03); a sub-floor feature jumps well
          past 2. Value-only: the denominator is a theta-staircase and is
          stop-gradiented (do not use as a loss term; use saturation).

    Returns diag(theta) -> {'saturation', 'inflation', 'tau',
                            'trust_threshold'}.

    include_inflation=False skips the hardened supersampled pass (the
    expensive, value-only half) — the cheap configuration for an optimizer
    subscribing to the saturation barrier every step.
    """
    region_fields = make_region_fields(asm)
    background = make_background_field(asm) if asm.mates else None
    ownership = make_hard_ownership(asm) if include_inflation else None
    pts = jnp.asarray(grid.points())
    s = int(supersample)
    pts_ss = jnp.asarray(grid.points(s)) if include_inflation else None
    tau = grid.tau
    ids = jnp.arange(len(asm.components))

    def diag(theta):
        phi = region_fields(theta, pts)
        phi_bg = background(theta, pts, BG_BRIDGE_TAUS * tau) \
            if background is not None else None
        w, _ = pou_weights(phi, tau, phi_bg)
        out = {
            "saturation": jnp.max(w, axis=1),
            "tau": tau,
            "trust_threshold": TRUST_SATURATION,
        }
        if include_inflation:
            v_soft = jnp.sum(w, axis=1) * grid.dV
            owner, solid = ownership(theta, pts_ss)
            occ = ((owner[None, :] == ids[:, None]) & solid[None, :]
                   ).astype(pts_ss.dtype)
            v_hard = jax.lax.stop_gradient(
                jnp.sum(occ, axis=1) * grid.dV / s ** 3)
            out["inflation"] = v_soft / jnp.maximum(v_hard, 1e-12)
        return out

    return diag


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
