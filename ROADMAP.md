## Goal

A differentiable, interpretable parametric geometry kernel — **"nTop, but autodiff-native."** Every shape resolves to a segmented signed-implicit field that is differentiable in its parameters, so the same model can be authored as CAD, dropped into an optimal-control / RL / co-design loop as a decision variable, and returned to CAD. The target round-trip:

**CAD → optimal-control-based co-design → CAD** (and out to any solver).

The differentiator vs nTop: nTop is field-based but **not** end-to-end differentiable; it cannot hand a gradient to a control/co-design loop. This kernel's entire reason to exist is that it can.

---

## What has been done

### Core (load-bearing, structurally locked)
- **Data-first DAG** — the model is *data* (`(op, children, param-slice)` into one flat unconstrained vector), not Python closures. A generative model or UI can author/regroup it directly. Invariants 1–2 hold in code, not just docs.
- **Pure op kernels** `f(params, children) → field`, per-graph JIT. No captured parameters anywhere.
- **Soft partition-of-unity + precedence composition** — point→region membership is a differentiable softmax with temperature τ; domains compose by declarable precedence; hard segmentation only on export (τ→0). Per-component mass is differentiable *through boundary ownership* (the validation test).
- **Metric discipline** — `metric_clean` flag propagates by construction; `offset`/`shell` refuse a non-metric input at build time; `redistance` node (first-order stub, honest flag).
- **Bijective reparameterization** (softplus for positive extents) so the optimizer cannot reach negative lengths/radii.
- **Topology version stamp** for cache invalidation across topology events.

### Ops
`sphere, box, capsule, polygon` · `revolve, extrude, loft` · `smooth_union, smooth_subtract, smooth_intersect` · `rigid` · `offset, shell, redistance` · `lattice`.
- **New this round: `polygon`** — exact 2D SDF sketch profile (arbitrary, concave-capable via winding sign), metric-**clean**, so `extrude(polygon)`/`revolve(polygon)` stay exact SDFs and are offset/shell-able with no redistance. Feeds all profile ops. This is the "shapes → parts" unlock.
- **New: `smooth_intersect`** — the missing third CSG boolean.

### Projections (the differentiable solver interface)
- **Mass properties** (per-component mass, COM, inertia) as smoothed-occupancy integrals; explicit **fidelity knob** (`soft` vs accurate-anchored `straddle`) — accurate absolute value, soft gradient.
- **Occupancy** (partition/indicator) — feeds an immersed/fictitious-domain FEM (matrix-free preconditioned CG under a custom adjoint).
- **Surface measure** (co-area) — wetted/projected area with no mesh.
- **Resolution diagnostics** — probe-derived rule (features trustworthy at half-thickness ≥ 2τ ↔ saturation ≥ 0.88) exposed as a differentiable barrier.

### Exposure layer
Flat decision vector + selection mask + affine relations θ(z) in unconstrained space, with symmetry coupling — the identical object a gradient optimizer *and* a black-box optimizer consume.

### End-to-end validation (the round-trip, demonstrated)
- **12-component UAV capstone** authored from a spec dict using all breadth ops (lofted fuselage, revolved nacelles, extruded fin, lattice deck), declared mate, 8 exposure macros.
- **Four-physics co-design objective** (flight plant / co-area drag / immersed-FEM compliance / resolution barrier) on one flat vector; **strict finite-difference gate passes ≤1e-6 in float64** on all 8 macros for the differentiable backbone.
- **Gradient co-design beats a tuned black-box (1+1)-ES** on the same vector; the saturation barrier holds controllable features in trust (with/without A/B).

### CAD-out endpoint (new this round)
- **Watertight mesh export** from the composed field — `export_solid` (outer shell) and `export_components` (per-material). Marching cubes with a positive pad ⇒ always a closed 2-manifold; `is_watertight` verifies (every edge shared by exactly two faces, no non-manifold edges); `mesh_volume` cross-checks against the accurate occupancy integral; outward normals; **binary STL + OBJ**. Verified on the 12-component capstone (watertight union, genus 20, 60k triangles).

### Tests
**69 passing**, including a whole-op-set finite-difference gradient CI, the precedence/mass validation, metric-flag refusal, topology-sweep, surface measure, sketch correctness, and watertight-export verification.

---

## What remains to reach the goal

### Pre-UI gate (do before slapping a CAD UI on it)
These are exactly the things a CAD UI stresses hardest; building the UI first means building it on a foundation that feels wrong/slow where users push.

1. **Adaptive / narrow-band resolution (highest leverage).** τ is currently a single scalar tied to the *coarsest* voxel axis, and evaluation is a uniform grid with no culling. A 1 mm wall on a 100 mm part is untrustworthy without a globally fine grid. Octree / narrow-band evaluation retires **three** drawbacks at once: thin-feature accuracy, meshing crispness (evaluate finely only near surfaces), and most of the interactive-performance cost.
2. **Interactive performance / incremental compilation.** JIT is per-graph-topology, so every *structural* edit recompiles (seconds). This is the consumer that finally justifies the deferred **tensor-interpreter evaluator** (or at least cached/incremental compilation).
3. **Robustness guards for arbitrary UI input.** Out-of-domain geometry silently returns garbage (a `domain_clipped` warning now exists as a start); the accurate path can divide-by-zero on an empty solid. Needs auto domain-sizing from the DAG, degenerate-input handling, and validity diagnostics surfaced rather than thrown.

### Strategic (for true nTop parity, can follow the UI)
4. **Field-driven parameters.** nTop's signature move — drive wall thickness / lattice density by a spatial field (e.g. stress-graded lattice) — is not expressible: `offset`/`shell`/`lattice` take *scalars*, not fields. Deeper architectural item.
5. **Meshing quality.** Export is watertight and faithful but uniform marching cubes — it rounds sharp creases and can't grade triangle density. Needs dual-contouring / **FlexiCubes** (probe G is the measured trigger).
6. **Metric robustness.** `redistance` is a single first-order step; offset/shell of heavily-booleaned geometry is only trustworthy near the surface. Needs a real redistancing pass.
7. **Field/op breadth.** No TPMS/gyroid, no graded lattice, sketch profiles are polygon-only (no arcs/splines), profile ops are axis-locked (revolve about x, extrude/loft along z — need `rigid()` today).
8. **First-class interface objects.** Per-component export interfaces are coincident, not vertex-shared; conformal interface meshing / named `(region_i, region_j)` interfaces stay deferred until a coupled sim (CFD/FSI/thermal-contact) needs them.
9. **Interop breadth.** Export is STL/OBJ; no STEP/B-rep out, no CAD import (so you must author in the DAG, can't start from a legacy file).

### Recommendation
The differentiable spine is sound and the CAD→co-design→CAD round-trip **closes today**. Two of the original three pre-UI blockers (sketch breadth, watertight export) are cleared. **Do #1 (adaptive resolution) before the UI** — it's the honest gate and retires the accuracy + crispness + perf drawbacks together. #2 and #3 close out UI-readiness. Everything else is post-UI, added as consumers demand.
