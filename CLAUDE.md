# CLAUDE.md — Differentiable Parametric Geometry Kernel

## What this is
A differentiable, interpretable parametric geometry kernel. The source of truth is an editable graph of primitives (a human can hand-author features; a generative model can author or regroup them). It resolves to a segmented, multi-domain signed-implicit field, and its purpose is to (a) feed a wide range of differentiable simulators without being tied to any one framework, and (b) act as a decision variable inside optimal-control / RL / co-design loops. First consumer / validation case: UAV flight-dynamics co-design. Build it simulator-agnostic; the UAV is not a special path.

## Substrate
JAX, autodiff-native. Everything downstream of `params → field` must be differentiable in `params`.

---

## Invariants — never violate, prototype status is not an exception
These poison the well if broken and cannot be cheaply retrofitted. Do not "simplify" them away.

1. **Data-first.** The DAG is *data*. Ops are pure kernels `f(params, children) -> field`. Parameters live in arrays. Reason: this is what keeps the tensor-interpreter evaluator and diffusion authoring reachable later.
2. **No closure-geometry.** Geometry is never a Python closure that captures parameters. Reason: a captured-lambda field cannot be lowered into an interpreter or a training tensor — it is the one shortcut that forecloses the whole generative goal, and it cannot be undone later.
3. **The field is a signed-implicit function, not a true SDF.** Sign is always correct; distance is only trustworthy where a subtree is flagged `metric_clean`. Ops needing true distance (offset, shell, uniform lattice thickness) must consume a `metric_clean` subtree or be preceded by an explicit redistance node. Never silently offset/shell a non-metric field. (Implementation: a boolean flag + redistance nodes — *not* symbolic Lipschitz-bound propagation.)
4. **Hard ledger / soft field / hard export.** Primitive→component membership is *hard* (exactly one owning component; authoring-time). Point→region membership is a *soft* partition-of-unity `w_i(x, θ)` with a temperature (evaluation-time, differentiable). Hard segmentation is produced only on export/display as the τ→0 projection.
5. **Domains compose by precedence.** A region's occupancy = its own field minus the smooth union of all higher-precedence regions. Precedence is a declarable discrete attribute; it is order-dependent (good for authoring, legible to a generator), differentiable, and gives well-defined interfaces on the higher-precedence boundary. This is composition semantics and lives *in the DAG*, not as a passive label.
6. **Intent labels are inert.** `intent` (structural / thermal / aero — the *why*) is decoupled annotation for humans and future models to reason about the design. It is not `domain` and never affects composition. Keep the two names separate.

## The contract every consumer relies on
The field is the contract, but the contract is thicker than a callable. Each projection publishes:
- **Differentiability spec:** what is differentiable in `θ`, where the kinks are, and at what smoothing bandwidth (e.g. occupancy is smoothed `Heaviside(−d)` with bandwidth tied to voxel size; mass properties are integrals of smoothed indicators, never of a hardened segmentation).
- **Metric status:** whether distance values are trustworthy (`metric_clean`).
- **Topology stamp:** an integer bumped on topology events. Region identity/count is **not** stable under `θ`; any consumer caching a mesh, region count, or connectivity must check the stamp.
- **Interface identity:** deferred (see below), but reserve the slot: interfaces will be named `(region_i, region_j)` queryable objects.

## Layers over the primitive set
- **Grouping** — components as hard labels over DAG nodes (leaves *and* interior nodes; define per-op label flow). Exactly one owner per node. A member shared by two components (e.g. a spar in wing and fuselage) is a reference/instance in the relations layer, not partial membership.
- **Intent** — inert labels (invariant 6).
- **Domain/medium** — solid / void / fluid-domain / interface, composed by precedence (invariant 5).
- **Exposure** — which params are decision variables; macro-parameters that drive many primitive params through a differentiable relations layer; symmetry/coupling (e.g. mirrored wings). Materializes as a flat parameter vector + selection mask — the same object a black-box/RL optimizer and a diffusion model consume.

---

## Deliberately deferred — named, deferred, DO NOT pre-build
Implementing these early is over-building. Stub behind a reported status; the reason is why it is safe to defer.
- **Tensor-interpreter evaluator.** Start with per-graph JIT tracing. Safe because invariants 1–2 make the later swap a contained *driver* rewrite, not a core rewrite. (The enabling invariants are locked now; the evaluator sophistication is not.)
- **Symbolic metric bounds.** Replaced permanently by invariant 3's flag + redistance nodes.
- **Culling / narrow-band evaluation.** At ~10–30 components on GPU, O(regions)/point is fine, and culling introduces band-edge discontinuities for no near-term payoff. Add when profiling demands, and smooth the band edges then.
- **FlexiCubes / differentiable meshing** and **boundary-integral shape derivatives.** Smoothed-occupancy integrals carry the prototype.
- **First-class interface objects and fluid domains.** Keep precedence composition now; defer interface *naming* until a sim (CFD/FSI) needs it. Rigid-body flight, the first consumer, does not.
- **Full redistancing algorithm.** Stub behind reported metric status early; implement properly later.

## Do now — cheap, from the first commit
Not machinery; each prevents an expensive class of bug.
- **Bijective reparameterization** (softplus for lengths/radii) so the optimizer cannot reach negative extents. Report other validity as differentiable diagnostics; do not pretend to be a feasible-set oracle.
- **Topology version stamp** (the integer above).
- **Finite-difference gradient checks as CI** across the whole op set — catches silently non-differentiable projections before they cost a week.

---

## Validation target — write this test FIRST, keep it in its hard form
Do not soften it. A gentle "body → inertia → gradient" passes without ever touching the load-bearing piece.
- Two overlapping components (wing + fuselage), different densities, **declared precedence**.
- Objective depends on per-component mass and total inertia tensor.
- Optimize a parameter that **moves the shared interface**.
- The gradient must run *through* boundary ownership (per-component mass = smoothed-indicator integrals over precedence-composed regions).

Standing CI alongside it: the finite-difference gradient checks above, and a parameter sweep that **crosses a topology event** (two components merge) so the behavior at merges is characterized, not discovered.

---

## Build sequence — multiple passes, each green before the next
Do **not** scaffold the whole kernel in one pass. An end-to-end build with closure-geometry buried in it is worse than a narrow slice that is structurally right.

- **Pass 1 — evaluator core + the validation test.** Data-first DAG, pure-kernel ops, per-graph JIT. Minimal op set: two primitives, smooth union, subtract, transform. Soft partition-of-unity field + precedence composition. Smoothed-occupancy mass properties. Write the validation test first; get it green. No offset/shell yet (they need metric).
- **Pass 2 — metric discipline.** `metric_clean` flag, redistance node, then the metric-sensitive ops (offset, shell, lattice).
- **Pass 3 — exposure.** Flat vector + mask, macro-parameter relations, symmetry coupling. (Reparam and topology stamp land here if not already in Pass 1.)
- **Pass 4+ — breadth.** More primitives/ops; more projections (mesh via smoothed occupancy first). Pull items from the deferred list only when a consumer forces it.

### Process rules
- When a spec item is ambiguous, **ask before implementing** rather than guessing — a wrong guess quietly discards a lot of prior reasoning.
- Keep each pass green; do not start the next pass on a red one.
- If you believe a deferred item actually needs to be structural, say so and why before building it — do not silently promote it.