"""Assemble out/fem_probe.html — the second-consumer (FEM) probe report."""
import base64
import json


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


E = json.load(open("out/probe_e.json"))
F = json.load(open("out/probe_f.json"))
eg, ec = E["gradcheck"], E["codesign"]
fg, fc, fb = F["gradcheck"], F["codesign"], F["brackets"]
fs, fstr = F["gradcheck_soft"], F["straddle_finding"]
fm = F["gradcheck_mix"]

fmt = lambda x: f"{x:.6e}"

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>geomk — FEM second-consumer probe</title>
<style>
  :root {{ --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e3e2de; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--surface); color:var(--ink);
         font:15px/1.55 system-ui,sans-serif; }}
  main {{ max-width:1080px; margin:0 auto; padding:28px 20px 64px; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  h2 {{ font-size:17px; margin:40px 0 6px; }}
  .sub {{ color:var(--ink2); margin:0 0 16px; }}
  .card {{ border:1px solid var(--line); border-radius:10px; overflow:hidden;
           margin:14px 0; }}
  img {{ max-width:100%; display:block; }}
  .verdict {{ border-left:4px solid #2a78d6; background:#f3f6fb;
              padding:12px 16px; border-radius:0 8px 8px 0; margin:12px 0; }}
  .finding {{ border-left:4px solid #eda100; background:#fbf7ec;
              padding:12px 16px; border-radius:0 8px 8px 0; margin:12px 0; }}
  table {{ border-collapse:collapse; width:100%; font-size:14px; }}
  th,td {{ text-align:right; padding:6px 12px; border-bottom:1px solid var(--line);
           font-variant-numeric:tabular-nums; }}
  th:first-child, td:first-child {{ text-align:left; }}
  th {{ color:var(--ink2); font-weight:600; }}
  code {{ background:#f1f0ec; padding:1px 5px; border-radius:4px; font-size:13px; }}
</style></head><body><main>

<h1>Second consumer: immersed structural FEM through the kernel</h1>
<p class="sub">Did the field/projections/contract architecture generalize to a
structurally different simulator, or was it quietly shaped around the
rigid-body plant? Finite-cell linear elasticity on the kernel's occupancy
field — no boundary mesh, no ad-hoc sampling; loads and supports keyed to
intent-tagged regions. Values accurate, gradients soft (locked rule).</p>

<div class="verdict"><b>Verdict: the architecture generalized — with three
honest amendments, all at the projection surface, none touching an
invariant.</b> Stiffness integrals do flow through the same disciplined
sampling as mass integrals (same GridSpec, jitter, PoU). But FEM needed
(1) the occupancy <i>field</i> itself — a new projection,
<code>make_occupancy</code> — where rigid-body only ever consumed
pre-integrated moments; (2) a second occupancy <i>flavor</i>: partition-of-
unity shares are matter, boundary-condition <i>predicates</i> must not
compete in the softmax; and (3) a scope limit on the accurate-value straddle:
exact for linear integrands, direction-unreliable under K⁻¹-nonlinear
functionals — nonlinear consumers optimize on the pure soft path and report
accurate values at anchors. All additive; every prior test passes unchanged
(43/43).</div>

<h2>Gate — gradient through the second physics</h2>
<div class="card"><table>
<tr><th>quantity</th><th>value</th></tr>
<tr><td>compliance C(z₀), accurate path (80 N hub load, body clamped)</td>
    <td>{fmt(eg['compliance0'])} J</td></tr>
<tr><td>∂C/∂(arm length) — AD vs central FD (h = 1e-7)</td>
    <td>{eg['ad'][0]:.8f} vs {eg['fd'][0]:.8f} → rel {eg['rel_err'][0]:.2e}</td></tr>
<tr><td>∂C/∂(arm radius θ) — AD vs FD</td>
    <td>{eg['ad'][1]:.8f} vs {eg['fd'][1]:.8f} → rel {eg['rel_err'][1]:.2e}</td></tr>
<tr><td>FD re-check at the optimized geometry</td>
    <td>rel {ec['fd_at_optimum_rel_err']:.2e}</td></tr>
<tr><td>CG relative residual / one value+gradient</td>
    <td>{eg['cg_residual']:.1e} / {eg['eval_grad_s']:.1f} s</td></tr>
<tr><td>co-design (arm radius vs mass penalty, {ec['steps']} Adam steps)</td>
    <td>C {fmt(ec['compliance'][0])} → {fmt(ec['compliance'][1])} J
        ({ec['compliance'][1]/ec['compliance'][0]:.3f}×); r
        {ec['r_arm'][0]*1e3:.1f} → {ec['r_arm'][1]*1e3:.1f} mm; m_struct
        {ec['m_struct'][0]:.3f} → {ec['m_struct'][1]:.3f} kg</td></tr>
</table></div>
<div class="card"><img alt="deformation and strain energy fields"
  src="data:image/png;base64,{b64('out/fem_field.png')}"></div>
<div class="card"><img alt="structural co-design curves"
  src="data:image/png;base64,{b64('out/fem_codesign.png')}"></div>

<h2>What FEM needed that rigid-body did not (the findings)</h2>
<div class="finding"><b>1 — A field projection, not just moments.</b> Added
<code>make_occupancy</code> (per-cell occupancy, same sampling discipline,
same accurate/soft straddle). Consistency is pinned by test: integrating it
reproduces the mass path's component volumes to 1e-12.</div>
<div class="finding"><b>2 — Matter vs predicate.</b> Partition-of-unity
occupancy is ownership-aware matter; a Dirichlet clamp is a spatial predicate.
Keyed to the partition share, the clamp got eroded by the arm halo in the
softmax (soft ∂C/∂r flipped positive while the accurate trend fell — measured:
C_acc 9.28→7.69e-4 J as r 20→32 mm, C_soft rising). Added
<code>mode="indicator"</code> (sigmoid of the composed field). Surface
tractions would have needed the deferred interface identity — volumetric body
force + volumetric penalty clamp deliberately avoided building it; that
deferral is still intact and still the right call for this consumer.</div>
<div class="finding"><b>3 — Nonlinear consumers amplify soft-field tails.</b>
Mass integrals are linear in occupancy; compliance is K⁻¹-nonlinear with
large coefficients. Two measured consequences: a clamp spring mis-scaled by a
dimensional-analysis default (β·E·dx ≈ 1e9/node) let the indicator's
exponential tail under the hubs short-circuit the load path to ground
(soft compliance 1400× too stiff, wrong gradient signs); and features below
Probe A's t ≥ 2τ floor made hardened compliance hypersensitive (25× change
from a 5e-5 m arm move; one CG divergence). Fixes: calibrate BC coefficients
against member stiffness (2e4 N/m·node), obey the kernel's own resolution
rule (r 16→24 mm), e_min 1e-2. The soft path still under-reports the
r-sensitivity ~12× (smeared arms never reach occupancy 1), so objective
weights must be sized against soft-path sensitivities — reported, not hidden.</div>
<div class="finding"><b>4 — The straddle is exact for linear integrands and
direction-unreliable for nonlinear ones.</b> The locked value-accurate /
gradient-soft rule is implemented as a stop-gradient straddle: soft
sensitivities applied at the hard state's Jacobian. For mass (linear in
occupancy) that is exact. For compliance, ∂C/∂occupancy at the hard state
weights cells very differently than at the soft state: the straddled
∂C/∂(arm length) = {fstr['dCdL_straddle']:+.4e} is FD-exact for its own
anchored function yet OPPOSES both the pure-soft gradient
({fstr['dCdL_soft']:+.4e}) and the accurate macro slope
({fstr['dCdL_accurate_macro']:+.4e}) — the first stretch run descended it and
made the combined objective worse (1.70 → 2.56). Soft supersampling did not
repair it (it is not a quadrature artifact). Resolution: nonlinear consumers
optimize on the pure soft path (gradients are still soft-path, per the lock)
and report accurate values at anchors — which is what the numbers below do.</div>
<div class="finding"><b>5 — Intent wired in cleanly.</b> 'structural' selects
the material region, 'propulsion' the load region — mapped to component
indices in the consumer; the kernel never reads intent (invariant 6 intact).
No new contract field was required, but the contract SHOULD eventually state
(a) that occupancy bandwidth/tails are calibrated for linear integrals, and
(b) the straddle's linear-consumers-only guarantee — both obligations
surfaced here.</div>

<h2>Stretch — one geometry, two physics</h2>
<div class="card"><table>
<tr><th>quantity</th><th>value</th></tr>
<tr><td>combined J(z₀) = flight/f₀ + {fg['lambda']}·C/C₀ (accurate anchors)</td>
    <td>{fg['J0']:.6f}  (flight₀ {fg['flight0']:.4f}, C₀ {fmt(fg['compliance0'])} J)</td></tr>
<tr><td>straddled combined gradient AD vs FD (h = 1e-7)</td>
    <td>rel err [{fg['rel_err'][0]:.2e}, {fg['rel_err'][1]:.2e}]</td></tr>
<tr><td>pure-soft combined gradient AD vs FD</td>
    <td>rel err [{fs['rel_err'][0]:.2e}, {fs['rel_err'][1]:.2e}]</td></tr>
<tr><td>mixed objective (flight: straddle, compliance: pure soft) AD vs FD —
        the optimizer's objective</td>
    <td>rel err [{fm['rel_err'][0]:.2e}, {fm['rel_err'][1]:.2e}]</td></tr>
<tr><td>FD re-check at the optimum (mixed objective)</td>
    <td>rel [{fc['fd_at_optimum_rel_err'][0]:.2e}, {fc['fd_at_optimum_rel_err'][1]:.2e}]</td></tr>
<tr><td>single-physics brackets</td>
    <td>flight-only L* = {fb['L_flight_only']:.3f} m; structure-only r* =
        {fb['r_struct_only']*1e3:.1f} mm</td></tr>
<tr><td>combined co-design ({fc['steps']} Adam steps over [L, r])</td>
    <td>J_acc {fc['J_accurate'][0]:.4f} → {fc['J_accurate'][1]:.4f};
        flight(acc) {fc['flight'][0]:.4f} → {fc['flight'][1]:.4f};
        C(acc) {fmt(fc['compliance'][0])} → {fmt(fc['compliance'][1])} J</td></tr>
<tr><td>geometry</td>
    <td>L {fc['L'][0]:.3f} → {fc['L'][1]:.3f} m; r {fc['r'][0]*1e3:.1f} →
        {fc['r'][1]*1e3:.1f} mm</td></tr>
</table></div>
<div class="finding"><b>6 — Per-consumer fidelity paths (the stretch's own
finding).</b> An all-soft combined objective is a trap from the other side:
the τ-halo's soft mass (5.75 vs 2.41 kg) cannot hover against the 10 N rotor
cap, so the soft flight plant is in a saturated regime whose optimum is
meaningless — an all-soft run drove the accurate combined objective from 1.70
to 112.7 while its own soft value fell. The working combination is flight on
the straddle (validated for near-linear mass/inertia consumers) and
compliance on the pure-soft path — one geometry, two physics, each on the
fidelity path its consumer class requires, all from the same two kernel
projections.</div>
<div class="card"><img alt="multi-physics co-design"
  src="data:image/png;base64,{b64('out/fem_multiphysics.png')}"></div>
<div class="card"><img alt="geometry before/after"
  src="data:image/png;base64,{b64('out/fem_geo.png')}"></div>

</main></body></html>"""

with open("out/fem_probe.html", "w") as f:
    f.write(html)
print("wrote out/fem_probe.html")
