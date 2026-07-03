"""Assemble out/probes.html — self-contained report of probes A/B/C."""
import base64
import json


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


A = json.load(open("out/probe_a.json"))
B = json.load(open("out/probe_b.json"))
C = json.load(open("out/probe_c.json"))

a1, a2r, a2g = A["compile_vs_nodes"], A["eval_vs_regions"], A["eval_vs_resolution"]
gc, opt = C["gradcheck"], C["optimization"]

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>geomk — probes A/B/C</title>
<style>
  :root {{ --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e3e2de; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--surface); color:var(--ink);
         font:15px/1.55 system-ui,sans-serif; }}
  main {{ max-width:1080px; margin:0 auto; padding:28px 20px 64px; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  h2 {{ font-size:17px; margin:40px 0 6px; }}
  .sub {{ color:var(--ink2); margin:0 0 18px; }}
  .card {{ border:1px solid var(--line); border-radius:10px; overflow:hidden;
           margin:14px 0; }}
  img {{ max-width:100%; display:block; }}
  .verdict {{ border-left:4px solid #2a78d6; background:#f3f6fb;
              padding:12px 16px; border-radius:0 8px 8px 0; margin:12px 0; }}
  .verdict b {{ display:block; margin-bottom:4px; }}
  table {{ border-collapse:collapse; width:100%; font-size:14px; }}
  th,td {{ text-align:right; padding:6px 12px; border-bottom:1px solid var(--line);
           font-variant-numeric:tabular-nums; }}
  th:first-child, td:first-child {{ text-align:left; }}
  th {{ color:var(--ink2); font-weight:600; }}
  code {{ background:#f1f0ec; padding:1px 5px; border-radius:4px; font-size:13px; }}
</style></head><body><main>

<h1>Probes A / B / C — where to spend effort next</h1>
<p class="sub">Measurements over the Pass 1–3 kernel (unchanged; probe code in
<code>bench/</code>). Each verdict is the data's reading, not a plan.</p>

<h2>Probe A — scale &amp; resolution</h2>
<div class="card"><img src="data:image/png;base64,{b64('out/probe_a.png')}"
  alt="scale and thin-feature curves"></div>
<div class="card"><table>
<tr><th>measurement</th><th>value</th></tr>
<tr><td>JIT trace+compile at N = 49 nodes</td><td>{a1['trace_s'][-1]+a1['compile_s'][-1]:.2f} s (linear, ≈ 8 ms/node + 30 ms)</td></tr>
<tr><td>mass-properties eval, R = 15 regions, 48³ grid</td><td>{a2r['eval_ms'][-1]:.0f} ms (linear in R: ≈ 5.3 ms/region)</td></tr>
<tr><td>eval, R = 6, 96³ = 884k points</td><td>{a2g['eval_ms'][-1]:.0f} ms (linear in points: ≈ 0.22 µs/pt·region, CPU f64)</td></tr>
<tr><td>thin-wall recovered thickness floor (t → 0)</td><td>≈ 1.39 τ  (= 2τ·ln2; thin features inflate, never vanish)</td></tr>
<tr><td>thickness trustworthy to ≤ 6 % / ≤ 0.3 %</td><td>t ≥ 2τ / t ≥ 4τ</td></tr>
<tr><td>quadrature (τ = 1.5·dx tied): wall volume error</td><td>27 % at dx = t/1.6 → 0.3 % at dx = t/6.4</td></tr>
</table></div>
<div class="verdict"><b>Verdict: the architecture survives the target scale with a wide margin.</b>
Compile and eval are both linear; at the spec's 10–30 components nothing is close to impractical
(worst measured: 0.51 s first-call, 81 ms/eval on CPU — a GPU only widens the margin).
Extrapolated trigger for the deferred culling/interpreter work: O(R·points) passes ≈ 10⁸
(e.g. ≥ 30 regions on ≥ 96³ inner-loop grids) or graphs of several hundred nodes recompiled
often — neither is in sight for the UAV case. Promote nothing. The real constraint Probe A
found is <i>resolution discipline</i>: features thinner than ~2τ (= 3·dx at the default tie)
are reported fatter than they are — see the Probe C mass row for this biting in practice.</div>

<h2>Probe B — coincident / mating surfaces</h2>
<div class="card"><img src="data:image/png;base64,{b64('out/probe_b.png')}"
  alt="interface sharpness and mass fidelity"></div>
<div class="card"><table>
<tr><th>measurement</th><th>value</th></tr>
<tr><td>ownership 10–90 smear width (coincident faces)</td><td>2.2 τ — set by τ alone, independent of k</td></tr>
<tr><td>void ridge on the mating plane (w<sub>background</sub> peak)</td><td>0.50–0.56 at τ = 0.03, k = 0.08 (should be ~0 inside mated solid)</td></tr>
<tr><td>mass error of the lower-precedence side, fine grid</td><td>−7.7 % floor at k = 0.08; −0.9 % at k = 0.02, τ = 0.01</td></tr>
<tr><td>does grid refinement fix it?</td><td>no — error plateaus at the k-erosion floor (≈ −k·ln2·A<sub>interface</sub>/V)</td></tr>
</table></div>
<div class="verdict"><b>Verdict: ownership stays sharp (2.2τ) but flush mates lose material —
and the loss is a composition-bandwidth effect, not a resolution effect.</b>
The LSE smooth-subtract erodes the lower-precedence side by O(k·ln2) along the whole mating
face, opening a void ridge (background weight up to 0.56) that grid refinement cannot close.
Gap and overlap behave identically to coincident at |δ| ≲ τ — the kernel does not distinguish
them, which is exactly the τ-band semantics the contract promises. Fix class to schedule, not
implemented here: <b>declared-mate sharpening</b> — a per-interface-pair local composition
bandwidth (k→0 or exact max for declared mates), plus <b>exact boolean at export</b> for the
hard τ→0 projection. Until then the working rule is k ≤ 0.02·(feature scale) at mating faces,
which the −0.9 % row shows is acceptable.</div>

<h2>Probe C — end-to-end differentiable sim loop</h2>
<div class="card"><img src="data:image/png;base64,{b64('out/probe_c.png')}"
  alt="sim-loop optimization"></div>
<div class="card"><table>
<tr><th>measurement</th><th>value</th></tr>
<tr><td>chain</td><td>θ → exposure → PoU mass/inertia/COM → 6-DOF quadrotor (RK4 × 250, geometric PD, mixing from rotor positions(θ)) → hover cost</td></tr>
<tr><td>∂J/∂θ: autodiff vs central FD</td><td>[{gc['ad'][0]:.8f}, {gc['ad'][1]:.8f}] — rel. err {gc['rel_err'][0]:.1e}, {gc['rel_err'][1]:.1e}</td></tr>
<tr><td>cost of one value+gradient</td><td>{gc['eval_grad_ms']:.0f} ms (compile {gc['compile_s']:.1f} s, once)</td></tr>
<tr><td>co-design optimization (60 Adam steps)</td><td>cost {opt['cost'][0]:.4f} → {opt['cost'][1]:.4f}; arm length {opt['L'][0]:.3f} → {opt['L'][1]:.3f} m; mass {opt['mass'][0]:.2f} → {opt['mass'][1]:.2f} kg</td></tr>
<tr><td>without actuator saturation in the cost</td><td>arms collapse to 0.02 m — the optimizer exploits the unmodeled limit, gradient stays clean throughout</td></tr>
<tr><td>caught along the way</td><td>quadrature nodes on the geometry's symmetry planes sample the box SDF's kinks with nonzero measure: AD-vs-FD off by ~1e-2; a sub-voxel grid shift restores 1e-10</td></tr>
<tr><td>absolute mass accuracy at dx = 1 cm, τ = 1.5 cm</td><td>5.75 kg vs ≈ 2.6 kg analytic — Probe A's thin-feature inflation biting (arm r = 1.8 cm ≈ τ); gradients are exact for the discretized objective regardless</td></tr>
</table></div>
<div class="verdict"><b>Verdict: current projections suffice for this class of objective — no
FlexiCubes trigger.</b> The gradient runs geometry → mass/inertia/COM/mixing → rollout →
cost and matches FD to 10⁻⁹; one value+grad costs 31 ms, so co-design iteration is cheap.
Everything this loop consumes is volumetric; a differentiable boundary mesh becomes necessary
only when an objective needs surface quantities (aero drag, wetted area, radiation) — leave it
consumer-triggered. Two operational findings to fold into the contract/defaults: (1) shift or
jitter projection grids off the geometry's symmetry planes (sub-voxel, documented in
<code>bench/probe_c_simloop.py</code>); (2) absolute mass calibration needs the Probe A
resolution rule (t ≥ 2τ) — at UAV part scales that means dx ≈ 5 mm grids, which Probe A prices
at ~100–200 ms/eval, still cheap.</div>

</main></body></html>"""

with open("out/probes.html", "w") as f:
    f.write(html)
print("wrote out/probes.html")
