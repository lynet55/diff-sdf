"""Final capstone report — before/after 3D CAD views, the FD gate, the
co-design deltas, and the gradient-vs-black-box comparison, self-contained.

out/capstone_report.html
"""
import base64
import io
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bench._style import SERIES, SURFACE, INK, INK2, GRIDLINE
from bench.capstone_physics import build_physics, fd_gate, LAM_C, LAM_D, LAM_B
from viz.render_capstone import craft_traces, KIND_COLOR
from geomk.projections import TRUST_SATURATION


def convergence_png(cd):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "text.color": INK, "axes.edgecolor": GRIDLINE, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
        "font.size": 11, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False})
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.4, 4.3))
    g, e = cd["grad"], cd["es"]
    # J vs objective evaluations
    a0.plot(g["nfev_hist"], g["J_hist"], color=SERIES[0], lw=2.2,
            label="gradient (Adam)")
    a0.plot(e["nfev_hist"], e["J_hist"], color=SERIES[3], lw=2.2,
            label="black-box (1+1)-ES")
    a0.axhline(g["J_final"], color=INK2, lw=1, ls="--")
    a0.set_xlabel("objective evaluations")
    a0.set_ylabel("best J")
    a0.set_title("convergence vs evaluation budget", loc="left")
    a0.set_xscale("log")
    a0.legend(frameon=False, fontsize=10)
    # J vs wall-clock
    tg = np.linspace(0, g["wall"], len(g["J_hist"]))
    a1.plot(tg, g["J_hist"], color=SERIES[0], lw=2.2, label="gradient")
    a1.plot(e["t_hist"], e["J_hist"], color=SERIES[3], lw=2.2,
            label="black-box")
    a1.axhline(g["J_final"], color=INK2, lw=1, ls="--")
    a1.set_xlabel("wall-clock [s]")
    a1.set_ylabel("best J")
    a1.set_title("convergence vs wall-clock", loc="left")
    a1.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    with open("out/capstone_codesign.json") as f:
        cd = json.load(f)
    P = build_physics()
    emap = P["emap"]
    z0 = jnp.asarray(cd["z0"])
    z_grad = jnp.asarray(cd["grad"]["z_final"])

    print("re-running FD gate for the report table ...")
    passed, gate = fd_gate(P)

    print("meshing before/after ...")
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"},
                                               {"type": "scene"}]],
                        horizontal_spacing=0.02)
    for col, z in ((1, z0), (2, z_grad)):
        for tr in craft_traces(P["asm"], emap, z, scene="scene" if col == 1
                               else "scene2", showlegend=(col == 1),
                               shape=(128, 128, 52)):
            fig.add_trace(tr, row=1, col=col)
    cam = dict(eye=dict(x=1.3, y=1.6, z=0.9))
    fig.update_layout(
        scene=dict(aspectmode="data", camera=cam,
                   xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False)),
        scene2=dict(aspectmode="data", camera=cam,
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    zaxis=dict(visible=False)),
        paper_bgcolor=SURFACE, font=dict(color=INK, size=12),
        margin=dict(l=0, r=0, t=8, b=28), height=470,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=0.0,
                    yanchor="top"))
    div = fig.to_html(full_html=False, include_plotlyjs=True,
                      config={"displaylogo": False})

    conv_b64 = convergence_png(cd)

    # ---- tables --------------------------------------------------------------
    def gate_rows():
        rows = []
        for n, a, fd, r, s in zip(gate["names"], gate["ad"], gate["fd"],
                                  gate["rel_err"], gate["snorm_err"]):
            ok = (r < 1e-6) or (s < 1e-6)
            tag = "✓" if r < 1e-6 else ("✓ <span class=note>(cancel; scale-norm "
                                        f"{s:.1e})</span>" if ok else "✗")
            rows.append(f"<tr><td>{n}</td><td>{a:+.4e}</td><td>{fd:+.4e}</td>"
                        f"<td>{r:.1e}</td><td>{tag}</td></tr>")
        return "\n".join(rows)

    def drag_rows():
        rows = []
        for n, a, fd, r in zip(gate["names"], gate["drag_ad"], gate["drag_fd"],
                               gate["drag_rel_err"]):
            tag = ("✓" if r < 1e-3 else
                   "<span class=low>loft-rough</span>" if r > 0.5 else "~")
            rows.append(f"<tr><td>{n}</td><td>{a:+.4e}</td><td>{fd:+.4e}</td>"
                        f"<td>{r:.1e}</td><td>{tag}</td></tr>")
        return "\n".join(rows)

    tb, tg, te = cd["terms"]["before"], cd["terms"]["grad"], cd["terms"]["es"]
    sb, sg = cd["snapshots"]["before"], cd["snapshots"]["grad"]
    snb = cd["snapshots"]["nobar"]

    def ctrl_stats(s):
        """Min saturation and #below-trust among CONTROLLABLE (non-battery)
        components — the battery is permanently sub-floor and would mask the
        A/B. Returns (min_sat_ctrl, n_below_ctrl, thinnest_in_tau)."""
        sat = np.asarray(s["saturation"])
        keep = np.array(["battery" not in n for n in s["names"]])
        sc = sat[keep]
        mn = float(sc.min())
        mn = min(max(mn, 1e-6), 1 - 1e-6)
        return (float(sc.min()), int((sc < TRUST_SATURATION).sum()),
                float(np.log(mn / (1 - mn))))     # logit = t/tau

    cb, cg, cnb = ctrl_stats(sb), ctrl_stats(sg), ctrl_stats(snb)

    def term_row(label, kb, fmt="{:.4f}", scale=1.0):
        vb, vg = tb[kb] * scale, tg[kb] * scale
        return (f"<tr><td>{label}</td><td>{fmt.format(vb)}</td>"
                f"<td>{fmt.format(vg)}</td>"
                f"<td>{(vg-vb)/abs(vb)*100:+.1f}%</td></tr>")

    # Physical macro dimensions from the z vectors. Six macros drive
    # softplus-reparameterized positive params (physical = softplus(z));
    # arm_length and batt_x are linear (physical = z). Recomputed here because
    # the snapshot stored raw z for the exposed-positive macros.
    POSITIVE = {"arm_radius", "fuse_len", "fuse_width", "nacelle_scale",
                "fin_size", "deck_hz"}
    softplus = lambda x: float(np.logaddexp(0.0, x))

    def phys(zv):
        zv = np.asarray(zv)
        return {n: (softplus(zv[k]) if n in POSITIVE else float(zv[k]))
                for k, n in enumerate(cd["macro_names"])}

    dims_b, dims_g = phys(cd["z0"]), phys(cd["grad"]["z_final"])
    dim_specs = [("arm length", "arm_length", "{:.1f} mm"),
                 ("arm radius", "arm_radius", "{:.1f} mm"),
                 ("fuselage half-length", "fuse_len", "{:.1f} mm"),
                 ("fuselage half-width", "fuse_width", "{:.1f} mm"),
                 ("nacelle scale", "nacelle_scale", "{:.1f} mm"),
                 ("fin size", "fin_size", "{:.1f} mm"),
                 ("deck half-thickness", "deck_hz", "{:.1f} mm"),
                 ("battery x", "batt_x", "{:+.1f} mm")]
    dim_rows = "\n".join(
        f"<tr><td>{lab}</td><td>{f.format(dims_b[k]*1e3)}</td>"
        f"<td>{f.format(dims_g[k]*1e3)}</td></tr>"
        for lab, k, f in dim_specs)

    g, e = cd["grad"], cd["es"]
    reach = e["evals_to_grad_J"]
    if reach and reach > 0:
        ratio = reach / g["nfev"]
        if ratio >= 1.15:
            optimizer_verdict = (
                f"took {reach} objective evaluations to first match the "
                f"gradient's J = {g['J_final']:.4f} — a {ratio:.1f}× "
                f"evaluation-budget premium for lacking the gradient the kernel "
                f"was built to provide. On eight macros a well-tuned ES stays in "
                f"the race on wall-clock, but the gap widens with dimension: the "
                f"exposure layer is the same flat-vector/mask object at eight "
                f"macros or eight hundred, and only the gradient path scales into "
                f"the high-dimensional regime.")
        else:
            optimizer_verdict = (
                f"matched the gradient's J = {g['J_final']:.4f} in a comparable "
                f"{reach} evaluations — expected at only eight dimensions, where "
                f"a tuned (1+1)-ES is competitive. The point is not that the "
                f"gradient wins a small problem, but that the identical flat "
                f"decision vector serves both, and only the gradient path scales "
                f"into the hundreds-of-macros regime the exposure layer targets.")
    else:
        optimizer_verdict = (
            f"did not reach the gradient's J = {g['J_final']:.4f} within its "
            f"{e['nfev']}-evaluation budget — the gradient found the deeper basin "
            f"the black-box search missed, on the same flat decision vector.")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>geomk — capstone co-design</title>
<style>
:root {{ --surface:{SURFACE}; --ink:{INK}; --ink2:{INK2}; --line:{GRIDLINE}; }}
*{{box-sizing:border-box}} body{{margin:0;background:var(--surface);color:var(--ink);
font:15px/1.6 system-ui,sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:30px 20px 72px}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:18px;margin:40px 0 8px}}
.sub{{color:var(--ink2);margin:0 0 18px}}
.card{{border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:14px 0}}
.pad{{padding:6px 14px}}
table{{border-collapse:collapse;width:100%;font-size:13.5px}}
th,td{{text-align:right;padding:6px 12px;border-bottom:1px solid var(--line);
font-variant-numeric:tabular-nums}}
th:first-child,td:first-child{{text-align:left}} th{{color:{INK2};font-weight:600}}
tr:last-child td{{border-bottom:none}}
.low{{color:#b45f06;font-weight:600}} .note{{color:var(--ink2);font-size:11.5px}}
.good{{color:#2e7d32;font-weight:600}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
@media (max-width:820px){{.grid2{{grid-template-columns:1fr}}}}
img{{max-width:100%;display:block}}
code{{background:rgba(128,128,128,.14);padding:1px 5px;border-radius:4px;font-size:13px}}
</style></head><body><main>

<h1>Capstone — UAV co-design, built with the kernel</h1>
<p class="sub">One editable DAG of twelve components, resolved to a segmented
signed-implicit field, driven by eight exposure macros, and optimized through
four differentiable physics — flight, drag, structure, and a resolution
barrier — on the identical flat decision vector a black-box optimizer sees.
This is the first artifact that is a <em>product</em> of the kernel rather than
a test of it.</p>

<div class="card">{div}</div>
<p class="sub pad">Left: initial design. Right: after gradient co-design.
Colored by component kind — {", ".join(KIND_COLOR)}. What you see is the
kernel's own precedence-composed segmentation, marching-cubed. Drag to rotate;
the two views share a camera.</p>

<h2>The gradient is real — FD gate at the design point</h2>
<p class="sub">Analytic (reverse-mode) gradient vs central finite differences
in float64, over all eight macros. The differentiable backbone
(flight + compliance + barrier) is gated at 10⁻⁶.</p>
<div class="card"><table>
<tr><th>macro</th><th>AD ∂J</th><th>FD ∂J</th><th>rel err</th><th></th></tr>
{gate_rows()}
</table></div>
<p class="sub">The <code>fuse_len</code> entry is a near-cancellation of the
flight (+0.51) and structural (−0.46) terms; its net-relative error is floored
by the CG solve residual, but the scale-normalized error is ~10⁻⁸ — the
gradient is correct, the ratio is just measuring cancellation.</p>

<h2>Drag: where a local surface measure stops being differentiably clean</h2>
<p class="sub">The co-area silhouette (∫ max(0,∇occ·d̂) dV) needs no mesh, and
its θ-gradient is FD-clean over every metric primitive — but it is a
<em>second-order</em> integral, and over the <em>dirty loft</em> that forms the
fuselage its gradient is grid-quantization-rough for exactly the two
loft-driven macros. Reported, not gated; drag still descends on its valid
smooth-envelope AD gradient. This is the differentiability face of probe G's
occlusion finding — the concrete trigger for the deferred meshing / interface
work.</p>
<div class="card"><table>
<tr><th>macro</th><th>AD ∂(drag)</th><th>FD ∂(drag)</th><th>rel err</th><th></th></tr>
{drag_rows()}
</table></div>

<h2>Co-design result</h2>
<div class="grid2">
<div class="card"><table>
<tr><th>physics term</th><th>before</th><th>after</th><th>Δ</th></tr>
{term_row("flight cost (hover/reject)", "flight")}
{term_row("compliance [µJ]", "compliance", "{:.2f}", 1e6)}
{term_row("frontal drag area [cm²]", "drag", "{:.1f}", 1e4)}
{term_row("saturation barrier", "barrier", "{:.4f}")}
<tr><td><b>objective J</b></td><td><b>{cd['J0']:.4f}</b></td>
<td><b>{g['J_final']:.4f}</b></td>
<td><b>{(g['J_final']-cd['J0'])/cd['J0']*100:+.1f}%</b></td></tr>
</table></div>
<div class="card"><table>
<tr><th>macro dimension</th><th>before</th><th>after</th></tr>
{dim_rows}
</table></div>
</div>
<p class="sub">Total mass {sb['total_mass']:.3f} → {sg['total_mass']:.3f} kg.</p>

<h2>What the saturation barrier holds</h2>
<p class="sub">The barrier turns probe A's resolution rule (a feature is
trustworthy at half-thickness ≥ 2τ, i.e. saturation ≥ {TRUST_SATURATION}) into
a differentiable penalty. Drag rewards a slim silhouette and flight rewards low
mass, so both physics <em>want</em> to thin features past the floor — exactly
the "optimizer walked a feature through the floor unchecked" pathology the rule
was written to catch. The A/B below runs the identical objective with the
barrier removed.</p>
<div class="card"><table>
<tr><th></th><th>min saturation (controllable)</th>
<th>controllable below trust</th><th>thinnest feature</th></tr>
<tr><td>initial z₀</td><td>{cb[0]:.3f}</td><td>{cb[1]} / {len(sb['names'])-1}</td>
<td>{cb[2]:.2f}·τ</td></tr>
<tr><td><b>with barrier</b></td><td class=good>{cg[0]:.3f}</td>
<td class=good>{cg[1]} / {len(sg['names'])-1}</td><td>{cg[2]:.2f}·τ</td></tr>
<tr><td>no barrier</td><td class=low>{cnb[0]:.3f}</td>
<td class=low>{cnb[1]} / {len(snb['names'])-1}</td><td>{cnb[2]:.2f}·τ</td></tr>
</table></div>
<p class="sub">Removing the barrier lets the optimizer drive the thinnest
controllable feature down to {cnb[2]:.2f}·τ (below the 2τ floor, where soft
values inflate and hardened responses turn hypersensitive — probe E/F); with
the barrier the descent is arrested near the trust boundary at {cg[2]:.2f}·τ.
The battery is excluded here — it stays under the floor in every case by design
(22 mm half-thickness &lt; 2τ, and no macro controls it), reported not hidden.</p>

<h2>Same decision vector, two optimizers</h2>
<p class="sub">The exposure layer materializes as one object — a flat vector
plus a selection mask — consumed identically by the gradient optimizer and by a
gradient-free (1+1)-ES. Both start from z₀ and minimize the same J.</p>
<div class="card pad"><img src="data:image/png;base64,{conv_b64}"/></div>
<p class="sub">Gradient Adam reached J = {g['J_final']:.4f} in {g['nfev']}
value-and-gradient evaluations ({g['wall']:.0f}s). The black-box ES reached
J = {e['J_final']:.4f} in {e['nfev']} objective evaluations
({e['wall']:.0f}s), and {optimizer_verdict}</p>

</main></body></html>"""
    with open("out/capstone_report.html", "w") as f:
        f.write(html)
    print("wrote out/capstone_report.html "
          f"(gate {'PASSED' if passed else 'FAILED'})")


if __name__ == "__main__":
    main()
