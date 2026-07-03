"""Render the Pass-1 validation geometry + optimization evidence.

Outputs (out/):
  slices.png        2D field slices, region-colored PoU + zero level sets,
                    before vs after the interface-moving optimization
  optimization.png  objective, gradient, radius, per-component masses per iter
  viewer.html       self-contained viewer: interactive 3D segmented geometry
                    (marching cubes per precedence-composed region) + the PNGs
                    + the validation numbers
"""
import base64
import io
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
from matplotlib.colors import to_rgb
from skimage import measure

from geomk.compose import Component, Assembly, make_region_fields, pou_weights
from geomk.dag import GraphBuilder
from geomk.optim import adam
from geomk.projections import GridSpec, make_mass_properties
from geomk.reparam import softplus_inverse
from geomk.topology import topology_signature, TopologyTracker

# ---- palette (validated, see dataviz skill) --------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRIDLINE = "#e3e2de"
C_FUS = "#2a78d6"   # series 1: fuselage
C_WING = "#1baf7a"  # series 2: wing
C_FUS_DK = "#1c5cab"
C_WING_DK = "#0f7a55"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": GRIDLINE, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

R_INIT, R_TARGET = 0.45, 0.62
GRID = GridSpec(lo=(-2.5, -2.3, -0.9), hi=(2.5, 2.3, 0.9), shape=(72, 64, 26))


def build_assembly():
    gb = GraphBuilder()
    fus = gb.capsule((-1.8, 0.0, 0.0), (1.8, 0.0, 0.0), R_INIT)
    wing = gb.box((0.0, 0.0, 0.0), (0.45, 2.0, 0.12))
    graph = gb.build()
    comps = (
        Component("fuselage", fus, density=2.0, precedence=1, intent="structural"),
        Component("wing", wing, density=1.0, precedence=0, intent="aero"),
    )
    return Assembly(graph, comps), int(fus.params[6])


# ---- solve the validation problem, recording everything --------------------
print("solving validation problem ...")
asm, r_idx = build_assembly()
props = jax.jit(make_mass_properties(asm, GRID))
region_fields = make_region_fields(asm)
theta0 = jnp.asarray(asm.graph.theta0)
theta_star = theta0.at[r_idx].set(softplus_inverse(R_TARGET))
target = props(theta_star)


def objective(theta):
    p = props(theta)
    dm = p["component_mass"] - target["component_mass"]
    dI = p["inertia"] - target["inertia"]
    return jnp.sum(dm ** 2) + 0.1 * jnp.sum(dI ** 2)


vg = jax.jit(jax.value_and_grad(objective))
mask = jnp.zeros_like(theta0).at[r_idx].set(1.0)
theta_opt, hist = adam(vg, theta0, mask, lr=0.03, steps=200)

iters = np.arange(len(hist["J"]))
Js = np.array(hist["J"])
gs = np.array([g[r_idx] for g in hist["grad"]])
radii = np.array([float(jax.nn.softplus(t[r_idx])) for t in hist["theta"]])
mass_hist = np.array([np.asarray(props(jnp.asarray(t))["component_mass"])
                      for t in hist["theta"]])

# gradient-check numbers for the report
h = 1e-5
m_wing = lambda th: props(th)["component_mass"][1]
g_wing_ad = float(jax.grad(m_wing)(theta0)[r_idx])
g_wing_fd = float((m_wing(theta0.at[r_idx].add(h)) - m_wing(theta0.at[r_idx].add(-h))) / (2 * h))
g_J_ad = float(jax.grad(objective)(theta0)[r_idx])
g_J_fd = float((objective(theta0.at[r_idx].add(h)) - objective(theta0.at[r_idx].add(-h))) / (2 * h))
p0, pf = props(theta0), props(theta_opt)
tracker = TopologyTracker()
tracker.update(topology_signature(asm, theta0, GRID))
tracker.update(topology_signature(asm, theta_opt, GRID))
r_final = float(jax.nn.softplus(theta_opt[r_idx]))
print(f"  J {Js[0]:.4g} -> {Js[-1]:.4g}   r {R_INIT} -> {r_final:.4f} (target {R_TARGET})")
print(f"  d m_wing/d theta_r: AD {g_wing_ad:.8f}  FD {g_wing_fd:.8f}")
print(f"  topology stamp after optimization: {tracker.stamp}")


# ---- 2D slices --------------------------------------------------------------
def slice_rgb(theta, plane, n=(460, 420)):
    """Soft-PoU region coloring of a coordinate plane + composed fields."""
    if plane == "plan":       # z = 0: x horizontal, y vertical
        u = np.linspace(GRID.lo[0], GRID.hi[0], n[0])
        v = np.linspace(GRID.lo[1], GRID.hi[1], n[1])
        U, V = np.meshgrid(u, v)
        pts = np.stack([U.ravel(), V.ravel(), np.zeros(U.size)], axis=-1)
    else:                     # cross-section x = 0: y horizontal, z vertical
        u = np.linspace(GRID.lo[1], GRID.hi[1], n[0])
        v = np.linspace(GRID.lo[2], GRID.hi[2], n[1])
        U, V = np.meshgrid(u, v)
        pts = np.stack([np.zeros(U.size), U.ravel(), V.ravel()], axis=-1)
    phi = region_fields(theta, jnp.asarray(pts))
    w, _ = pou_weights(phi, GRID.tau)
    w = np.asarray(w).reshape(2, *U.shape)
    phi = np.asarray(phi).reshape(2, *U.shape)
    rgb = np.ones((*U.shape, 3)) * np.array(to_rgb(SURFACE))
    for wi, c in zip(w, (C_FUS, C_WING)):
        rgb += wi[..., None] * (np.array(to_rgb(c)) - np.array(to_rgb(SURFACE)))
    return u, v, np.clip(rgb, 0, 1), phi


def draw_slice(ax, theta, plane, labels):
    u, v, rgb, phi = slice_rgb(theta, plane)
    ax.imshow(rgb, origin="lower", extent=[u[0], u[-1], v[0], v[-1]], aspect="equal")
    ax.contour(u, v, phi[0], levels=[0.0], colors=[C_FUS_DK], linewidths=1.6)
    ax.contour(u, v, phi[1], levels=[0.0], colors=[C_WING_DK], linewidths=1.6)
    if labels:
        if plane == "plan":
            ax.annotate("fuselage", (1.9, 0.0), color=INK, ha="center", va="center", fontsize=9)
            ax.annotate("wing", (0.0, 1.5), color=INK, ha="center", va="center", fontsize=9)
        else:
            ax.annotate("fuselage", (0.0, 0.62), color=INK, ha="center", fontsize=9)
            ax.annotate("wing", (1.35, 0.28), color=INK, ha="center", fontsize=9)
    ax.set_xlabel("x [m]" if plane == "plan" else "y [m]")
    ax.set_ylabel("y [m]" if plane == "plan" else "z [m]")


print("rendering slices ...")
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0),
                         gridspec_kw={"height_ratios": [1, 1]})
for row, (theta, tag, r) in enumerate([(theta0, "before", R_INIT),
                                       (theta_opt, "after", r_final)]):
    draw_slice(axes[row, 0], theta, "plan", labels=(row == 0))
    axes[row, 0].set_title(f"{tag} — plan view (z = 0), r = {r:.3f} m", loc="left")
    draw_slice(axes[row, 1], theta, "xsec", labels=(row == 0))
    axes[row, 1].set_title(f"{tag} — cross-section (x = 0)", loc="left")
fig.suptitle("Precedence-composed segmented field — soft region membership, "
             "zero level sets; fuselage (precedence 1) owns the overlap",
             fontsize=12, x=0.02, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("out/slices.png", dpi=150)
plt.close(fig)


# ---- optimization panels -----------------------------------------------------
print("rendering optimization panels ...")
fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.8))
(axJ, axG), (axR, axM) = axes

axJ.semilogy(iters, Js, color=C_FUS, lw=2)
axJ.set_title("objective  J(θ)", loc="left")
axJ.set_xlabel("iteration")
axJ.grid(True, color=GRIDLINE, lw=0.6)

axG.plot(iters, gs, color=C_FUS, lw=2)
axG.axhline(0, color=GRIDLINE, lw=1)
axG.set_title("gradient  dJ/dθ_r  (through boundary ownership)", loc="left")
axG.set_xlabel("iteration")
axG.grid(True, color=GRIDLINE, lw=0.6)

axR.plot(iters, radii, color=C_FUS, lw=2)
axR.axhline(R_TARGET, color=INK2, lw=1.2, ls="--")
axR.annotate(f"target r* = {R_TARGET}", (iters[-1], R_TARGET), ha="right",
             va="bottom", color=INK2, fontsize=9)
axR.set_title("fuselage radius (softplus-reparameterized)", loc="left")
axR.set_xlabel("iteration")
axR.set_ylabel("r [m]")
axR.grid(True, color=GRIDLINE, lw=0.6)

rel = mass_hist / mass_hist[0]
rel_tgt = np.asarray(target["component_mass"]) / mass_hist[0]
axM.plot(iters, rel[:, 0], color=C_FUS, lw=2)
axM.plot(iters, rel[:, 1], color=C_WING, lw=2)
for j, (name, c) in enumerate([("fuselage", C_FUS), ("wing", C_WING)]):
    axM.annotate(f"{name}  m: {mass_hist[0, j]:.3f} → {mass_hist[-1, j]:.3f} kg",
                 (iters[-1], rel[-1, j]), ha="right", va="bottom",
                 color=INK, fontsize=9)
    axM.axhline(rel_tgt[j], color=c, lw=1, ls=":", alpha=0.7)
axM.axhline(1.0, color=GRIDLINE, lw=1)
axM.set_title("per-component mass m/m(0) — ownership transfer", loc="left")
axM.set_xlabel("iteration")
axM.set_ylabel("m / m(0)")
axM.grid(True, color=GRIDLINE, lw=0.6)

fig.suptitle("Optimizing the interface-moving parameter (fuselage radius), "
             "Adam on one masked entry of the flat θ vector",
             fontsize=12, x=0.02, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("out/optimization.png", dpi=150)
plt.close(fig)


# ---- 3D view -----------------------------------------------------------------
print("rendering 3D meshes ...")
GRID3 = GridSpec(lo=GRID.lo, hi=GRID.hi, shape=(110, 100, 44))
pts3 = jnp.asarray(GRID3.points())


def region_meshes(theta):
    phi = np.asarray(region_fields(theta, pts3)).reshape(2, *GRID3.shape)
    out = []
    for k in range(2):
        verts, faces, _, _ = measure.marching_cubes(phi[k], level=0.0,
                                                    spacing=GRID3.dx)
        verts += np.array(GRID3.lo) + 0.5 * np.array(GRID3.dx)
        out.append((verts, faces))
    return out


def scene_traces(theta, scene, showlegend):
    traces = []
    for (verts, faces), name, color in zip(region_meshes(theta),
                                           ("fuselage", "wing"), (C_FUS, C_WING)):
        traces.append(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color, name=name, showlegend=showlegend, opacity=1.0,
            flatshading=False, scene=scene,
            lighting=dict(ambient=0.45, diffuse=0.7, specular=0.15, roughness=0.7),
            hovertemplate=name + "<extra></extra>",
        ))
    return traces


fig3 = go.Figure(scene_traces(theta0, "scene", True)
                 + scene_traces(theta_opt, "scene2", False))
scene_kw = dict(aspectmode="data",
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                camera=dict(eye=dict(x=1.15, y=1.35, z=0.85)))
fig3.update_layout(
    scene=scene_kw, scene2=scene_kw,
    paper_bgcolor=SURFACE, font=dict(color=INK, size=12),
    margin=dict(l=0, r=0, t=60, b=0), height=460,
    legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02),
    annotations=[
        dict(text=f"before — r = {R_INIT:.3f} m", x=0.22, y=0.98, xref="paper",
             yref="paper", showarrow=False, font=dict(size=13)),
        dict(text=f"after — r = {r_final:.3f} m", x=0.78, y=0.98, xref="paper",
             yref="paper", showarrow=False, font=dict(size=13)),
    ],
)
fig3.update_layout(scene_domain=dict(x=[0.0, 0.5]), scene2_domain=dict(x=[0.5, 1.0]))
plotly_div = fig3.to_html(full_html=False, include_plotlyjs=True,
                          config={"displaylogo": False})


# ---- self-contained HTML viewer ---------------------------------------------
def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


fmt = lambda x: f"{float(x):.6f}"
rows = [
    ("fuselage mass [kg]", fmt(p0['component_mass'][0]), fmt(pf['component_mass'][0]), fmt(target['component_mass'][0])),
    ("wing mass [kg]", fmt(p0['component_mass'][1]), fmt(pf['component_mass'][1]), fmt(target['component_mass'][1])),
    ("total mass [kg]", fmt(p0['total_mass']), fmt(pf['total_mass']), fmt(target['total_mass'])),
    ("inertia Ixx", fmt(p0['inertia'][0, 0]), fmt(pf['inertia'][0, 0]), fmt(target['inertia'][0, 0])),
    ("inertia Iyy", fmt(p0['inertia'][1, 1]), fmt(pf['inertia'][1, 1]), fmt(target['inertia'][1, 1])),
    ("inertia Izz", fmt(p0['inertia'][2, 2]), fmt(pf['inertia'][2, 2]), fmt(target['inertia'][2, 2])),
    ("fuselage radius [m]", f"{R_INIT:.4f}", f"{r_final:.4f}", f"{R_TARGET:.4f}"),
]
table_rows = "\n".join(
    f"<tr><td>{n}</td><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for n, a, b, c in rows)

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>geomk — Pass-1 validation viewer</title>
<style>
  :root {{ --surface:{SURFACE}; --ink:{INK}; --ink2:{INK2}; --line:{GRIDLINE};
           --fus:{C_FUS}; --wing:{C_WING}; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--surface); color:var(--ink);
         font:15px/1.55 system-ui, sans-serif; }}
  main {{ max-width:1080px; margin:0 auto; padding:28px 20px 64px; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  h2 {{ font-size:16px; margin:36px 0 10px; }}
  .sub {{ color:var(--ink2); margin:0 0 20px; }}
  .card {{ border:1px solid var(--line); border-radius:10px; overflow:hidden;
           background:var(--surface); }}
  img {{ max-width:100%; display:block; }}
  table {{ border-collapse:collapse; width:100%; font-size:14px; }}
  th, td {{ text-align:right; padding:7px 12px; border-bottom:1px solid var(--line);
            font-variant-numeric: tabular-nums; }}
  th:first-child, td:first-child {{ text-align:left; }}
  th {{ color:var(--ink2); font-weight:600; }}
  .chip {{ display:inline-block; width:10px; height:10px; border-radius:3px;
           margin-right:6px; vertical-align:baseline; }}
  code {{ background:#f1f0ec; padding:1px 5px; border-radius:4px; font-size:13px; }}
</style></head><body><main>
<h1>Differentiable geometry kernel — Pass-1 validation</h1>
<p class="sub">
  <span class="chip" style="background:var(--fus)"></span>fuselage (ρ = 2.0, precedence 1) ·
  <span class="chip" style="background:var(--wing)"></span>wing (ρ = 1.0, precedence 0) ·
  overlap owned by the fuselage; optimizing its radius moves the shared interface
  and the gradient runs <em>through</em> boundary ownership.
</p>

<h2>3D segmented geometry (interactive — drag to rotate)</h2>
<div class="card">{plotly_div}</div>

<h2>Field slices — soft partition of unity + zero level sets</h2>
<div class="card"><img alt="2D field slices before and after optimization"
  src="data:image/png;base64,{b64('out/slices.png')}"></div>

<h2>Optimization through the interface</h2>
<div class="card"><img alt="objective, gradient, radius and masses over iterations"
  src="data:image/png;base64,{b64('out/optimization.png')}"></div>

<h2>Numbers</h2>
<div class="card"><table>
<tr><th>quantity</th><th>before</th><th>after (optimized)</th><th>target</th></tr>
{table_rows}
</table></div>

<h2>Gradient checks (float64, central FD, h = 1e-5)</h2>
<div class="card"><table>
<tr><th>gradient</th><th>autodiff</th><th>finite difference</th><th>rel. error</th></tr>
<tr><td>d m_wing / d θ_r</td><td>{g_wing_ad:.10f}</td><td>{g_wing_fd:.10f}</td>
    <td>{abs(g_wing_ad-g_wing_fd)/abs(g_wing_fd):.2e}</td></tr>
<tr><td>d J / d θ_r</td><td>{g_J_ad:.10f}</td><td>{g_J_fd:.10f}</td>
    <td>{abs(g_J_ad-g_J_fd)/abs(g_J_fd):.2e}</td></tr>
</table></div>
<p class="sub">d m_wing/d θ_r &lt; 0: growing the fuselage <em>steals</em> wing mass —
the ownership path is differentiable. Objective J: {Js[0]:.4f} → {Js[-1]:.3e}
in {len(Js)} Adam iterations on one masked entry of the flat θ vector.
Topology stamp after optimization: {tracker.stamp}
{"(no topology event — the wing stays split into two panels by the fuselage throughout)"
 if tracker.stamp == 0 else "(a topology event occurred during the optimization)"}.</p>
</main></body></html>"""

with open("out/viewer.html", "w") as f:
    f.write(html)
print("wrote out/slices.png, out/optimization.png, out/viewer.html")
