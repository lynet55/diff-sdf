"""Render the Pass-4 breadth ops (loft / revolve / extrude / lattice) and the
6-DOF co-design craft as interactive 3D CAD views.

Outputs:
  out/loft_morph.png   loft profile morph: z-slices circle -> rectangle
  out/breadth.html     self-contained viewer: breadth-op gallery, the loft in
                       depth, the probe-C/D quadrotor before/after co-design,
                       and a quadrotor rebuilt from the new ops
"""
import base64
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

from bench._style import SERIES, SURFACE, INK, INK2, GRIDLINE
from geomk.dag import GraphBuilder
from geomk.compose import Component, Assembly, make_region_fields, pou_weights
from geomk.evaluate import make_field
from geomk.exposure import Exposure
from geomk.projections import GridSpec
from geomk.reparam import softplus_inverse

C_BLUE, C_AQUA, C_YELLOW = SERIES[0], SERIES[1], SERIES[2]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": GRIDLINE, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

LIGHTING = dict(ambient=0.45, diffuse=0.7, specular=0.15, roughness=0.7)
SCENE = dict(aspectmode="data",
             xaxis=dict(visible=False), yaxis=dict(visible=False),
             zaxis=dict(visible=False))


def mc_mesh(field_fn, theta, grid):
    """Marching-cubes mesh of a field's zero set on a GridSpec."""
    d = np.asarray(field_fn(theta, jnp.asarray(grid.points()))).reshape(grid.shape)
    verts, faces, _, _ = measure.marching_cubes(d, 0.0, spacing=grid.dx)
    verts += (np.array(grid.lo)
              + np.array(grid.dx) * (0.5 + np.array(grid.jitter)))
    return verts, faces


def mesh_trace(verts, faces, color, name, scene="scene", showlegend=False):
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, name=name, showlegend=showlegend, scene=scene,
        flatshading=False, lighting=LIGHTING,
        hovertemplate=name + "<extra></extra>")


def fig_layout(fig, height=420, **kw):
    fig.update_layout(paper_bgcolor=SURFACE, font=dict(color=INK, size=12),
                      margin=dict(l=0, r=0, t=34, b=0), height=height, **kw)
    return fig


HTML_FIGS = []          # (title, subtitle, div) in page order
_PLOTLY_INCLUDED = False


def add_fig(fig, title, subtitle):
    global _PLOTLY_INCLUDED
    div = fig.to_html(full_html=False,
                      include_plotlyjs=(not _PLOTLY_INCLUDED),
                      config={"displaylogo": False})
    _PLOTLY_INCLUDED = True
    HTML_FIGS.append((title, subtitle, div))


# =============================================================== loft in depth
print("loft: duct (circle -> rectangle) ...")
H_DUCT = 0.5
gb = GraphBuilder()
prof_circle = gb.sphere((0.0, 0.0, 0.0), 0.42)
prof_rect = gb.box((0.0, 0.0, 0.0), (0.50, 0.26, 0.50))
duct = gb.loft(prof_circle, prof_rect, H_DUCT)
g_duct = gb.build()
th_duct = jnp.asarray(g_duct.theta0)
f_duct = make_field(g_duct, duct)

grid_duct = GridSpec(lo=(-0.62, -0.62, -0.58), hi=(0.62, 0.62, 0.58),
                     shape=(110, 110, 104))
fig = go.Figure([mesh_trace(*mc_mesh(f_duct, th_duct, grid_duct), C_BLUE, "loft")])
# translucent profile frames at five stations
u = np.linspace(-0.62, 0.62, 240)
U, V = np.meshgrid(u, u)
for zc in (-H_DUCT, -H_DUCT / 2, 0.0, H_DUCT / 2, H_DUCT):
    pl = np.stack([U.ravel(), V.ravel(), np.full(U.size, zc)], axis=-1)
    d2 = np.asarray(f_duct(th_duct, jnp.asarray(pl))).reshape(U.shape)
    cs = measure.find_contours(d2, 0.0)
    for c in cs:
        xs = np.interp(c[:, 1], np.arange(len(u)), u)
        ys = np.interp(c[:, 0], np.arange(len(u)), u)
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=np.full(len(xs), zc), mode="lines",
            line=dict(color=INK2, width=3), showlegend=False,
            hoverinfo="skip"))
fig_layout(fig, height=480, scene=dict(
    **SCENE, camera=dict(eye=dict(x=1.5, y=1.1, z=0.75))))
add_fig(fig, "The loft op, in depth",
        "loft(circle, rectangle, h): the two children's z = 0 field slices are "
        "linearly blended in t = (z + h)/2h and capped — one differentiable op, "
        "profiles are ordinary subtrees (here a sphere slice and a box slice). "
        "Rings mark the blended profile at five stations. Sign is exact; "
        "distance is honestly flagged non-metric (invariant 3).")

# profile-morph slices PNG
print("loft: profile morph slices ...")
fig2, axes = plt.subplots(1, 5, figsize=(12.5, 2.9))
for ax, tfrac in zip(axes, (0.0, 0.25, 0.5, 0.75, 1.0)):
    zc = -H_DUCT + 2 * H_DUCT * tfrac
    pl = np.stack([U.ravel(), V.ravel(), np.full(U.size, zc)], axis=-1)
    d2 = np.asarray(f_duct(th_duct, jnp.asarray(pl))).reshape(U.shape)
    occ = 1.0 / (1.0 + np.exp(d2 / 0.02))
    rgb = (np.array(to_rgb(SURFACE)) +
           occ[..., None] * (np.array(to_rgb(C_BLUE)) - np.array(to_rgb(SURFACE))))
    ax.imshow(rgb, origin="lower", extent=[u[0], u[-1], u[0], u[-1]])
    ax.contour(u, u, d2, levels=[0.0], colors=[INK2], linewidths=1.4)
    ax.set_title(f"t = {tfrac:.2f}   (z = {zc:+.2f})", loc="left", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig2.suptitle("loft profile morph — the blended field's zero set from the circle "
              "(t = 0) to the rectangle (t = 1)", fontsize=12, x=0.02, ha="left")
fig2.tight_layout(rect=[0, 0, 1, 0.90])
fig2.savefig("out/loft_morph.png", dpi=150)
plt.close(fig2)

# =========================================================== breadth gallery
print("gallery: revolve vase, extrude clover, lattice-cored block ...")
# revolve: bulged vase profile (union of disks along x), revolved about x,
# stood upright with rigid
gb = GraphBuilder()
prof = gb.smooth_union(
    gb.sphere((-0.30, 0.16, 0.0), 0.16),
    gb.sphere((0.00, 0.26, 0.0), 0.22),
    gb.sphere((0.33, 0.12, 0.0), 0.09), k=0.07)
vase = gb.rigid(gb.revolve(prof), rotvec=(0.0, -np.pi / 2, 0.0))
g_vase = gb.build()
grid_vase = GridSpec(lo=(-0.55, -0.55, -0.5), hi=(0.55, 0.55, 0.5),
                     shape=(96, 96, 88))
mesh_vase = mc_mesh(make_field(g_vase, vase), jnp.asarray(g_vase.theta0), grid_vase)

# extrude: clover of three disks
gb = GraphBuilder()
ang = [np.pi / 2 + i * 2 * np.pi / 3 for i in range(3)]
clover_prof = gb.smooth_union(
    *[gb.sphere((0.2 * np.cos(a), 0.2 * np.sin(a), 0.0), 0.22) for a in ang],
    k=0.05)
clover = gb.extrude(clover_prof, 0.11)
g_clover = gb.build()
grid_clover = GridSpec(lo=(-0.52, -0.52, -0.2), hi=(0.52, 0.52, 0.2),
                       shape=(96, 96, 40))
mesh_clover = mc_mesh(make_field(g_clover, clover), jnp.asarray(g_clover.theta0),
                      grid_clover)

# lattice: block drilled by the strut lattice (smooth_subtract keeps it legal:
# lattice is a dirty generator, never offset/shelled)
gb = GraphBuilder()
block = gb.smooth_subtract(gb.box((0, 0, 0), (0.55, 0.38, 0.17)),
                           gb.lattice(0.27, 0.055), k=0.015)
g_block = gb.build()
grid_block = GridSpec(lo=(-0.62, -0.45, -0.24), hi=(0.62, 0.45, 0.24),
                      shape=(120, 88, 48))
mesh_block = mc_mesh(make_field(g_block, block), jnp.asarray(g_block.theta0),
                     grid_block)

fig = go.Figure(
    [mesh_trace(*mesh_vase, C_BLUE, "revolve", "scene"),
     mesh_trace(*mesh_clover, C_AQUA, "extrude", "scene2"),
     mesh_trace(*mesh_block, C_YELLOW, "lattice ∖ box", "scene3")])
cam = dict(eye=dict(x=1.45, y=1.2, z=0.9))
fig_layout(fig, height=380,
           scene=dict(**SCENE, camera=cam, domain=dict(x=[0.0, 0.33])),
           scene2=dict(**SCENE, camera=cam, domain=dict(x=[0.34, 0.66])),
           scene3=dict(**SCENE, camera=cam, domain=dict(x=[0.67, 1.0])),
           annotations=[
               dict(text="revolve — vase (profile: 3 blended disks)", x=0.15,
                    y=0.99, xref="paper", yref="paper", showarrow=False),
               dict(text="extrude — clover plate", x=0.5, y=0.99,
                    xref="paper", yref="paper", showarrow=False),
               dict(text="smooth_subtract(box, lattice)", x=0.85, y=0.99,
                    xref="paper", yref="paper", showarrow=False)])
add_fig(fig, "The rest of the breadth set",
        "revolve and extrude preserve a clean profile's metric; loft and "
        "lattice are honestly dirty — offset/shell refuse them without an "
        "explicit redistance (invariant 3). The lattice is an infinite "
        "generator; a finite part composes it with a domain shape.")

# ===================================================== the 6-DOF craft (C/D)
print("craft: probe-C/D quadrotor before/after co-design ...")
L0, BODY_HX0 = 0.16, 0.07
L_OPT, BODY_HX_OPT = 0.21013742351636228, 0.03513658063975797
DIRS = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]]) / np.sqrt(2)


def build_craft():
    gb = GraphBuilder()
    body = gb.box((0, 0, 0), (BODY_HX0, BODY_HX0, 0.03))
    arms, hubs = [], []
    for d in DIRS:
        arms.append(gb.capsule((0, 0, 0), tuple(L0 * d), 0.018))
        hubs.append(gb.sphere(tuple(L0 * d), 0.032))
    arm_u = gb.smooth_union(*arms, k=0.02)
    hub_u = gb.smooth_union(*hubs, k=0.02)
    graph = gb.build()
    asm = Assembly(graph, (
        Component("hubs", hub_u, density=2600.0, precedence=2, intent="structural"),
        Component("body", body, density=700.0, precedence=1, intent="structural"),
        Component("arms", arm_u, density=400.0, precedence=0, intent="structural"),
    ), k_compose=0.01)
    ex = Exposure(graph)
    zL = ex.macro("arm_length", init=L0)
    for arm, hub, d in zip(arms, hubs, DIRS):
        for axis in range(3):
            if abs(d[axis]) > 1e-12:
                ex.drive(arm.params[3 + axis], zL, scale=float(d[axis]))
                ex.drive(hub.params[axis], zL, scale=float(d[axis]))
    zB = ex.expose(body.params[3], "body_hx")
    ex.tie(body.params[4], zB)
    return asm, ex.build(), (zL, zB)


asm_c, emap_c, (ZL, ZB) = build_craft()
region_fields_c = make_region_fields(asm_c)
CRAFT_COLORS = {"hubs": C_YELLOW, "body": C_BLUE, "arms": C_AQUA}


def craft_traces(z, scene, showlegend):
    ext = float(z[ZL]) + 0.06
    grid = GridSpec(lo=(-ext, -ext, -0.075), hi=(ext, ext, 0.075),
                    shape=(120, 120, 26))
    theta = emap_c.theta(z)
    phi = np.asarray(region_fields_c(theta, jnp.asarray(grid.points())))
    traces = []
    for i, comp in enumerate(asm_c.components):
        d = phi[i].reshape(grid.shape)
        verts, faces, _, _ = measure.marching_cubes(d, 0.0, spacing=grid.dx)
        verts += (np.array(grid.lo)
                  + np.array(grid.dx) * (0.5 + np.array(grid.jitter)))
        traces.append(mesh_trace(verts, faces, CRAFT_COLORS[comp.name],
                                 comp.name, scene, showlegend))
    return traces


z_before = jnp.asarray(emap_c.z0)
z_after = z_before.at[ZL].set(L_OPT).at[ZB].set(float(softplus_inverse(BODY_HX_OPT)))
fig = go.Figure(craft_traces(z_before, "scene", True)
                + craft_traces(z_after, "scene2", False))
cam = dict(eye=dict(x=1.7, y=2.0, z=1.3))
fig_layout(fig, height=440,
           scene=dict(**SCENE, camera=cam, domain=dict(x=[0.0, 0.5])),
           scene2=dict(**SCENE, camera=cam, domain=dict(x=[0.5, 1.0])),
           legend=dict(orientation="h", x=0.5, xanchor="center", y=1.04),
           annotations=[
               dict(text=f"before — L = {L0:.3f} m, body = {BODY_HX0*2:.3f} m, "
                         f"2.41 kg", x=0.22, y=0.97, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=13)),
               dict(text=f"after — L = {L_OPT:.3f} m, body = {BODY_HX_OPT*2:.3f} m, "
                         f"2.07 kg", x=0.78, y=0.97, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=13))])
add_fig(fig, "The 6-DOF co-design craft (probes C/D)",
        "The assembly the differentiable sim loop optimizes: hubs / body / arms "
        "with precedence composition, arm length and body width driven by two "
        "exposure macros. Under the honest saturating plant (probe D) the "
        "optimum is interior: the arms GROW 0.160 → 0.210 m to buy "
        "disturbance-rejection authority while the body slims for mass, "
        "cost 0.162 → 0.110 in 120 Adam steps, gradient through mass, inertia, "
        "COM and the mixing matrix at once.")

# ====================================== remix: the craft from the breadth ops
print("craft remix with breadth ops ...")
L_RMX = 0.20
gb = GraphBuilder()
# fuselage: loft from circular nose profile to rectangular tail profile,
# lofted along z then laid down the x-axis
fus_prof0 = gb.sphere((0.0, 0.0, 0.0), 0.052)
fus_prof1 = gb.box((0.0, 0.0, 0.0), (0.055, 0.03, 0.2))
fus = gb.rigid(gb.loft(fus_prof0, fus_prof1, 0.14), rotvec=(0.0, np.pi / 2, 0.0))
# nacelles: revolved teardrop profile (two blended disks), motor axis vertical
nacs = []
for d in DIRS:
    prof = gb.smooth_union(gb.sphere((-0.012, 0.0, 0.0), 0.030),
                           gb.sphere((0.030, 0.0, 0.0), 0.016), k=0.02)
    nacs.append(gb.rigid(gb.revolve(prof),
                         translation=(L_RMX * d[0], L_RMX * d[1], 0.012),
                         rotvec=(0.0, -np.pi / 2, 0.0)))
nac_u = gb.smooth_union(*nacs, k=0.02)
# arms: capsules out to the nacelles
arms = [gb.capsule((0.0, 0.0, 0.0), tuple(L_RMX * d), 0.014) for d in DIRS]
arm_u = gb.smooth_union(*arms, k=0.015)
g_rmx = gb.build()
asm_rmx = Assembly(g_rmx, (
    Component("nacelles", nac_u, density=2600.0, precedence=2, intent="structural"),
    Component("fuselage", fus, density=700.0, precedence=1, intent="structural"),
    Component("arms", arm_u, density=400.0, precedence=0, intent="structural"),
), k_compose=0.01)
region_fields_rmx = make_region_fields(asm_rmx)
grid_rmx = GridSpec(lo=(-0.27, -0.27, -0.08), hi=(0.27, 0.27, 0.09),
                    shape=(130, 130, 34))
phi = np.asarray(region_fields_rmx(jnp.asarray(g_rmx.theta0),
                                   jnp.asarray(grid_rmx.points())))
RMX_COLORS = {"nacelles": C_YELLOW, "fuselage": C_BLUE, "arms": C_AQUA}
traces = []
for i, comp in enumerate(asm_rmx.components):
    d = phi[i].reshape(grid_rmx.shape)
    verts, faces, _, _ = measure.marching_cubes(d, 0.0, spacing=grid_rmx.dx)
    verts += (np.array(grid_rmx.lo)
              + np.array(grid_rmx.dx) * (0.5 + np.array(grid_rmx.jitter)))
    traces.append(mesh_trace(verts, faces, RMX_COLORS[comp.name], comp.name,
                             "scene", True))
fig = go.Figure(traces)
fig_layout(fig, height=480,
           scene=dict(**SCENE, camera=dict(eye=dict(x=1.35, y=1.75, z=1.0))),
           legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02))
add_fig(fig, "Remix — the same craft authored with the breadth ops",
        "Lofted fuselage (circular nose profile → rectangular tail profile), "
        "revolved teardrop motor nacelles, capsule arms — all still one "
        "data-first DAG with precedence composition, so the whole thing stays "
        "differentiable and exposure-drivable exactly like the box-and-sphere "
        "version above.")

# ==================================================================== page
def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


sections = []
for i, (title, subtitle, div) in enumerate(HTML_FIGS):
    sections.append(f"<h2>{title}</h2>\n<p class='sub'>{subtitle}</p>\n"
                    f"<div class='card'>{div}</div>")
    if i == 0:
        sections.append(
            "<div class='card'><img alt='loft profile morph slices' "
            f"src='data:image/png;base64,{b64('out/loft_morph.png')}'></div>")

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>geomk — breadth ops &amp; the 6-DOF CAD</title>
<style>
  :root {{ --surface:{SURFACE}; --ink:{INK}; --ink2:{INK2}; --line:{GRIDLINE}; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--surface); color:var(--ink);
         font:15px/1.55 system-ui,sans-serif; }}
  main {{ max-width:1080px; margin:0 auto; padding:28px 20px 64px; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  h2 {{ font-size:17px; margin:40px 0 6px; }}
  .sub {{ color:var(--ink2); margin:0 0 14px; }}
  .card {{ border:1px solid var(--line); border-radius:10px; overflow:hidden;
           margin:12px 0; }}
  img {{ max-width:100%; display:block; }}
</style></head><body><main>
<h1>Breadth ops &amp; the 6-DOF CAD</h1>
<p class="sub">Everything below is the kernel's own field evaluated on a grid
and marching-cubed — the actual signed-implicit geometry, not an illustration.
Drag any view to rotate.</p>
{chr(10).join(sections)}
</main></body></html>"""

with open("out/breadth.html", "w") as f:
    f.write(html)
print("wrote out/loft_morph.png, out/breadth.html")
