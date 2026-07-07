"""Render the capstone craft as an interactive 3D CAD view for inspection.

out/capstone.html — marching-cubes meshes of every precedence-composed
region (the kernel's own segmentation), colored by component kind, with the
mass/diagnostics table. Pass --z to render a specific macro vector (used
again after the co-design for the before/after pair).
"""
import base64
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, ".")

import plotly.graph_objects as go
from skimage import measure

from bench._style import SERIES, SURFACE, INK, INK2, GRIDLINE
from bench.capstone_craft import build_capstone
from geomk.compose import make_region_fields
from geomk.projections import (GridSpec, make_mass_properties,
                               make_resolution_diagnostics, TRUST_SATURATION)

KIND_COLOR = {
    "nacelle": SERIES[2], "fuselage": SERIES[0], "arm": SERIES[1],
    "deck": SERIES[4], "battery": SERIES[5], "fin": SERIES[3],
}
LIGHTING = dict(ambient=0.45, diffuse=0.7, specular=0.15, roughness=0.7)
MASS_GRID = GridSpec(lo=(-0.36, -0.36, -0.12), hi=(0.36, 0.36, 0.16),
                     shape=(72, 72, 28))


def kind_of(name):
    return name.split("_")[0] if name.split("_")[0] in KIND_COLOR else name


def craft_traces(asm, emap, z, scene="scene", showlegend=True,
                 shape=(150, 150, 60)):
    grid = GridSpec(lo=MASS_GRID.lo, hi=MASS_GRID.hi, shape=shape)
    phi = np.asarray(make_region_fields(asm)(
        emap.theta(jnp.asarray(z)), jnp.asarray(grid.points())))
    traces, seen = [], set()
    for i, comp in enumerate(asm.components):
        d = phi[i].reshape(grid.shape)
        if d.min() > 0:
            continue
        verts, faces, _, _ = measure.marching_cubes(d, 0.0, spacing=grid.dx)
        verts += (np.array(grid.lo)
                  + np.array(grid.dx) * (0.5 + np.array(grid.jitter)))
        kind = kind_of(comp.name)
        traces.append(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=KIND_COLOR[kind], name=kind, legendgroup=kind,
            showlegend=showlegend and kind not in seen,
            flatshading=False, lighting=LIGHTING, scene=scene,
            hovertemplate=comp.name + "<extra></extra>"))
        seen.add(kind)
    return traces


def main():
    asm, emap, macros, sel, index = build_capstone()
    z0 = jnp.asarray(emap.z0)

    print("meshing regions ...")
    fig = go.Figure(craft_traces(asm, emap, z0))
    fig.update_layout(
        scene=dict(aspectmode="data",
                   xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False),
                   camera=dict(eye=dict(x=1.3, y=1.6, z=0.9))),
        paper_bgcolor=SURFACE, font=dict(color=INK, size=12),
        margin=dict(l=0, r=0, t=10, b=0), height=560,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02))
    div = fig.to_html(full_html=False, include_plotlyjs=True,
                      config={"displaylogo": False})

    print("props + diagnostics ...")
    props = jax.jit(make_mass_properties(asm, MASS_GRID, fidelity="straddle"))
    p = props(emap.theta(z0))
    diag = jax.jit(make_resolution_diagnostics(asm, MASS_GRID))(emap.theta(z0))
    sat = np.asarray(diag["saturation"])
    infl = np.asarray(diag["inflation"])
    masses = np.asarray(p["component_mass"])
    rows = "\n".join(
        f"<tr><td>{c.name}</td><td>{c.intent}</td><td>{m*1e3:.0f} g</td>"
        f"<td>{s:.3f} {'✓' if s >= TRUST_SATURATION else '<span class=low>low</span>'}</td>"
        f"<td>{iv:.2f}</td></tr>"
        for c, m, s, iv in zip(asm.components, masses, sat, infl))
    macro_rows = "\n".join(
        f"<tr><td>{name}</td><td>{float(emap.z0[k]):+.4f}</td></tr>"
        for name, k in macros.items())

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>geomk — capstone craft</title>
<style>
:root {{ --surface:{SURFACE}; --ink:{INK}; --ink2:{INK2}; --line:{GRIDLINE}; }}
*{{box-sizing:border-box}} body{{margin:0;background:var(--surface);color:var(--ink);
font:15px/1.55 system-ui,sans-serif}} main{{max-width:1080px;margin:0 auto;padding:28px 20px 64px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:36px 0 6px}}
.sub{{color:var(--ink2);margin:0 0 16px}}
.card{{border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:14px 0}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{text-align:right;padding:6px 12px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}}
th:first-child,td:first-child{{text-align:left}} th{{color:{INK2};font-weight:600}}
.low{{color:#b45f06;font-weight:600}}
.grid2{{display:grid;grid-template-columns:2fr 1fr;gap:14px}}
@media (max-width:800px){{.grid2{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Capstone craft — for inspection</h1>
<p class="sub">Twelve components, all breadth ops in use (lofted fuselage,
revolved nacelles, extruded fin, lattice-cored deck), declared deck–fuselage
mate, eight exposure macros. What you see is the kernel's own
precedence-composed segmentation, marching-cubed. Drag to rotate.</p>
<div class="card">{div}</div>
<div class="grid2">
<div class="card"><table>
<tr><th>component</th><th>intent</th><th>mass</th><th>saturation</th><th>inflation</th></tr>
{rows}
<tr><td><b>total</b></td><td></td><td><b>{float(p['total_mass'])*1e3:.0f} g</b></td>
<td colspan=2>τ = {float(diag['tau']):.4f}, trust ≥ {TRUST_SATURATION}</td></tr>
</table></div>
<div class="card"><table>
<tr><th>macro (z)</th><th>init</th></tr>
{macro_rows}
</table></div>
</div>
<p class="sub">COM {np.asarray(p['com']).round(4).tolist()} m. Mass and
diagnostics on the straddle/accurate path (72×72×28 grid). The battery's low
saturation is real: at 22 mm half-thickness it sits under the 2τ = 30 mm
trust floor — reported, not hidden.</p>
</main></body></html>"""
    with open("out/capstone.html", "w") as f:
        f.write(html)
    print("wrote out/capstone.html")


if __name__ == "__main__":
    main()
