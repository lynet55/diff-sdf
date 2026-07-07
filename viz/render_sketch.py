"""Gallery for the sketch/profile + intersection op breadth.

out/sketch.html — four parts that the pre-sketch op set could NOT express:
a concave L-bracket from a 2D polygon (with a bolt hole), a star boss (concave
winding, extruded), a chamfered block (box ∩ octahedron via smooth_intersect),
and a flanged pulley (an L-profile REVOLVED). All exact-SDF where marked, so
each is metric-clean and could be offset/shelled. Drag to rotate.
"""
import sys

import numpy as np

sys.path.insert(0, ".")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage import measure

from bench._style import SERIES, SURFACE, INK, INK2, GRIDLINE
from geomk.dag import GraphBuilder
from geomk.evaluate import make_field

LIGHTING = dict(ambient=0.5, diffuse=0.72, specular=0.18, roughness=0.6)


def star(n=5, r_out=0.9, r_in=0.42):
    v = []
    for i in range(2 * n):
        a = np.pi / 2 + np.pi * i / n
        r = r_out if i % 2 == 0 else r_in
        v.append((r * np.cos(a), r * np.sin(a)))
    return v


def bracket_L():
    return [(-0.7, -0.7), (0.7, -0.7), (0.7, -0.2), (-0.2, -0.2),
            (-0.2, 0.7), (-0.7, 0.7)]


def build_bracket(gb):
    # concave L polygon extruded, then a round bolt hole punched through
    plate = gb.extrude(gb.polygon(bracket_L()), 0.18)
    hole = gb.rigid(gb.extrude(gb.polygon(
        [(0.28 * np.cos(t), 0.28 * np.sin(t))
         for t in np.linspace(0, 2 * np.pi, 24, endpoint=False)]), 0.5),
        translation=(0.28, -0.45, 0.0))
    return gb.smooth_subtract(plate, hole, k=0.01)


def build_star(gb):
    return gb.extrude(gb.polygon(star()), 0.22)


def build_chamfer(gb):
    # box ∩ rotated box (an octahedron-ish clip) = chamfered block: the third
    # boolean, impossible before smooth_intersect
    a = gb.box((0, 0, 0), (0.6, 0.6, 0.6))
    b = gb.rigid(gb.box((0, 0, 0), (0.8, 0.8, 0.8)),
                 rotvec=(0.0, 0.0, np.pi / 4))
    c = gb.rigid(gb.box((0, 0, 0), (0.9, 0.9, 0.9)),
                 rotvec=(np.pi / 4, 0.0, 0.0))
    return gb.smooth_intersect(a, b, c, k=0.03)


def build_pulley(gb):
    # an L / stepped profile revolved about x -> a flanged hub with a groove
    prof = [(-0.5, 0.15), (0.5, 0.15), (0.5, 0.35), (0.2, 0.35),
            (0.2, 0.6), (0.4, 0.6), (0.4, 0.8), (-0.4, 0.8),
            (-0.4, 0.6), (-0.2, 0.6), (-0.2, 0.35), (-0.5, 0.35)]
    return gb.rigid(gb.revolve(gb.polygon(prof)), rotvec=(0.0, np.pi / 2, 0.0))


PARTS = [("L-bracket — concave sketch + bolt hole", build_bracket, SERIES[0]),
         ("star boss — concave winding, extruded", build_star, SERIES[3]),
         ("chamfered block — box ∩ box ∩ box", build_chamfer, SERIES[1]),
         ("flanged pulley — profile revolved", build_pulley, SERIES[2])]

LO, HI, N = (-1.05, -1.05, -1.05), (1.05, 1.05, 1.05), 120


def mesh(build, color, scene):
    gb = GraphBuilder()
    root = build(gb)
    graph = gb.build()
    f = jax.jit(make_field(graph, root))
    axes = [np.linspace(l, h, N) for l, h in zip(LO, HI)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    d = np.asarray(f(jnp.asarray(graph.theta0), jnp.asarray(G))).reshape(N, N, N)
    dx = [(h - l) / (N - 1) for l, h in zip(LO, HI)]
    verts, faces, _, _ = measure.marching_cubes(d, 0.0, spacing=dx)
    verts += np.array(LO)
    return go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                     i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                     color=color, flatshading=False, lighting=LIGHTING,
                     scene=scene, hoverinfo="skip")


def main():
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}],
               [{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[p[0] for p in PARTS], vertical_spacing=0.06,
        horizontal_spacing=0.02)
    scenes = ["scene", "scene2", "scene3", "scene4"]
    for idx, ((title, build, color), sc) in enumerate(zip(PARTS, scenes)):
        print(f"meshing {title} ...")
        fig.add_trace(mesh(build, color, sc), row=idx // 2 + 1,
                      col=idx % 2 + 1)
    cam = dict(eye=dict(x=1.5, y=1.5, z=1.1))
    for sc in scenes:
        fig.update_layout(**{sc: dict(
            aspectmode="data", camera=cam,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False))})
    fig.update_layout(paper_bgcolor=SURFACE, font=dict(color=INK, size=13),
                      margin=dict(l=0, r=0, t=28, b=0), height=760,
                      showlegend=False)
    div = fig.to_html(full_html=False, include_plotlyjs=True,
                      config={"displaylogo": False})
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>geomk — sketch profiles + intersection</title>
<style>body{{margin:0;background:{SURFACE};color:{INK};
font:15px/1.6 system-ui,sans-serif}} main{{max-width:1080px;margin:0 auto;
padding:28px 20px 56px}} h1{{font-size:22px;margin:0 0 4px}}
.sub{{color:{INK2};margin:0 0 14px}} .card{{border:1px solid {GRIDLINE};
border-radius:10px;overflow:hidden;margin:12px 0}}</style></head><body><main>
<h1>Sketch profiles + the third boolean</h1>
<p class="sub">Four parts the pre-sketch op set could not express. The
<b>polygon</b> primitive is an exact 2D SDF (concave-capable, winding sign), so
extrude/revolve of a sketch stays metric-clean and offset/shell-able;
<b>smooth_intersect</b> completes CSG. What you see is the raw field,
marching-cubed at 120³. Drag to rotate.</p>
<div class="card">{div}</div>
<p class="sub">Left→right, top→bottom: a concave L-bracket from a 6-vertex
sketch with a punched bolt hole; a 5-point star boss (reflex vertices, extruded);
a chamfered block as the intersection of three boxes; and a stepped/grooved
profile revolved into a flanged pulley. Every profile is an authored vertex
list — the same data a sketch UI would emit.</p>
</main></body></html>"""
    with open("out/sketch.html", "w") as fp:
        fp.write(html)
    print("wrote out/sketch.html")


if __name__ == "__main__":
    main()
