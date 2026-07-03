"""Probe A — scale: compile time vs node count, eval time vs region count and
grid resolution, thin-feature recovery vs voxel size and smoothing bandwidth.

Measurements only; no kernel changes. The output decides when the deferred
tensor-interpreter / culling items get promoted.
"""
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, ".")

import matplotlib.pyplot as plt
from bench._style import SERIES, SEQ, INK, INK2, GRIDLINE

from geomk.dag import GraphBuilder
from geomk.evaluate import make_field
from geomk.compose import Component, Assembly, make_region_fields, pou_weights
from geomk.projections import GridSpec, make_mass_properties

RESULTS = {}


def median_time(fn, n=5):
    fn()  # warm
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# ---------- A1: JIT trace+compile time vs node count -------------------------
def chain_graph(n_leaves):
    """1 + 3*(n_leaves-1) nodes: sphere, then (primitive+rigid+union) per leaf."""
    gb = GraphBuilder()
    root = gb.sphere((0, 0, 0), 0.4)
    for i in range(1, n_leaves):
        a = 2 * np.pi * i / n_leaves
        if i % 2:
            prim = gb.box((0, 0, 0), (0.25, 0.2, 0.15))
        else:
            prim = gb.capsule((-0.2, 0, 0), (0.2, 0, 0), 0.12)
        moved = gb.rigid(prim, translation=(0.8 * np.cos(a), 0.8 * np.sin(a), 0.0),
                         rotvec=(0.0, 0.0, a))
        root = gb.smooth_union(root, moved, k=0.08)
    return gb.build(), root


print("A1: compile time vs node count")
pts = jnp.asarray(np.random.default_rng(0).uniform(-1.5, 1.5, (2048, 3)))
a1 = {"N": [], "trace_s": [], "compile_s": [], "eval_ms": []}
for n_leaves in (1, 3, 5, 9, 13, 17):
    graph, root = chain_graph(n_leaves)
    N = len(graph.nodes)
    theta = jnp.asarray(graph.theta0)
    f = jax.jit(make_field(graph, root))
    t0 = time.perf_counter()
    lowered = f.lower(theta, pts)
    t1 = time.perf_counter()
    compiled = lowered.compile()
    t2 = time.perf_counter()
    ev = median_time(lambda: compiled(theta, pts).block_until_ready())
    a1["N"].append(N)
    a1["trace_s"].append(t1 - t0)
    a1["compile_s"].append(t2 - t1)
    a1["eval_ms"].append(ev * 1e3)
    print(f"  N={N:3d}  trace {t1-t0:6.3f}s  compile {t2-t1:6.3f}s  eval {ev*1e3:7.3f}ms")
RESULTS["compile_vs_nodes"] = a1


# ---------- A2: eval time vs region count and vs grid resolution -------------
def multi_region_assembly(R):
    gb = GraphBuilder()
    comps = []
    for i in range(R):
        a = 2 * np.pi * i / R
        cx, cy = 1.1 * np.cos(a), 1.1 * np.sin(a)
        prim = gb.smooth_union(
            gb.box((cx, cy, 0.0), (0.35, 0.25, 0.2)),
            gb.capsule((cx - 0.3, cy, 0.0), (cx + 0.3, cy, 0.0), 0.15), k=0.08)
        comps.append(Component(f"c{i}", prim, density=1.0 + 0.1 * i, precedence=i))
    graph = gb.build()
    return Assembly(graph, tuple(comps))


print("A2: eval time vs region count (grid 48^3)")
a2r = {"R": [], "eval_ms": [], "compile_s": []}
grid48 = GridSpec(lo=(-2, -2, -2), hi=(2, 2, 2), shape=(48, 48, 48))
for R in (2, 4, 8, 12, 15):
    asm = multi_region_assembly(R)
    theta = jnp.asarray(asm.graph.theta0)
    props = jax.jit(make_mass_properties(asm, grid48))
    t0 = time.perf_counter()
    props(theta)["total_mass"].block_until_ready()
    tc = time.perf_counter() - t0
    ev = median_time(lambda: props(theta)["total_mass"].block_until_ready())
    a2r["R"].append(R)
    a2r["eval_ms"].append(ev * 1e3)
    a2r["compile_s"].append(tc)
    print(f"  R={R:3d}  first-call {tc:6.2f}s  eval {ev*1e3:8.2f}ms")
RESULTS["eval_vs_regions"] = a2r

print("A2b: eval time vs grid resolution (R=6)")
a2g = {"points": [], "n": [], "eval_ms": []}
asm6 = multi_region_assembly(6)
theta6 = jnp.asarray(asm6.graph.theta0)
for n in (16, 24, 32, 48, 64, 96):
    grid = GridSpec(lo=(-2, -2, -2), hi=(2, 2, 2), shape=(n, n, n))
    props = jax.jit(make_mass_properties(asm6, grid))
    ev = median_time(lambda: props(theta6)["total_mass"].block_until_ready())
    a2g["points"].append(n ** 3)
    a2g["n"].append(n)
    a2g["eval_ms"].append(ev * 1e3)
    print(f"  {n:3d}^3 = {n**3:8d} pts  eval {ev*1e3:8.2f}ms")
RESULTS["eval_vs_resolution"] = a2g


# ---------- A3: thin-wall recovery vs tau (fine quadrature) -------------------
# Wall: box of half-thickness t. Occupancy = sigmoid(-phi/tau); recovered
# thickness t_hat = V_hat / area. Analytic prediction: t_hat = tau*ln(1+e^(2t... )
# measured, not assumed. Anisotropic grid: fine along x, coarse in y,z.
print("A3: thin-wall recovered thickness vs t and tau")
WALL_Y, WALL_Z = 0.8, 0.8


def wall_recovered_thickness(t, tau, nx=600):
    gb = GraphBuilder()
    wall = gb.box((0, 0, 0), (t, WALL_Y, WALL_Z))
    graph = gb.build()
    asm = Assembly(graph, (Component("wall", wall, density=1.0, precedence=0),))
    grid = GridSpec(lo=(-0.5, -0.7, -0.7), hi=(0.5, 0.7, 0.7),
                    shape=(nx, 20, 20), tau=tau)
    props = make_mass_properties(asm, grid)
    v = float(props(jnp.asarray(graph.theta0))["component_volume"][0])
    return v / (4 * 0.7 * 0.7)  # slab area inside the y,z window


ts = np.array([0.005, 0.01, 0.02, 0.04, 0.08, 0.16])
taus = [0.01, 0.02, 0.04, 0.08]
a3 = {"t": ts.tolist(), "tau": taus, "t_hat": []}
for tau in taus:
    row = [wall_recovered_thickness(t, tau) for t in ts]
    a3["t_hat"].append(row)
    print(f"  tau={tau}: t_hat/t = " + " ".join(f"{r/(2*t):5.2f}" for r, t in zip(row, ts)))
RESULTS["thin_wall"] = a3

# ---------- A4: thin-wall error vs voxel size (quadrature + tied tau) --------
print("A4: thin-wall volume error vs dx (t=0.04)")
T_FIX = 0.04
a4 = {"dx": [], "err_fixed_tau": [], "err_tied_tau": []}
for nx in (10, 20, 40, 80, 160, 320):
    dx = 1.0 / nx
    fixed = wall_recovered_thickness(T_FIX, 0.03, nx=nx)
    tied = wall_recovered_thickness(T_FIX, 1.5 * dx, nx=nx)
    a4["dx"].append(dx)
    a4["err_fixed_tau"].append(abs(fixed - 2 * T_FIX) / (2 * T_FIX))
    a4["err_tied_tau"].append(abs(tied - 2 * T_FIX) / (2 * T_FIX))
    print(f"  dx={dx:6.4f}  err(tau=0.03) {a4['err_fixed_tau'][-1]:8.2e}  "
          f"err(tau=1.5dx) {a4['err_tied_tau'][-1]:8.2e}")
RESULTS["wall_vs_dx"] = a4


# ---------- plots -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
(axN, axR), (axT, axD) = axes

axN.plot(a1["N"], a1["compile_s"], color=SERIES[0], lw=2, marker="o", ms=5)
axN.plot(a1["N"], a1["trace_s"], color=SERIES[1], lw=2, marker="o", ms=5)
axN.annotate("XLA compile", (a1["N"][-1], a1["compile_s"][-1]),
             ha="right", va="bottom", color=INK, fontsize=9)
axN.annotate("trace + lower", (a1["N"][-1], a1["trace_s"][-1]),
             ha="right", va="bottom", color=INK, fontsize=9)
axN.set_title("A1 — per-graph JIT cost vs node count", loc="left")
axN.set_xlabel("DAG nodes")
axN.set_ylabel("seconds")

axR.plot(a2r["R"], a2r["eval_ms"], color=SERIES[0], lw=2, marker="o", ms=5)
axR.set_title("A2 — mass-properties eval vs region count (48³ grid)", loc="left")
axR.set_xlabel("regions R")
axR.set_ylabel("ms / eval")
axR.set_ylim(bottom=0)

for i, (tau, row) in enumerate(zip(taus, a3["t_hat"])):
    axT.loglog(ts, np.array(row), color=SEQ[i + 1], lw=2, marker="o", ms=4)
    axT.annotate(f"τ={tau}", (ts[0], row[0]), ha="right", va="center",
                 color=INK2, fontsize=8.5, xytext=(-4, 0), textcoords="offset points")
axT.loglog(ts, 2 * ts, color=INK2, lw=1.2, ls="--")
axT.annotate("exact 2t", (ts[-1], 2 * ts[-1]), ha="left", va="top", color=INK2,
             fontsize=9, xytext=(2, -2), textcoords="offset points")
axT.set_title("A3 — recovered wall thickness vs true half-thickness t", loc="left")
axT.set_xlabel("wall half-thickness t")
axT.set_ylabel("recovered thickness  V̂/A")

axD.loglog(a4["dx"], a4["err_fixed_tau"], color=SERIES[0], lw=2, marker="o", ms=5)
axD.loglog(a4["dx"], a4["err_tied_tau"], color=SERIES[1], lw=2, marker="o", ms=5)
axD.axvline(T_FIX, color=INK2, lw=1, ls=":")
axD.annotate("dx = t", (T_FIX, 1e-4), color=INK2, fontsize=9, rotation=90,
             ha="right", va="bottom")
axD.annotate("fixed τ=0.03", (a4["dx"][0], a4["err_fixed_tau"][0]),
             ha="left", va="bottom", color=INK, fontsize=9)
axD.annotate("tied τ=1.5·dx", (a4["dx"][0], a4["err_tied_tau"][0]),
             ha="left", va="top", color=INK, fontsize=9)
axD.set_title(f"A4 — wall volume error vs voxel size (t={T_FIX})", loc="left")
axD.set_xlabel("voxel size dx")
axD.set_ylabel("relative volume error")

fig.suptitle("Probe A — scale & resolution limits of the per-graph-JIT evaluator",
             fontsize=12, x=0.02, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("out/probe_a.png", dpi=150)

with open("out/probe_a.json", "w") as f:
    json.dump(RESULTS, f, indent=1)
print("wrote out/probe_a.png, out/probe_a.json")
