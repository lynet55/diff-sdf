"""Probe B — coincident surfaces: two components mating on a flush plane.

Cases: exact-coincident, small gap, small overlap. Measures whether ownership
stays sharp across the mating plane and whether per-component mass matches the
hard/analytic reference, as a function of tau (PoU temperature), k (composition
bandwidth), and grid resolution. Measurements only; the fix class is reported,
not implemented.
"""
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, ".")

import matplotlib.pyplot as plt
from bench._style import SERIES, SEQ, SURFACE, INK, INK2, GRIDLINE
from matplotlib.colors import to_rgb

from geomk.dag import GraphBuilder
from geomk.compose import (Component, Assembly, make_region_fields,
                           make_background_field, pou_weights)
from geomk.projections import GridSpec, make_mass_properties, BG_BRIDGE_TAUS

HX, HY, HZ = 0.5, 0.3, 0.2      # box half extents
RHO_A, RHO_B = 2.0, 1.0
RESULTS = {}


def assembly(shift, k, mated=False):
    """A: x in [-1, 0] (precedence 1). B: x in [shift, 1+shift] (precedence 0).
    shift > 0 gap, = 0 exact-coincident, < 0 overlap (owned by A).
    mated=True declares the A/B interface a mate: exact-max composition for
    B's higher-precedence contribution + bridged PoU background at the seam."""
    gb = GraphBuilder()
    a = gb.box((-HX, 0, 0), (HX, HY, HZ))
    b = gb.box((HX + shift, 0, 0), (HX, HY, HZ))
    graph = gb.build()
    return Assembly(graph, (
        Component("A", a, density=RHO_A, precedence=1),
        Component("B", b, density=RHO_B, precedence=0),
    ), k_compose=k, mates=((0, 1),) if mated else ())


def exact_masses(shift):
    va = 2 * HX * 2 * HY * 2 * HZ
    vb = (2 * HX + min(shift, 0.0)) * 2 * HY * 2 * HZ  # A owns the overlap
    return RHO_A * va, RHO_B * vb


CASES = {"gap +0.03": 0.03, "coincident 0": 0.0, "overlap −0.03": -0.03}

# ---------- B1: ownership profile along the mating normal --------------------
print("B1: ownership profiles (tau=0.03, k=0.08)")
xs = np.linspace(-0.25, 0.25, 2001)
line = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=-1)
profiles = {}
for name, s in CASES.items():
    asm = assembly(s, k=0.08)
    phi = make_region_fields(asm)(jnp.asarray(asm.graph.theta0), jnp.asarray(line))
    w, wbg = pou_weights(phi, 0.03)
    profiles[name] = (np.asarray(w), np.asarray(wbg))
    own_b = np.asarray(w[1]) / (np.asarray(w[0]) + np.asarray(w[1]) + 1e-300)
    i10 = np.argmax(own_b > 0.1)
    i90 = np.argmax(own_b > 0.9)
    print(f"  {name:14s} ownership 10–90 width {xs[i90]-xs[i10]:.4f}  "
          f"max background weight {np.asarray(wbg).max():.3f}")


def smear_and_dip(s, tau, k):
    asm = assembly(s, k=k)
    phi = make_region_fields(asm)(jnp.asarray(asm.graph.theta0), jnp.asarray(line))
    w, wbg = pou_weights(phi, tau)
    w, wbg = np.asarray(w), np.asarray(wbg)
    own_b = w[1] / (w[0] + w[1] + 1e-300)
    width = xs[np.argmax(own_b > 0.9)] - xs[np.argmax(own_b > 0.1)]
    return float(width), float(wbg.max())


taus = np.array([0.01, 0.02, 0.04, 0.08])
b1 = {"tau": taus.tolist(), "width_k008": [], "width_k002": [],
      "dip_k008": [], "dip_k002": []}
for tau in taus:
    for kk, wkey, dkey in ((0.08, "width_k008", "dip_k008"),
                           (0.02, "width_k002", "dip_k002")):
        w_, d_ = smear_and_dip(0.0, tau, kk)
        b1[wkey].append(w_)
        b1[dkey].append(d_)
RESULTS["smear"] = b1

# ---------- B2: per-component mass error vs tau, k, resolution ----------------
print("B2: mass error vs tau and k (fine grid), vs resolution (tied tau)")


def masses(s, tau, k, nx=192, mated=False, accurate=False, supersample=3):
    asm = assembly(s, k=k, mated=mated)
    ny = max(12, nx // 8)
    grid = GridSpec(lo=(-1.15, -0.45, -0.35), hi=(1.15 + max(s, 0), 0.45, 0.35),
                    shape=(nx, ny, ny), tau=tau)
    p = make_mass_properties(asm, grid, accurate=accurate,
                             supersample=supersample)(jnp.asarray(asm.graph.theta0))
    return np.asarray(p["component_mass"])


b2 = {"tau": taus.tolist(), "cases": list(CASES), "k": [0.02, 0.08],
      "err_B": {}, "err_A": {}}
for kk in b2["k"]:
    for name, s in CASES.items():
        ea, eb = [], []
        ma_x, mb_x = exact_masses(s)
        for tau in taus:
            m = masses(s, tau, kk)
            ea.append(float((m[0] - ma_x) / ma_x))
            eb.append(float((m[1] - mb_x) / mb_x))
        b2["err_B"][f"k={kk} {name}"] = eb
        b2["err_A"][f"k={kk} {name}"] = ea
        print(f"  k={kk} {name:14s} errB(tau) = "
              + " ".join(f"{e:+.4f}" for e in eb))
RESULTS["mass_err"] = b2

print("B2b: coincident-case mass error vs resolution (tau tied 1.5dx, k=0.08)")
b2r = {"nx": [], "dx": [], "err_B": [], "err_A": []}
for nx in (24, 48, 96, 192, 384):
    dx = 2.3 / nx
    m = masses(0.0, 1.5 * dx, 0.08, nx=nx)
    ma_x, mb_x = exact_masses(0.0)
    b2r["nx"].append(nx)
    b2r["dx"].append(dx)
    b2r["err_A"].append(float((m[0] - ma_x) / ma_x))
    b2r["err_B"].append(float((m[1] - mb_x) / mb_x))
    print(f"  nx={nx:4d} dx={dx:.4f}  errA {b2r['err_A'][-1]:+.4f}  errB {b2r['err_B'][-1]:+.4f}")
RESULTS["mass_vs_resolution"] = b2r

# ---------- B3: declared mate (A,B) — the fix, gated ---------------------------
print("B3: declared mate — coincident-case gates (k=0.08)")
TAU_LINE = 0.03
ma_x, mb_x = exact_masses(0.0)

# (i) lower-precedence (B) mass error at the finest baseline setting
# (nx=384, tau=1.5dx): accurate straddle path (hardened supersampled
# occupancy values) and the soft mated path alongside.
nx_g = 384
dx_g = 2.3 / nx_g
m_acc = masses(0.0, 1.5 * dx_g, 0.08, nx=nx_g, mated=True,
               accurate=True, supersample=2)
m_soft = masses(0.0, 1.5 * dx_g, 0.08, nx=nx_g, mated=True)
errB_acc = float((m_acc[1] - mb_x) / mb_x)
errA_acc = float((m_acc[0] - ma_x) / ma_x)
errB_soft = float((m_soft[1] - mb_x) / mb_x)
baseline_errB = b2r["err_B"][b2r["nx"].index(nx_g)]
print(f"  errB accurate path {errB_acc:+.5f}  (soft mated {errB_soft:+.5f}, "
      f"unmated baseline {baseline_errB:+.5f})")

# (ii) max background (void) weight along the mating normal, mated
# composition + bridged background, vs the unmated baseline at the same tau.
asm_m = assembly(0.0, k=0.08, mated=True)
theta_m = jnp.asarray(asm_m.graph.theta0)
phi_m = make_region_fields(asm_m)(theta_m, jnp.asarray(line))
phi_bg = make_background_field(asm_m)(theta_m, jnp.asarray(line),
                                      BG_BRIDGE_TAUS * TAU_LINE)
_, wbg_m = pou_weights(phi_m, TAU_LINE, phi_bg)
ridge_mated = float(np.asarray(wbg_m).max())
ridge_unmated = float(np.asarray(profiles["coincident 0"][1]).max())
print(f"  void ridge max w_bg {ridge_mated:.4f}  (unmated baseline "
      f"{ridge_unmated:.4f}, tau={TAU_LINE})")

RESULTS["mate_gates"] = {
    "coincident_errB_accurate_path": errB_acc,
    "coincident_errA_accurate_path": errA_acc,
    "coincident_errB_soft_mated": errB_soft,
    "coincident_errB_unmated_baseline": float(baseline_errB),
    "mating_plane_max_background_weight": ridge_mated,
    "mating_plane_max_background_weight_unmated_baseline": ridge_unmated,
    "settings": {"nx": nx_g, "tau_mass": 1.5 * dx_g, "k": 0.08,
                 "supersample": 2, "tau_line": TAU_LINE,
                 "k_bridge": BG_BRIDGE_TAUS * TAU_LINE},
}
assert abs(errB_acc) <= 0.01, f"gate: |errB| accurate path {errB_acc}"
assert ridge_mated <= 0.05, f"gate: void ridge {ridge_mated}"

# ---------- plots -------------------------------------------------------------
fig = plt.figure(figsize=(11.5, 10.0))
gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1, 1], hspace=0.45, wspace=0.3)

# row 1: 2D slices of the interface zone
u = np.linspace(-0.35, 0.35, 320)
v = np.linspace(-0.42, 0.42, 300)
U, V = np.meshgrid(u, v)
pln = np.stack([U.ravel(), V.ravel(), np.zeros(U.size)], axis=-1)
for j, (name, s) in enumerate(CASES.items()):
    ax = fig.add_subplot(gs[0, j])
    asm = assembly(s, k=0.08)
    phi = make_region_fields(asm)(jnp.asarray(asm.graph.theta0), jnp.asarray(pln))
    w, _ = pou_weights(phi, 0.03)
    w = np.asarray(w).reshape(2, *U.shape)
    rgb = np.ones((*U.shape, 3)) * np.array(to_rgb(SURFACE))
    for wi, c in zip(w, (SERIES[0], SERIES[1])):
        rgb += wi[..., None] * (np.array(to_rgb(c)) - np.array(to_rgb(SURFACE)))
    ax.imshow(np.clip(rgb, 0, 1), origin="lower", extent=[u[0], u[-1], v[0], v[-1]],
              aspect="equal")
    ax.set_title(f"{name}", loc="left")
    ax.set_xlabel("x [m]")
    if j == 0:
        ax.set_ylabel("y [m]")
        ax.annotate("A", (-0.25, 0.0), color="white", fontsize=13, ha="center")
        ax.annotate("B", (0.25, 0.0), color="white", fontsize=13, ha="center")
    ax.grid(False)

# row 2: ownership profiles + smear/dip vs tau
ax = fig.add_subplot(gs[1, 0])
for j, (name, (w, wbg)) in enumerate(profiles.items()):
    ax.plot(xs, w[1] / (w[0] + w[1] + 1e-300), color=SERIES[j], lw=2,
            label=name.split()[0])
ax.legend(loc="upper left", fontsize=8.5)
ax.set_title("B ownership fraction along x (τ=0.03)", loc="left")
ax.set_xlabel("x [m]")
ax.set_ylabel("w_B / (w_A + w_B)")
ax.set_xlim(-0.12, 0.12)

ax = fig.add_subplot(gs[1, 1])
for name, (w, wbg) in profiles.items():
    ax.plot(xs, wbg, lw=2,
            color={"gap +0.03": SERIES[0], "coincident 0": SERIES[1],
                   "overlap −0.03": SERIES[2]}[name])
ax.annotate("gap", (0.015, 0.9), color=SERIES[0], fontsize=9)
ax.annotate("coincident", (0.03, 0.35), color=SERIES[1], fontsize=9)
ax.annotate("overlap", (-0.06, 0.12), color=SERIES[2], fontsize=9)
ax.set_title("background (void) weight along x (τ=0.03)", loc="left")
ax.set_xlabel("x [m]")
ax.set_ylabel("w_background")
ax.set_xlim(-0.12, 0.12)

ax = fig.add_subplot(gs[1, 2])
ax.loglog(taus, b1["width_k008"], color=SERIES[0], lw=3.5)
ax.loglog(taus, b1["width_k002"], color=SERIES[1], lw=1.6, marker="o", ms=4)
ax.loglog(taus, 2.2 * taus, color=INK2, lw=1.2, ls="--")
ax.annotate("k=0.08 ≡ k=0.02\n(τ alone sets the width ≈ 2.2τ)",
            (taus[1], b1["width_k008"][1]), color=INK, fontsize=9,
            ha="left", va="top", xytext=(6, -6), textcoords="offset points")
ax.set_title("ownership 10–90 smear width vs τ (coincident)", loc="left")
ax.set_xlabel("τ")
ax.set_ylabel("width [m]")
ax.set_xticks(taus.tolist(), [str(t) for t in taus])
ax.minorticks_off()

# row 3: mass errors
for col, kk in enumerate((0.08, 0.02)):
    ax = fig.add_subplot(gs[2, col])
    for j, name in enumerate(CASES):
        ax.semilogx(taus, np.abs(b2["err_B"][f"k={kk} {name}"]), color=SERIES[j],
                    lw=2, marker="o", ms=5, label=name.split()[0])
    if col == 0:
        ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title(f"|mass error| of B vs τ  (k={kk}, fine grid)", loc="left")
    ax.set_xlabel("τ")
    ax.set_ylabel("|Δm_B| / m_B")
    ax.set_xticks(taus.tolist(), [str(t) for t in taus])
    ax.minorticks_off()

ax = fig.add_subplot(gs[2, 2])
ax.loglog(b2r["dx"], np.abs(b2r["err_B"]), color=SERIES[0], lw=2, marker="o",
          ms=5, label="B (eroded side)")
ax.loglog(b2r["dx"], np.abs(b2r["err_A"]), color=SERIES[1], lw=2, marker="o",
          ms=5, label="A")
ax.legend(loc="lower right", fontsize=8.5)
ax.set_title("coincident: mass error vs dx (τ=1.5dx, k=0.08)", loc="left")
ax.set_xlabel("dx")
ax.set_ylabel("|Δm| / m")

fig.suptitle("Probe B — flush mating faces: ownership sharpness and mass fidelity",
             fontsize=12, x=0.02, ha="left")
fig.savefig("out/probe_b.png", dpi=150)

with open("out/probe_b.json", "w") as f:
    json.dump(RESULTS, f, indent=1)
print("wrote out/probe_b.png, out/probe_b.json")
