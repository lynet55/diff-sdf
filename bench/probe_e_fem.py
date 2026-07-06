"""Probe E — second consumer: immersed structural FEM through the kernel.

Question: does the projection/contract architecture generalize beyond the
rigid-body plant it grew up with? The FEM path consumes ONLY kernel
projections: the occupancy field of the intent-selected 'structural' region
is the material indicator, the 'propulsion' region's occupancy carries the
load, the body's occupancy anchors the volumetric Dirichlet penalty. No
boundary mesh, no ad-hoc geometry sampling.

LOCKED fidelity rule holds: absolute compliance values come from the accurate
(hardened supersampled) occupancy; gradients come from the soft PoU path via
the stop-gradient straddle. FD methodology as in probe D: the FD reference is
the accurate-anchored soft-path objective, which at the anchor matches the
straddled objective's value AND gradient exactly and is smooth.

Load case: 80 N downward maneuver load distributed over the rotor hubs
('propulsion'), craft held at the body ('structural' core) — four cantilever
arms in bending. Gate co-design: fatten the arm-radius macro to stiffen
against a mass penalty.
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
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgb

from bench._style import SERIES, SEQ, SURFACE, INK, INK2, GRIDLINE
from bench._fem import ImmersedFEM
from bench._craft import build_craft, MASS_GRID, FEM_GRID, R_ARM0
from geomk.compose import make_region_fields, pou_weights
from geomk.optim import adam
from geomk.projections import make_mass_properties, make_occupancy

RESULTS = {}
F_LOAD = jnp.array([0.0, 0.0, -80.0])   # fixed total maneuver load on the hubs

asm, emap, (ZL, ZR), sel = build_craft()
print(f"intent -> indices: {sel}")
# e_min_ratio 1e-2 (not 1e-3): the ersatz void carries ~1% parasitic
# stiffness but cuts the operator's condition number ~10x. With the 1e-3
# contrast AND sub-resolution arms, perturbed solves ranged from converged-
# but-hypersensitive to outright CG failure (residual 4.5) — measured.
fem = ImmersedFEM(FEM_GRID, e_solid=5.0e9, e_min_ratio=1.0e-2)

# occupancy projections (the new kernel surface) — accurate straddle + soft.
# Material regions use the partition-of-unity mode (matter must not double-
# count); the CLAMP uses the indicator mode: a boundary-condition predicate
# keyed to the body's partition share gets eroded in the softmax by the very
# arm halo being sized, flipping d(compliance)/d(arm radius) positive on the
# soft path while the accurate trend falls — measured; see make_occupancy.
occs = {}
for name, comps, mode in (("structural", sel["structural"], "partition"),
                          ("load", sel["propulsion"], "partition"),
                          ("support", [1], "indicator")):   # body clamps
    occs[name] = {
        "acc": make_occupancy(asm, FEM_GRID, comps, accurate=True, mode=mode),
        "soft": make_occupancy(asm, FEM_GRID, comps, mode=mode),
    }

props_acc = make_mass_properties(asm, MASS_GRID, accurate=True)
props_soft = make_mass_properties(asm, MASS_GRID)
STRUCT = jnp.asarray(sel["structural"])


def compliance_from(occf):
    def compliance(z):
        theta = emap.theta(z)
        o_s = occf["structural"](theta)
        o_l = occf["load"](theta)
        o_c = occf["support"](theta)
        f = fem.body_force(o_l, F_LOAD)
        C, _ = fem.solve(o_s, o_c, f)
        return C
    return compliance


compliance_str = compliance_from({k: v["acc"] for k, v in occs.items()})

# accurate-anchored soft path (FD reference; probe-D methodology)
z0 = jnp.asarray(emap.z0)
theta0 = emap.theta(z0)
ANCH = {k: (jnp.asarray(v["acc"](theta0)), jnp.asarray(v["soft"](theta0)))
        for k, v in occs.items()}


def anchored(name):
    acc0, soft0 = ANCH[name]
    fn = occs[name]["soft"]
    return lambda theta: acc0 + (fn(theta) - soft0)


compliance_anch = compliance_from({k: anchored(k) for k in occs})

# ---------------- E1: gate gradient check -------------------------------------
print("E1: FD gate on d(compliance)/d(theta_geo)")
t0 = time.perf_counter()
vg_str = jax.jit(jax.value_and_grad(compliance_str))
C0, g_str = vg_str(z0)
C0, g_str = float(C0), np.asarray(g_str)
t_compile = time.perf_counter() - t0
t0 = time.perf_counter()
vg_str(z0)[0].block_until_ready()
t_eval = time.perf_counter() - t0

g_anch = np.asarray(jax.jit(jax.grad(compliance_anch))(z0))
assert np.allclose(g_str, g_anch, rtol=1e-10), \
    f"straddled grad != anchored soft grad: {g_str} vs {g_anch}"

comp_anch_j = jax.jit(compliance_anch)
fd = np.zeros_like(g_str)
# h = 1e-7, not the usual 1e-5: C''' along arm length is ~1e8 (thin features
# sweeping the background grid), so truncation dominates until h ~ 1e-7; the
# 1e-12-residual CG keeps the roundoff floor below that. Verified sweep:
# h=1e-5 -> 3.3e-3, 1e-6 -> 3.3e-5, 1e-7 -> 3.3e-7 (pure h^2 truncation).
h = 1e-7
for k in range(z0.size):
    fd[k] = float((comp_anch_j(z0.at[k].add(h))
                   - comp_anch_j(z0.at[k].add(-h))) / (2 * h))
rel = np.abs(g_str - fd) / np.maximum(np.abs(fd), 1e-300)
print(f"  C(z0) = {C0:.6e} J   (compile {t_compile:.1f}s, eval+grad {t_eval:.1f}s)")
print(f"  AD {g_str}")
print(f"  FD {fd}   rel err {rel}")
assert np.all(np.isfinite(g_str)) and np.all(rel < 1e-6), "FD gate failed"
assert g_str[ZR] < 0, "fatter arms must reduce compliance"
RESULTS["gradcheck"] = {
    "names": list(emap.names), "ad": g_str.tolist(), "fd": fd.tolist(),
    "rel_err": rel.tolist(), "compliance0": C0,
    "compile_s": t_compile, "eval_grad_s": t_eval}

# solver honesty at the anchor
theta_j = emap.theta(z0)
o_s0, o_l0, o_c0 = (occs[k]["acc"](theta_j) for k in ("structural", "load", "support"))
f0 = fem.body_force(o_l0, F_LOAD)
_, u0 = jax.jit(fem.solve)(o_s0, o_c0, f0)
res0 = float(fem.residual_norm(o_s0, o_c0, f0, u0))
print(f"  CG relative residual at anchor: {res0:.2e}")
RESULTS["gradcheck"]["cg_residual"] = res0

# ---------------- E2: structural co-design ------------------------------------
print("E2: co-design — fatten arm radius against a mass penalty")
m_struct = lambda th: jnp.sum(make_mass_properties(
    asm, MASS_GRID, accurate=True)(th)["component_mass"][STRUCT])
m0 = float(jax.jit(m_struct)(theta0))
# GAMMA calibrated to the SOFT path's r-sensitivity: the smeared arm (interior
# occupancy ~0.7 at tau = 1.5dx) underestimates d(C)/d(r) ~12x vs the accurate
# trend, so a mass weight sized against the accurate sensitivity would
# overpower the soft gradient and shrink the arms. Reported as a finding.
GAMMA = 0.02


def objective(z):
    return (compliance_str(z) / C0
            + GAMMA * m_struct(emap.theta(z)) / m0)


vg = jax.jit(jax.value_and_grad(objective))
mask = jnp.zeros_like(z0).at[ZR].set(1.0)
t0 = time.perf_counter()
z_opt, hist = adam(vg, z0, mask, lr=0.05, steps=40)
print(f"  optimization wall time {time.perf_counter()-t0:.0f}s")

r_traj = np.asarray(jax.nn.softplus(jnp.asarray(np.array(hist["theta"])[:, ZR])))
Js = np.array(hist["J"])
gR = np.array([g[ZR] for g in hist["grad"]])
C_final = float(jax.jit(compliance_str)(z_opt))
m_final = float(jax.jit(m_struct)(emap.theta(z_opt)))
r_final = float(jax.nn.softplus(z_opt[ZR]))
print(f"  compliance {C0:.4e} -> {C_final:.4e} J  ({C_final/C0:.3f}x)")
print(f"  arm radius {R_ARM0:.4f} -> {r_final:.4f} m   "
      f"m_struct {m0:.4f} -> {m_final:.4f} kg")
RESULTS["codesign"] = {
    "compliance": [C0, C_final], "r_arm": [R_ARM0, r_final],
    "m_struct": [m0, m_final], "J": [float(Js[0]), float(Js[-1])],
    "gamma": GAMMA, "steps": len(Js)}

# FD spot-check at the optimum too (gate says geometry is a REAL decision var)
ANCH_F = {k: (jnp.asarray(occs[k]["acc"](emap.theta(z_opt))),
              jnp.asarray(occs[k]["soft"](emap.theta(z_opt))))
          for k in occs}
comp_anch_f = jax.jit(compliance_from(
    {k: (lambda k=k: (lambda th: ANCH_F[k][0] + occs[k]["soft"](th) - ANCH_F[k][1]))()
     for k in occs}))
g_f = float(jax.jit(jax.grad(compliance_str))(z_opt)[ZR])
fd_f = float((comp_anch_f(z_opt.at[ZR].add(h))
              - comp_anch_f(z_opt.at[ZR].add(-h))) / (2 * h))
rel_f = abs(g_f - fd_f) / abs(fd_f)
print(f"  FD at optimum (r entry): AD {g_f:.6e}  FD {fd_f:.6e}  rel {rel_f:.2e}")
assert rel_f < 1e-6
RESULTS["codesign"]["fd_at_optimum_rel_err"] = rel_f

# ---------------- visuals ------------------------------------------------------
print("rendering ...")
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": GRIDLINE, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUES = LinearSegmentedColormap.from_list("seq", [SURFACE] + SEQ)
region_fields = make_region_fields(asm)
NX, NY, NZ = FEM_GRID.shape
KMID = NZ // 2


def solve_fields(z):
    theta = emap.theta(z)
    o_s = occs["structural"]["acc"](theta)
    o_l = occs["load"]["acc"](theta)
    o_c = occs["support"]["acc"](theta)
    f = fem.body_force(o_l, F_LOAD)
    C, u = fem.solve(o_s, o_c, f)
    e = fem.strain_energy_density(o_s, u)
    return (float(C), np.asarray(o_s).reshape(FEM_GRID.shape),
            np.asarray(u).reshape(NX + 1, NY + 1, NZ + 1, 3),
            np.asarray(e).reshape(FEM_GRID.shape))


ext = [FEM_GRID.lo[0], FEM_GRID.hi[0], FEM_GRID.lo[1], FEM_GRID.hi[1]]
fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.6))
emax = None
for row, (z, tag) in enumerate([(z0, "before"), (z_opt, "after")]):
    C, o_s, u, e = solve_fields(z)
    umag = np.linalg.norm(u, axis=-1)[:, :, KMID] * 1e3   # mm, node mid-layer
    esl = e[:, :, KMID] + 1e-30
    if emax is None:
        emax, umax = esl.max(), umag.max()
    ax = axes[row, 0]
    im = ax.imshow(umag.T, origin="lower", extent=ext, cmap=BLUES,
                   vmin=0, vmax=umax, aspect="equal")
    ax.contour(np.linspace(*ext[:2], NX), np.linspace(*ext[2:], NY),
               o_s[:, :, KMID].T, levels=[0.5], colors=[INK2], linewidths=1.2)
    ax.set_title(f"{tag} — |u| [mm], C = {C:.3e} J", loc="left")
    ax.grid(False)
    plt.colorbar(im, ax=ax, shrink=0.85)
    ax = axes[row, 1]
    im = ax.imshow(esl.T, origin="lower", extent=ext, cmap=BLUES,
                   norm=LogNorm(vmin=emax * 1e-5, vmax=emax), aspect="equal")
    ax.contour(np.linspace(*ext[:2], NX), np.linspace(*ext[2:], NY),
               o_s[:, :, KMID].T, levels=[0.5], colors=[INK2], linewidths=1.2)
    ax.set_title(f"{tag} — strain energy density [J/m³] (log)", loc="left")
    ax.grid(False)
    plt.colorbar(im, ax=ax, shrink=0.85)
fig.suptitle("Immersed FEM on the kernel's occupancy field — 80 N hub load, "
             "body clamped, structural region outlined", fontsize=12,
             x=0.02, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("out/fem_field.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
axes[0].plot(Js, color=SERIES[0], lw=2)
axes[0].set_title(f"objective  C/C₀ + {GAMMA}·m/m₀", loc="left")
axes[0].set_xlabel("Adam iteration")
axes[1].plot(gR, color=SERIES[0], lw=2)
axes[1].axhline(0, color=GRIDLINE, lw=1)
axes[1].set_title("dJ/dθ_r (FD-checked at both ends)", loc="left")
axes[1].set_xlabel("Adam iteration")
axes[2].plot(r_traj * 1e3, color=SERIES[0], lw=2)
axes[2].set_title(f"arm radius  {R_ARM0*1e3:.1f} → {r_final*1e3:.1f} mm", loc="left")
axes[2].set_xlabel("Adam iteration")
axes[2].set_ylabel("r [mm]")
fig.suptitle("Structural co-design through the second physics", fontsize=12,
             x=0.02, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("out/fem_codesign.png", dpi=150)
plt.close(fig)

with open("out/probe_e.json", "w") as f:
    json.dump(RESULTS, f, indent=1)
print("wrote out/fem_field.png, out/fem_codesign.png, out/probe_e.json")
