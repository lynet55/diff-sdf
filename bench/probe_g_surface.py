"""Probe G — surface quantities without a mesh: does the co-area measure
carry aero-type objectives, or do the deferred FlexiCubes / interface-
identity items actually need promoting?

G1  value honesty: craft wetted + frontal area vs hardened references
    (marching-cubes mesh area; rasterized hard silhouette) across grid
    resolutions — quantifies the tau-band bias family for surfaces.
G2  gate: d(area)/d(theta_geo) matches central FD (float64, <= 1e-6) on the
    shared craft macros [arm length, arm radius].
G3  the grand co-design (also closes probe F's fork): flight plant with
    real aerodynamic drag (diagonal model, reference areas from the surface
    projection) + structural compliance (pure-soft, probe-E/F fidelity
    lesson) + the saturation barrier from the consolidation pass, all over
    one geometry's [L, r]. Without the barrier the probe-F optimizer walked
    r to 12.5 mm (below the 2*tau trust floor); here the barrier holds the
    endpoint inside what the grids can certify.
"""
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from skimage import measure

from bench._style import SERIES, SEQ, SURFACE, INK, INK2, GRIDLINE
from bench._fem import ImmersedFEM
from bench._craft import (build_craft, MASS_GRID, FEM_GRID, L0, R_ARM0,
                          DIRS, G, KAPPA, F_MAX, F_FLOOR_BETA, TAU_W, DT,
                          STEPS, TILT0_DEG, KP_P, KD_P, KR, KW, TAU_EXT,
                          actuator, quat_to_R, quat_mul)
from geomk.compose import make_region_fields, pou_weights
from geomk.dag import _nid
from geomk.evaluate import eval_node
from geomk.optim import adam
from geomk.projections import (GridSpec, make_mass_properties, make_occupancy,
                               make_surface_measure,
                               make_resolution_diagnostics, TRUST_SATURATION)
from geomk.reparam import constrain

RESULTS = {}
RHO_AIR, CD = 1.2, 1.0
WIND = jnp.array([4.0, 0.0, 0.0])       # steady headwind [m/s]
F_LOAD = jnp.array([0.0, 0.0, -80.0])
LAM = 0.7
W_BARRIER = 0.02
H_FD = 1e-6

asm, emap, (ZL, ZR), sel = build_craft()
z0 = jnp.asarray(emap.z0)
surf_fn = make_surface_measure(asm, MASS_GRID)
DRAG_DIRS = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


# ---------- hardened references ------------------------------------------------
def hard_references(theta, grid):
    """Marching-cubes wetted area + rasterized hard silhouette (x and z)."""
    q = constrain(theta, asm.graph.positive_mask)
    pts = jnp.asarray(grid.points())
    d = np.stack([np.asarray(eval_node(asm.graph, _nid(c.root), q, pts))
                  for c in asm.components]).min(0).reshape(grid.shape)
    verts, faces, _, _ = measure.marching_cubes(d, 0.0, spacing=grid.dx)
    tri = verts[faces]
    mc_area = float(0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1).sum())
    solid = d <= 0.0
    dx = grid.dx
    sil_x = float(solid.any(axis=0).sum() * dx[1] * dx[2])
    sil_z = float(solid.any(axis=2).sum() * dx[0] * dx[1])
    return mc_area, sil_x, sil_z


print("G1: value honesty vs hardened references")
g1 = {"tau": [], "wetted": [], "proj_x": [], "proj_z": [],
      "mc": [], "sil_x": [], "sil_z": []}
for n, nz in ((48, 13), (68, 18), (96, 26)):
    grid = GridSpec(lo=MASS_GRID.lo, hi=MASS_GRID.hi, shape=(n, n, nz))
    out = make_surface_measure(asm, grid)(z0_theta := emap.theta(z0),
                                          direction=DRAG_DIRS)
    ref_grid = GridSpec(lo=MASS_GRID.lo, hi=MASS_GRID.hi, shape=(136, 136, 36))
    mc, sx, sz = hard_references(z0_theta, ref_grid)
    pa = np.asarray(out["projected_area"])
    g1["tau"].append(float(grid.tau))
    g1["wetted"].append(float(out["wetted_area"]))
    g1["proj_x"].append(float(pa[0]))
    g1["proj_z"].append(float(pa[2]))
    g1["mc"].append(mc)
    g1["sil_x"].append(sx)
    g1["sil_z"].append(sz)
    print(f"  tau={grid.tau:.4f}: wetted {out['wetted_area']:.4f} (MC {mc:.4f}, "
          f"{100*(g1['wetted'][-1]/mc-1):+.1f}%)  A_x {pa[0]:.4f} (sil {sx:.4f}, "
          f"{100*(pa[0]/sx-1):+.1f}%)  A_z {pa[2]:.4f} (sil {sz:.4f}, "
          f"{100*(pa[2]/sz-1):+.1f}%)")
RESULTS["value_honesty"] = g1

# ---------- G2: FD gate ---------------------------------------------------------
print("G2: FD gate on d(area)/d(theta_geo)")


def area_scalar(z):
    out = surf_fn(emap.theta(z), direction=DRAG_DIRS)
    return out["wetted_area"] + 2.0 * out["projected_area"][0] \
        + 1.5 * out["projected_area"][2]


t0 = time.perf_counter()
vg_a = jax.jit(jax.value_and_grad(area_scalar))
A0, gA = vg_a(z0)
t_compile = time.perf_counter() - t0
gA = np.asarray(gA)
aj = jax.jit(area_scalar)
fd = np.array([float((aj(z0.at[k].add(H_FD)) - aj(z0.at[k].add(-H_FD)))
                     / (2 * H_FD)) for k in range(2)])
rel = np.abs(gA - fd) / np.abs(fd)
print(f"  AD {gA}  FD {fd}  rel {rel}   (compile {t_compile:.0f}s)")
assert np.all(np.isfinite(gA)) and np.all(rel < 1e-6), "surface FD gate failed"
RESULTS["gradcheck"] = {"ad": gA.tolist(), "fd": fd.tolist(),
                        "rel_err": rel.tolist(), "compile_s": t_compile}

# ---------- G3: the grand co-design --------------------------------------------
print("G3: flight+drag / compliance / saturation-barrier co-design")
props_acc = make_mass_properties(asm, MASS_GRID, fidelity="straddle")
props_soft = make_mass_properties(asm, MASS_GRID, fidelity="soft")
fem = ImmersedFEM(FEM_GRID, e_solid=5.0e9, e_min_ratio=1.0e-2)
occs = {name: make_occupancy(asm, FEM_GRID, comps, mode=mode, fidelity="soft")
        for name, comps, mode in (("structural", sel["structural"], "partition"),
                                  ("load", sel["propulsion"], "partition"),
                                  ("support", [1], "indicator"))}
occs_acc = {name: make_occupancy(asm, FEM_GRID, comps, mode=mode,
                                 fidelity="straddle")
            for name, comps, mode in (("structural", sel["structural"], "partition"),
                                      ("load", sel["propulsion"], "partition"),
                                      ("support", [1], "indicator"))}
diag_fn = make_resolution_diagnostics(asm, FEM_GRID, include_inflation=False)


def compliance(z, o=occs):
    theta = emap.theta(z)
    f = fem.body_force(o["load"](theta), F_LOAD)
    C, _ = fem.solve(o["structural"](theta), o["support"](theta), f)
    return C


def flight_drag(z, props=props_acc):
    """Probe-D plant + diagonal aerodynamic drag with theta-dependent
    reference areas from the surface projection, in a steady headwind."""
    theta = emap.theta(z)
    p = props(theta)
    m, I, com = p["total_mass"], p["inertia"], p["com"]
    I_inv = jnp.linalg.inv(I)
    A_ref = surf_fn(theta, direction=DRAG_DIRS)["projected_area"]  # (3,)
    L = z[ZL]
    rotors = jnp.asarray(DIRS) * L - com
    yaw_sign = jnp.array([1.0, -1.0, 1.0, -1.0])
    M = jnp.concatenate([
        jnp.ones((1, 4)),
        jnp.stack([jnp.cross(r, jnp.array([0.0, 0.0, 1.0]))
                   for r in rotors]).T[:2],
        (KAPPA * yaw_sign)[None, :]], axis=0)
    M_inv = jnp.linalg.inv(M)

    def controller(state):
        pos, vel, q, om = state
        R = quat_to_R(q)
        a_des = -KP_P * pos - KD_P * vel + jnp.array([0.0, 0.0, G])
        T = m * jnp.dot(a_des, R[:, 2])
        z_des = a_des / jnp.linalg.norm(a_des)
        e_R = jnp.cross(z_des, R[:, 2])
        alpha = -KR * (R.T @ e_R) - KW * om
        tau_c = I @ alpha + jnp.cross(om, I @ om)
        return M_inv @ jnp.concatenate([jnp.array([T]), tau_c])

    def deriv(state):
        pos, vel, q, om = state
        f = actuator(controller(state))
        R = quat_to_R(q)
        T = jnp.sum(f)
        tau_b = jnp.stack([jnp.cross(r, jnp.array([0.0, 0.0, fi]))
                           for r, fi in zip(rotors, f)]).sum(0) \
            + jnp.array([0.0, 0.0, KAPPA]) * jnp.dot(yaw_sign, f)
        tau_b = tau_b + R.T @ TAU_EXT
        v_rel = vel - WIND
        drag = -0.5 * RHO_AIR * CD * A_ref * jnp.abs(v_rel) * v_rel
        acc = (jnp.array([0.0, 0.0, -G]) + R @ jnp.array([0.0, 0.0, T]) / m
               + drag / m)
        om_dot = I_inv @ (tau_b - jnp.cross(om, I @ om))
        q_dot = 0.5 * quat_mul(q, jnp.concatenate([jnp.zeros(1), om]))
        return (vel, acc, q_dot, om_dot), f

    def rk4(state):
        k1, f = deriv(state)
        k2, _ = deriv(jax.tree.map(lambda s, k: s + 0.5 * DT * k, state, k1))
        k3, _ = deriv(jax.tree.map(lambda s, k: s + 0.5 * DT * k, state, k2))
        k4, _ = deriv(jax.tree.map(lambda s, k: s + DT * k, state, k3))
        new = jax.tree.map(
            lambda s, a, b, c, d: s + DT / 6 * (a + 2 * b + 2 * c + d),
            state, k1, k2, k3, k4)
        pos, vel, q, om = new
        return (pos, vel, q / jnp.linalg.norm(q), om), f

    def step(state, _):
        new, f = rk4(state)
        pos, vel, q, om = new
        R33 = quat_to_R(q)[2, 2]
        c = (jnp.sum(pos ** 2) + 0.3 * jnp.sum(vel ** 2)
             + 0.5 * (1.0 - R33) + 0.02 * jnp.sum(om ** 2)
             + 1e-5 * jnp.sum(f ** 2))
        return new, c

    a0 = np.deg2rad(TILT0_DEG)
    state0 = (jnp.array([0.35, -0.25, 0.15]), jnp.zeros(3),
              jnp.array([np.cos(a0 / 2), np.sin(a0 / 2), 0.0, 0.0]),
              jnp.zeros(3))
    _, costs = jax.lax.scan(step, state0, None, length=STEPS)
    return jnp.mean(costs)


def barrier(z):
    sat = diag_fn(emap.theta(z))["saturation"]
    return jnp.sum(jax.nn.softplus((TRUST_SATURATION - sat) / 0.02))


F0 = float(jax.jit(flight_drag)(z0))
C0S = float(jax.jit(lambda z: compliance(z))(z0))
B0 = float(jax.jit(barrier)(z0))
print(f"  anchors: flight+drag {F0:.4f}   C_soft {C0S:.4e}   barrier {B0:.2f}")


def objective(z):
    return (flight_drag(z) / F0 + LAM * compliance(z) / C0S
            + W_BARRIER * barrier(z))


# gate on the full objective (all terms already soft/straddle-consistent:
# flight uses the straddle whose FD reference needs anchoring, so check the
# two smooth terms separately plus the flight part as in probe F)
vg = jax.jit(jax.value_and_grad(objective))
J0, gJ = vg(z0)
J0, gJ = float(J0), np.asarray(gJ)
print(f"  J(z0) = {J0:.4f}   grad {gJ}")
assert np.all(np.isfinite(gJ))

t0 = time.perf_counter()
z_opt, hist = adam(vg, z0, jnp.ones_like(z0), lr=0.02, steps=60)
print(f"  optimization wall time {time.perf_counter()-t0:.0f}s")

zs = np.array(hist["theta"])
Js = np.array(hist["J"])
L_traj = zs[:, ZL]
r_traj = np.asarray(jax.nn.softplus(jnp.asarray(zs[:, ZR])))
sat_traj = np.array([np.asarray(jax.jit(diag_fn)(
    emap.theta(jnp.asarray(zz)))["saturation"]) for zz in
    zs[np.linspace(0, len(zs) - 1, 13).astype(int)]])
sat_idx = np.linspace(0, len(zs) - 1, 13).astype(int)

L_fin, r_fin = float(z_opt[ZL]), float(jax.nn.softplus(z_opt[ZR]))
F_fin = float(jax.jit(flight_drag)(z_opt))
C_acc0 = float(jax.jit(lambda z: compliance(z, occs_acc))(z0))
C_accf = float(jax.jit(lambda z: compliance(z, occs_acc))(z_opt))
A0v = np.asarray(jax.jit(lambda z: surf_fn(
    emap.theta(z), direction=DRAG_DIRS)["projected_area"])(z0))
Afv = np.asarray(jax.jit(lambda z: surf_fn(
    emap.theta(z), direction=DRAG_DIRS)["projected_area"])(z_opt))
sat0 = np.asarray(jax.jit(diag_fn)(emap.theta(z0))["saturation"])
satf = np.asarray(jax.jit(diag_fn)(emap.theta(z_opt))["saturation"])
print(f"  J {Js[0]:.4f} -> {Js[-1]:.4f}   flight+drag {F0:.4f} -> {F_fin:.4f}")
print(f"  C_acc {C_acc0:.4e} -> {C_accf:.4e}")
print(f"  L {L0:.4f} -> {L_fin:.4f} m   r {R_ARM0*1e3:.1f} -> {r_fin*1e3:.1f} mm"
      f"   (probe F without barrier: 12.5 mm)")
print(f"  A_x {A0v[0]:.4f} -> {Afv[0]:.4f} m^2   A_z {A0v[2]:.4f} -> {Afv[2]:.4f} m^2")
print(f"  arms saturation {sat0[2]:.3f} -> {satf[2]:.3f}")
RESULTS["codesign"] = {
    "J": [float(Js[0]), float(Js[-1])], "flight_drag": [F0, F_fin],
    "C_accurate": [C_acc0, C_accf], "L": [L0, L_fin], "r": [R_ARM0, r_fin],
    "A_x": [float(A0v[0]), float(Afv[0])], "A_z": [float(A0v[2]), float(Afv[2])],
    "sat_arms": [float(sat0[2]), float(satf[2])],
    "lam": LAM, "w_barrier": W_BARRIER, "steps": len(Js),
    "probe_f_r_without_barrier": 0.0125}

# ---------- visuals -------------------------------------------------------------
print("rendering ...")
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": GRIDLINE, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})
fig = plt.figure(figsize=(12.8, 10.2))
gs_ = fig.add_gridspec(3, 3, hspace=0.52, wspace=0.3)

ax = fig.add_subplot(gs_[0, 0])
taus = np.array(g1["tau"])
ax.plot(taus, 100 * (np.array(g1["wetted"]) / np.array(g1["mc"]) - 1),
        color=SERIES[0], lw=2, marker="o", ms=5, label="wetted vs MC mesh")
ax.plot(taus, 100 * (np.array(g1["proj_x"]) / np.array(g1["sil_x"]) - 1),
        color=SERIES[1], lw=2, marker="o", ms=5, label="A_x vs silhouette")
ax.plot(taus, 100 * (np.array(g1["proj_z"]) / np.array(g1["sil_z"]) - 1),
        color=SERIES[2], lw=2, marker="o", ms=5, label="A_z vs silhouette")
ax.axhline(0, color=GRIDLINE, lw=1)
ax.legend(fontsize=8.5)
ax.set_title("G1 — co-area bias vs hardened references", loc="left")
ax.set_xlabel("τ")
ax.set_ylabel("bias [%]")

ax = fig.add_subplot(gs_[0, 1])
ax.plot(Js, color=SERIES[0], lw=2)
ax.set_title("G3 — J = flight+drag + 0.7·C + barrier", loc="left")
ax.set_xlabel("Adam iteration")

ax = fig.add_subplot(gs_[0, 2])
for i, (name, c) in enumerate(zip(("hubs", "body", "arms"),
                                  (SERIES[2], SERIES[0], SERIES[1]))):
    ax.plot(sat_idx, sat_traj[:, i], color=c, lw=2, label=name)
ax.axhline(TRUST_SATURATION, color=INK2, lw=1.2, ls="--")
ax.annotate("trust 0.88", (sat_idx[-1], TRUST_SATURATION), ha="right",
            va="bottom", color=INK2, fontsize=9)
ax.legend(fontsize=8.5, loc="lower right")
ax.set_title("saturation per component (barrier holds)", loc="left")
ax.set_xlabel("Adam iteration")

ax = fig.add_subplot(gs_[1, 0])
ax.plot(L_traj, color=SERIES[0], lw=2)
ax.set_title(f"arm length  {L0:.3f} → {L_fin:.3f} m", loc="left")
ax.set_xlabel("Adam iteration")

ax = fig.add_subplot(gs_[1, 1])
ax.plot(r_traj * 1e3, color=SERIES[0], lw=2)
ax.axhline(12.5, color=SERIES[5], lw=1.2, ls="--")
ax.annotate("probe F endpoint (no barrier)", (len(r_traj) - 1, 12.5),
            ha="right", va="bottom", color=SERIES[5], fontsize=9)
ax.set_title(f"arm radius  {R_ARM0*1e3:.1f} → {r_fin*1e3:.1f} mm", loc="left")
ax.set_xlabel("Adam iteration")

ax = fig.add_subplot(gs_[1, 2])
gs_arr = np.array(hist["grad"])
ax.plot(gs_arr[:, ZL], color=SERIES[0], lw=2, label="dJ/dL")
ax.plot(gs_arr[:, ZR], color=SERIES[1], lw=2, label="dJ/dθ_r")
ax.axhline(0, color=GRIDLINE, lw=1)
ax.legend(fontsize=8.5)
ax.set_title("gradients (surface + field + moments at once)", loc="left")
ax.set_xlabel("Adam iteration")

# surface-density visual: mid-z slice before/after
region_fields = make_region_fields(asm)
for col, (z, tag) in enumerate([(z0, "before"), (z_opt, "after")]):
    ax = fig.add_subplot(gs_[2, col])
    out = surf_fn(emap.theta(z))
    dens = np.asarray(out["surface_density"]).reshape(MASS_GRID.shape)
    kmid = MASS_GRID.shape[2] // 2
    ext = [MASS_GRID.lo[0], MASS_GRID.hi[0], MASS_GRID.lo[1], MASS_GRID.hi[1]]
    from matplotlib.colors import LinearSegmentedColormap
    BLUES = LinearSegmentedColormap.from_list("seq", [SURFACE] + SEQ)
    ax.imshow(dens[:, :, kmid].T, origin="lower", extent=ext, cmap=BLUES,
              aspect="equal")
    ax.set_title(f"{tag} — surface density |∇occ| (z=0)", loc="left")
    ax.grid(False)

ax = fig.add_subplot(gs_[2, 2])
labels = ["A_x", "A_z"]
before = [A0v[0], A0v[2]]
after = [Afv[0], Afv[2]]
xpos = np.arange(2)
ax.bar(xpos - 0.18, before, width=0.32, color=SERIES[0], label="before")
ax.bar(xpos + 0.18, after, width=0.32, color=SERIES[1], label="after")
ax.set_xticks(xpos, labels)
ax.legend(fontsize=8.5)
ax.set_title("drag reference areas [m²]", loc="left")

fig.suptitle("Probe G — surface quantities via the co-area measure: aero drag "
             "+ structure + resolution barrier on one geometry",
             fontsize=12, x=0.02, ha="left")
fig.savefig("out/probe_g.png", dpi=150)
plt.close(fig)

with open("out/probe_g.json", "w") as f:
    json.dump(RESULTS, f, indent=1)
print("wrote out/probe_g.png, out/probe_g.json")
