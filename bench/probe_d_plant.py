"""Probe D — plant realism: real actuator saturation makes arm length interior.

Probe C's plant limited rotor thrust only through a soft *cost penalty*
(calibrated against the tau-inflated 5.75 kg craft). After Step 2 corrected
the mass to ~2.41 kg the penalty no longer binds, and the optimizer collapses
the arm length 0.16 -> 0.029 m: shedding arm mass is free because control
torque authority is never actually limited.

Step 4 puts the limit in the PLANT. The applied per-rotor thrust is a smooth
C-inf saturating map of the commanded thrust (softplus floor at zero,
asymptotic tanh cap at F_MAX) applied BEFORE it produces any force or torque.
Control torque per unit differential thrust scales with the rotor lever arm
(prop L), so a steady disturbance moment requires differential thrust ~ 1/L:
short arms saturate, cannot reject the disturbance, tumble, and the cost
explodes. Long arms reject it easily but pay a mass/inertia penalty (the heavy
rotor hubs sit at the arm tips). Those two forces balance at an INTERIOR
optimum in arm length, produced by physics, not by a constraint.

This is genuine co-design coupling: geometry(theta) sets BOTH the mass /
inertia / com (Step-2 accurate straddle: value accurate, gradient soft) AND
the rotor lever arms that enter the dynamics and the control allocation. The
disturbance-rejection authority (propto L) trades against the tip inertia
(the heavy hubs, propto L^2) inside one differentiable rollout.

Disturbance: a mild initial tilt plus a STEADY world-frame crosswind moment
tau_w about body x. The steady moment is what makes authority bind as ~1/L
across the whole arm-length range (an initial tilt alone only bites in a
narrow transient window, because attitude recovery is otherwise inertia-
cancelling and L-independent until saturation).

Honest 6-DOF plant on purpose (not MJX): mujoco-mjx installs and steps here,
but wiring the theta-dependent full inertia tensor into an mjx.Model needs a
differentiable eigendecomposition into body_inertia + body_iquat that is
singular for this near-symmetric craft (Ixx ~= Iyy) -- exactly the fragile
geometry-coupled build the plan says not to gamble the gate on. The existing
Probe-C sim already IS an honest differentiable 6-DOF plant; Step 4 upgrades
its actuator model, which is the load-bearing change.

Same FD methodology as Probe C: central differences of the accurate-anchored
soft-path cost (value=accurate at z0, sensitivity=soft, smooth, FD-valid).
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
from bench._style import SERIES, INK, INK2, GRIDLINE

from geomk.dag import GraphBuilder
from geomk.compose import Component, Assembly
from geomk.exposure import Exposure
from geomk.projections import GridSpec, make_mass_properties
from geomk.optim import adam

RESULTS = {}

# ---------- geometry: identical craft to Probe C (body + 4 arms + 4 hubs) ----
L0 = 0.16          # initial arm length (m, along the diagonal)
BODY_HX0 = 0.07
DIRS = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]]) / np.sqrt(2)


def build():
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
        for ax in range(3):
            if abs(d[ax]) > 1e-12:
                ex.drive(arm.params[3 + ax], zL, scale=float(d[ax]))
                ex.drive(hub.params[ax], zL, scale=float(d[ax]))
    zB = ex.expose(body.params[3], "body_hx")
    ex.tie(body.params[4], zB)
    return asm, ex.build(), (zL, zB)


asm, emap, (ZL, ZB) = build()
# Grid spans L up to ~0.30 (hub tip at 0.30/sqrt2 + r ~ 0.24 < 0.34).
GRID = GridSpec(lo=(-0.34, -0.34, -0.09), hi=(0.34, 0.34, 0.09),
                shape=(68, 68, 18))
# Accurate straddle (LOCKED: value accurate, gradient smooth), as in Probe C.
props_fn = make_mass_properties(asm, GRID, accurate=True)
props_soft_fn = make_mass_properties(asm, GRID)

# ---------- quadrotor plant with real actuator saturation ---------------------
G = 9.81
KAPPA = 0.016      # rotor drag-torque / thrust coefficient
# F_MAX calibrated to the accurate ~2.41 kg craft: hover ~5.9 N/rotor, so
# F_MAX=10 N leaves ~4 N of differential headroom. The steady crosswind moment
# tau_w below needs differential thrust ~ tau_w/L, so headroom is exceeded
# (saturation) below L ~ 0.1 m -> the disturbance can no longer be rejected.
F_MAX = 10.0       # per-rotor thrust limit [N] -- enforced in the plant
F_FLOOR_BETA = 0.5  # softness [N] of the nonnegativity floor
TAU_W = 0.30       # steady world-frame crosswind moment about x [N.m]
DT, STEPS = 0.01, 250
TILT0_DEG = 8.0    # mild initial tilt about body x
KP_P, KD_P = 6.0, 4.0
KR, KW = 220.0, 30.0
TAU_EXT = jnp.array([TAU_W, 0.0, 0.0])


def actuator(f_cmd):
    """Smooth (C-inf) applied-thrust map: nonnegativity floor via softplus,
    asymptotic upper limit via tanh. Applied thrust lives in (0, F_MAX); the
    limit is real plant behaviour, not a penalty, and gradients stay finite."""
    f_pos = F_FLOOR_BETA * jax.nn.softplus(f_cmd / F_FLOOR_BETA)
    return F_MAX * jnp.tanh(f_pos / F_MAX)


def quat_to_R(q):
    w, x, y, z = q
    return jnp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return jnp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def make_rollout(z, props=None):
    """Physical quantities from geometry(z), then a scanned RK4 rollout."""
    theta = emap.theta(z)
    p = (props_fn if props is None else props)(theta)
    m, I, com = p["total_mass"], p["inertia"], p["com"]
    I_inv = jnp.linalg.inv(I)
    L = z[ZL]
    rotors = jnp.asarray(DIRS) * L - com          # lever arms about the COM
    # mixing: [T, tau] = M f  (thrust along body z, alternating yaw signs)
    yaw_sign = jnp.array([1.0, -1.0, 1.0, -1.0])
    M = jnp.concatenate([
        jnp.ones((1, 4)),
        jnp.stack([jnp.cross(r, jnp.array([0.0, 0.0, 1.0])) for r in rotors]).T[:2],
        (KAPPA * yaw_sign)[None, :]], axis=0)
    M_inv = jnp.linalg.inv(M)

    def controller(state):
        pos, vel, q, om = state
        R = quat_to_R(q)
        a_des = -KP_P * pos - KD_P * vel + jnp.array([0.0, 0.0, G])
        z_b = R[:, 2]
        T = m * jnp.dot(a_des, z_b)
        z_des = a_des / jnp.linalg.norm(a_des)
        e_R = jnp.cross(z_des, z_b)
        alpha = -KR * (R.T @ e_R) - KW * om
        tau = I @ alpha + jnp.cross(om, I @ om)
        return M_inv @ jnp.concatenate([jnp.array([T]), tau])

    def deriv(state):
        pos, vel, q, om = state
        f = actuator(controller(state))   # <- saturation applied IN the plant
        R = quat_to_R(q)
        T = jnp.sum(f)
        tau = jnp.stack([jnp.cross(r, jnp.array([0.0, 0.0, fi]))
                         for r, fi in zip(rotors, f)]).sum(0) \
            + jnp.array([0.0, 0.0, KAPPA]) * jnp.dot(jnp.array([1.0, -1.0, 1.0, -1.0]), f)
        tau = tau + R.T @ TAU_EXT          # steady crosswind moment (body frame)
        acc = jnp.array([0.0, 0.0, -G]) + R @ jnp.array([0.0, 0.0, T]) / m
        om_dot = I_inv @ (tau - jnp.cross(om, I @ om))
        q_dot = 0.5 * quat_mul(q, jnp.concatenate([jnp.zeros(1), om]))
        return (vel, acc, q_dot, om_dot), f

    def rk4(state):
        k1, f = deriv(state)
        k2, _ = deriv(jax.tree.map(lambda s, k: s + 0.5 * DT * k, state, k1))
        k3, _ = deriv(jax.tree.map(lambda s, k: s + 0.5 * DT * k, state, k2))
        k4, _ = deriv(jax.tree.map(lambda s, k: s + DT * k, state, k3))
        new = jax.tree.map(lambda s, a, b, c, d: s + DT / 6 * (a + 2 * b + 2 * c + d),
                           state, k1, k2, k3, k4)
        pos, vel, q, om = new
        return (pos, vel, q / jnp.linalg.norm(q), om), f

    def step(state, _):
        new, f = rk4(state)
        pos, vel, q, om = new
        R33 = quat_to_R(q)[2, 2]
        # No saturation penalty: the limit is enforced by the actuator map.
        c = (jnp.sum(pos ** 2) + 0.3 * jnp.sum(vel ** 2) + 0.5 * (1.0 - R33)
             + 0.02 * jnp.sum(om ** 2) + 1e-5 * jnp.sum(f ** 2))
        return new, (c, pos, jnp.max(f))

    a0 = np.deg2rad(TILT0_DEG)
    state0 = (jnp.array([0.35, -0.25, 0.15]), jnp.zeros(3),
              jnp.array([np.cos(a0 / 2), np.sin(a0 / 2), 0.0, 0.0]), jnp.zeros(3))
    final, (costs, traj, fmax) = jax.lax.scan(step, state0, None, length=STEPS)
    return jnp.mean(costs), (traj, fmax), (m, I, com)


def cost(z):
    return make_rollout(z)[0]


# ---------- D1: end-to-end gradient, verified against FD ----------------------
print("D1: end-to-end gradient check (saturated plant + steady wind)")
z0 = jnp.asarray(emap.z0)
vg = jax.jit(jax.value_and_grad(cost))
t0 = time.perf_counter()
J0, g0 = vg(z0)
t_compile = time.perf_counter() - t0
J0, g0 = float(J0), np.asarray(g0)
t0 = time.perf_counter()
vg(z0)[0].block_until_ready()
t_eval = time.perf_counter() - t0

m0, I0, com0 = [np.asarray(v) for v in jax.jit(lambda z: make_rollout(z)[2])(z0)]
print(f"  mass {m0:.4f} kg (accurate straddle)  Ixx {I0[0,0]:.3e}  cost {J0:.5f}")
print(f"  grad {g0}   (compile {t_compile:.1f}s, eval+grad {t_eval*1e3:.0f}ms)")

# FD reference: the accurate-anchored soft-path cost (Probe C technique). The
# straddled cost's VALUE is a theta-staircase, so FD runs on the soft-path
# variant p_acc(z0) + p_soft(z) - p_soft(z0); at z0 its value and gradient
# equal the straddled cost's exactly, and it is smooth.
P_ACC0 = jax.jit(props_fn)(emap.theta(z0))
P_SOFT0 = jax.jit(props_soft_fn)(emap.theta(z0))


def props_soft_path(theta):
    s = props_soft_fn(theta)
    return {key: P_ACC0[key] + (s[key] - P_SOFT0[key]) for key in s}


def cost_soft_path(z):
    return make_rollout(z, props_soft_path)[0]


cost_soft_j = jax.jit(cost_soft_path)
g_soft = np.asarray(jax.jit(jax.grad(cost_soft_path))(z0))
assert np.allclose(g0, g_soft, rtol=1e-12, atol=1e-15), \
    f"straddled grad != soft-path grad: {g0} vs {g_soft}"

fd = np.zeros_like(g0)
h = 1e-5
for k in range(z0.size):
    fd[k] = float((cost_soft_j(z0.at[k].add(h))
                   - cost_soft_j(z0.at[k].add(-h))) / (2 * h))
rel = np.abs(g0 - fd) / np.maximum(np.abs(fd), 1e-12)
print(f"  FD(soft-path cost) {fd}   rel err {rel}")
assert np.all(np.isfinite(g0)) and np.all(rel < 1e-5), "gradient check failed"
assert 2.29 <= float(m0) <= 2.53, f"accurate mass {m0} outside [2.29, 2.53] kg"
RESULTS["gradcheck"] = {"names": list(emap.names), "ad": g0.tolist(),
                        "ad_soft_path": g_soft.tolist(),
                        "fd_soft_path": fd.tolist(), "rel_err": rel.tolist(),
                        "cost0": J0, "mass0_accurate": float(m0),
                        "mass0_soft": float(np.asarray(P_SOFT0["total_mass"])),
                        "compile_s": t_compile, "eval_grad_ms": t_eval * 1e3}

# ---------- D2a: sweep cost over arm length ------------------------------------
print("D2a: cost(L) sweep -- looking for an interior minimum")
L_sweep = np.round(np.arange(0.06, 0.301, 0.01), 4)
cost_j = jax.jit(cost)
J_sweep = np.array([float(cost_j(z0.at[ZL].set(L))) for L in L_sweep])
i_star = int(np.argmin(J_sweep))
L_star, J_star = float(L_sweep[i_star]), float(J_sweep[i_star])
print("    L [m]   cost")
for L, J in zip(L_sweep, J_sweep):
    mark = "  <-- min" if float(L) == L_star else ""
    print(f"    {L:5.3f}   {J:.5f}{mark}")
assert 0 < i_star < len(L_sweep) - 1, \
    f"swept minimum at the edge: L*={L_star}, not interior"
assert J_sweep[0] > 2 * J_star, "floor is not clearly worse than the interior min"

# ---------- D2b: optimize through the saturated rollout ------------------------
print("D2b: optimizing arm length + body width through the rollout")
z_opt, hist = adam(vg, z0, jnp.ones_like(z0), lr=0.01, steps=120)
Js = np.array(hist["J"])
zs = np.array(hist["theta"])
L_traj = zs[:, ZL]
hx_traj = np.asarray(jax.nn.softplus(jnp.asarray(zs[:, ZB])))
L_final = float(L_traj[-1])
m_f, I_f, _ = [np.asarray(v) for v in jax.jit(lambda z: make_rollout(z)[2])(z_opt)]
for s in range(0, len(Js), 20):
    print(f"    step {s:3d}  J {Js[s]:.5f}  L {L_traj[s]:.4f}"
          f"  dJ/dL {hist['grad'][s][ZL]:+.3f}")
print(f"  cost {Js[0]:.5f} -> {Js[-1]:.5f}   L {L_traj[0]:.4f} -> {L_final:.4f} m"
      f"   body_hx {hx_traj[0]:.4f} -> {hx_traj[-1]:.4f} m"
      f"   mass {m0:.3f} -> {m_f:.3f} kg")
print(f"  swept L* = {L_star:.3f} (cost {J_star:.5f});"
      f" floor L={L_sweep[0]:.3f} cost {J_sweep[0]:.5f}")
assert L_final > 0.10, f"arm length collapsed: L_final={L_final}"
assert abs(L_final - L_star) <= 0.04, \
    f"optimizer did not converge near the swept minimum: {L_final} vs {L_star}"
RESULTS["sweep"] = {"L": L_sweep.tolist(), "cost": J_sweep.tolist(),
                    "L_star": L_star, "cost_star": J_star,
                    "cost_floor": float(J_sweep[0])}
RESULTS["optimization"] = {
    "cost": [float(Js[0]), float(Js[-1])],
    "L": [float(L_traj[0]), L_final],
    "body_hx": [float(hx_traj[0]), float(hx_traj[-1])],
    "mass": [float(m0), float(m_f)], "steps": len(Js)}
RESULTS["plant"] = {"F_MAX": F_MAX, "floor_beta": F_FLOOR_BETA, "tau_w": TAU_W,
                    "tilt0_deg": TILT0_DEG, "dt": DT, "steps": STEPS,
                    "model": "honest 6-DOF RK4 quaternion plant; smooth "
                             "softplus-floor + tanh-cap actuator saturation; "
                             "steady crosswind moment disturbance"}

roll = jax.jit(lambda z: make_rollout(z)[1])
(traj0, fmax0), (trajf, fmaxf) = [tuple(np.asarray(a) for a in roll(z))
                                  for z in (z0, z_opt)]

# ---------- plots --------------------------------------------------------------
fig = plt.figure(figsize=(11.5, 7.0))
gs_ = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.28)

ax = fig.add_subplot(gs_[0, 0])
ax.plot(L_sweep, J_sweep, color=SERIES[0], lw=2, marker="o", ms=3.5)
ax.plot([L_star], [J_star], marker="o", ms=9, mfc="none", mec=INK, mew=1.4)
ax.set_ylim(0, min(J_sweep.max() * 1.05, J_star * 6))
ytop = ax.get_ylim()[1]
ax.annotate(f"interior minimum\nL* = {L_star:.2f} m", (L_star, J_star),
            xytext=(L_star + 0.02, J_star + 0.35 * (ytop - J_star)),
            color=INK, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
ax.axvline(L_final, color=SERIES[1], lw=1.4, ls="--")
ax.annotate(f"optimizer\nL_final = {L_final:.3f}", (L_final, ytop),
            ha="left", va="top", color=SERIES[1], fontsize=9,
            xytext=(L_final - 0.075, ytop * 0.98))
ax.set_title("hover+wind recovery cost vs arm length (saturated plant)",
             loc="left")
ax.set_xlabel("L [m]")
ax.set_ylabel("J")

ax = fig.add_subplot(gs_[0, 1])
ax.plot(L_traj, color=SERIES[0], lw=2)
ax.axhline(L_star, color=INK2, lw=1.2, ls="--")
ax.annotate(f"swept L* = {L_star:.2f} m", (0, L_star), va="bottom",
            color=INK2, fontsize=9)
ax.set_title(f"arm length under Adam  {L_traj[0]:.3f} -> {L_final:.3f} m",
             loc="left")
ax.set_xlabel("Adam iteration")
ax.set_ylabel("L [m]")

ax = fig.add_subplot(gs_[1, 0])
t_axis = np.arange(STEPS) * DT
for j, (fm, tag) in enumerate([(fmax0, "before (L=0.16)"),
                               (fmaxf, f"after (L={L_final:.2f})")]):
    ax.plot(t_axis, fm, color=SERIES[j], lw=2, label=tag)
ax.axhline(F_MAX, color=INK2, lw=1.2, ls="--")
ax.annotate(f"actuator limit F_MAX = {F_MAX:.0f} N (in the plant)",
            (t_axis[0], F_MAX), ha="left", va="bottom", color=INK2, fontsize=9)
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(0, F_MAX * 1.15)
ax.set_title("peak applied rotor thrust during recovery", loc="left")
ax.set_xlabel("t [s]")
ax.set_ylabel("max_i f_i [N]")

ax = fig.add_subplot(gs_[1, 1])
ax.plot(Js, color=SERIES[0], lw=2)
ax.set_title("cost through the sim loop", loc="left")
ax.set_xlabel("Adam iteration")
ax.set_ylabel("J")

fig.suptitle("Probe D -- real actuator saturation in the plant makes the "
             "arm-length optimum interior", fontsize=12, x=0.02, ha="left")
fig.savefig("out/probe_d.png", dpi=150)

with open("out/probe_d.json", "w") as f:
    json.dump(RESULTS, f, indent=1)
print("wrote out/probe_d.png, out/probe_d.json")
