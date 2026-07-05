"""Probe C — one real differentiable-sim loop.

geometry(theta) -> smoothed-occupancy mass/inertia/com -> 6-DOF quadrotor
dynamics (JAX, RK4, quaternions) with a geometric PD controller whose mixing
matrix depends on the rotor positions -> hover-recovery cost -> gradient back
to geometry decision variables (arm length, body width) via the exposure layer.

The arm-length macro drives BOTH the geometry (capsule endpoints, rotor hub
spheres) and the control allocation (rotor lever arms), so the gradient path
is genuinely coupled co-design, not a pass-through. Verified against central
finite differences, then optimized for a few dozen Adam steps.

Minimal honest quadrotor implemented here on purpose: the probe's requirement
is an end-to-end gradient, not a specific simulator. No motor clipping (noted).
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
from bench._style import SERIES, SURFACE, INK, INK2, GRIDLINE
from matplotlib.colors import to_rgb

from geomk.dag import GraphBuilder
from geomk.compose import Component, Assembly, make_region_fields, pou_weights
from geomk.exposure import Exposure
from geomk.projections import GridSpec, make_mass_properties
from geomk.optim import adam

RESULTS = {}

# ---------- geometry: body + 4 arms + 4 rotor hubs, exposure-driven ----------
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
    zL = ex.macro("arm_length", init=L0)      # unconstrained; drives many params
    for arm, hub, d in zip(arms, hubs, DIRS):
        for ax in range(3):
            if abs(d[ax]) > 1e-12:
                ex.drive(arm.params[3 + ax], zL, scale=float(d[ax]))  # capsule end
                ex.drive(hub.params[ax], zL, scale=float(d[ax]))      # hub center
    zB = ex.expose(body.params[3], "body_hx")  # softplus-side (positive param)
    ex.tie(body.params[4], zB)                 # keep body square
    return asm, ex.build(), (zL, zB)


asm, emap, (ZL, ZB) = build()
# Plain symmetric bounds: GridSpec's default deterministic sub-voxel jitter
# (promoted from this probe's original hand-coded EPS shift) keeps quadrature
# nodes off the box SDF's max/abs tie sets (x = ±y diagonals), where AD
# subgradients would otherwise disagree with central FD by ~1e-2.
GRID = GridSpec(lo=(-0.28, -0.28, -0.09), hi=(0.28, 0.28, 0.09),
                shape=(56, 56, 18))
# Accurate straddle (LOCKED: value accurate, gradient smooth): the rollout's
# physical constants (m, I, com) VALUES come from hardened supersampled
# occupancy (~2.41 kg, vs ~5.75 kg for the tau-inflated soft values); their
# GRADIENTS stay exactly the soft partition-of-unity path via stop_gradient.
props_fn = make_mass_properties(asm, GRID, accurate=True)
props_soft_fn = make_mass_properties(asm, GRID)

# ---------- quadrotor dynamics + geometric PD controller ---------------------
G = 9.81
KAPPA = 0.016      # rotor drag-torque / thrust coefficient
F_MAX = 22.0       # rotor thrust limit [N] (smooth penalty, not a clip)
DT, STEPS = 0.01, 250
KP_P, KD_P = 6.0, 4.0
KR, KW = 220.0, 30.0


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
        f = controller(state)
        R = quat_to_R(q)
        T = jnp.sum(f)
        tau = jnp.stack([jnp.cross(r, jnp.array([0.0, 0.0, fi]))
                         for r, fi in zip(rotors, f)]).sum(0) \
            + jnp.array([0.0, 0.0, KAPPA]) * jnp.dot(jnp.array([1.0, -1.0, 1.0, -1.0]), f)
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
        # smooth actuator saturation: without it the optimizer collapses the
        # arms to ~2 cm (measured: L 0.16 -> 0.02, cost 0.103 -> 0.098),
        # shedding mass while torque authority stays free via unlimited f.
        sat = jnp.sum(jax.nn.softplus(f - F_MAX) ** 2
                      + jax.nn.softplus(-f) ** 2)
        c = (jnp.sum(pos ** 2) + 0.3 * jnp.sum(vel ** 2) + 0.5 * (1.0 - R33)
             + 0.02 * jnp.sum(om ** 2) + 1e-5 * jnp.sum(f ** 2) + 0.05 * sat)
        return new, (c, pos, jnp.max(f))

    # hover recovery: tilted 20 deg about x, offset, at rest
    a0 = np.deg2rad(20.0)
    state0 = (jnp.array([0.35, -0.25, 0.15]), jnp.zeros(3),
              jnp.array([np.cos(a0 / 2), np.sin(a0 / 2), 0.0, 0.0]), jnp.zeros(3))
    final, (costs, traj, fmax) = jax.lax.scan(step, state0, None, length=STEPS)
    return jnp.mean(costs), (traj, fmax), (m, I, com)


def cost(z):
    return make_rollout(z)[0]


# ---------- C1: end-to-end gradient, verified against FD ---------------------
print("C1: end-to-end gradient check")
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

# The straddled cost's VALUE is a theta-staircase (hardened occupancy), so it
# must not be finite-differenced. The FD reference is the soft-path cost
# variant: m/I/com follow the SOFT projection in theta, anchored at the
# accurate values at z0 (p_acc(z0) + p_soft(z) - p_soft(z0)) so its rollout
# linearizes at the same physical constants the straddled rollout uses. At z0
# its value and its gradient equal the straddled cost's exactly (that is what
# the stop-gradient straddle means), and it is smooth, so central FD applies.
P_ACC0 = jax.jit(props_fn)(emap.theta(z0))       # accurate values at z0
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
assert np.all(np.isfinite(g0)) and np.all(rel < 1e-6), "gradient check failed"
assert 2.29 <= float(m0) <= 2.53, f"accurate mass {m0} outside [2.29, 2.53] kg"
RESULTS["gradcheck"] = {"names": list(emap.names), "ad": g0.tolist(),
                        "ad_soft_path": g_soft.tolist(),
                        "fd_soft_path": fd.tolist(), "rel_err": rel.tolist(),
                        "cost0": J0, "mass0_accurate": float(m0),
                        "mass0": float(m0),
                        "mass0_soft": float(np.asarray(P_SOFT0["total_mass"])),
                        "compile_s": t_compile, "eval_grad_ms": t_eval * 1e3}

# ---------- C2: optimize geometry through the sim loop ------------------------
print("C2: optimizing arm length + body width through the rollout")
z_opt, hist = adam(vg, z0, jnp.ones_like(z0), lr=0.01, steps=60)
Js = np.array(hist["J"])
zs = np.array(hist["theta"])
gs = np.array(hist["grad"])
L_traj = zs[:, ZL]
hx_traj = np.asarray(jax.nn.softplus(jnp.asarray(zs[:, ZB])))
m_f, I_f, _ = [np.asarray(v) for v in jax.jit(lambda z: make_rollout(z)[2])(z_opt)]
print(f"  cost {Js[0]:.5f} -> {Js[-1]:.5f}   L {L_traj[0]:.4f} -> {L_traj[-1]:.4f} m"
      f"   body_hx {hx_traj[0]:.4f} -> {hx_traj[-1]:.4f} m   mass {m0:.3f} -> {m_f:.3f} kg")
RESULTS["optimization"] = {
    "cost": [float(Js[0]), float(Js[-1])], "L": [float(L_traj[0]), float(L_traj[-1])],
    "body_hx": [float(hx_traj[0]), float(hx_traj[-1])],
    "mass": [float(m0), float(m_f)], "steps": len(Js)}

roll = jax.jit(lambda z: make_rollout(z)[1])
(traj0, fmax0), (trajf, fmaxf) = [tuple(np.asarray(a) for a in roll(z))
                                  for z in (z0, z_opt)]

# ---------- plots -------------------------------------------------------------
fig = plt.figure(figsize=(11.5, 9.6))
gs_ = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.28)

ax = fig.add_subplot(gs_[0, 0])
ax.plot(Js, color=SERIES[0], lw=2)
ax.set_title("hover-recovery cost through the sim loop", loc="left")
ax.set_xlabel("Adam iteration")
ax.set_ylabel("J")

ax = fig.add_subplot(gs_[0, 1])
ax.plot(gs[:, ZL], color=SERIES[0], lw=2)
ax.plot(gs[:, ZB], color=SERIES[1], lw=2)
ax.axhline(0, color=GRIDLINE, lw=1)
ax.annotate("dJ/d(arm length)", (len(gs) - 1, gs[-1, ZL]), ha="right",
            va="bottom", color=INK, fontsize=9)
ax.annotate("dJ/d(body θ)", (len(gs) - 1, gs[-1, ZB]), ha="right",
            va="top", color=INK, fontsize=9)
ax.set_title("geometry gradients (through mass, inertia, COM, mixing)", loc="left")
ax.set_xlabel("Adam iteration")

ax = fig.add_subplot(gs_[1, 0])
ax.plot(L_traj, color=SERIES[0], lw=2)
ax.set_title(f"arm length  {L_traj[0]:.3f} → {L_traj[-1]:.3f} m", loc="left")
ax.set_xlabel("Adam iteration")
ax.set_ylabel("L [m]")

ax = fig.add_subplot(gs_[1, 1])
t_axis = np.arange(STEPS) * DT
for j, (fm, tag) in enumerate([(fmax0, "before"), (fmaxf, "after")]):
    ax.plot(t_axis, fm, color=SERIES[j], lw=2, label=tag)
ax.axhline(F_MAX, color=INK2, lw=1.2, ls="--")
ax.annotate(f"rotor limit {F_MAX:.0f} N", (t_axis[-1], F_MAX), ha="right",
            va="bottom", color=INK2, fontsize=9)
ax.legend(loc="center right", fontsize=9)
ax.set_title("peak rotor thrust during recovery (‖p(t)‖ is unchanged)", loc="left")
ax.set_xlabel("t [s]")
ax.set_ylabel("max_i f_i [N]")

# geometry before/after, plan view
region_fields = make_region_fields(asm)
u = np.linspace(-0.28, 0.28, 360)
U, V = np.meshgrid(u, u)
pln = jnp.asarray(np.stack([U.ravel(), V.ravel(), np.zeros(U.size)], axis=-1))
for j, (z, tag) in enumerate([(z0, "before"), (z_opt, "after")]):
    ax = fig.add_subplot(gs_[2, j])
    phi = region_fields(emap.theta(z), pln)
    w, _ = pou_weights(phi, GRID.tau)
    w = np.asarray(w).reshape(3, *U.shape)
    rgb = np.ones((*U.shape, 3)) * np.array(to_rgb(SURFACE))
    for wi, c in zip(w, (SERIES[2], SERIES[0], SERIES[1])):  # hubs, body, arms
        rgb += wi[..., None] * (np.array(to_rgb(c)) - np.array(to_rgb(SURFACE)))
    ax.imshow(np.clip(rgb, 0, 1), origin="lower", extent=[u[0], u[-1], u[0], u[-1]],
              aspect="equal")
    Lv = float(z[ZL])
    ax.set_title(f"{tag} — plan view, L = {Lv:.3f} m", loc="left")
    ax.set_xlabel("x [m]")
    ax.grid(False)
    if j == 0:
        ax.set_ylabel("y [m]")
        ax.annotate("body", (0, 0), color="white", ha="center", fontsize=9)
        ax.annotate("hub", (Lv / np.sqrt(2), Lv / np.sqrt(2) + 0.05),
                    color=INK, ha="center", fontsize=9)
        ax.annotate("arm", (-Lv / 2 / np.sqrt(2), -Lv / 2 / np.sqrt(2) - 0.045),
                    color=INK, ha="center", fontsize=9)

fig.suptitle("Probe C — geometry(θ) → mass/inertia/COM → 6-DOF quadrotor rollout "
             "→ hover cost → ∂J/∂θ_geometry", fontsize=12, x=0.02, ha="left")
fig.savefig("out/probe_c.png", dpi=150)

with open("out/probe_c.json", "w") as f:
    json.dump(RESULTS, f, indent=1)
print("wrote out/probe_c.png, out/probe_c.json")
