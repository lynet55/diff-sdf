"""Shared craft for the FEM / multi-physics probes.

Same body + 4 arms + 4 hubs quadrotor as probes C/D, with TWO geometry
macros: z = [arm_length, arm_radius]. Intent tags are load-bearing for the
first time: 'structural' (body, arms) selects the FEM material region;
'propulsion' (hubs) is the load region. Intent maps to component indices
HERE, in the consumer — the kernel never reads it (invariant 6).

Also carries the probe-D flight plant (saturating actuator, steady crosswind)
so the multi-physics stretch runs both physics on one assembly.
"""
import jax
import jax.numpy as jnp
import numpy as np

from geomk.dag import GraphBuilder
from geomk.compose import Component, Assembly
from geomk.exposure import Exposure
from geomk.projections import GridSpec

L0 = 0.16
# Arm radius obeys the kernel's own resolution rule (Probe A: features are
# trustworthy at t >= 2*tau). On the FEM grid tau = 0.018, so the arm diameter
# 2r = 0.048 > 2*tau = 0.041. At r = 0.016 (first attempt) the hard load path
# hung on a few throat cells and compliance was hypersensitive (25x from a
# 5e-5 arm-length change, kappa ~ 1e4-1e5) — measured, see probe E notes.
R_ARM0 = 0.024
BODY_HX0 = 0.07
HUB_R = 0.032
DIRS = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]]) / np.sqrt(2)

MASS_GRID = GridSpec(lo=(-0.34, -0.34, -0.09), hi=(0.34, 0.34, 0.09),
                     shape=(68, 68, 18))
FEM_GRID = GridSpec(lo=(-0.30, -0.30, -0.06), hi=(0.30, 0.30, 0.06),
                    shape=(44, 44, 10))


def build_craft():
    gb = GraphBuilder()
    body = gb.box((0, 0, 0), (BODY_HX0, BODY_HX0, 0.03))
    arms, hubs = [], []
    for d in DIRS:
        arms.append(gb.capsule((0, 0, 0), tuple(L0 * d), R_ARM0))
        hubs.append(gb.sphere(tuple(L0 * d), HUB_R))
    arm_u = gb.smooth_union(*arms, k=0.02)
    hub_u = gb.smooth_union(*hubs, k=0.02)
    graph = gb.build()

    asm = Assembly(graph, (
        Component("hubs", hub_u, density=2600.0, precedence=2, intent="propulsion"),
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
    zR = ex.macro_positive("arm_radius", init=R_ARM0)
    for arm in arms:
        ex.drive(arm.params[6], zR)     # unconstrained-space tie, softplus after
    emap = ex.build()

    # intent -> component indices (consumer-side; kernel stays intent-inert)
    sel = {tag: [i for i, c in enumerate(asm.components) if c.intent == tag]
           for tag in ("structural", "propulsion")}
    return asm, emap, (zL, zR), sel


# ---------------- flight plant (probe D, verbatim physics) -------------------
G = 9.81
KAPPA = 0.016
F_MAX = 10.0
F_FLOOR_BETA = 0.5
TAU_W = 0.30
DT, STEPS = 0.01, 250
TILT0_DEG = 8.0
KP_P, KD_P = 6.0, 4.0
KR, KW = 220.0, 30.0
TAU_EXT = jnp.array([TAU_W, 0.0, 0.0])


def actuator(f_cmd):
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


def make_flight_cost(emap, zL_index, props_fn):
    """Hover/disturbance-rejection cost of the probe-D plant as a function of
    the shared decision vector z. props_fn supplies mass/inertia/com."""

    def rollout(z):
        theta = emap.theta(z)
        p = props_fn(theta)
        m, I, com = p["total_mass"], p["inertia"], p["com"]
        I_inv = jnp.linalg.inv(I)
        L = z[zL_index]
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
            tau = I @ alpha + jnp.cross(om, I @ om)
            return M_inv @ jnp.concatenate([jnp.array([T]), tau])

        def deriv(state):
            pos, vel, q, om = state
            f = actuator(controller(state))
            R = quat_to_R(q)
            T = jnp.sum(f)
            tau = jnp.stack([jnp.cross(r, jnp.array([0.0, 0.0, fi]))
                             for r, fi in zip(rotors, f)]).sum(0) \
                + jnp.array([0.0, 0.0, KAPPA]) * jnp.dot(
                    jnp.array([1.0, -1.0, 1.0, -1.0]), f)
            tau = tau + R.T @ TAU_EXT
            acc = jnp.array([0.0, 0.0, -G]) + R @ jnp.array([0.0, 0.0, T]) / m
            om_dot = I_inv @ (tau - jnp.cross(om, I @ om))
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

    return rollout
