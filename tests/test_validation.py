"""Pass-1 hard validation target (CLAUDE.md — do not soften).

Two overlapping components (wing + fuselage), different densities, declared
precedence. Objective depends on per-component mass and the total inertia
tensor. We optimize a parameter (fuselage radius) that moves the shared
interface; the gradient must run *through* boundary ownership: per-component
mass is a smoothed-indicator integral over precedence-composed regions, so
growing the fuselage must *steal* mass from the wing, and that transfer must
be visible to autodiff and agree with finite differences.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geomk.dag import GraphBuilder
from geomk.compose import Component, Assembly
from geomk.projections import GridSpec, make_mass_properties
from geomk.reparam import softplus_inverse
from geomk.optim import adam

R_INIT = 0.45
R_TARGET = 0.62

GRID = GridSpec(lo=(-2.5, -2.3, -0.9), hi=(2.5, 2.3, 0.9), shape=(72, 64, 26))


def build_assembly():
    gb = GraphBuilder()
    fus = gb.capsule((-1.8, 0.0, 0.0), (1.8, 0.0, 0.0), R_INIT)
    wing = gb.box((0.0, 0.0, 0.0), (0.45, 2.0, 0.12))
    graph = gb.build()
    components = (
        Component("fuselage", fus, density=2.0, precedence=1, intent="structural"),
        Component("wing", wing, density=1.0, precedence=0, intent="aero"),
    )
    # capsule params: (ax, ay, az, bx, by, bz, r) -> radius is index 6
    return Assembly(graph, components), int(fus.params[6])


@pytest.fixture(scope="module")
def problem():
    asm, r_idx = build_assembly()
    props = jax.jit(make_mass_properties(asm, GRID))
    theta0 = jnp.asarray(asm.graph.theta0)
    theta_star = theta0.at[r_idx].set(softplus_inverse(R_TARGET))
    target = props(theta_star)

    def objective(theta):
        p = props(theta)
        dm = p["component_mass"] - target["component_mass"]
        dI = p["inertia"] - target["inertia"]
        return jnp.sum(dm ** 2) + 0.1 * jnp.sum(dI ** 2)

    return asm, r_idx, props, theta0, theta_star, objective


def _central_fd(f, theta, i, h=1e-5):
    tp = theta.at[i].add(h)
    tm = theta.at[i].add(-h)
    return (f(tp) - f(tm)) / (2 * h)


def test_ownership_transfer_gradient(problem):
    """d(wing mass)/d(fuselage radius) must be significantly negative:
    the only path from fuselage radius to wing mass is boundary ownership."""
    asm, r_idx, props, theta0, *_ = problem
    m_wing = lambda th: props(th)["component_mass"][1]
    m_fus = lambda th: props(th)["component_mass"][0]

    g_wing = jax.grad(m_wing)(theta0)[r_idx]
    g_fus = jax.grad(m_fus)(theta0)[r_idx]

    assert g_wing < -0.02, f"wing mass gradient wrt fuselage radius not negative enough: {g_wing}"
    assert g_fus > 0.02, f"fuselage mass gradient wrt its radius should be positive: {g_fus}"

    # and both must agree with central finite differences
    fd_wing = _central_fd(m_wing, theta0, r_idx)
    fd_fus = _central_fd(m_fus, theta0, r_idx)
    np.testing.assert_allclose(g_wing, fd_wing, rtol=1e-5)
    np.testing.assert_allclose(g_fus, fd_fus, rtol=1e-5)


def test_objective_gradient_matches_fd(problem):
    asm, r_idx, props, theta0, theta_star, objective = problem
    g = jax.grad(objective)(theta0)[r_idx]
    fd = _central_fd(objective, theta0, r_idx)
    assert np.isfinite(g)
    np.testing.assert_allclose(g, fd, rtol=1e-5)


def test_optimization_moves_interface(problem):
    """Gradient descent on the fuselage radius alone must recover the target
    radius by moving the shared wing/fuselage interface."""
    asm, r_idx, props, theta0, theta_star, objective = problem
    mask = jnp.zeros_like(theta0).at[r_idx].set(1.0)
    vg = jax.jit(jax.value_and_grad(objective))

    theta_opt, hist = adam(vg, theta0, mask, lr=0.03, steps=200)

    J0 = float(hist["J"][0])
    Jf = float(hist["J"][-1])
    r_final = float(jax.nn.softplus(theta_opt[r_idx]))

    assert Jf < 1e-4 * J0, f"objective did not collapse: {J0} -> {Jf}"
    assert abs(r_final - R_TARGET) < 0.02, f"recovered radius {r_final} != {R_TARGET}"

    # mass must have actually transferred across the interface
    p0 = props(theta0)
    pf = props(theta_opt)
    assert pf["component_mass"][0] > p0["component_mass"][0]  # fuselage gained
    assert pf["component_mass"][1] < p0["component_mass"][1]  # wing lost
