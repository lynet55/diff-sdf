"""Standing CI: finite-difference gradient checks across the whole op set.

Catches silently non-differentiable projections. Each case builds a small
graph, evaluates a smooth scalar of the field at fixed probe points (plus a
mass-properties projection for the composed case), and compares jax.grad
against central finite differences in float64.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geomk.dag import GraphBuilder
from geomk.evaluate import make_field
from geomk.compose import Component, Assembly
from geomk.projections import GridSpec, make_mass_properties

RNG = np.random.default_rng(7)
PROBES = jnp.asarray(RNG.uniform(-1.6, 1.6, size=(24, 3)))


def _sphere(gb):
    return gb.sphere((0.1, -0.2, 0.3), 0.7)


def _box(gb):
    return gb.box((0.05, 0.1, -0.15), (0.6, 0.4, 0.3))


def _capsule(gb):
    return gb.capsule((-0.8, 0.1, 0.0), (0.7, -0.2, 0.3), 0.35)


def _union(gb):
    return gb.smooth_union(_sphere(gb), _box(gb), k=0.1)


def _subtract(gb):
    return gb.smooth_subtract(_box(gb), _sphere(gb), k=0.1)


def _rigid(gb):
    return gb.rigid(_capsule(gb), translation=(0.2, -0.3, 0.1), rotvec=(0.3, -0.2, 0.5))


def _rigid_zero_rotation(gb):
    return gb.rigid(_box(gb), translation=(0.1, 0.0, -0.2), rotvec=(0.0, 0.0, 0.0))


def _redistance(gb):
    return gb.redistance(_union(gb))


def _offset(gb):
    return gb.offset(gb.redistance(_union(gb)), 0.15)


def _shell(gb):
    return gb.shell(_sphere(gb), 0.2)


CASES = {
    "sphere": _sphere,
    "box": _box,
    "capsule": _capsule,
    "smooth_union": _union,
    "smooth_subtract": _subtract,
    "rigid": _rigid,
    "rigid_zero_rotation": _rigid_zero_rotation,
    "redistance": _redistance,
    "offset": _offset,
    "shell": _shell,
}


def _check_all_params(f, theta0, h=1e-6, rtol=5e-6, atol=1e-8):
    g = jax.grad(f)(theta0)
    assert np.all(np.isfinite(np.asarray(g))), "non-finite gradient"
    for i in range(theta0.size):
        fd = (f(theta0.at[i].add(h)) - f(theta0.at[i].add(-h))) / (2 * h)
        np.testing.assert_allclose(
            g[i], fd, rtol=rtol, atol=atol,
            err_msg=f"param {i}: AD {g[i]} vs FD {fd}")


@pytest.mark.parametrize("name", CASES)
def test_field_gradients(name):
    gb = GraphBuilder()
    root = CASES[name](gb)
    graph = gb.build()
    field = jax.jit(make_field(graph, root))
    theta0 = jnp.asarray(graph.theta0)

    def scalar(theta):
        return jnp.sum(jnp.sin(3.0 * field(theta, PROBES)))

    _check_all_params(scalar, theta0)


def test_mass_projection_gradients():
    """FD check of the full composed pipeline: precedence + PoU + integrals,
    including gradients w.r.t. every primitive parameter of both regions."""
    gb = GraphBuilder()
    a = gb.sphere((-0.3, 0.0, 0.0), 0.55)
    b = gb.box((0.35, 0.0, 0.0), (0.45, 0.35, 0.3))
    graph = gb.build()
    asm = Assembly(graph, (
        Component("A", a, density=2.0, precedence=1),
        Component("B", b, density=1.0, precedence=0),
    ))
    grid = GridSpec(lo=(-1.2, -0.9, -0.9), hi=(1.4, 0.9, 0.9), shape=(30, 22, 22))
    props = jax.jit(make_mass_properties(asm, grid))
    theta0 = jnp.asarray(graph.theta0)

    def scalar(theta):
        p = props(theta)
        return (jnp.sum(p["component_mass"] ** 2)
                + jnp.trace(p["inertia"]) + jnp.sum(p["com"] ** 2))

    _check_all_params(scalar, theta0, h=1e-5, rtol=1e-5)
