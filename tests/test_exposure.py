"""Pass 3 CI — exposure: flat decision vector + mask, macro relations,
symmetry coupling, all differentiable end to end (FD-checked)."""
import jax
import jax.numpy as jnp
import numpy as np

from geomk.dag import GraphBuilder
from geomk.compose import Component, Assembly
from geomk.evaluate import make_field
from geomk.exposure import Exposure
from geomk.projections import GridSpec, make_mass_properties


def build():
    gb = GraphBuilder()
    fus = gb.capsule((-1.5, 0, 0), (1.5, 0, 0), 0.4)
    wing_r = gb.box((0.0, 1.1, 0.0), (0.4, 1.0, 0.1))
    wing_l = gb.box((0.0, -1.1, 0.0), (0.4, 1.0, 0.1))
    wings = gb.smooth_union(wing_r, wing_l, k=0.08)
    graph = gb.build()

    ex = Exposure(graph)
    z_r = ex.expose(fus.params[6], "fuselage_radius")
    # macro: wing y-position magnitude, mirrored across the xz-plane
    z_y = ex.macro("wing_y_offset", init=1.1)
    ex.drive(wing_r.params[1], z_y, scale=+1.0)   # right wing center-y
    ex.tie(wing_l.params[1], z_y, scale=-1.0)     # mirrored left wing
    emap = ex.build()

    asm = Assembly(graph, (
        Component("fuselage", fus, density=2.0, precedence=1, intent="structural"),
        Component("wings", wings, density=1.0, precedence=0, intent="aero"),
    ))
    return graph, asm, emap, wings, (z_r, z_y)


def test_z0_reproduces_base_theta():
    graph, asm, emap, wings, _ = build()
    np.testing.assert_allclose(np.asarray(emap.theta(jnp.asarray(emap.z0))),
                               graph.theta0, atol=1e-12)
    # mask selects exactly the three driven params
    assert emap.mask.sum() == 3


def test_mirror_symmetry_holds_under_macro_change():
    graph, asm, emap, wings, (z_r, z_y) = build()
    z = jnp.asarray(emap.z0).at[z_y].add(0.35)  # move both wings outboard
    theta = emap.theta(z)
    f = make_field(graph, wings)
    pts = jnp.asarray(np.random.default_rng(0).uniform(-1.5, 1.5, size=(50, 3)))
    mirrored = pts * jnp.array([1.0, -1.0, 1.0])
    np.testing.assert_allclose(f(theta, pts), f(theta, mirrored), atol=1e-12)


def test_gradients_through_relations_layer():
    graph, asm, emap, wings, (z_r, z_y) = build()
    grid = GridSpec(lo=(-2.0, -2.4, -0.7), hi=(2.0, 2.4, 0.7), shape=(36, 44, 14))
    props = jax.jit(make_mass_properties(asm, grid))

    def scalar(z):
        p = props(emap.theta(z))
        return jnp.sum(p["component_mass"] ** 2) + jnp.trace(p["inertia"])

    z0 = jnp.asarray(emap.z0)
    g = jax.grad(scalar)(z0)
    assert np.all(np.isfinite(np.asarray(g)))
    assert abs(g[z_r]) > 1e-3   # radius drives mass
    assert abs(g[z_y]) > 1e-3   # span position drives inertia

    h = 1e-5
    for k in range(z0.size):
        fd = (scalar(z0.at[k].add(h)) - scalar(z0.at[k].add(-h))) / (2 * h)
        np.testing.assert_allclose(g[k], fd, rtol=1e-5,
                                   err_msg=f"z[{k}] ({emap.names[k]})")
