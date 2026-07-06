"""make_occupancy — the field-level projection added for the FEM consumer.

Pinned properties:
1. Consistency with the mass path: integrating the occupancy field must
   reproduce component volumes bit-comparably on the soft path and to
   straddle semantics on the accurate path (same GridSpec, same jitter,
   same PoU / hardened ownership).
2. Selection is by component index (intent mapping is the consumer's);
   selected occupancies add.
3. Differentiable: FD check of a smooth functional of the field; the
   accurate straddle's gradient equals the soft gradient exactly.
"""
import jax
import jax.numpy as jnp
import numpy as np

from geomk.dag import GraphBuilder
from geomk.compose import Component, Assembly
from geomk.projections import GridSpec, make_mass_properties, make_occupancy


def build():
    gb = GraphBuilder()
    a = gb.sphere((-0.3, 0.0, 0.0), 0.55)
    b = gb.box((0.35, 0.0, 0.0), (0.45, 0.35, 0.3))
    graph = gb.build()
    asm = Assembly(graph, (
        Component("A", a, density=2.0, precedence=1, intent="structural"),
        Component("B", b, density=1.0, precedence=0, intent="aero"),
    ))
    return asm


GRID = GridSpec(lo=(-1.2, -0.9, -0.9), hi=(1.4, 0.9, 0.9), shape=(30, 22, 22))


def test_soft_occupancy_integrates_to_component_volume():
    asm = build()
    theta = jnp.asarray(asm.graph.theta0)
    props = make_mass_properties(asm, GRID)(theta)
    for i in range(2):
        occ = make_occupancy(asm, GRID, components=[i])(theta)
        v = float(jnp.sum(occ)) * GRID.dV
        np.testing.assert_allclose(v, float(props["component_volume"][i]),
                                   rtol=1e-12)


def test_accurate_occupancy_integrates_to_accurate_volume():
    asm = build()
    theta = jnp.asarray(asm.graph.theta0)
    props = make_mass_properties(asm, GRID, accurate=True, supersample=3)(theta)
    for i in range(2):
        occ = make_occupancy(asm, GRID, components=[i], accurate=True,
                             supersample=3)(theta)
        v = float(jnp.sum(occ)) * GRID.dV
        np.testing.assert_allclose(v, float(props["component_volume"][i]),
                                   rtol=1e-12)


def test_selection_adds():
    asm = build()
    theta = jnp.asarray(asm.graph.theta0)
    both = make_occupancy(asm, GRID)(theta)
    sep = (make_occupancy(asm, GRID, components=[0])(theta)
           + make_occupancy(asm, GRID, components=[1])(theta))
    np.testing.assert_allclose(np.asarray(both), np.asarray(sep), atol=1e-12)


def test_indicator_mode_is_a_predicate_not_a_share():
    """Indicator occupancy ignores softmax competition: where a component owns
    material it reports ~1 even if another component's halo is near, and it
    integrates to at least the partition volume. Hard indicator == region
    membership by the composed field's sign."""
    asm = build()
    theta = jnp.asarray(asm.graph.theta0)
    for i in range(2):
        part = np.asarray(make_occupancy(asm, GRID, [i])(theta))
        ind = np.asarray(make_occupancy(asm, GRID, [i], mode="indicator")(theta))
        assert np.all(ind >= part - 1e-9)   # predicate dominates share
        assert float(ind.sum()) >= float(part.sum())
    # accurate indicator: exact membership fraction, straddle-gradable
    ind_acc = make_occupancy(asm, GRID, [0], accurate=True, mode="indicator")
    v = float(jnp.sum(ind_acc(theta))) * GRID.dV
    assert v > 0
    g = jax.grad(lambda th: jnp.sum(ind_acc(th)))(theta)
    assert np.all(np.isfinite(np.asarray(g)))


def test_occupancy_gradients():
    asm = build()
    theta0 = jnp.asarray(asm.graph.theta0)
    occ_soft = make_occupancy(asm, GRID, components=[0])
    occ_acc = make_occupancy(asm, GRID, components=[0], accurate=True)

    def scalar(theta):
        return jnp.sum(jnp.sin(3.0 * occ_soft(theta)))

    g = jax.grad(scalar)(theta0)
    assert np.all(np.isfinite(np.asarray(g)))
    h = 1e-6
    for i in range(theta0.size):
        fd = (scalar(theta0.at[i].add(h)) - scalar(theta0.at[i].add(-h))) / (2 * h)
        np.testing.assert_allclose(g[i], fd, rtol=5e-6, atol=1e-8)

    # straddle gradient == soft gradient exactly (locked fidelity rule)
    g_acc = jax.grad(lambda th: jnp.sum(jnp.sin(3.0 * occ_acc(th))))(theta0)
    # note: sin is applied to different VALUES (hard vs soft), so compare the
    # linear functional instead, where the straddle guarantee is exact
    g_lin_soft = jax.grad(lambda th: jnp.sum(occ_soft(th)))(theta0)
    g_lin_acc = jax.grad(lambda th: jnp.sum(occ_acc(th)))(theta0)
    np.testing.assert_allclose(np.asarray(g_lin_acc), np.asarray(g_lin_soft),
                               rtol=1e-12, atol=1e-15)
    assert np.all(np.isfinite(np.asarray(g_acc)))
