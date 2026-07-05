"""Value accurate, gradient smooth + declared-mate sharpening + grid jitter.

The accurate absolute-quantity path (hardened supersampled occupancy behind a
stop-gradient straddle) must return values near the exact geometry while its
gradients remain exactly the soft partition-of-unity gradients. Declared
mates must remove the O(k) erosion of a flush lower-precedence region and the
background void ridge at the mating plane, and must be strictly opt-in.
"""
import jax
import jax.numpy as jnp
import numpy as np

from geomk.dag import GraphBuilder
from geomk.compose import (Component, Assembly, make_region_fields,
                           make_background_field, pou_weights)
from geomk.projections import (GridSpec, make_mass_properties, make_contract,
                               BG_BRIDGE_TAUS, GRID_JITTER)

HX, HY, HZ = 0.5, 0.3, 0.2


def flush_boxes(mated):
    """A: x in [-1,0] (precedence 1, rho 2). B: x in [0,1] (precedence 0)."""
    gb = GraphBuilder()
    a = gb.box((-HX, 0, 0), (HX, HY, HZ))
    b = gb.box((HX, 0, 0), (HX, HY, HZ))
    graph = gb.build()
    return Assembly(graph, (
        Component("A", a, density=2.0, precedence=1),
        Component("B", b, density=1.0, precedence=0),
    ), k_compose=0.08, mates=((0, 1),) if mated else ())


GRID = GridSpec(lo=(-1.15, -0.45, -0.35), hi=(1.15, 0.45, 0.35),
                shape=(96, 24, 24))
V_EXACT = 2 * HX * 2 * HY * 2 * HZ  # each box, exact


def test_accurate_values_land_on_exact_geometry():
    asm = flush_boxes(mated=True)
    theta = jnp.asarray(asm.graph.theta0)
    p = make_mass_properties(asm, GRID, accurate=True, supersample=3)(theta)
    np.testing.assert_allclose(np.asarray(p["component_volume"]),
                               [V_EXACT, V_EXACT], rtol=0.01)
    np.testing.assert_allclose(float(p["total_mass"]), 3.0 * V_EXACT, rtol=0.01)
    np.testing.assert_allclose(np.asarray(p["com"]),
                               [-1.0 / 6.0, 0.0, 0.0], atol=0.01)


def test_straddle_gradient_is_exactly_the_soft_gradient():
    """Per straddled output O: value(O) = O_accurate, grad(O) = grad(O_soft)
    exactly. (A nonlinear consumer of several outputs then gets the soft
    sensitivities chained through its own Jacobian at the accurate values —
    the intended co-design linearization.)"""
    asm = flush_boxes(mated=False)
    theta = jnp.asarray(asm.graph.theta0)
    soft = make_mass_properties(asm, GRID)
    acc = make_mass_properties(asm, GRID, accurate=True, supersample=2)

    w = jnp.asarray([1.7, -0.4])
    W = jnp.asarray(np.random.default_rng(0).normal(size=(3, 3)))

    def linear(props):  # linear functional of the straddled outputs
        return lambda th: (props(th)["total_mass"]
                           + w @ props(th)["component_mass"]
                           + jnp.sum(W * props(th)["inertia"])
                           + jnp.sum(props(th)["com"]))

    g_soft = jax.grad(linear(soft))(theta)
    g_acc = jax.grad(linear(acc))(theta)
    np.testing.assert_allclose(np.asarray(g_acc), np.asarray(g_soft),
                               rtol=1e-12, atol=1e-14)
    # values, by contrast, are the accurate ones
    p_acc = acc(theta)
    p_soft = soft(theta)
    assert abs(float(p_acc["total_mass"]) - 3.0 * V_EXACT) < 0.01
    assert abs(float(p_soft["total_mass"] - p_acc["total_mass"])) > 0.005
    # and the soft gradient itself still agrees with central FD
    i = 3  # A half-extent hx (positive param)
    h = 1e-5
    f = linear(soft)
    fd = (f(theta.at[i].add(h)) - f(theta.at[i].add(-h))) / (2 * h)
    np.testing.assert_allclose(float(g_soft[i]), float(fd), rtol=1e-5)


def test_mate_removes_erosion_and_void_ridge():
    theta = jnp.asarray(flush_boxes(False).graph.theta0)
    grid = GridSpec(lo=(-1.15, -0.45, -0.35), hi=(1.15, 0.45, 0.35),
                    shape=(192, 24, 24), tau=0.018)  # ~1.5 * dx_x
    m_un = make_mass_properties(flush_boxes(False), grid)(theta)
    m_ma = make_mass_properties(flush_boxes(True), grid)(theta)
    err_un = (np.asarray(m_un["component_volume"]) - V_EXACT) / V_EXACT
    err_ma = (np.asarray(m_ma["component_volume"]) - V_EXACT) / V_EXACT
    # erosion = how much worse the flush lower-precedence B fares than A
    # (A carries only the generic grid/smoothing error, B additionally the
    # O(k) interface erosion). The mate must remove the differential.
    erosion_un = err_un[1] - err_un[0]
    erosion_ma = err_ma[1] - err_ma[0]
    assert erosion_un < -0.05, f"unmated erosion vanished? {erosion_un}"
    assert abs(erosion_ma) < 0.005, f"mate left erosion: {erosion_ma}"

    # background void ridge along the mating normal
    xs = np.linspace(-0.25, 0.25, 2001)
    ln = jnp.asarray(np.stack([xs, 0 * xs, 0 * xs], axis=-1))
    tau = 0.03
    asm_m = flush_boxes(True)
    phi = make_region_fields(asm_m)(theta, ln)
    phi_bg = make_background_field(asm_m)(theta, ln, BG_BRIDGE_TAUS * tau)
    _, wbg = pou_weights(phi, tau, phi_bg)
    assert float(jnp.max(wbg)) <= 0.05

    asm_u = flush_boxes(False)
    _, wbg_u = pou_weights(make_region_fields(asm_u)(theta, ln), tau)
    assert float(jnp.max(wbg_u)) > 0.3  # the defect the mate declaration fixes


def test_no_mates_is_byte_for_byte_unchanged():
    """Default composition must be exactly the smooth log-sum-exp path."""
    asm = flush_boxes(False)
    theta = jnp.asarray(asm.graph.theta0)
    pts = jnp.asarray(GRID.points()[::97])
    phi = make_region_fields(asm)(theta, pts)
    from geomk.evaluate import eval_node
    from geomk.reparam import constrain
    q = constrain(theta, asm.graph.positive_mask)
    d = [eval_node(asm.graph, c.root.node, q, pts) for c in asm.components]
    k = asm.k_compose
    np.testing.assert_array_equal(np.asarray(phi[0]), np.asarray(d[0]))
    np.testing.assert_array_equal(
        np.asarray(phi[1]),
        np.asarray(k * jnp.logaddexp(d[1] / k, -d[0] / k)))


def test_grid_jitter_deterministic_subvoxel():
    p1, p2 = GRID.points(), GRID.points()
    np.testing.assert_array_equal(p1, p2)
    # jittered centers are the unjittered ones shifted by jitter*dx per axis
    plain = GridSpec(lo=GRID.lo, hi=GRID.hi, shape=GRID.shape,
                     jitter=(0.0, 0.0, 0.0)).points()
    off = (p1 - plain)
    for ax in range(3):
        np.testing.assert_allclose(off[:, ax], GRID_JITTER[ax] * GRID.dx[ax],
                                   rtol=0, atol=1e-12)
        assert 0.0 < GRID_JITTER[ax] * GRID.dx[ax] < GRID.dx[ax]
    # supersampled points refine each cell with the same deterministic rule
    ss = GRID.points(3)
    assert ss.shape == (p1.shape[0] * 27, 3)


def test_contract_publishes_accurate_path_and_mates():
    asm = flush_boxes(True)
    c = make_contract(asm, GRID, topology_stamp=0, accurate=True, supersample=3)
    assert "hardened supersampled occupancy" in c.absolute_value_path
    assert "stop-gradient straddle" in c.absolute_value_path
    assert c.sharpened_mates == ((0, 1),)

    c_soft = make_contract(flush_boxes(False), GRID, topology_stamp=0)
    assert "soft smoothed-occupancy" in c_soft.absolute_value_path
    assert c_soft.sharpened_mates == ()
