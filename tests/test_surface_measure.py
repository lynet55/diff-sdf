"""make_surface_measure — surface quantities via the co-area formula.

Pins: (1) value honesty against closed-form areas (sphere, box, torus) at
resolved scales; (2) projected (silhouette) area of a convex body; (3) the
FD gradient gate through the second-order path (theta-grad of an x-grad);
(4) internal interfaces contribute ~nothing (wetted area of two mated boxes
== wetted area of the equivalent single box, not the sum of both).
"""
import jax
import jax.numpy as jnp
import numpy as np

from geomk.dag import GraphBuilder
from geomk.compose import Component, Assembly
from geomk.projections import GridSpec, make_surface_measure


def single(builder_fn, lo, hi, shape):
    gb = GraphBuilder()
    root = builder_fn(gb)
    graph = gb.build()
    asm = Assembly(graph, (Component("s", root, density=1.0, precedence=0),))
    return asm, graph, GridSpec(lo=lo, hi=hi, shape=shape)


def test_sphere_area_and_silhouette():
    R = 0.5
    asm, graph, grid = single(lambda gb: gb.sphere((0, 0, 0), R),
                              (-0.8, -0.8, -0.8), (0.8, 0.8, 0.8),
                              (56, 56, 56))
    surf = make_surface_measure(asm, grid)
    out = surf(jnp.asarray(graph.theta0), direction=(1.0, 0.0, 0.0))
    area = float(out["wetted_area"])
    proj = float(out["projected_area"])
    np.testing.assert_allclose(area, 4 * np.pi * R ** 2, rtol=0.03)
    np.testing.assert_allclose(proj, np.pi * R ** 2, rtol=0.03)


def test_box_area():
    h = (0.45, 0.3, 0.2)
    asm, graph, grid = single(lambda gb: gb.box((0, 0, 0), h),
                              (-0.75, -0.6, -0.5), (0.75, 0.6, 0.5),
                              (52, 44, 36))
    surf = make_surface_measure(asm, grid)
    area = float(surf(jnp.asarray(graph.theta0))["wetted_area"])
    hx, hy, hz = h
    analytic = 8 * (hx * hy + hy * hz + hx * hz)
    np.testing.assert_allclose(area, analytic, rtol=0.04)


def test_torus_area():
    R, a = 0.55, 0.2
    asm, graph, grid = single(lambda gb: gb.revolve(gb.sphere((0, R, 0), a)),
                              (-0.35, -0.85, -0.85), (0.35, 0.85, 0.85),
                              (28, 64, 64))
    surf = make_surface_measure(asm, grid)
    area = float(surf(jnp.asarray(graph.theta0))["wetted_area"])
    np.testing.assert_allclose(area, 4 * np.pi ** 2 * R * a, rtol=0.04)


def test_internal_interfaces_do_not_count():
    """A flush DECLARED-MATE seam is not exterior surface: its co-area
    density is quiet (<10% of a real face), while the same geometry unmated
    shows the probe-B void ridge loudly (>50% of a face). Two separate
    effects are pinned: the seam silence, and the mate BRIDGE's outer-
    surface bias — the bridge deepens the background by up to
    k_bridge*ln2 = 5*tau*ln2 everywhere the two fields' sum is O(k_bridge),
    which fattens the measured area by ~13% at this deliberately coarse
    tau = 0.049 (bias shrinks proportionally with tau)."""
    gb = GraphBuilder()
    a = gb.box((-0.25, 0, 0), (0.25, 0.3, 0.2))
    b = gb.box((0.25, 0, 0), (0.25, 0.3, 0.2))
    graph = gb.build()
    grid = GridSpec(lo=(-0.85, -0.6, -0.5), hi=(0.85, 0.6, 0.5),
                    shape=(52, 40, 32))
    theta = jnp.asarray(graph.theta0)

    def measure(mates):
        asm = Assembly(graph, (
            Component("A", a, density=1.0, precedence=1),
            Component("B", b, density=1.0, precedence=0)), mates=mates)
        out = make_surface_measure(asm, grid)(theta)
        dens = np.asarray(out["surface_density"]).reshape(grid.shape)
        prof = dens[:, grid.shape[1] // 2, grid.shape[2] // 2]
        xs = np.asarray(grid.points()).reshape(*grid.shape, 3)[
            :, grid.shape[1] // 2, grid.shape[2] // 2, 0]
        seam = prof[np.abs(xs) < 0.15].max()
        face = prof[np.abs(xs) >= 0.15].max()
        return float(out["wetted_area"]), seam, face

    area_m, seam_m, face_m = measure(((0, 1),))
    area_u, seam_u, face_u = measure(())
    assert seam_m < 0.10 * face_m          # mated seam: quiet
    assert seam_u > 0.50 * face_u          # unmated: void-ridge walls, loud

    asm1, graph1, grid1 = single(lambda gb: gb.box((0, 0, 0), (0.5, 0.3, 0.2)),
                                 (-0.85, -0.6, -0.5), (0.85, 0.6, 0.5),
                                 (52, 40, 32))
    area1 = float(make_surface_measure(asm1, grid1)(
        jnp.asarray(graph1.theta0))["wetted_area"])
    assert area_m / area1 < 1.16           # bridge bias, bounded + documented


def test_surface_gradients_match_fd():
    """theta-gradient of the co-area integrals (a second-order quantity)
    against central FD — the probe-G gate at test scale."""
    asm, graph, grid = single(lambda gb: gb.sphere((0.05, -0.1, 0.02), 0.42),
                              (-0.7, -0.7, -0.7), (0.7, 0.7, 0.7),
                              (30, 30, 30))
    surf = make_surface_measure(asm, grid)
    theta0 = jnp.asarray(graph.theta0)

    def scalar(theta):
        out = surf(theta, direction=(0.3, 0.9, 0.2))
        return out["wetted_area"] + 2.0 * out["projected_area"]

    g = jax.grad(scalar)(theta0)
    assert np.all(np.isfinite(np.asarray(g)))
    h = 1e-6
    for i in range(theta0.size):
        fd = (scalar(theta0.at[i].add(h)) - scalar(theta0.at[i].add(-h))) / (2 * h)
        np.testing.assert_allclose(g[i], fd, rtol=1e-6, atol=1e-10)
