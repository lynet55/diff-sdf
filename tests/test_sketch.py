"""2D sketch profiles (polygon) + smooth_intersect (the third boolean).

Pins: (1) polygon is an EXACT 2D SDF read as an infinite prism — value-checked
against closed form for a square; (2) it handles CONCAVE profiles (winding
sign), so a point in a reflex notch reads outside; (3) the metric-clean chain
holds — polygon / extrude(polygon) / revolve(polygon) are metric_clean and can
be offset/shelled with no redistance node; (4) extrude(polygon rectangle)
reproduces the exact box SDF; (5) smooth_intersect is the smooth max (sign of
the intersected shape, exact-max limit as k->0).
"""
import jax.numpy as jnp
import numpy as np
import pytest

from geomk.dag import GraphBuilder, metric_clean
from geomk.evaluate import make_field

SQUARE = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
LSHAPE = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]   # reflex at (1,1)


def field_of(build):
    gb = GraphBuilder()
    root = build(gb)
    graph = gb.build()
    return make_field(graph, root), graph, root


def test_polygon_square_exact_sdf():
    f, graph, _ = field_of(lambda gb: gb.polygon(SQUARE))
    theta = jnp.asarray(graph.theta0)
    pts = jnp.array([[0.0, 0.0, 0.0],     # center: -0.5 to nearest edge
                     [1.0, 0.0, 0.3],     # right, z ignored: +0.5
                     [0.5, 0.0, 0.0],     # on edge: 0
                     [1.0, 1.0, 0.0]])    # off corner: +sqrt(0.5)
    d = np.asarray(f(theta, pts))
    np.testing.assert_allclose(d, [-0.5, 0.5, 0.0, np.sqrt(0.5)], atol=1e-9)


def test_polygon_concave_winding():
    """Reflex notch of an L reads OUTSIDE; interior reads inside."""
    f, graph, _ = field_of(lambda gb: gb.polygon(LSHAPE))
    theta = jnp.asarray(graph.theta0)
    pts = jnp.array([[0.5, 0.5, 0.0],     # solid arm: inside
                     [1.5, 1.5, 0.0],     # the missing quadrant: outside
                     [1.5, 0.5, 0.0]])    # other arm: inside
    d = np.asarray(f(theta, pts))
    assert d[0] < 0 and d[2] < 0 and d[1] > 0


def test_extrude_polygon_matches_box():
    """extrude(rectangle) is the exact box SDF (metric preserved)."""
    h = (0.5, 0.5, 0.4)
    fp, gp, _ = field_of(lambda gb: gb.extrude(gb.polygon(SQUARE), h[2]))
    fb, gb2, _ = field_of(lambda gb: gb.box((0, 0, 0), h))
    pts = jnp.asarray(np.random.default_rng(0).uniform(-1.2, 1.2, (200, 3)))
    dp = np.asarray(fp(jnp.asarray(gp.theta0), pts))
    db = np.asarray(fb(jnp.asarray(gb2.theta0), pts))
    np.testing.assert_allclose(dp, db, atol=1e-9)


def test_polygon_metric_clean_chain():
    gb = GraphBuilder()
    poly = gb.polygon(SQUARE)
    ext = gb.extrude(poly, 0.5)
    rev = gb.revolve(gb.polygon([(-0.2, 0.4), (0.2, 0.4), (0.0, 0.8)]))
    off = gb.offset(ext, 0.1)          # must NOT raise: extrude(polygon) clean
    sh = gb.shell(ext, 0.05)
    graph = gb.build()
    assert metric_clean(graph, poly)
    assert metric_clean(graph, ext)
    assert metric_clean(graph, rev)
    assert metric_clean(graph, off)
    assert metric_clean(graph, sh)


def test_offset_refuses_but_polygon_allows():
    """A polygon needs no redistance for offset; a smooth boolean still does."""
    gb = GraphBuilder()
    gb.offset(gb.extrude(gb.polygon(SQUARE), 0.5), 0.1)   # fine
    with pytest.raises(ValueError, match="metric_clean"):
        gb.offset(gb.smooth_intersect(gb.sphere((0, 0, 0), 0.5),
                                      gb.sphere((0.3, 0, 0), 0.5)), 0.1)


def test_smooth_intersect_sign_and_limit():
    a = lambda gb: gb.sphere((-0.3, 0, 0), 0.5)
    b = lambda gb: gb.sphere((0.3, 0, 0), 0.5)
    f, graph, _ = field_of(lambda gb: gb.smooth_intersect(a(gb), b(gb), k=0.02))
    theta = jnp.asarray(graph.theta0)
    pts = jnp.array([[0.0, 0.0, 0.0],     # inside both -> inside
                     [-0.5, 0.0, 0.0],    # inside A only -> outside intersect
                     [0.5, 0.0, 0.0]])    # inside B only -> outside intersect
    d = np.asarray(f(theta, pts))
    assert d[0] < 0 and d[1] > 0 and d[2] > 0
    # k->0 limit approaches the exact max of the two sphere SDFs: at the center
    # both are -0.2, so the intersection ~ -0.2 (+ O(k ln2) smoothing bias)
    np.testing.assert_allclose(d[0], -0.2, atol=0.02)
