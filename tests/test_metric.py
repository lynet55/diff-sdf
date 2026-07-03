"""Pass 2 — metric discipline (invariant 3).

Sign is always correct; distance is trustworthy only where metric_clean.
Offset/shell must refuse a non-metric input at construction; an explicit
redistance node restores (approximately, first-order) metric distances and
the honest flag.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geomk.dag import GraphBuilder, metric_clean
from geomk.evaluate import make_field


def test_metric_flag_propagation():
    gb = GraphBuilder()
    s = gb.sphere((0, 0, 0), 0.5)
    b = gb.box((0.4, 0, 0), (0.4, 0.3, 0.3))
    moved = gb.rigid(s, translation=(0.1, 0.2, 0.0), rotvec=(0.1, 0.0, 0.3))
    u = gb.smooth_union(s, b, k=0.1)
    rd = gb.redistance(u)
    graph = gb.build()

    assert metric_clean(graph, s)          # exact SDF
    assert metric_clean(graph, moved)      # rigid transform preserves metric
    assert not metric_clean(graph, u)      # smooth boolean is signed-implicit only
    assert metric_clean(graph, rd)         # redistance restores (reported) metric


def test_offset_refuses_non_metric_input():
    gb = GraphBuilder()
    u = gb.smooth_union(gb.sphere((0, 0, 0), 0.5), gb.sphere((0.6, 0, 0), 0.5), k=0.1)
    with pytest.raises(ValueError, match="metric_clean"):
        gb.offset(u, 0.1)
    with pytest.raises(ValueError, match="metric_clean"):
        gb.shell(u, 0.1)
    # with an explicit redistance node both are allowed
    gb.offset(gb.redistance(u), 0.1)
    gb.shell(gb.redistance(u), 0.1)


def test_offset_and_shell_values_on_sphere():
    gb = GraphBuilder()
    s = gb.sphere((0, 0, 0), 0.5)
    off = gb.offset(s, 0.2)
    sh = gb.shell(s, 0.1)
    graph = gb.build()
    theta = jnp.asarray(graph.theta0)
    pts = jnp.array([[1.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.5, 0.0, 0.0]])

    d_off = make_field(graph, off)(theta, pts)
    np.testing.assert_allclose(d_off, [0.3, -0.4, -0.2], atol=1e-12)  # sphere r=0.7

    d_sh = make_field(graph, sh)(theta, pts)  # shell around r=0.5, thickness 0.1
    np.testing.assert_allclose(d_sh, [0.45, 0.15, -0.05], atol=1e-12)


def test_redistance_restores_unit_gradient_near_surface():
    """First-order redistance (d / |grad d|) is a *near-surface* guarantee:
    on the zero level set the redistanced field has unit spatial gradient, so
    offsets/shells of small magnitude are trustworthy. Far-field distance
    (e.g. the LSE union's -k*ln2 deep-overlap offset) is NOT repaired — that
    is why full redistancing stays on the deferred list and the flag's
    honesty level is documented."""
    gb1 = GraphBuilder()
    u1 = gb1.smooth_union(gb1.sphere((-0.4, 0, 0), 0.5), gb1.sphere((0.4, 0, 0), 0.5), k=0.15)
    g1 = gb1.build()

    gb2 = GraphBuilder()
    u2 = gb2.smooth_union(gb2.sphere((-0.4, 0, 0), 0.5), gb2.sphere((0.4, 0, 0), 0.5), k=0.15)
    rd = gb2.redistance(u2)
    g2 = gb2.build()

    rng = np.random.default_rng(3)
    pts = jnp.asarray(rng.uniform(-1.2, 1.2, size=(1500, 3)))

    theta1 = jnp.asarray(g1.theta0)
    f_raw = make_field(g1, u1)
    band = np.abs(np.asarray(f_raw(theta1, pts))) < 0.15  # near-surface probes
    pts_band = pts[band]

    def grad_norms(graph, root):
        theta = jnp.asarray(graph.theta0)
        f = make_field(graph, root)
        g = jax.vmap(jax.grad(lambda p: f(theta, p)))(pts_band)
        return np.asarray(jnp.linalg.norm(g, axis=-1))

    err_raw = np.abs(grad_norms(g1, u1) - 1.0)
    err_rd = np.abs(grad_norms(g2, rd) - 1.0)

    assert pts_band.shape[0] > 100      # the band actually sampled the surface
    assert err_raw.max() > 0.15         # the union really is non-metric
    assert err_rd.mean() < 0.5 * err_raw.mean()

    # right on the surface the guarantee is tight (second-order term ~ d)
    tight = np.abs(np.asarray(f_raw(theta1, pts_band))) < 0.03
    assert tight.sum() > 20
    assert err_rd[tight].max() < 0.1
