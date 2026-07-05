"""Pass 4 op breadth: revolve / extrude / loft / lattice.

Two things are pinned here beyond the gradcheck CI (test_gradcheck.py):

1. Metric honesty (invariant 3): revolve and extrude PRESERVE a clean
   profile's metric; loft and lattice are DIRTY and the require_clean ops
   (offset/shell) must keep refusing them at construction.
2. Sign correctness under the accurate mass path: loft's blend and revolve's
   meridian map have geometrically exact zero sets for circle profiles, so
   the Step-2 hardened supersampled occupancy must reproduce closed-form
   volumes (frustum, torus) within tolerance.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geomk.dag import GraphBuilder, metric_clean
from geomk.evaluate import make_field
from geomk.compose import Component, Assembly
from geomk.projections import GridSpec, make_mass_properties


# --- metric flags (invariant 3) ---------------------------------------------

def test_metric_flags_of_new_ops():
    gb = GraphBuilder()
    disk = gb.sphere((0.0, 0.6, 0.0), 0.25)          # clean profile
    rev = gb.revolve(disk)
    ext = gb.extrude(disk, 0.4)
    dirty_profile = gb.smooth_union(disk, gb.sphere((0.3, 0.0, 0.0), 0.4), k=0.1)
    rev_dirty = gb.revolve(dirty_profile)
    ext_dirty = gb.extrude(dirty_profile, 0.4)
    lof = gb.loft(disk, gb.sphere((0.0, 0.0, 0.0), 0.4), 0.5)
    lat = gb.lattice(1.0, 0.2)
    graph = gb.build()

    assert metric_clean(graph, rev)            # revolve preserves clean
    assert metric_clean(graph, ext)            # extrude preserves clean
    assert not metric_clean(graph, rev_dirty)  # ... and preserves dirty
    assert not metric_clean(graph, ext_dirty)
    assert not metric_clean(graph, lof)        # linear blend is never metric
    assert not metric_clean(graph, lat)        # folded lattice is never metric


def test_require_clean_still_refuses_dirty_inputs():
    gb = GraphBuilder()
    lof = gb.loft(gb.sphere((0, 0, 0), 0.5), gb.sphere((0, 0, 0), 0.3), 0.5)
    lat = gb.lattice(1.0, 0.2)
    with pytest.raises(ValueError, match="metric_clean"):
        gb.shell(lof, 0.05)
    with pytest.raises(ValueError, match="metric_clean"):
        gb.offset(lof, 0.05)
    with pytest.raises(ValueError, match="metric_clean"):
        gb.shell(lat, 0.05)
    # clean-preserving new ops feed require_clean ops directly ...
    gb.shell(gb.extrude(gb.sphere((0, 0, 0), 0.5), 0.4), 0.05)
    gb.offset(gb.revolve(gb.sphere((0.0, 0.6, 0.0), 0.25)), 0.05)
    # ... and an explicit redistance readmits the loft (invariant 3)
    gb.shell(gb.redistance(lof), 0.05)


# --- sign spot-checks --------------------------------------------------------

def test_loft_sign_is_exact_frustum():
    """For concentric circle profiles the blend's zero set IS the frustum:
    sign(d) == sign(r - R(z)) inside the slab, positive outside it."""
    R0, R1, h = 0.5, 0.3, 0.4
    gb = GraphBuilder()
    body = gb.loft(gb.sphere((0, 0, 0), R0), gb.sphere((0, 0, 0), R1), h)
    graph = gb.build()
    f = make_field(graph, body)
    theta = jnp.asarray(graph.theta0)

    rng = np.random.default_rng(11)
    pts = jnp.asarray(rng.uniform(-0.8, 0.8, size=(4000, 3)))
    d = np.asarray(f(theta, pts))
    x, y, z = (np.asarray(pts[:, i]) for i in range(3))
    r = np.hypot(x, y)
    t = np.clip((z + h) / (2 * h), 0.0, 1.0)
    inside = (np.abs(z) < h) & (r < (1 - t) * R0 + t * R1)
    np.testing.assert_array_equal(d < 0, inside)


# --- accurate-mass validation against closed-form volumes -------------------

def _accurate_total_mass(body, graph, grid):
    asm = Assembly(graph, (Component("body", body, density=1.0, precedence=0),))
    props = jax.jit(make_mass_properties(asm, grid, accurate=True, supersample=3))
    return float(props(jnp.asarray(graph.theta0))["total_mass"])


def test_lofted_fuselage_frustum_accurate_mass():
    """loft(circle R0 @ z=-h -> circle R1 @ z=+h) is the exact conical
    frustum: V = (pi * 2h / 3) (R0^2 + R0 R1 + R1^2)."""
    R0, R1, h = 0.5, 0.3, 0.4
    gb = GraphBuilder()
    body = gb.loft(gb.sphere((0, 0, 0), R0), gb.sphere((0, 0, 0), R1), h)
    graph = gb.build()
    grid = GridSpec(lo=(-0.65, -0.65, -0.55), hi=(0.65, 0.65, 0.55),
                    shape=(52, 52, 44))
    mass = _accurate_total_mass(body, graph, grid)
    analytic = np.pi * (2 * h) / 3 * (R0 ** 2 + R0 * R1 + R1 ** 2)
    rel = abs(mass - analytic) / analytic
    print(f"\nfrustum: analytic V = {analytic:.6f}, accurate mass = "
          f"{mass:.6f}, rel err = {100 * rel:.3f}%")
    assert rel <= 0.03


def test_revolved_torus_accurate_mass():
    """revolve(disk radius a centered at radius R off the x-axis) is the
    exact torus: V = 2 pi^2 R a^2."""
    R, a = 0.6, 0.25
    gb = GraphBuilder()
    body = gb.revolve(gb.sphere((0.0, R, 0.0), a))
    graph = gb.build()
    grid = GridSpec(lo=(-0.35, -0.95, -0.95), hi=(0.35, 0.95, 0.95),
                    shape=(28, 76, 76))
    mass = _accurate_total_mass(body, graph, grid)
    analytic = 2 * np.pi ** 2 * R * a ** 2
    rel = abs(mass - analytic) / analytic
    print(f"\ntorus: analytic V = {analytic:.6f}, accurate mass = "
          f"{mass:.6f}, rel err = {100 * rel:.3f}%")
    assert rel <= 0.03
