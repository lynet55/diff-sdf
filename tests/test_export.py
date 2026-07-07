"""Watertight export (geomk.export).

Pins: (1) a sphere exports a closed 2-manifold (no boundary/non-manifold edges)
of the right genus (Euler 2) and its enclosed volume matches the analytic
value; (2) 'watertight' is topology-general — a revolved torus is closed and
manifold with Euler 0 (genus 1), not 2; (3) the mesh is FAITHFUL — its enclosed
volume matches the field's own occupancy integral; (4) a composed 2-component
assembly exports per-component closed meshes AND a closed union whose volume is
the occupancy union; (5) binary STL round-trips its triangle count.
"""
import struct

import jax.numpy as jnp
import numpy as np

from geomk.dag import GraphBuilder
from geomk.compose import Component, Assembly
from geomk.export import (export_solid, export_components, is_watertight,
                          mesh_volume, write_stl, domain_clipped)
from geomk.projections import GridSpec, make_mass_properties


def _asm(build, **kw):
    gb = GraphBuilder()
    root = build(gb)
    graph = gb.build()
    return Assembly(graph, (Component("s", root, density=1.0, precedence=0),))


def test_sphere_watertight_and_volume():
    R = 0.6
    asm = _asm(lambda gb: gb.sphere((0, 0, 0), R))
    grid = GridSpec(lo=(-1, -1, -1), hi=(1, 1, 1), shape=(96, 96, 96))
    theta = jnp.asarray(asm.graph.theta0)
    assert not domain_clipped(asm, theta, grid)
    m = export_solid(asm, theta, grid)
    ok, st = is_watertight(m)
    assert ok, st
    assert st["euler"] == 2                     # genus 0
    np.testing.assert_allclose(mesh_volume(m), 4 / 3 * np.pi * R ** 3, rtol=0.02)


def test_torus_watertight_genus_one():
    asm = _asm(lambda gb: gb.revolve(gb.sphere((0.0, 0.6, 0.0), 0.22)))
    grid = GridSpec(lo=(-0.4, -1, -1), hi=(0.4, 1, 1), shape=(40, 100, 100))
    m = export_solid(asm, jnp.asarray(asm.graph.theta0), grid)
    ok, st = is_watertight(m)
    assert ok, st                               # closed + manifold ...
    assert st["euler"] == 0                     # ... but genus 1, not a sphere


def test_mesh_volume_matches_occupancy():
    """The exported shell's volume equals the field's own occupancy integral."""
    asm = _asm(lambda gb: gb.capsule((-0.4, 0, 0), (0.4, 0.1, 0.0), 0.3))
    grid = GridSpec(lo=(-1, -0.8, -0.8), hi=(1, 0.8, 0.8), shape=(110, 90, 90))
    theta = jnp.asarray(asm.graph.theta0)
    m = export_solid(asm, theta, grid)
    occ_vol = float(make_mass_properties(asm, grid, fidelity="straddle")(theta)["total_mass"])  # rho=1
    np.testing.assert_allclose(mesh_volume(m), occ_vol, rtol=0.03)


def test_two_component_export():
    gb = GraphBuilder()
    a = gb.sphere((-0.3, 0, 0), 0.5)
    b = gb.box((0.35, 0, 0), (0.45, 0.35, 0.3))
    graph = gb.build()
    asm = Assembly(graph, (
        Component("A", a, density=1.0, precedence=1),
        Component("B", b, density=1.0, precedence=0)))
    grid = GridSpec(lo=(-1.2, -0.9, -0.9), hi=(1.4, 0.9, 0.9), shape=(90, 66, 66))
    theta = jnp.asarray(graph.theta0)

    comps = export_components(asm, theta, grid)
    assert len(comps) == 2
    for m in comps:
        assert is_watertight(m)[0], m.name

    solid = export_solid(asm, theta, grid)
    assert is_watertight(solid)[0]
    # union volume ~ occupancy total (both regions, rho=1)
    occ = float(make_mass_properties(asm, grid, fidelity="straddle")(theta)["total_mass"])
    np.testing.assert_allclose(mesh_volume(solid), occ, rtol=0.04)


def test_stl_roundtrip(tmp_path):
    asm = _asm(lambda gb: gb.box((0, 0, 0), (0.4, 0.3, 0.5)))
    grid = GridSpec(lo=(-0.8, -0.8, -0.8), hi=(0.8, 0.8, 0.8), shape=(64, 64, 64))
    m = export_solid(asm, jnp.asarray(asm.graph.theta0), grid)
    p = tmp_path / "box.stl"
    n = write_stl(str(p), m)
    with open(p, "rb") as f:
        f.read(80)
        count = struct.unpack("<I", f.read(4))[0]
    assert count == n == len(m.faces)
    assert p.stat().st_size == 84 + 50 * n
