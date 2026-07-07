"""Contract hardening — the probe findings encoded as standing CI.

Pins three things the probes paid to learn:
1. Resolution diagnostics: saturation maps probe A's t >= 2*tau trust rule to
   a per-component, differentiable signal (threshold ~0.88); inflation flags
   the tau-halo bias. An optimizer can subscribe to saturation as a barrier.
2. Straddle scope: gradients of the accurate straddle equal soft gradients
   for LINEAR functionals exactly, and measurably diverge for nonlinear
   ones — that divergence is documented behavior, not a bug to 'fix' by
   changing straddle semantics silently.
3. The fidelity knob: 'soft'/'straddle' is the explicit spelling of the
   accurate flag, bit-identical to it, and rejects unknown values.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geomk.dag import GraphBuilder
from geomk.compose import Component, Assembly
from geomk.projections import (GridSpec, make_mass_properties, make_occupancy,
                               make_resolution_diagnostics, make_contract,
                               TRUST_SATURATION)

TAU = 0.03


def slab_assembly():
    """Three z-slabs with half-thickness 0.5*tau / 2*tau / 4*tau."""
    gb = GraphBuilder()
    slabs = [gb.box((x, 0.0, 0.0), (0.28, 0.35, hz))
             for x, hz in ((-0.7, 0.5 * TAU), (0.0, 2 * TAU), (0.7, 4 * TAU))]
    graph = gb.build()
    comps = tuple(Component(f"s{i}", s, density=1.0, precedence=i)
                  for i, s in enumerate(slabs))
    return Assembly(graph, comps), graph


GRID = GridSpec(lo=(-1.1, -0.5, -0.25), hi=(1.1, 0.5, 0.25),
                shape=(56, 26, 26), tau=TAU)


def test_saturation_encodes_the_trust_rule():
    asm, graph = slab_assembly()
    diag = make_resolution_diagnostics(asm, GRID)(jnp.asarray(graph.theta0))
    sat = np.asarray(diag["saturation"])
    # monotone in thickness; the 2*tau slab sits near the threshold
    assert sat[0] < sat[1] < sat[2]
    assert sat[0] < 0.80              # half tau: clearly below trust
    assert abs(sat[1] - TRUST_SATURATION) < 0.06   # at the floor
    assert sat[2] > 0.93              # well resolved
    assert float(diag["trust_threshold"]) == TRUST_SATURATION
    assert float(diag["tau"]) == TAU


def test_saturation_is_a_usable_barrier():
    """Differentiable and pointing the right way: thickening the thin slab
    raises its saturation."""
    asm, graph = slab_assembly()
    diag = make_resolution_diagnostics(asm, GRID)
    hz_idx = 5   # slab 0's z half-extent (box params 0..5, hz last)

    def sat_thin(theta):
        return diag(theta)["saturation"][0]

    theta0 = jnp.asarray(graph.theta0)
    g = jax.grad(sat_thin)(theta0)
    assert np.all(np.isfinite(np.asarray(g)))
    assert g[hz_idx] > 1e-3


def test_inflation_flags_the_tau_halo():
    """Inflation is a comparative signal: resolved bodies carry an
    edge/corner-halo baseline (~1.17 here), a sub-floor feature jumps
    well past 2 (measured 2.36 for the 0.5*tau slab)."""
    asm, graph = slab_assembly()
    diag = make_resolution_diagnostics(asm, GRID)(jnp.asarray(graph.theta0))
    infl = np.asarray(diag["inflation"])
    assert infl[0] > infl[1] > infl[2]      # monotone in thinness
    assert infl[0] > 2.0                    # sub-floor: unmistakable
    assert infl[2] < 1.30                   # resolved: baseline only


def test_straddle_scope_is_pinned():
    """Linear functional: straddle grad == soft grad to machine precision.
    Nonlinear functional: they measurably diverge (documented behavior)."""
    asm, graph = slab_assembly()
    theta0 = jnp.asarray(graph.theta0)
    occ_soft = make_occupancy(asm, GRID, [0], fidelity="soft")
    occ_str = make_occupancy(asm, GRID, [0], fidelity="straddle")

    g_lin_soft = jax.grad(lambda th: jnp.sum(occ_soft(th)))(theta0)
    g_lin_str = jax.grad(lambda th: jnp.sum(occ_str(th)))(theta0)
    np.testing.assert_allclose(np.asarray(g_lin_str), np.asarray(g_lin_soft),
                               rtol=1e-12, atol=1e-15)

    g_nl_soft = np.asarray(jax.grad(
        lambda th: jnp.sum(occ_soft(th) ** 2))(theta0))
    g_nl_str = np.asarray(jax.grad(
        lambda th: jnp.sum(occ_str(th) ** 2))(theta0))
    rel = (np.linalg.norm(g_nl_str - g_nl_soft)
           / np.linalg.norm(g_nl_soft))
    assert rel > 0.05, (
        "nonlinear straddle/soft gradients now agree — if straddle semantics "
        "changed, update the contract's straddle_scope statement too")


def test_fidelity_knob():
    asm, graph = slab_assembly()
    theta0 = jnp.asarray(graph.theta0)
    for fid, acc in (("soft", False), ("straddle", True)):
        a = make_mass_properties(asm, GRID, fidelity=fid)(theta0)
        b = make_mass_properties(asm, GRID, accurate=acc)(theta0)
        for k in a:
            np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))
    with pytest.raises(ValueError, match="fidelity"):
        make_mass_properties(asm, GRID, fidelity="exact")
    with pytest.raises(ValueError, match="fidelity"):
        make_occupancy(asm, GRID, [0], fidelity="hard")


def test_contract_publishes_the_obligations():
    asm, graph = slab_assembly()
    c = make_contract(asm, GRID, topology_stamp=0)
    assert "LINEAR integrands" in c.occupancy_semantics
    assert "fidelity='soft'" in c.straddle_scope
    assert "2*tau" in c.resolution_rule
