"""Standing CI: parameter sweep across a topology event (CLAUDE.md).

Two sphere components approach until they merge. The behavior at the merge is
characterized, not discovered: the stamp bumps where the hard connectivity
changes, gradients of the smoothed projections stay finite through the event,
and the smoothed objective itself stays continuous (no jumps beyond what the
sweep step explains).
"""
import jax
import jax.numpy as jnp
import numpy as np

from geomk.dag import GraphBuilder
from geomk.compose import Component, Assembly
from geomk.projections import GridSpec, make_mass_properties, make_contract
from geomk.topology import TopologyTracker, topology_signature

GRID = GridSpec(lo=(-1.6, -0.8, -0.8), hi=(1.6, 0.8, 0.8), shape=(48, 24, 24))


def build():
    gb = GraphBuilder()
    left = gb.sphere((-0.65, 0.0, 0.0), 0.42)
    right = gb.sphere((0.65, 0.0, 0.0), 0.42)
    graph = gb.build()
    asm = Assembly(graph, (
        Component("left", left, density=1.0, precedence=1),
        Component("right", right, density=1.0, precedence=0),
    ))
    # left sphere center-x is param 0 of node 0
    return asm, int(left.params[0])


def test_merge_event_is_characterized():
    asm, cx_idx = build()
    props = jax.jit(make_mass_properties(asm, GRID))
    theta0 = jnp.asarray(asm.graph.theta0)

    def total_mass(theta):
        return props(theta)["total_mass"]

    def inertia_trace(theta):
        return jnp.trace(props(theta)["inertia"])

    g_mass = jax.jit(jax.grad(total_mass))
    g_inertia = jax.jit(jax.grad(inertia_trace))

    tracker = TopologyTracker()
    sweep = np.linspace(-0.65, -0.05, 25)  # spheres touch at cx = -0.19
    stamps, masses = [], []
    for cx in sweep:
        theta = theta0.at[cx_idx].set(cx)
        sig = topology_signature(asm, theta, GRID)
        stamps.append(tracker.update(sig))
        masses.append(float(total_mass(theta)))
        for g in (g_mass(theta), g_inertia(theta)):
            assert np.all(np.isfinite(np.asarray(g))), f"non-finite grad at cx={cx}"

    # the merge happened and the stamp recorded it
    assert stamps[0] == 0
    assert stamps[-1] >= 1, "stamp never bumped across a merge"
    first_bump = int(np.argmax(np.diff(stamps) > 0))
    cx_bump = sweep[first_bump + 1]
    assert abs(cx_bump - (-0.19)) < 0.12, f"stamp bumped far from the merge: {cx_bump}"

    # smoothed projections stay continuous through the event
    steps = np.abs(np.diff(masses))
    assert steps.max() < 0.05, f"mass jumped across the sweep: {steps.max()}"

    # the contract publishes the stamp and honest metric status
    contract = make_contract(asm, GRID, tracker.stamp)
    assert contract.topology_stamp == tracker.stamp
    assert contract.metric_clean  # two exact-SDF roots, no smooth booleans
    assert contract.interface_identity is None  # reserved slot, deferred
