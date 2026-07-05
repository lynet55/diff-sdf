"""Domains compose by precedence (invariant 5); soft membership (invariant 4).

Hard ledger: each Component owns exactly one subtree root (authoring-time).
Soft field: a region's composed field is its own field minus the smooth union
of all strictly-higher-precedence regions' fields; point->region membership is
then a partition of unity with temperature tau (evaluation-time), hardened
only on export as the tau->0 projection.

`intent` is inert annotation (invariant 6): stored, displayed, never read by
any composition code. `domain` participates: a "void" region carves lower-
precedence solids simply by owning the overlap and carrying zero density.
"""
from dataclasses import dataclass

import jax.numpy as jnp

from .dag import Graph, _nid
from .evaluate import eval_node
from .ops import smin
from .reparam import constrain


@dataclass(frozen=True)
class Component:
    name: str
    root: object            # Handle or node id
    density: float
    precedence: int         # higher wins the overlap
    intent: str = ""        # inert (invariant 6)
    domain: str = "solid"   # solid | void  (fluid/interface deferred)


@dataclass(frozen=True)
class Assembly:
    graph: Graph
    components: tuple
    k_compose: float = 0.08  # smoothing bandwidth of precedence composition
    # Declared mates: unordered component-index pairs the author asserts are
    # flush/coincident. For a mated pair the higher-precedence contribution is
    # composed with an EXACT max (no O(k) erosion of the lower-precedence
    # region) and the PoU background is bridged across the mated seam (no
    # spurious void ridge at the mating plane). Default: none — composition
    # is byte-for-byte the smooth log-sum-exp path.
    mates: tuple = ()

    @property
    def densities(self):
        return jnp.array([c.density if c.domain == "solid" else 0.0
                          for c in self.components])

    @property
    def mate_set(self):
        return frozenset(frozenset(m) for m in self.mates)


def make_region_fields(asm: Assembly):
    """Returns f(theta, points) -> (n_regions, ...) precedence-composed fields
    phi_i, in component order. phi_i <= 0 means point belongs to region i.

    Non-mate higher-precedence contributions use the smooth log-sum-exp
    subtract (bandwidth k_compose). Declared-mate contributions use the exact
    max(d_i, -d_j): the ownership boundary sits exactly on the mate's surface,
    so a flush lower-precedence region is not eroded by O(k)."""
    roots = [_nid(c.root) for c in asm.components]
    prec = [c.precedence for c in asm.components]
    k = asm.k_compose
    mates = asm.mate_set

    def region_fields(theta, points):
        q = constrain(theta, asm.graph.positive_mask)
        d = [eval_node(asm.graph, r, q, points) for r in roots]
        phi = []
        for i in range(len(d)):
            hi = [j for j in range(len(d)) if prec[j] > prec[i]]
            higher = [d[j] for j in hi if frozenset((i, j)) not in mates]
            mated = [d[j] for j in hi if frozenset((i, j)) in mates]
            f = d[i]
            if higher:
                h = smin(jnp.stack(higher), k) if len(higher) > 1 else higher[0]
                # smooth subtract: smax(d_i, -h)
                f = k * jnp.logaddexp(f / k, -h / k)
            for dj in mated:                       # exact subtract (mate)
                f = jnp.maximum(f, -dj)
            phi.append(f)
        return jnp.stack(phi)

    return region_fields


def make_background_field(asm: Assembly):
    """Mate-aware background (void) field: bg(theta, points, k_bridge) ->
    phi_bg, negative outside the solid, positive inside.

    The plain PoU background (zero logit) sees a spurious void ridge at a
    flush mate: both region fields are exactly zero on the mating plane, a
    three-way softmax tie. A declared mate asserts there is no void at that
    interface, so the background is the raw union of the components with the
    mated seams bridged: for each mate pair (i, j) the union is deepened by
    k_bridge * softplus(-(d_i + d_j) / k_bridge). d_i + d_j vanishes exactly
    where the two surfaces oppose each other (the flush seam, filled by
    k_bridge*ln 2) and grows away from it, so — unlike a smooth union of the
    pair — the bridge does not inflate the union outside the seam rim, where
    d_i ~ d_j but both are positive."""
    roots = [_nid(c.root) for c in asm.components]
    pairs = tuple(tuple(sorted(m)) for m in asm.mate_set)

    def background(theta, points, k_bridge):
        q = constrain(theta, asm.graph.positive_mask)
        d = [eval_node(asm.graph, r, q, points) for r in roots]
        u = jnp.min(jnp.stack(d), axis=0) if len(d) > 1 else d[0]
        for i, j in pairs:
            u = u - k_bridge * jnp.logaddexp(0.0, -(d[i] + d[j]) / k_bridge)
        return -u

    return background


def pou_weights(phi, tau, phi_bg=None):
    """Soft partition of unity over regions + background (invariant 4).

    w_i = exp(-phi_i/tau) / (1 + sum_j exp(-phi_j/tau)); the appended zero
    logit is the background (void) share, so weights sum to 1 exactly.
    Returns (w_regions, w_background). tau -> 0 is the hard segmentation.

    phi_bg (optional): a signed background field (make_background_field);
    its logit -phi_bg/tau replaces the zero logit. Used by mate-declaring
    assemblies to keep the void weight out of bridged (flush) interfaces.
    Default None is byte-for-byte the previous behavior.
    """
    if phi_bg is None:
        bg_logit = jnp.zeros_like(phi[:1])
    else:
        bg_logit = -phi_bg[None] / tau
    logits = jnp.concatenate([-phi / tau, bg_logit], axis=0)
    m = jnp.max(logits, axis=0, keepdims=True)
    e = jnp.exp(logits - m)
    w = e / jnp.sum(e, axis=0, keepdims=True)
    return w[:-1], w[-1]
