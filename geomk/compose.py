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

    @property
    def densities(self):
        return jnp.array([c.density if c.domain == "solid" else 0.0
                          for c in self.components])


def make_region_fields(asm: Assembly):
    """Returns f(theta, points) -> (n_regions, ...) precedence-composed fields
    phi_i, in component order. phi_i <= 0 means point belongs to region i."""
    roots = [_nid(c.root) for c in asm.components]
    prec = [c.precedence for c in asm.components]
    k = asm.k_compose

    def region_fields(theta, points):
        q = constrain(theta, asm.graph.positive_mask)
        d = [eval_node(asm.graph, r, q, points) for r in roots]
        phi = []
        for i in range(len(d)):
            higher = [d[j] for j in range(len(d)) if prec[j] > prec[i]]
            if higher:
                h = smin(jnp.stack(higher), k) if len(higher) > 1 else higher[0]
                # smooth subtract: smax(d_i, -h)
                phi.append(k * jnp.logaddexp(d[i] / k, -h / k))
            else:
                phi.append(d[i])
        return jnp.stack(phi)

    return region_fields


def pou_weights(phi, tau):
    """Soft partition of unity over regions + background (invariant 4).

    w_i = exp(-phi_i/tau) / (1 + sum_j exp(-phi_j/tau)); the appended zero
    logit is the background (void) share, so weights sum to 1 exactly.
    Returns (w_regions, w_background). tau -> 0 is the hard segmentation.
    """
    logits = jnp.concatenate([-phi / tau, jnp.zeros_like(phi[:1])], axis=0)
    m = jnp.max(logits, axis=0, keepdims=True)
    e = jnp.exp(logits - m)
    w = e / jnp.sum(e, axis=0, keepdims=True)
    return w[:-1], w[-1]
