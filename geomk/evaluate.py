"""Per-graph JIT evaluator.

The deliberate starting point from CLAUDE.md's deferred list: Python recursion
over the *static* node table happens at JAX trace time, so each graph compiles
to one XLA program. Parameters are always an explicit argument — the compiled
field is a function of (theta, points), never a closure over theta. The
tensor-interpreter evaluator stays a contained driver rewrite later because
nothing here depends on Python behavior at run time.
"""
import jax
import jax.numpy as jnp

from .dag import Graph, _nid
from .ops import OPS
from .reparam import constrain


def eval_node(graph: Graph, nid: int, q, p):
    """Evaluate subtree `nid` at points p (..., 3) given physical params q."""
    node = graph.nodes[nid]
    spec = OPS[node.op]
    params = q[node.p0:node.p1]
    if spec.kind == "primitive":
        return spec.fn(p, params)
    if spec.kind == "warp":
        return eval_node(graph, node.children[0], q, spec.fn(p, params))
    if spec.kind == "combine":
        vals = [eval_node(graph, c, q, p) for c in node.children]
        return spec.fn(vals, params)
    if spec.kind == "unary":
        return spec.fn(eval_node(graph, node.children[0], q, p), params)
    if node.op == "redistance":
        return _redistance(graph, node.children[0], q, p)
    raise KeyError(node.op)


def _redistance(graph, child, q, p):
    """First-order redistance d/|grad d| (the honest stub behind the
    metric_clean flag; full redistancing is deferred, see CLAUDE.md)."""
    def scalar(pt):
        return eval_node(graph, child, q, pt)

    d = eval_node(graph, child, q, p)
    if p.ndim == 1:
        g = jax.grad(scalar)(p)
        return d / (jnp.linalg.norm(g) + 1e-12)
    flat = p.reshape(-1, 3)
    g = jax.vmap(jax.grad(scalar))(flat).reshape(p.shape)
    return d / (jnp.linalg.norm(g, axis=-1) + 1e-12)


def make_field(graph: Graph, root):
    """Compile-ready field: f(theta, points) -> signed-implicit values."""
    rid = _nid(root)

    def field(theta, points):
        q = constrain(theta, graph.positive_mask)
        return eval_node(graph, rid, q, points)

    return field
