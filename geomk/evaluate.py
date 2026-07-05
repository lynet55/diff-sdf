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
from .ops import OPS, _safe_norm
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
    if node.op == "extrude":
        return _extrude(graph, node.children[0], q, p, params[0])
    if node.op == "loft":
        return _loft(graph, node.children, q, p, params[0])
    raise KeyError(node.op)


def _at_z0(p):
    """Profile-plane points: (x, y, z) -> (x, y, 0)."""
    return jnp.concatenate([p[..., :2], jnp.zeros_like(p[..., 2:3])], axis=-1)


def _cap_z(d2, wz):
    """Exact-SDF z-capping of a profile field d2 against the slab wz = |z|-h:
    min(max(d2, wz), 0) + |max((d2, wz), 0)|. _safe_norm keeps the gradient
    finite in the interior, where both maxed terms are exactly zero."""
    outside = _safe_norm(jnp.stack(
        [jnp.maximum(d2, 0.0), jnp.maximum(wz, 0.0)], axis=-1))
    return jnp.minimum(jnp.maximum(d2, wz), 0.0) + outside


def _extrude(graph, child, q, p, h):
    """Linear extrusion of the child's z=0 slice along z to half-height h.
    Exact 3D SDF whenever the profile slice is an exact 2D SDF ('preserve')."""
    d2 = eval_node(graph, child, q, _at_z0(p))
    return _cap_z(d2, jnp.abs(p[..., 2]) - h)


def _loft(graph, children, q, p, h):
    """Loft between two z=0 profiles from z=-h to z=+h: linear blend of the
    profile fields in t = clip((z+h)/2h, 0, 1), capped like extrude. The
    blend's SIGN is geometrically meaningful (for concentric circles the zero
    set is the exact frustum) but its DISTANCE is not metric -> 'dirty'."""
    z = p[..., 2]
    t = jnp.clip((z + h) / (2.0 * h), 0.0, 1.0)
    p0 = _at_z0(p)
    d2 = ((1.0 - t) * eval_node(graph, children[0], q, p0)
          + t * eval_node(graph, children[1], q, p0))
    return _cap_z(d2, jnp.abs(z) - h)


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
