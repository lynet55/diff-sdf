"""Pure op kernels: f(params, children) -> field. (Invariants 1-2.)

Every kernel is a pure function of an explicit physical-parameter vector `q`
and either points or child field values. No kernel ever captures a parameter;
graph structure is data (see dag.py) and parameters arrive as arguments.

Kernel kinds
  primitive : fn(p, q) -> d          p is (..., 3) points
  warp      : fn(p, q) -> p'         point map applied before the child
  combine   : fn([d_i], q) -> d      boolean over child field values
  unary     : fn(d, q) -> d          pointwise map of one child value
  special   : handled by the evaluator (redistance needs grad wrt the point)

Metric flow (invariant 3): 'clean' (exact SDF), 'preserve' (child's status),
'dirty' (signed-implicit only), 'require_clean' (refuses a non-metric child,
output clean). Smooth booleans use log-sum-exp blending: C-infinity in both
params and points, sign correct for the smoothed shape, distances NOT metric.
"""
from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from jax.scipy.special import logsumexp


@dataclass(frozen=True)
class OpSpec:
    kind: str
    param_kinds: object       # per-param kind chars ('f' free, 'p' positive):
                              # a fixed str, or a callable(n_values) -> str for
                              # variable-arity ops (polygon).
    fn: Callable
    metric: str               # clean | preserve | dirty | require_clean


# --- primitives (exact SDFs, metric clean) ---------------------------------

def _safe_norm(v):
    """|v| with a zero (not NaN) gradient where v == 0; norm(max(a,0)) inside
    a box and |p - c| at a sphere center otherwise poison autodiff."""
    s = jnp.sum(v * v, axis=-1)
    pos = s > 0.0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, s, 1.0)), 0.0)


def _sphere(p, q):
    c, r = q[:3], q[3]
    return _safe_norm(p - c) - r


def _box(p, q):
    c, h = q[:3], q[3:6]
    a = jnp.abs(p - c) - h
    outside = _safe_norm(jnp.maximum(a, 0.0))
    inside = jnp.minimum(jnp.max(a, axis=-1), 0.0)
    return outside + inside


def _capsule(p, q):
    a, b, r = q[:3], q[3:6], q[6]
    ab = b - a
    ap = p - a
    t = jnp.clip(jnp.sum(ap * ab, axis=-1) / jnp.sum(ab * ab), 0.0, 1.0)
    return _safe_norm(ap - t[..., None] * ab) - r


def _safe_sqrt(d):
    """sqrt with a zero (not inf) gradient at d == 0 — the scalar analogue of
    _safe_norm, for squared-distance fields (polygon)."""
    pos = d > 0.0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, d, 1.0)), 0.0)


def _polygon(p, q):
    """Exact 2D signed-distance field of a closed polygon in the (x, y) plane,
    read as an infinite prism in z (z is ignored). Vertices are the node's
    param slice, flattened (x0, y0, x1, y1, ...); any n >= 3, convex OR concave
    (winding-number sign, so self-consistent for non-convex sketches).

    This is the sketch-profile primitive: extrude(polygon) is the exact prism
    SDF and revolve(polygon) the exact solid of revolution ('preserve' both),
    so a sketched profile stays metric-clean and can be offset/shelled without
    a redistance node. Exact distance = min over edges of the point-segment
    distance; sign = parity of the QL winding test. Metric 'clean'.
    """
    V = q.reshape(-1, 2)                       # (n, 2) vertices
    Vj = jnp.roll(V, 1, axis=0)                # previous vertex (edge Vj->Vi)
    e = Vj - V                                 # (n, 2) edge vectors
    P = p[..., :2]                             # (..., 2), z ignored
    w = P[..., None, :] - V                    # (..., n, 2)
    ee = jnp.sum(e * e, axis=-1)               # (n,)
    t = jnp.clip(jnp.sum(w * e, axis=-1) / (ee + 1e-30), 0.0, 1.0)   # (..., n)
    b = w - e * t[..., None]                   # (..., n, 2)
    dsq = jnp.min(jnp.sum(b * b, axis=-1), axis=-1)                  # (...,)
    c1 = P[..., None, 1] >= V[:, 1]
    c2 = P[..., None, 1] < Vj[:, 1]
    c3 = e[:, 0] * w[..., 1] > e[:, 1] * w[..., 0]
    flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)                        # (..., n)
    s = jnp.where(jnp.sum(flip, axis=-1) % 2 == 1, -1.0, 1.0)        # (...,)
    return s * _safe_sqrt(dsq)


def _lattice(p, q):
    """Infinite axis-aligned strut lattice: cell size c, strut radius t.
    Fold space into the unit cell and take the min distance to the three
    strut families (infinite cylinders along x/y/z through the cell center).
    Approximately metric inside a cell but NOT across cell boundaries (the
    fold discards farther periodic copies' interiors), so metric is 'dirty' —
    it is a signed-implicit generator, not an offsettable distance field.
    A finite lattice = compose this with a domain shape (invariant 5 /
    smooth_subtract); no clean child is consumed, so require_clean does not
    apply — the op simply never CLAIMS metric (invariant 3)."""
    c, t = q[0], q[1]
    pf = p - c * jnp.round(p / c)
    dist_x = _safe_norm(pf[..., 1:3]) - t
    dist_y = _safe_norm(pf[..., ::2]) - t
    dist_z = _safe_norm(pf[..., :2]) - t
    return jnp.minimum(dist_x, jnp.minimum(dist_y, dist_z))


# --- smooth booleans (log-sum-exp; C-infinity, metric dirty) ----------------

def smin(vals, k):
    """Smooth minimum of stacked field values, axis 0. vals: (n, ...)."""
    return -k * logsumexp(-vals / k, axis=0)


def _smooth_union(child_vals, q):
    return smin(jnp.stack(child_vals), q[0])


def _smooth_subtract(child_vals, q):
    a, b = child_vals
    return q[0] * jnp.logaddexp(a / q[0], -b / q[0])  # smax(a, -b)


def _smooth_intersect(child_vals, q):
    """Smooth intersection = smooth max of the children (the third boolean).
    C-infinity, sign-correct for the intersected shape, metric dirty."""
    return q[0] * logsumexp(jnp.stack(child_vals) / q[0], axis=0)


# --- rigid transform (warp; metric preserved) -------------------------------

def _rotate_vec(v, rotvec):
    """Rodrigues rotation, smooth (and NaN-free in grad) at rotvec = 0."""
    t2 = jnp.sum(rotvec * rotvec)
    small = t2 < 1e-16
    t = jnp.sqrt(jnp.where(small, 1.0, t2))       # double-where: safe grads
    a = jnp.where(small, 1.0 - t2 / 6.0, jnp.sin(t) / t)
    b = jnp.where(small, 0.5 - t2 / 24.0, (1.0 - jnp.cos(t)) / jnp.where(small, 1.0, t2))
    c1 = jnp.cross(jnp.broadcast_to(rotvec, v.shape), v)
    c2 = jnp.cross(jnp.broadcast_to(rotvec, v.shape), c1)
    return v + a * c1 + b * c2


def _rigid(p, q):
    """Point map for a body placed at translation t with rotation rotvec:
    evaluate the child in body frame, p' = R^T (p - t)."""
    t, rotvec = q[:3], q[3:6]
    return _rotate_vec(p - t, -rotvec)


def _revolve(p, q):
    """Point map: revolve the child 2D profile about the x-axis (fixed axis;
    reorient the result with rigid()). The profile lives in the (x, y) plane
    with y the radial coordinate: p = (x, y, z) -> (x, sqrt(y^2 + z^2), 0).
    The nearest surface point of a revolved 2D SDF stays in the meridian
    half-plane, so a metric-clean profile yields a metric-clean solid
    ('preserve'). _safe_norm keeps the gradient finite on the axis r = 0."""
    r = _safe_norm(p[..., 1:3])
    return jnp.stack([p[..., 0], r, jnp.zeros_like(r)], axis=-1)


# --- metric-sensitive unary ops (Pass 2; require a clean child) -------------

def _offset(d, q):
    return d - q[0]


def _shell(d, q):
    return jnp.abs(d) - 0.5 * q[0]


OPS = {
    "sphere": OpSpec("primitive", "fffp", _sphere, "clean"),
    "box": OpSpec("primitive", "fffppp", _box, "clean"),
    "capsule": OpSpec("primitive", "ffffffp", _capsule, "clean"),
    # polygon: variable arity (2 free coords per vertex) -> param_kinds is a
    # callable of the value count. Exact 2D SDF, metric clean.
    "polygon": OpSpec("primitive", lambda n: "f" * n, _polygon, "clean"),
    "lattice": OpSpec("primitive", "pp", _lattice, "dirty"),
    "smooth_union": OpSpec("combine", "p", _smooth_union, "dirty"),
    "smooth_subtract": OpSpec("combine", "p", _smooth_subtract, "dirty"),
    "smooth_intersect": OpSpec("combine", "p", _smooth_intersect, "dirty"),
    "rigid": OpSpec("warp", "ffffff", _rigid, "preserve"),
    "revolve": OpSpec("warp", "", _revolve, "preserve"),
    "offset": OpSpec("unary", "f", _offset, "require_clean"),
    "shell": OpSpec("unary", "p", _shell, "require_clean"),
    "redistance": OpSpec("special", "", None, "clean"),
    # extrude/loft evaluate their children at (x, y, 0) and cap in z: special.
    # Extrude of an exact 2D SDF is the exact 3D SDF -> 'preserve'.
    # Loft's linear blend of two profiles is sign-correct but not metric
    # (distance shrinks under the blend) -> 'dirty', honest per invariant 3.
    "extrude": OpSpec("special", "p", None, "preserve"),
    "loft": OpSpec("special", "p", None, "dirty"),
}
