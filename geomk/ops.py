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
    param_kinds: str          # one char per param: 'f' free, 'p' positive
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


# --- smooth booleans (log-sum-exp; C-infinity, metric dirty) ----------------

def smin(vals, k):
    """Smooth minimum of stacked field values, axis 0. vals: (n, ...)."""
    return -k * logsumexp(-vals / k, axis=0)


def _smooth_union(child_vals, q):
    return smin(jnp.stack(child_vals), q[0])


def _smooth_subtract(child_vals, q):
    a, b = child_vals
    return q[0] * jnp.logaddexp(a / q[0], -b / q[0])  # smax(a, -b)


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


# --- metric-sensitive unary ops (Pass 2; require a clean child) -------------

def _offset(d, q):
    return d - q[0]


def _shell(d, q):
    return jnp.abs(d) - 0.5 * q[0]


OPS = {
    "sphere": OpSpec("primitive", "fffp", _sphere, "clean"),
    "box": OpSpec("primitive", "fffppp", _box, "clean"),
    "capsule": OpSpec("primitive", "ffffffp", _capsule, "clean"),
    "smooth_union": OpSpec("combine", "p", _smooth_union, "dirty"),
    "smooth_subtract": OpSpec("combine", "p", _smooth_subtract, "dirty"),
    "rigid": OpSpec("warp", "ffffff", _rigid, "preserve"),
    "offset": OpSpec("unary", "f", _offset, "require_clean"),
    "shell": OpSpec("unary", "p", _shell, "require_clean"),
    "redistance": OpSpec("special", "", None, "clean"),
}
