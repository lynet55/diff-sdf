"""The DAG is data (invariant 1).

A Graph is a flat node table plus one flat unconstrained parameter vector.
Node = (op name, child ids, slice into the vector). Nothing geometric is a
Python object with behavior; ops are looked up in geomk.ops.OPS at evaluation.
A generative model can author or regroup this table directly.
"""
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .ops import OPS
from .reparam import softplus_inverse


@dataclass(frozen=True)
class Node:
    op: str
    children: tuple  # node ids, children always precede parents
    p0: int          # [p0, p1) slice into the flat parameter vector
    p1: int


@dataclass(frozen=True)
class Graph:
    nodes: tuple            # tuple[Node]
    theta0: np.ndarray      # unconstrained initial parameters
    positive_mask: np.ndarray  # bool, which params map through softplus


@dataclass(frozen=True)
class Handle:
    """Returned by builder methods: node id + global indices of its params."""
    node: int
    params: np.ndarray

    def __index__(self):
        return self.node


def _nid(x):
    return x.node if isinstance(x, Handle) else int(x)


class GraphBuilder:
    def __init__(self):
        self._nodes = []
        self._theta0 = []
        self._positive = []

    def _add(self, op, children, values):
        spec = OPS[op]
        kinds = spec.param_kinds
        if callable(kinds):                       # variable-arity op (polygon)
            kinds = kinds(len(values))
        assert len(values) == len(kinds), f"{op}: expected {len(kinds)} params"
        p0 = len(self._theta0)
        for v, kind in zip(values, kinds):
            if kind == "p":
                assert v > 0, f"{op}: positive param initialized to {v}"
                self._theta0.append(float(softplus_inverse(v)))
            else:
                self._theta0.append(float(v))
            self._positive.append(kind == "p")
        nid = len(self._nodes)
        self._nodes.append(Node(op, tuple(_nid(c) for c in children), p0, len(self._theta0)))
        return Handle(nid, np.arange(p0, len(self._theta0)))

    # primitives
    def sphere(self, center, radius):
        return self._add("sphere", (), [*center, radius])

    def box(self, center, half_extents):
        return self._add("box", (), [*center, *half_extents])

    def capsule(self, a, b, radius):
        return self._add("capsule", (), [*a, *b, radius])

    def polygon(self, vertices):
        """Closed 2D sketch profile in the (x, y) plane (infinite prism in z).
        `vertices`: sequence of (x, y), n >= 3, convex or concave. Exact 2D SDF
        -> metric clean; feed to extrude/revolve/loft or offset/shell it."""
        v = np.asarray(vertices, dtype=np.float64)
        assert v.ndim == 2 and v.shape[1] == 2 and v.shape[0] >= 3, \
            "polygon needs an (n>=3, 2) vertex array"
        return self._add("polygon", (), v.reshape(-1).tolist())

    # booleans
    def smooth_union(self, *children, k=0.08):
        return self._add("smooth_union", children, [k])

    def smooth_subtract(self, a, b, k=0.08):
        return self._add("smooth_subtract", (a, b), [k])

    def smooth_intersect(self, *children, k=0.08):
        return self._add("smooth_intersect", children, [k])

    # transform
    def rigid(self, child, translation=(0.0, 0.0, 0.0), rotvec=(0.0, 0.0, 0.0)):
        return self._add("rigid", (child,), [*translation, *rotvec])

    # profile ops (Pass 4 breadth). Profiles are the child's z=0 slice in the
    # (x, y) plane; revolve's axis is fixed to x (y = radial coordinate) —
    # reorient the solid with rigid().
    def revolve(self, profile):
        return self._add("revolve", (profile,), [])

    def extrude(self, profile, half_height):
        return self._add("extrude", (profile,), [half_height])

    def loft(self, profile0, profile1, half_height):
        return self._add("loft", (profile0, profile1), [half_height])

    # generators
    def lattice(self, cell, strut_radius):
        return self._add("lattice", (), [cell, strut_radius])

    # metric ops (Pass 2) — construction-time refusal of non-metric children
    def redistance(self, child):
        return self._add("redistance", (child,), [])

    def offset(self, child, distance):
        self._require_clean(child, "offset")
        return self._add("offset", (child,), [distance])

    def shell(self, child, thickness):
        self._require_clean(child, "shell")
        return self._add("shell", (child,), [thickness])

    def _require_clean(self, child, op):
        if not _metric_clean(self._nodes, _nid(child)):
            raise ValueError(
                f"{op} needs true distance but its input is not metric_clean; "
                f"insert an explicit redistance node (invariant 3)."
            )

    def build(self):
        return Graph(
            nodes=tuple(self._nodes),
            theta0=np.array(self._theta0, dtype=np.float64),
            positive_mask=np.array(self._positive, dtype=bool),
        )


def _metric_clean(nodes, nid):
    node = nodes[nid]
    rule = OPS[node.op].metric
    if rule == "clean":
        return True
    if rule == "dirty":
        return False
    # preserve / require_clean: status flows from children
    return all(_metric_clean(nodes, c) for c in node.children)


def metric_clean(graph: Graph, nid) -> bool:
    """Reported metric status of a subtree (invariant 3). A True flag on a
    redistance output means 'approximately metric' at the honesty level of the
    redistance implementation in use (first-order for now, see metric notes)."""
    return _metric_clean(graph.nodes, _nid(nid))
