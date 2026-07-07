"""Watertight mesh export — the CAD-out endpoint (params -> field -> mesh).

This is the *non*-differentiable boundary: a hardened surface handed to an
external tool (printer, mesher, body-fitted solver). The differentiable loop
never comes through here — it consumes the field/occupancy projections
directly. Export exists so a design that was optimized in field space can leave
the kernel as a manufacturable/meshable artifact.

Meshing is marching cubes on a regular lattice of the *composed* field, with a
one-cell positive pad so any surface reaching the domain box is capped — the
output is always a closed 2-manifold. `is_watertight` verifies it (every edge
shared by exactly two faces, no non-manifold edges); `mesh_volume` cross-checks
the result against the field's own occupancy integral so the mesh is provably
faithful, not just closed.

Two granularities:
  export_solid      — the outer shell of the whole assembly (union of all
                      components): one watertight mesh, for single-material
                      export / printing / external-CFD wetted surface.
  export_components — per-component precedence-composed region, each a closed
                      mesh, for multi-material handoff. Shared interfaces are
                      COINCIDENT, not vertex-shared (conformal interface meshing
                      is the deferred interface-identity work); most external
                      volume meshers re-triangulate anyway.
"""
import struct
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from skimage import measure

from .compose import Assembly, make_region_fields
from .dag import _nid
from .evaluate import eval_node
from .reparam import constrain


@dataclass
class Mesh:
    verts: np.ndarray        # (V, 3) float
    faces: np.ndarray        # (F, 3) int, outward-oriented
    name: str = "solid"


def _axes(grid):
    return [np.linspace(l, h, n) for l, h, n in
            zip(grid.lo, grid.hi, grid.shape)]


def _sample(field, theta, axes):
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    d = np.asarray(field(jnp.asarray(theta), jnp.asarray(G)))
    return d.reshape(len(axes[0]), len(axes[1]), len(axes[2]))


def _orient_outward(verts, faces):
    """Flip winding so the enclosed signed volume is positive (outward normals).
    Signed volume = (1/6) sum tri (v0 . v1 x v2)."""
    tri = verts[faces]
    vol6 = np.einsum("fi,fi->f", tri[:, 0],
                     np.cross(tri[:, 1], tri[:, 2])).sum()
    if vol6 < 0:
        faces = faces[:, ::-1]
    return faces


def _mesh_scalar(vol, axes, name):
    """Marching cubes on a scalar volume (exterior positive), padded to close.
    Lewiner method (skimage default) is manifold; padding guarantees closure."""
    if vol.min() > 0:
        return None                                   # empty: no surface
    dx = tuple(float(a[1] - a[0]) for a in axes)
    pad = float(np.abs(vol).max()) + 1.0
    volp = np.pad(vol, 1, mode="constant", constant_values=pad)
    verts, faces, _, _ = measure.marching_cubes(volp, 0.0, spacing=dx)
    origin = np.array([a[0] for a in axes]) - np.array(dx)   # undo the pad shift
    verts = verts + origin
    faces = _orient_outward(verts, faces.astype(np.int64))
    return Mesh(verts, faces, name)


def _solid_field(asm: Assembly):
    """Raw union min_i d_i — the exterior boundary of the whole solid (the
    same 'solid' predicate the hard-ownership / topology paths use)."""
    roots = [_nid(c.root) for c in asm.components]

    def field(theta, pts):
        q = constrain(theta, asm.graph.positive_mask)
        d = jnp.stack([eval_node(asm.graph, r, q, pts) for r in roots])
        return jnp.min(d, axis=0)

    return field


def domain_clipped(asm: Assembly, theta, grid) -> bool:
    """True if the solid reaches the domain box (mesh would be capped by the
    box, not the geometry) — a caller should enlarge the grid."""
    axes = _axes(grid)
    vol = _sample(_solid_field(asm), theta, axes)
    faces = [vol[0], vol[-1], vol[:, 0], vol[:, -1], vol[:, :, 0], vol[:, :, -1]]
    return bool(min(f.min() for f in faces) <= 0.0)


def export_solid(asm: Assembly, theta, grid) -> Mesh:
    """Watertight outer shell of the whole assembly (union of components)."""
    axes = _axes(grid)
    vol = _sample(_solid_field(asm), theta, axes)
    return _mesh_scalar(vol, axes, "solid")


def export_components(asm: Assembly, theta, grid):
    """Per-component watertight meshes (precedence-composed regions)."""
    axes = _axes(grid)
    shp = tuple(len(a) for a in axes)
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    phi = np.asarray(make_region_fields(asm)(jnp.asarray(theta),
                                             jnp.asarray(G)))
    out = []
    for i, c in enumerate(asm.components):
        m = _mesh_scalar(phi[i].reshape(shp), axes, c.name)
        if m is not None:
            out.append(m)
    return out


# ---- validation ---------------------------------------------------------------

def is_watertight(mesh: Mesh):
    """(bool, stats). Watertight == closed 2-manifold: every undirected edge is
    shared by exactly two faces (no boundary edges, no non-manifold edges)."""
    f = mesh.faces
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]),
                axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    boundary = int((counts == 1).sum())
    nonmanifold = int((counts > 2).sum())
    ok = boundary == 0 and nonmanifold == 0
    return ok, {
        "watertight": ok, "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "euler": int(len(mesh.verts) - len(uniq) + len(mesh.faces)),
        "n_verts": int(len(mesh.verts)), "n_faces": int(len(mesh.faces)),
    }


def mesh_volume(mesh: Mesh) -> float:
    """Enclosed volume via the divergence theorem (faithfulness cross-check
    against the field's occupancy integral)."""
    tri = mesh.verts[mesh.faces]
    return float(abs(np.einsum("fi,fi->f", tri[:, 0],
                               np.cross(tri[:, 1], tri[:, 2])).sum()) / 6.0)


# ---- writers ------------------------------------------------------------------

def write_stl(path, mesh: Mesh):
    """Binary STL (outward per-facet normals)."""
    tri = mesh.verts[mesh.faces].astype("<f4")               # (F, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = (n / np.where(ln > 0, ln, 1.0)).astype("<f4")
    rec = np.zeros(len(tri), dtype=np.dtype([
        ("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")]))
    rec["n"] = n
    rec["v"] = tri
    with open(path, "wb") as fp:
        fp.write(b"geomk watertight export".ljust(80, b"\0"))
        fp.write(struct.pack("<I", len(tri)))
        fp.write(rec.tobytes())
    return len(tri)


def write_obj(path, meshes):
    """OBJ, one `o` group per mesh (multi-material handoff keeps components)."""
    if isinstance(meshes, Mesh):
        meshes = [meshes]
    with open(path, "w") as fp:
        base = 1
        for m in meshes:
            fp.write(f"o {m.name}\n")
            for v in m.verts:
                fp.write(f"v {v[0]:.6g} {v[1]:.6g} {v[2]:.6g}\n")
            for tri in m.faces:
                fp.write(f"f {tri[0]+base} {tri[1]+base} {tri[2]+base}\n")
            base += len(m.verts)
