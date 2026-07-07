"""Capstone craft — the first artifact built WITH the kernel rather than to
test it. Twelve components at CLAUDE.md's intended scale, authored
programmatically from a spec dict (the generative-reachability pattern):

  fuselage   loft(rectangle tail -> circle nose) laid along x   structural
  battery    box embedded in the fuselage                       payload
  fin        extruded stadium, vertical at the tail             aero
  deck       lattice-cored plate under the fuselage             structural
  arm_*      4 capsules on the +-45 deg diagonals               structural
  nacelle_*  4 revolved teardrops at the arm tips               propulsion

Declared mate: (deck, fuselage). Precedence: nacelles > battery > fuselage >
fin > arms > deck. Eight exposure macros drive ~30 primitive parameters
through the relations layer, with the diagonal symmetry coupling (the
arm-length macro places eight bodies).
"""
import numpy as np

from geomk.dag import GraphBuilder
from geomk.compose import Component, Assembly
from geomk.exposure import Exposure

DIRS = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]]) / np.sqrt(2)
ARM_TAGS = ("fr", "fl", "rl", "rr")

# ---- spec (initial physical values; first 8 entries are the macros) ----------
SPEC = {
    "arm_length": 0.20,      # macro: diagonal arm length [m]
    "arm_radius": 0.024,     # macro: arm capsule radius [m]
    "fuse_len": 0.14,        # macro: fuselage loft half-length [m]
    "fuse_width": 0.040,     # macro: fuselage tail half-width (y) [m]
    "nacelle_scale": 0.030,  # macro: nacelle main-sphere radius [m]
    "fin_size": 0.050,       # macro: fin stadium radius [m]
    "deck_hz": 0.016,        # macro: deck half-thickness [m]
    "batt_x": -0.02,         # macro: battery x position (COM trim) [m]
    # fixed dimensions
    "fuse_nose_r": 0.042, "fuse_hz_profile": 0.036,
    "batt_half": (0.05, 0.032, 0.022),
    "nacelle_z": 0.012, "fin_x": -0.13, "fin_z": 0.070,
    "fin_half_th": 0.015, "deck_z": -0.045, "deck_half": (0.14, 0.14),
    "lattice_cell": 0.075, "lattice_strut": 0.016,
}

# Densities sized so the assembled craft hovers under the flight plant's rotor
# cap (4 x F_MAX = 40 N > m*g): a ~0.4 m quad at ~2.5 kg, realistic and
# hover-feasible, so flight does not trivially dominate by forcing mass-shedding.
DENSITY = {"fuselage": 260.0, "battery": 900.0, "fin": 220.0, "deck": 360.0,
           "arm": 280.0, "nacelle": 1200.0}
INTENT = {"fuselage": "structural", "battery": "payload", "fin": "aero",
          "deck": "structural", "arm": "structural", "nacelle": "propulsion"}
MACRO_NAMES = ("arm_length", "arm_radius", "fuse_len", "fuse_width",
               "nacelle_scale", "fin_size", "deck_hz", "batt_x")


def build_capstone(spec=None):
    s = dict(SPEC, **(spec or {}))
    gb = GraphBuilder()

    # fuselage: loft rectangle (tail, z=-h) -> circle (nose, z=+h), then laid
    # along x (loft z-axis -> world x, nose forward)
    prof_tail = gb.box((0.0, 0.0, 0.0),
                       (s["fuse_width"], s["fuse_hz_profile"], 0.2))
    prof_nose = gb.sphere((0.0, 0.0, 0.0), s["fuse_nose_r"])
    fuse_loft = gb.loft(prof_tail, prof_nose, s["fuse_len"])
    fuse = gb.rigid(fuse_loft, rotvec=(0.0, np.pi / 2, 0.0))

    battery = gb.box((s["batt_x"], 0.0, -0.004), s["batt_half"])

    # fin: stadium profile in (x, y), extruded thin along z, rotated into the
    # x-z plane (vertical) at the tail
    fin_prof = gb.capsule((-0.05, -0.01, 0.0), (0.045, 0.045, 0.0),
                          s["fin_size"])
    fin = gb.rigid(gb.extrude(fin_prof, s["fin_half_th"]),
                   translation=(s["fin_x"], 0.0, s["fin_z"]),
                   rotvec=(np.pi / 2, 0.0, 0.0))

    deck_box = gb.box((0.0, 0.0, s["deck_z"]), (*s["deck_half"], s["deck_hz"]))
    deck = gb.smooth_subtract(deck_box,
                              gb.lattice(s["lattice_cell"], s["lattice_strut"]),
                              k=0.008)

    arms, nacelles, nac_spheres = [], [], []
    for d in DIRS:
        tip = s["arm_length"] * d
        arms.append(gb.capsule((0.0, 0.0, -0.01),
                               (tip[0], tip[1], 0.002), s["arm_radius"]))
        sph_a = gb.sphere((-0.012, 0.0, 0.0), s["nacelle_scale"])
        sph_b = gb.sphere((0.030, 0.0, 0.0), 0.55 * s["nacelle_scale"])
        prof = gb.smooth_union(sph_a, sph_b, k=0.02)
        nac_spheres.append(sph_a)
        nacelles.append(gb.rigid(
            gb.revolve(prof),
            translation=(tip[0], tip[1], s["nacelle_z"]),
            rotvec=(0.0, -np.pi / 2, 0.0)))
    graph = gb.build()

    comps, index = [], {}

    def add(name, kind, handle, prec):
        index[name] = len(comps)
        comps.append(Component(name, handle, density=DENSITY[kind],
                               precedence=prec, intent=INTENT[kind]))

    for tag, nac in zip(ARM_TAGS, nacelles):
        add(f"nacelle_{tag}", "nacelle", nac, 6)
    add("battery", "battery", battery, 5)
    add("fuselage", "fuselage", fuse, 4)
    add("fin", "fin", fin, 3)
    for tag, arm in zip(ARM_TAGS, arms):
        add(f"arm_{tag}", "arm", arm, 2)
    add("deck", "deck", deck, 1)

    asm = Assembly(graph, tuple(comps), k_compose=0.01,
                   mates=((index["deck"], index["fuselage"]),))

    # ---- exposure: 8 macros through the relations layer ---------------------
    ex = Exposure(graph)
    zL = ex.macro("arm_length", init=s["arm_length"])
    for arm, nac, d in zip(arms, nacelles, DIRS):
        for ax in range(2):                      # x, y of tips (z fixed)
            ex.drive(arm.params[3 + ax], zL, scale=float(d[ax]))
            ex.drive(nac.params[ax], zL, scale=float(d[ax]))
    zR = ex.macro_positive("arm_radius", init=s["arm_radius"])
    for arm in arms:
        ex.drive(arm.params[6], zR)
    ex.expose(fuse_loft.params[0], "fuse_len")       # loft half-length
    ex.expose(prof_tail.params[3], "fuse_width")     # tail profile y-half
    zN = ex.macro_positive("nacelle_scale", init=s["nacelle_scale"])
    for sph in nac_spheres:                          # main teardrop sphere;
        ex.drive(sph.params[3], zN)                  # the small one stays put
    ex.expose(fin_prof.params[6], "fin_size")
    ex.expose(deck_box.params[5], "deck_hz")
    ex.expose(battery.params[0], "batt_x")
    emap = ex.build()
    assert emap.names == MACRO_NAMES, emap.names

    sel = {tag: [i for i, c in enumerate(asm.components) if c.intent == tag]
           for tag in ("structural", "propulsion", "payload", "aero")}
    macros = {name: k for k, name in enumerate(MACRO_NAMES)}
    return asm, emap, macros, sel, index


if __name__ == "__main__":
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import sys
    sys.path.insert(0, ".")
    from geomk.projections import (GridSpec, make_mass_properties,
                                   make_resolution_diagnostics,
                                   TRUST_SATURATION)

    asm, emap, macros, sel, index = build_capstone()
    print(f"components: {len(asm.components)}   nodes: {len(asm.graph.nodes)}"
          f"   params: {asm.graph.theta0.size}   macros: {len(emap.z0)}"
          f"   driven params: {int(emap.mask.sum())}")
    print(f"intent -> {dict((k, len(v)) for k, v in sel.items())}")
    grid = GridSpec(lo=(-0.36, -0.36, -0.12), hi=(0.36, 0.36, 0.16),
                    shape=(72, 72, 28))
    z0 = jnp.asarray(emap.z0)
    props = jax.jit(make_mass_properties(asm, grid, fidelity="straddle"))
    p = props(emap.theta(z0))
    print(f"total mass {float(p['total_mass']):.3f} kg   "
          f"com {np.asarray(p['com']).round(4)}")
    for c, m in zip(asm.components, np.asarray(p["component_mass"])):
        print(f"  {c.name:12s} {c.intent:11s} {m*1e3:8.1f} g")
    d = jax.jit(make_resolution_diagnostics(asm, grid))(emap.theta(z0))
    sat = np.asarray(d["saturation"])
    infl = np.asarray(d["inflation"])
    print(f"tau {float(d['tau']):.4f}  (trust >= {TRUST_SATURATION})")
    for c, sv, iv in zip(asm.components, sat, infl):
        flag = "OK " if sv >= TRUST_SATURATION else "low"
        print(f"  {c.name:12s} saturation {sv:.3f} [{flag}]  inflation {iv:.2f}")
