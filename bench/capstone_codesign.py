"""Capstone co-design — the full 8-macro optimization, and a gradient-vs-
black-box comparison on the identical flat decision vector.

Step 4: descend the four-physics objective J over all eight exposure macros
with Adam (gradients on the fidelity paths locked across probes D-F: flight
straddle, compliance/barrier/drag pure-soft). The saturation barrier holds
every component inside probe A's 2*tau trust floor throughout. Report the
before/after craft (masses, key dimensions, per-physics values, saturations).

Step 5: run a gradient-FREE optimizer — a self-adapting (1+1)-ES with the 1/5
success rule — on the SAME z, mask, and J, from the same z0. This is the
exposure layer's whole point: one object (flat vector + selection mask) that a
gradient optimizer and a black-box optimizer consume identically. Plot best-J
vs wall-clock for both; the gradient path reaches the same basin in a fraction
of the objective evaluations. Everything is saved for the report page.
"""
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, ".")

from bench.capstone_physics import (build_physics, make_objective, F_LOAD,
                                     LAM_C, LAM_D, LAM_B, MACRO_NAMES)
from bench.capstone_craft import MACRO_NAMES as _CHK
from geomk.optim import adam
from geomk.projections import TRUST_SATURATION

assert tuple(_CHK) == MACRO_NAMES


def terms(P, z):
    """The four physics values at z (flight straddle=accurate; others soft)."""
    return dict(
        flight=float(jax.jit(P["flight"])(z)),
        compliance=float(jax.jit(P["compliance"])(z)),
        drag=float(jax.jit(P["drag"])(z)),
        barrier=float(jax.jit(P["barrier"])(z)))


def snapshot(P, z):
    """Physical read-out of a design: masses, COM, saturations, key dims."""
    emap = P["emap"]
    theta = emap.theta(z)
    p = jax.jit(P["props_acc"])(theta)
    diag = jax.jit(P["diag"])(theta)
    sat = np.asarray(diag["saturation"])
    names = [c.name for c in P["asm"].components]
    z = np.asarray(z)
    # physical meters: six macros drive softplus-reparameterized positive params
    # (physical = softplus(z)); arm_length and batt_x are linear (physical = z).
    positive = {"arm_radius", "fuse_len", "fuse_width", "nacelle_scale",
                "fin_size", "deck_hz"}
    dims = {n: (float(jax.nn.softplus(jnp.asarray(z[k]))) if n in positive
                else float(z[k])) for k, n in enumerate(MACRO_NAMES)}
    return dict(
        total_mass=float(p["total_mass"]),
        com=np.asarray(p["com"]).tolist(),
        component_mass=np.asarray(p["component_mass"]).tolist(),
        saturation=sat.tolist(), min_saturation=float(sat.min()),
        n_below_trust=int((sat < TRUST_SATURATION).sum()),
        names=names, dims=dims)


def run_gradient(P, J, z0, steps=90, lr=0.02):
    vg = jax.jit(jax.value_and_grad(J))
    vg(z0)                                   # compile before timing
    mask = jnp.ones_like(z0)
    hist = {"t": [], "J": [], "nfev": []}
    z = z0
    t0 = time.perf_counter()

    # light wrapper around adam to log wall-clock / eval count per step
    from geomk.optim import adam as _adam
    z_opt, h = _adam(vg, z0, mask, lr=lr, steps=steps)
    Js = np.asarray(h["J"])
    zs = np.asarray(h["theta"])
    # each Adam step is one value_and_grad = reverse-mode: ~1 forward + adjoint
    for i in range(len(Js)):
        hist["J"].append(float(Js[i]))
        hist["nfev"].append(i + 1)
    wall = time.perf_counter() - t0
    hist["wall"] = wall
    hist["theta"] = zs.tolist()
    return z_opt, hist


def run_es(P, Jf, z0, budget=520, sigma0=0.25, seed=0):
    """Self-adapting (1+1)-ES, 1/5 success rule. Jf: jitted scalar J(z)."""
    rng = np.random.default_rng(seed)
    z = np.asarray(z0, dtype=float)
    fbest = float(Jf(jnp.asarray(z)))
    sigma = sigma0
    n = z.size
    scale = np.ones(n)                        # per-coord step scale (z-space)
    hist = {"J": [fbest], "nfev": [1], "t": [0.0]}
    t0 = time.perf_counter()
    succ = 0
    for it in range(1, budget):
        cand = z + sigma * scale * rng.standard_normal(n)
        fc = float(Jf(jnp.asarray(cand)))
        if fc < fbest:
            z, fbest, succ = cand, fc, succ + 1
        hist["J"].append(fbest)
        hist["nfev"].append(it + 1)
        hist["t"].append(time.perf_counter() - t0)
        if it % 20 == 0:                      # 1/5 rule on a window
            rate = succ / 20.0
            sigma *= np.exp((rate - 0.2) / 0.8)
            sigma = float(np.clip(sigma, 1e-3, 1.0))
            succ = 0
    hist["wall"] = time.perf_counter() - t0
    hist["z"] = z.tolist()
    return jnp.asarray(z), hist


def main():
    print("building capstone physics ...")
    P = build_physics()
    J, backbone, backbone_fd, anchors = make_objective(P)
    z0 = P["z0"]
    J_jit = jax.jit(J)
    J0 = float(J_jit(z0))
    print(f"J(z0) = {J0:.5f}   anchors F0={anchors['F0']:.3f} "
          f"C0={anchors['C0']:.3e} D0={anchors['D0']:.3e}")

    print("Step 4: gradient co-design (Adam, 8 macros) ...")
    z_grad, hg = run_gradient(P, J, z0, steps=90, lr=0.02)
    Jg = float(J_jit(z_grad))
    print(f"  J {J0:.5f} -> {Jg:.5f}  ({hg['wall']:.0f}s, {hg['nfev'][-1]} "
          f"value+grad evals)")

    # A/B: same objective WITHOUT the saturation barrier, to isolate what the
    # barrier holds. Rebuild physics with lam_b=0 (own compiled J).
    print("Step 4b: no-barrier baseline (isolates the barrier's effect) ...")
    P0 = build_physics()
    P0["lam_b"] = 0.0
    J_nb, _, _, _ = make_objective(P0)
    z_nb, hnb = run_gradient(P0, J_nb, z0, steps=90, lr=0.02)
    print(f"  no-barrier J* {float(jax.jit(J_nb)(z_nb)):.5f}")

    print("Step 5: black-box (1+1)-ES on the same z/mask/J ...")
    z_es, he = run_es(P, J_jit, z0, budget=360)
    Je = float(J_jit(z_es))
    print(f"  J {J0:.5f} -> {Je:.5f}  ({he['wall']:.0f}s, {he['nfev'][-1]} "
          f"objective evals)")

    # how many ES evals to first reach the gradient optimum's J?
    he_J = np.asarray(he["J"])
    reach = int(np.argmax(he_J <= Jg)) if np.any(he_J <= Jg) else -1
    print(f"  ES evals to reach gradient's J={Jg:.5f}: "
          f"{reach if reach > 0 else '>budget'}")

    snaps = {"before": snapshot(P, z0), "grad": snapshot(P, z_grad),
             "es": snapshot(P, z_es), "nobar": snapshot(P, z_nb)}
    trm = {"before": terms(P, z0), "grad": terms(P, z_grad),
           "es": terms(P, z_es), "nobar": terms(P, z_nb)}
    for tag in ("before", "grad", "es", "nobar"):
        s, t = snaps[tag], trm[tag]
        print(f"  [{tag:6s}] J-terms flight {t['flight']:.3f} "
              f"compliance {t['compliance']:.3e} drag {t['drag']:.4f} "
              f"barrier {t['barrier']:.4f} | mass {s['total_mass']:.3f} kg "
              f"min-sat {s['min_saturation']:.3f} "
              f"below-trust {s['n_below_trust']}")

    out = dict(
        macro_names=list(MACRO_NAMES), J0=J0, anchors=anchors,
        weights=dict(LAM_C=LAM_C, LAM_D=LAM_D, LAM_B=LAM_B),
        grad=dict(J_final=Jg, wall=hg["wall"], nfev=hg["nfev"][-1],
                  J_hist=hg["J"], nfev_hist=hg["nfev"],
                  z_final=np.asarray(z_grad).tolist(),
                  theta_hist=hg["theta"]),
        es=dict(J_final=Je, wall=he["wall"], nfev=he["nfev"][-1],
                J_hist=he["J"], nfev_hist=he["nfev"], t_hist=he["t"],
                z_final=np.asarray(z_es).tolist(),
                evals_to_grad_J=reach),
        nobar=dict(z_final=np.asarray(z_nb).tolist()),
        snapshots=snaps, terms=trm,
        z0=np.asarray(z0).tolist(), trust=TRUST_SATURATION)
    with open("out/capstone_codesign.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote out/capstone_codesign.json")
    return out


if __name__ == "__main__":
    main()
