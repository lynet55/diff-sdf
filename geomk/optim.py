"""Minimal Adam over the flat unconstrained parameter vector.

`mask` is the exposure selection: which entries are decision variables. This
is the same flat-vector-plus-mask object a black-box/RL optimizer consumes.
"""
import jax.numpy as jnp
import numpy as np


def adam(value_and_grad_fn, theta0, mask, lr=0.02, steps=100,
         beta1=0.9, beta2=0.999, eps=1e-8):
    theta = jnp.asarray(theta0)
    mask = jnp.asarray(mask)
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    hist = {"J": [], "grad": [], "theta": []}
    for t in range(1, steps + 1):
        J, g = value_and_grad_fn(theta)
        g = g * mask
        hist["J"].append(float(J))
        hist["grad"].append(np.asarray(g))
        hist["theta"].append(np.asarray(theta))
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        mhat = m / (1 - beta1 ** t)
        vhat = v / (1 - beta2 ** t)
        theta = theta - lr * mask * mhat / (jnp.sqrt(vhat) + eps)
    return theta, hist
