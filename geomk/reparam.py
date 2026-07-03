"""Bijective reparameterization (CLAUDE.md "do now").

The flat parameter vector theta lives in unconstrained space. Params declared
kind 'p' (positive: lengths, radii, smoothing bandwidths, thicknesses) map
through softplus so no optimizer step can reach a negative extent. Params of
kind 'f' (free: coordinates, rotations, offsets) pass through unchanged.

This is the only feasibility *guarantee*; everything else is reported as
differentiable diagnostics, not enforced.
"""
import jax.numpy as jnp
import numpy as np


def softplus(x):
    return jnp.logaddexp(x, 0.0)


def softplus_inverse(y):
    """Physical -> unconstrained. y must be > 0."""
    y = np.asarray(y, dtype=np.float64)
    return y + np.log(-np.expm1(-y))


def constrain(theta, positive_mask):
    """Unconstrained theta -> physical params. positive_mask is a static bool array."""
    mask = jnp.asarray(positive_mask)
    return jnp.where(mask, softplus(theta), theta)
