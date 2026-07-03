"""Pass 3 — exposure: decision variables over the parameter vector.

Which params are decision variables, macro-parameters that drive many
primitive params through a differentiable relations layer, and
symmetry/coupling (mirrored wings). Materializes as exactly the object a
black-box/RL optimizer or a diffusion model consumes: a flat decision vector
z, a differentiable map theta(z), and a selection mask over the full
parameter vector.

The relations layer is affine in *unconstrained* space: theta_i = offset_i +
sum_k S_ik z_k for driven params, theta_i = base_i otherwise. Positive params
still pass through softplus afterwards (reparam.py), so no relation can
produce a negative extent.
"""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .dag import Graph
from .reparam import softplus_inverse


@dataclass(frozen=True)
class ExposureMap:
    z0: np.ndarray          # initial decision vector
    names: tuple            # one name per z entry
    S: np.ndarray           # (n_params, n_z) relations matrix
    offsets: np.ndarray     # (n_params,)
    mask: np.ndarray        # bool: which params are driven (decision-selected)
    base: np.ndarray        # defaults for undriven params

    def theta(self, z):
        """Differentiable z -> full unconstrained parameter vector."""
        driven = self.offsets + self.S @ jnp.asarray(z)
        return jnp.where(jnp.asarray(self.mask), driven, jnp.asarray(self.base))


class Exposure:
    def __init__(self, graph: Graph):
        self.graph = graph
        self._base = graph.theta0.copy()
        self._z0 = []
        self._names = []
        self._entries = []  # (param_index, z_index, scale, offset)

    def expose(self, param_index, name):
        """Expose one existing param directly; z starts at its current value."""
        k = self._new_z(name, self._base[int(param_index)])
        self.drive(param_index, k)
        return k

    def macro(self, name, init):
        """A new macro decision variable (unconstrained space), driving nothing
        until wired with drive()/tie()."""
        return self._new_z(name, float(init))

    def macro_positive(self, name, init):
        """Macro for a positive physical quantity (length/radius); stored
        unconstrained so scale-1 relations to positive params are exact."""
        return self._new_z(name, float(softplus_inverse(init)))

    def drive(self, param_index, z_index, scale=1.0, offset=0.0):
        self._entries.append((int(param_index), int(z_index), float(scale), float(offset)))

    def tie(self, param_index, z_index, scale=1.0):
        """Symmetry coupling: param follows an existing z entry (scale=-1 mirrors)."""
        self.drive(param_index, z_index, scale=scale)

    def _new_z(self, name, init):
        self._names.append(name)
        self._z0.append(float(init))
        return len(self._z0) - 1

    def build(self) -> ExposureMap:
        n_p, n_z = len(self._base), len(self._z0)
        S = np.zeros((n_p, n_z))
        offsets = np.zeros(n_p)
        mask = np.zeros(n_p, dtype=bool)
        for i, k, scale, offset in self._entries:
            S[i, k] += scale
            offsets[i] += offset
            mask[i] = True
        return ExposureMap(
            z0=np.array(self._z0), names=tuple(self._names),
            S=S, offsets=offsets, mask=mask, base=self._base,
        )
