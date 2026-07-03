"""geomk — differentiable parametric geometry kernel (see CLAUDE.md).

Data-first DAG of pure op kernels resolving to a segmented, multi-domain
signed-implicit field. JAX substrate; everything downstream of
``params -> field`` is differentiable in the unconstrained parameter vector.
"""
