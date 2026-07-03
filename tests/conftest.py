import jax

# Float64 everywhere: finite-difference CI is meaningless in float32.
jax.config.update("jax_enable_x64", True)
