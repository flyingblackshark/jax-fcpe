
# FCPE jax version 
## This version is working perfectly fine. 😀 
### Original https://github.com/CNChTu/FCPE

# Example
```Python
import jax_fcpe
import jax.numpy as jnp

a = jnp.ones((16000))
f0 = jax_fcpe.get_f0(a,16000)
print(f0)
