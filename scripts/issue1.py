import jax
import jax.numpy as jnp

def f(b, x):
  out = jax.nn.dot_product_attention(x, x, x, bias=b, implementation='cudnn')
  return jnp.sum(out)

f_grad = jax.jit(jax.grad(f))

seq_len = 128
batch = 8
heads = 4

x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, heads, 32), dtype=jnp.bfloat16)
bias = jax.random.normal(jax.random.PRNGKey(0), (1, heads, seq_len, seq_len), dtype=jnp.float32)

grad = f_grad(bias, x)
jnp.isfinite(grad).all()
print(grad)