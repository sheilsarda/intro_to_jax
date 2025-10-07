from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp

from jax.sharding import PartitionSpec as P, NamedSharding

# Create a Sharding object to distribute a value across devices:
mesh = jax.make_mesh((1, 1), ('x', 'y'))

# Create an array of random values:
x = jax.random.normal(jax.random.key(0), (8192, 8192))
# and use jax.device_put to distribute it across devices:
y = jax.device_put(x, NamedSharding(mesh, P('x', 'y')))
jax.debug.visualize_array_sharding(y)