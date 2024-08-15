# Licensed under the MIT license.
import jax
import jax.numpy as jnp
import numpy as np

def generate_circles1(
    rng: jax.Array,
    num_circles: int,
    min_radius: float = 0.1,
    max_radius: float = 1.5,
    min_points: int = 3,
    max_points: int = 20,
    min_x: float = -5.0,
    max_x: float = 5.0,
    min_y: float = -5.0,
    max_y: float = 5.0,
):
    circles = []
    masks = []
    for _ in range(num_circles):
        r_rng, x_rng, y_rng, n_rng, theta_rng = jax.random.split(rng, 5)
        
        radius = jax.random.uniform(r_rng, (1,)) * (max_radius - min_radius) + min_radius
        x = jax.random.uniform(x_rng, (1,)) * (max_x - min_x) + min_x
        y = jax.random.uniform(y_rng, (1,)) * (max_y - min_y) + min_y
        num_points = jax.random.randint(n_rng, (1,), min_points, max_points).item()
        
        theta = jax.random.uniform(theta_rng, (num_points,)) * 2 * 3.14159
        x = radius * jnp.cos(theta) + x
        y = radius * jnp.sin(theta) + y
        circle = jnp.stack([x, y], axis=-1)

        # do some padding with zeros
        num_pad = max_points - num_points
        circle = jnp.concat([circle, jnp.zeros((num_pad, 2))], axis=0)
        mask = jnp.concat([jnp.ones((num_points,), dtype=bool), jnp.zeros((max_points-num_points,), dtype=bool)]) 

        circles.append(circle)
        masks.append(mask)

    return circles, masks


def generate_circles(
    rng: jax.Array,
    num_circles: int,
    min_radius: float = 0.1,
    max_radius: float = 1.5,
    min_points: int = 3,
    max_points: int = 20,
    min_x: float = -5.0,
    max_x: float = 5.0,
    min_y: float = -5.0,
    max_y: float = 5.0,
):
    r_rng, x_rng, y_rng, n_rng, theta_rng = jax.random.split(rng, 5)
    
    radii = jax.random.uniform(r_rng, (num_circles,)) * (max_radius - min_radius) + min_radius
    x = jax.random.uniform(x_rng, (num_circles,)) * (max_x - min_x) + min_x
    y = jax.random.uniform(y_rng, (num_circles,)) * (max_y - min_y) + min_y
    num_points = jax.random.randint(n_rng, (num_circles,), min_points, max_points)
    
    theta = jax.random.uniform(theta_rng, (jnp.sum(num_points),)) * 2 * 3.14159
    i = jnp.arange(0,max_points)
    masks = jax.vmap(lambda n: jnp.where(i<n, True, False))(num_points) 
    masks = jnp.array(masks, dtype=jnp.bool)
    
    angles = np.zeros((max_points*num_circles,))
    angles[jnp.concat(masks, axis=-1)] = theta    
    
    def gen_circle(x, y, radius, angles, mask):
        x = mask * (radius * jnp.cos(angles) + x)
        y = mask * (radius * jnp.sin(angles) + y)
        circle = jnp.stack([x, y], axis=-1)
        return circle
    
    circles = jax.vmap(gen_circle, in_axes=(0,0,0,0,0))(x,y,radii,jnp.array(jnp.split(angles, num_circles)), masks)

    return list(np.array(circles)), list(np.array(masks))
