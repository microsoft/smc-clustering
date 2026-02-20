import jax
import jax.numpy as jnp
import numpy as np

from smc_clustering.surrogate_models import Gaussian, GaussianCluster
from smc_clustering.clustering import DirichletProcess

d = 2
a=2; b=.5; mu=0; lam=0.0002
theta = np.array([a,a,b,b,mu,mu,lam,lam])
alpha = 20
surrogate = Gaussian(a, b, mu, lam)
prior = DirichletProcess(alpha)
ClusterClass = GaussianCluster

def generate_gauss_dataset():
    a = theta[0:2]; b = theta[2:4]; mu = theta[4:6]; lam = jnp.abs(theta[6:8])

    rng = jax.random.PRNGKey(1)
    n_c = 100
    n_s = 16
    n = 700

    c_rng, rng = jax.random.split(rng)

    sd = (jax.random.gamma(c_rng, a, (n_c, d))*b)**(-0.5)

    c_rng, rng = jax.random.split(rng)
    centers = jax.random.normal(c_rng, (n_c, d))*sd/(125*lam)**0.5 + mu

    s_rng, rng = jax.random.split(rng)
    s_ids = jax.random.choice(s_rng, jnp.arange(0,n_s), (n_c,))
    s_rng, rng = jax.random.split(rng)
    s_centers = jax.random.normal(s_rng, (n_s, d))*1/lam**0.5 + mu

    means = centers + s_centers[s_ids]

    b_rng, rng = jax.random.split(rng)
    beta = jax.random.beta(b_rng, shape=(n_c,), a=1, b=alpha)
    logprops = jnp.array([jnp.log(beta[k]) + jnp.sum(jnp.log(1-beta[:k])) for k in range(len(beta))])

    z_rng, rng = jax.random.split(rng)
    z = jax.random.choice(z_rng, n_c, shape=(n,), p = jnp.exp(logprops - jax.scipy.special.logsumexp(logprops)))

    c_rng, rng = jax.random.split(rng)
    cluster_data = sd[z]*jax.random.normal(c_rng, (n,2)) + means[z]

    labels = np.array(z)
    data = np.array(cluster_data)
    
    return data, labels

def batched_score_eval(rng, clusters):
    n = np.array([cl.shape[0] for cl in clusters])
    summary = np.concatenate([np.array([np.sum(cl, axis=0) if len(cl.shape)>1 else cl for cl in clusters])[:,None,:],
               np.array([np.sum(cl**2, axis=0) if len(cl.shape)>1 else cl**2 for cl in clusters])[:,None,:]], axis=1)
    return surrogate.evidence(n, summary)

