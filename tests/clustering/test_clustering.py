# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Test that the clustering algorithms give the expected output on a toy problem.
"""
import jax
import numpy as np
import math
import pytest

from smc_clustering.clustering.surrogate_models import Gaussian, GaussianCluster
from smc_clustering.clustering.cluster import DirichletProcess
from smc_clustering.clustering.smc import SMCClusterer, resample_greedy
from smc_clustering.clustering.mcmc import GibbsClusterer
from smc_clustering.clustering.agglomerative import Clusterer


@pytest.fixture
def data():
    return np.array(
          [[12.015805 ,  1.9518057],
           [10.9575205, -2.55881  ],
           [10.999484 , -0.4894154],
           [-3.5586088,  2.3852081],
           [12.014804 ,  2.1909385],
           [12.6026745,  3.9515743],
           [-4.5497727,  4.4243646],
           [-4.880345 ,  2.9680119],
           [ 9.406058 ,  2.3700042],
           [13.361339 , -1.0973097]]
          )

@pytest.fixture
def prior():
    return DirichletProcess(1)

@pytest.fixture
def surrogate():
    return Gaussian(1, 1, 0, 0.01)


def test_clustering_smc(data, prior, surrogate):
    """Test SMC clustering, without split step"""
    clusterer = SMCClusterer(data = data, 
                             max_particles = 5,
                             max_evals = 0,
                             score_fn = None,
                             prior = prior, 
                             surrogate = surrogate, 
                             resample_fn = resample_greedy,
                             ClusterClass = GaussianCluster
                             )
    
    rng = jax.random.PRNGKey(0)
    evals, subprobs = clusterer.cluster(rng,callback_interval=0) 
    
    # check number of subproblems at each point in time
    assert subprobs == [1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    # check the top-weighted clustering
    reference_labels = np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 1])
    _, clusterer_labels = np.unique(clusterer.list_cluster_labels(), return_inverse=True)
    assert list(clusterer_labels) == list(reference_labels)
    
    # check the log-posterior density of the top-weighted clustering
    assert math.isclose(clusterer.best_logpost, -42.167816, rel_tol=1e-4, abs_tol=1e-4)
    
    # check all particle weights
    reference_weights = [np.array([-8.43127867, -8.29379147, -7.82196373, -7.40201631, -4.40918871])]
    assert [len(w) for w in clusterer.state.weights] == [len(w) for w in reference_weights]
    
    weight_diff = np.sum([np.abs(w - reference_weights[i]) for i, w in enumerate(clusterer.state.weights)])
    assert math.isclose(weight_diff, 0, rel_tol=1e-4, abs_tol=1e-4)


def test_clustering_split_smc(data, prior, surrogate):
    """Test SMC clustering, with split step"""
    clusterer = SMCClusterer(data = data, 
                             split = 1,
                             max_particles = 5,
                             max_evals = 0,
                             score_fn = None,
                             prior = prior, 
                             surrogate = surrogate, 
                             resample_fn = resample_greedy,
                             ClusterClass = GaussianCluster
                             )
    
    rng = jax.random.PRNGKey(0)
    evals, subprobs = clusterer.cluster(rng,callback_interval=0) 
    
    # check number of subproblems at each point in time
    assert subprobs == [1, 1, 1, 2, 2, 2, 2, 2, 2]
    
    # check the top-weighted clustering
    reference_labels = np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 1])
    _, clusterer_labels = np.unique(clusterer.list_cluster_labels(), return_inverse=True)
    assert list(clusterer_labels) == list(reference_labels)
    
    # check the log-posterior density of the top-weighted clustering
    assert math.isclose(clusterer.best_logpost, -42.167816, rel_tol=1e-4, abs_tol=1e-4)
    
    # check all particle weights
    reference_weights = [np.array([-10.53582581,  -7.74093588,  -6.8029681 ,  -6.78348787,  -2.76139792]),
                         np.array([-8.41763892, -7.91175309, -7.43992534, -7.01997792, -4.02715032])]
    assert [len(w) for w in clusterer.state.weights] == [len(w) for w in reference_weights]
    
    weight_diff = np.sum([np.abs(w - reference_weights[i]) for i, w in enumerate(clusterer.state.weights)])
    assert math.isclose(weight_diff, 0, rel_tol=1e-4, abs_tol=1e-4)


def test_clustering_mcmc(data, prior, surrogate):
    """Test Gibbs clustering"""  
    def batched_likelihood(rng, clusters):
        n = np.array([cl.shape[0] for cl in clusters])
        summary = np.concatenate([np.array([np.sum(cl, axis=0) if len(cl.shape)>1 else cl for cl in clusters])[:,None,:],
                   np.array([np.sum(cl**2, axis=0) if len(cl.shape)>1 else cl**2 for cl in clusters])[:,None,:]], axis=1)
        return surrogate.evidence(n, summary)
    
    clusterer = GibbsClusterer(data = data, 
                               score_fn = batched_likelihood,
                               prior = prior, 
                               surrogate = surrogate,
                               ClusterClass = GaussianCluster)
    
    rng = jax.random.PRNGKey(0)
    _ = clusterer.cluster(rng, sweeps=10)

    # check current state's clusters and log posterior density
    assert math.isclose(clusterer.logpost, -42.25396, rel_tol=1e-4, abs_tol=1e-4)

    reference_labels = np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 1])
    _, clusterer_labels = np.unique(clusterer.list_cluster_labels(best=False), return_inverse=True)
    assert list(clusterer_labels) == list(reference_labels)
    
    # check MAP clustering and log posterior density 
    assert math.isclose(clusterer.best_logpost, -42.25396, rel_tol=1e-4, abs_tol=1e-4)

    reference_labels = np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 1])
    _, clusterer_labels = np.unique(clusterer.list_cluster_labels(), return_inverse=True)
    assert list(clusterer_labels) == list(reference_labels)
    
    

def test_clustering_agglomerative(data, prior, surrogate):
    """Test agglomerative clustering"""
    def batched_likelihood(rng, clusters):
        n = np.array([cl.shape[0] for cl in clusters])
        summary = np.concatenate([np.array([np.sum(cl, axis=0) if len(cl.shape)>1 else cl for cl in clusters])[:,None,:],
                   np.array([np.sum(cl**2, axis=0) if len(cl.shape)>1 else cl**2 for cl in clusters])[:,None,:]], axis=1)
        return surrogate.evidence(n, summary)
    
    clusterer = Clusterer(data = data, 
                          score_fn = batched_likelihood, 
                          prior = prior, 
                          cluster_batch_size = 16)
    
    rng = jax.random.PRNGKey(0)
    _, done = clusterer.cluster(rng, max_iter=500)
    
    # check algorithm has terminated
    assert done
    
    # check clustering
    reference_labels = np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 1])
    _, clusterer_labels = np.unique(clusterer.list_cluster_labels(), return_inverse=True)
    assert list(clusterer_labels) == list(reference_labels)
    
    # check log posterior density
    assert math.isclose(clusterer.objective, -42.25396, rel_tol=1e-4, abs_tol=1e-4)

