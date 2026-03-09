# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Clustering algorithms and utilities for entity linking.

The subpackage exposes agglomerative, Gibbs, and SMC clustering implementations together with shared data structures, metrics, and helper functions.
"""

from smc_clustering.clustering import agglomerative, cluster, mcmc, metrics, smc, utils
