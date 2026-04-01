# Scalable Model-Based Clustering with Sequential Monte Carlo

This repository contains an implementation of the split SMC algorithm proposed in the paper _Scalable Model-Based Clustering with Sequential Monte Carlo_ as well as code for reproducing the experiments.

Split SMC is an online, Bayesian clustering algorithm that decomposes large problems into approximately independent subproblems, producing a compact representation of the posterior distribution over clusterings. This allows it to scale to tasks with complex cluster-likelihoods and a large number of clusters, where existing methods struggle due to prohibitive memory or computational cost. Our approach is motivated by the knowledge base construction problem, which requires clustering large datasets of free-text fragments of information.

Authors: Connie Trojan, Pavel Myshkov, Paul Fearnhead, James Hensman, Tom Minka, Christopher Nemeth

This work is done in collaboration between Microsoft Research and Lancaster University.


## Setup

The recommended way to install the code is to clone the repository and install everything via `uv`.

1. Create the virtual environment and install the package:

   ```bash
   uv sync --locked
   ```

2. Optional: install the MS-KeBAB package for REBEL-based experiments:
   This dependency is only needed for the evaluation scripts that use the REBEL datasets.
   Assuming the MS-KeBAB project is cloned in the parent directory of this repository, you can install it as follows:

   ```bash
   uv pip install --no-deps -e ../MS-KeBAB
   ```
