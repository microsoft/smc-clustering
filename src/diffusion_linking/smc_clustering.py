# Licensed under the MIT license.
from math import ceil, floor
import functools, collections, warnings
from itertools import chain
from unidecode import unidecode
import nltk
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from matplotlib import pyplot as plt
from tqdm import tqdm

from diffusion_linking.utils import batched_eval
from diffusion_linking.clustering import Cluster

#====================== Cluster types ====================== 
class GaussianCluster(Cluster):
    """
    Cluster subclass with summary statistics for a Gaussian model
    """
    def __init__(self, data_ids, dim=2, Sx=None, Sxx=None, data=None):
        super().__init__(data_ids)
        
        self.dim = dim
        
        if Sx is not None:
            self.Sx = Sx
        elif data is not None:
            self.Sx = data
        else:
            self.Sx = jnp.zeros((dim,))
            
        if Sxx is not None:
            self.Sxx = Sxx
        elif data is not None:
            self.Sxx = data**2
        else:
            self.Sxx = jnp.zeros((dim,)) 
    
    @property
    def summary(self):
        return [self.Sx, self.Sxx]
    
    def merge_point(self, data_id, data):
        data_ids = self.data.union({data_id})            
        Sx = self.Sx + data
        Sxx = self.Sxx + data**2
        
        return GaussianCluster(data_ids, self.dim, Sx=Sx, Sxx=Sxx)
    
    
def get_counts(strings):
    """
    Convert strings to ASCII and get character counts
    """
    counts = np.zeros((26,), dtype=np.int32)
    for string in strings:
        string = unidecode(string).lower()
        count_dict = collections.Counter(string)
        for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
            counts[i] += count_dict[char]
        
    return counts

class WordCluster(Cluster):
    """
    Cluster subclass with summary statistics for a unigram model
    """
    def __init__(self, data_ids, dim=26, counts=None, data=None):
        super().__init__(data_ids)
        self.dim = dim
        if counts is not None:
            self.counts = counts
        elif data is not None:
            self.counts = get_counts(data)
        else:
            self.counts = jnp.zeros((dim,))          

    @property
    def summary(self):
        return self.counts
        
    def merge_point(self, data_id, data):
        data_counts = get_counts(data)        
        return WordCluster(self.data.union({data_id}), self.dim, counts=self.counts + data_counts) 

class CountDict(dict):
    """
    Dictionary with a default value. Does not insert new keys into the dictionary.
    """
    def __init__(self, default_val, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_val = default_val
    
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.default_val
        
    def copy(self):
        return CountDict(self.default_val, super().copy())

def get_ngrams(string, n):
    """
    Convert string to ASCII and get n-gram counts
    """
    return nltk.everygrams(' '*(n-1) + unidecode(string.strip()).lower().replace(' ', ' '*(n-1)) + ' '*(n-1), max_len=n, min_len=n-1)

def get_ngram_counts(strings, n=2):
    if len(strings) == 1:
        ngrams = get_ngrams(strings[0], n)
        return collections.Counter(ngrams)
    
    else:            
        ngrams = [get_ngrams(string, n) for string in strings if len(string.strip())>0]
        counts = collections.Counter(ngrams[0])
        for ns in ngrams[1:]:
            counts.update(ns)
        return counts

class NgramCluster(Cluster):
    """
    Cluster subclass with summary statistics for an n-gram model
    """
    def __init__(self, data_ids, n=2, counts=None, data=None):
        super().__init__(data_ids)
        self.n = n
        if counts is not None:
            self.counts = counts
        elif data is not None:
            self.counts = get_ngram_counts(data, self.n)
        else:
            self.counts = collections.Counter()

    @property
    def summary(self):
        return self.counts
        
    def merge_point(self, data_id, data):
        new_counts = self.counts + get_ngram_counts(data, self.n)           
        return NgramCluster(self.data.union({data_id}), self.n, counts=new_counts) 

class BigramCluster(NgramCluster):
    def __init__(self, data_ids, counts=None, data=None):
        super().__init__(data_ids, 2, counts, data)
        
class TrigramCluster(NgramCluster):
    def __init__(self, data_ids, counts=None, data=None):
        super().__init__(data_ids, 3, counts, data)

#====================== Mixture models ====================== 
    
class DirichletProcess:
    """
    Dirichlet process prior on cluster sizes
    """
    def __init__(self, alpha):
        self.alpha = alpha
        
    def __call__(self, n_obs, cluster_size, n_clusters):
        # Prior probability of a clustering, based on cluster size and hyperparameter alpha        
        return jnp.where(cluster_size > 1, jnp.log(cluster_size/(n_obs + self.alpha)), cluster_size*jnp.log(self.alpha/(n_obs + self.alpha)))

class PitmanYorProcess:
    """
    Pitman-Yor process prior on cluster sizes
    """
    def __init__(self, alpha, d):
        self.alpha = alpha
        self.d = d
        
    def __call__(self, n_obs, cluster_size, n_clusters):
        # Prior probability of a clustering, based on cluster size and hyperparameters
        return jnp.where(cluster_size > 1, jnp.log((cluster_size - self.d)/(n_obs + self.alpha)), cluster_size*jnp.log((self.alpha+self.d*n_clusters)/(n_obs + self.alpha)))
        
 
class GaussianMixture:
    """
    Gaussian model with normal-inverse-gamma prior on cluster parameters
    """
    def __init__(self, a, b, mu, lam):        
        self.alpha_0 = a
        self.beta_0 = b
        self.mu_0 = mu
        self.lam_0 = lam
            
    @functools.partial(jax.jit, static_argnums = (0))
    @functools.partial(jax.vmap, in_axes = (None, None, 0, 0))
    def _post_predictive(self, x, n, summary):
        Sx = summary[0]
        Sxx = summary[1]
        alpha = self.alpha_0 + n
        lam = self.lam_0 + n
        mu = (self.lam_0 + Sx)/(self.lam_0 + n)
        beta = self.beta_0 + jnp.where(n>0, 1/n, 0) * 1/2*( Sxx*n - Sx**2 + self.lam_0/lam * (Sx - self.mu_0*n)**2 )
                            
        return jnp.sum(jax.scipy.stats.t.logpdf(x, 
                                        df = 2*alpha, 
                                        loc = mu, 
                                        scale = beta * (lam + 1)/(alpha*lam)
                                        ))
    
    def post_predictive(self, x, n, summary):
        batch_size = 2**int(jnp.log2(n.shape[0]).item())
        return batched_eval(self._post_predictive, batch_size, (1,2), x, n, np.array(summary))
    
    
class BagOfWordsMixture:
    """
    Unigram model with Dirichlet prior on character frequencies
    """
    def __init__(self, alpha_0):
        self.alpha_0 = alpha_0
            
    @functools.partial(jax.jit, static_argnums = (0))
    @functools.partial(jax.vmap, in_axes = (None, None, 0))
    def _post_predictive(self, x, counts):
        p = (counts + self.alpha_0) / (jnp.sum(counts) + jnp.sum(self.alpha_0))        
        return jax.scipy.stats.multinomial.logpmf(x, jnp.sum(x), p)
    
    def post_predictive(self, x, n, summary):
        batch_size = 2**int(jnp.log2(n.shape[0]).item())
        return batched_eval(self._post_predictive, batch_size, (1,2), get_counts(x), summary)
    
class NgramMixture:
    """
    N-gram model with Dirichlet prior on n-gram frequencies
    """
    def __init__(self, alpha_0, n=2, prior_counts=None):
        self.alpha_0 = alpha_0
        self.n = n
        
        if prior_counts is not None:
            self.prior_counts = prior_counts
            self.V = len([key for key in prior_counts.keys() if len(key)==1]) + 1
            if self.V < 2:
                raise ValueError("Count dictionary must contain unigram counts.")
        else:
            self.V = 28
            self.prior_counts = CountDict(1)
    
    def post_predictive(self, x, n, summary):
        counts = get_ngram_counts(x, self.n)
        LL = []
        for i in range(len(summary)):
            LL.append( sum([ counts[nj]*jnp.log(summary[i][nj] + self.alpha_0*self.prior_counts[nj]) - counts[nj]*jnp.log(summary[i][nj[:-1]] + self.alpha_0*(self.prior_counts[nj[:-1]] + self.V*self.prior_counts['<UNK>'])) for nj in counts.keys() if len(nj)==self.n]) )
        
        return jnp.array(LL)

class BigramMixture(NgramMixture):
    def __init__(self, alpha_0, prior_counts=None):
        super().__init__(alpha_0, 2, prior_counts) 

class TrigramMixture(NgramMixture):
    def __init__(self, alpha_0, prior_counts=None):
        super().__init__(alpha_0, 3, prior_counts)

#====================== Clusterer ======================  

class SMCClustererState():
    """
    Represents particles as sets of cluster hashes, with a shared lookup table 
    for the ids of the datapoints in each cluster.
    """
    def __init__(self, data, ClusterClass):
        self.data = data
        self.ClusterClass = ClusterClass
        empty_cluster = ClusterClass([])
        initial_cluster = ClusterClass([1], data=self.data[1])
        self.cluster_partition = [{initial_cluster.hash}]
        self.clusters = {empty_cluster.hash: empty_cluster,
                         initial_cluster.hash: initial_cluster
                         }
        self.particles = [[{initial_cluster.hash}]]
        self.weights = [[0]]     
        self.score_cache = {list(self.clusters.keys())[0]: 0}
        self.n_obs = 0
        
        
    def get_descendants(self, particle_id):
        """
        Compute possible descendants of a particle, retrieve summary statistics and weights
        """
        old_cluster_ids = list(self.particles[particle_id[0]][particle_id[1]]) + [self.ClusterClass([]).hash]
        old_clusters = [self.clusters[cluster_hash] for cluster_hash in self.particles[particle_id[0]][particle_id[1]]] + [self.ClusterClass([])]
        
        n = [cluster.size for cluster in old_clusters]
        summary = [cluster.summary for cluster in old_clusters]
        
        return [[particle_id[1]]*len(old_clusters), old_cluster_ids], n, summary, [self.weights[particle_id[0]][particle_id[1]]]*len(old_clusters)
    
 
    def merge_point(self, particle_id, datapoint_id, cluster_hash):
        """
        Add a datapoint to its assigned cluster on a given particle
        """
        new_hash = hash(self.clusters[cluster_hash].add(datapoint_id)) # this could be computed directly
        self.particles[particle_id[0]][particle_id[1]].add(new_hash)          
        
        if new_hash not in self.clusters:
            # create new cluster
            self.clusters[new_hash] = self.clusters[cluster_hash].merge_point(datapoint_id, self.data[datapoint_id])
            
        # delete old cluster from particle
        if cluster_hash in self.particles[particle_id[0]][particle_id[1]]:
            self.particles[particle_id[0]][particle_id[1]].remove(cluster_hash)
                
    
    def update_particle_set(self, subprob, particle_ids, weights, cluster_assignments, datapoint_id):
        """
        Update the particle set with assignments for the new datapoint
        """
        
        # resample the particles
        self.particles[subprob] = [self.particles[subprob][i].copy() for i in particle_ids]
        self.weights[subprob] = weights
        
        # assign the new datapoint to its clusters
        for i, cluster_hash in enumerate(cluster_assignments):
            self.merge_point((subprob,i), datapoint_id, cluster_hash)
            
        # delete the clusters that are no longer in use
        used_clusters = set.union(*self.particles[subprob])
        unused_clusters = self.cluster_partition[subprob] - used_clusters - set({self.ClusterClass([]).hash})
        for cl in unused_clusters:
            del self.clusters[cl]
        self.cluster_partition[subprob] = used_clusters
        
        
    def update_and_merge_particle_set(self, subprobs, particle_ids, weights, cluster_assignments, datapoint_id):
        """
        Update the particle set with assignments for the new datapoint, merging two subproblems together
        """
        # create particles
        new_particles = [ set.union(self.particles[subprobs[0]][p_i], self.particles[subprobs[1]][p_j]) for p_i, p_j in  particle_ids]
        del self.particles[max(subprobs)]
        del self.particles[min(subprobs)]
        del self.weights[max(subprobs)]
        del self.weights[min(subprobs)]
        
        self.particles.append(new_particles)
        self.weights.append(weights)
        subprob = len(self.particles) - 1
        
        # assign the new datapoint to its clusters
        for i, cluster_hash in enumerate(cluster_assignments):
            self.merge_point((subprob,i), datapoint_id, cluster_hash)
            
        # delete the clusters that are no longer in use
        used_clusters = set.union(*self.particles[subprob])
        unused_clusters = set.union(self.cluster_partition[subprobs[0]], self.cluster_partition[subprobs[1]]) - used_clusters - set({self.ClusterClass([]).hash})
        del self.cluster_partition[max(subprobs)]
        del self.cluster_partition[min(subprobs)]
        for cl in unused_clusters:
            del self.clusters[cl]
        self.cluster_partition.append(used_clusters)
        
    def add_subproblem(self, data_id):
        """
        Add datapoint to new subproblem by itself
        """
        new_cluster = self.ClusterClass([data_id], data=self.data[data_id])
        self.cluster_partition.append({new_cluster.hash})
        self.clusters[new_cluster.hash] = new_cluster
        self.particles.append([{new_cluster.hash}])
        self.weights.append([0]) 
            
    def retrieve_cluster_data(self, cluster_hash):
        """
        Return the datapoints in a given cluster
        """        
        return self.data[self.clusters[cluster_hash].ids]
    
    def split_problem(self, s):
        """
        Attempt to split problem s into subproblems
        """
        
        # list datapoints on partition
        data_idx = np.concatenate([self.clusters[cl].ids for cl in self.particles[s][0]])
        
        # construct adjacency matrix
        n = len(data_idx)       
        E = np.zeros((n,n))
        for p, particle in enumerate(self.particles[s]):
            for cl in particle:
                for n_i, i in enumerate(self.clusters[cl].ids):
                    for j in self.clusters[cl].ids[n_i:]:
                            E[jnp.where(data_idx==i)[0], jnp.where(data_idx==j)[0]] += 1
        E = scipy.sparse.csr_matrix(E)
        
        # compute subgraphs
        n_c, c = scipy.sparse.csgraph.connected_components(E, directed=False)        
        if n_c > 1:            
            # list out subproblems
            data_partition = [set(data_idx[np.where(c == i)[0]]) for i in range(n_c)]
            cluster_partition = [set([cl for cl in self.cluster_partition[s] if self.clusters[cl].data.issubset(data_partition[i])]) for i in range(n_c)]
    
            # compute new particles and weights
            new_weights = []
            new_particles = []
            for subprob in cluster_partition:
                new_particles.append([])
                new_weights.append([])
                for p in self.particles[s]:
                    new_particle = p.intersection(subprob)
                    if new_particle not in new_particles[-1]:
                        new_particles[-1].append(new_particle)
                        new_weights[-1].append(jax.scipy.special.logsumexp(jnp.array([self.weights[s][i] for i in range(len(self.particles[s])) if new_particle.issubset(self.particles[s][i])])))
    
            # update state
            del self.cluster_partition[s]
            del self.particles[s]
            del self.weights[s]
            self.cluster_partition += cluster_partition
            self.particles += new_particles
            self.weights += new_weights
            
        return n_c
        
         
class SMCClusterer:
    def __init__(self, data, score_fn, max_evals, max_particles, prior, surrogate, ClusterClass, resample_fn, split_interval=None, resample_inner=None, threshold=None, callback = None):
        self.score_fn = score_fn
        self.prior = prior
        self.surrogate = surrogate         
        self.max_evals = max_evals
        self.max_particles = max_particles
        self.resample = resample_fn
        if resample_inner is not None:
            self.resample_inner = resample_inner
        else:
            self.resample_inner = resample_fn
        self.split_interval = split_interval
        self.threshold = threshold
        self.callback = callback   

        self.state = SMCClustererState(data, ClusterClass)
        
        
    def compute_scores(self, rng, clusters, force_recompute=False):
        """
        For a list of clusters, compute the score for each cluster
        We use a cache to avoid recomputing scores for clusters that have already been computed.

        """
        # remove the score from the cache if we're forcing a recompute
        if force_recompute:
            [self.state.score_cache.pop(hash(cluster), None) for cluster in clusters]
            
        compute_clusters = [ cluster for cluster in clusters if hash(cluster) not in self.state.score_cache]        
        if len(compute_clusters)==0:
            return
        
        hashes = [hash(cluster) for cluster in compute_clusters]
        scores = self.score_fn(rng, [self.state.data[np.fromiter(cluster, dtype=np.int64)] for cluster in compute_clusters])
        for score, hash_ in zip(scores, hashes):
            self.state.score_cache[hash_] = score

    
    def update_step(self, rng, new_obs, verbose):
        """
        Update particle set given new observation
        """
        rng, update_rng, move_rng = jax.random.split(rng, 3)        
        n_probs = len(self.state.weights)
        resampled = False
        
        # Construct putative particle set
        putative_particles = []
        cluster_sizes = []
        n_clusters = []
        summary_stats = []
        old_weights = []       
        for i in range(n_probs):
            putative_particles.append([[],[]])
            cluster_sizes.append([])
            n_clusters.append([])
            summary_stats.append([])
            old_weights.append([])
            for j in range(len(self.state.weights[i])):
                p, n, s, w = self.state.get_descendants((i,j))
                putative_particles[i][0] += p[0]
                putative_particles[i][1] += p[1]
                cluster_sizes[i] += n
                n_clusters[i] += [len(self.state.particles[i][j])]*len(n)
                summary_stats[i] += s
                old_weights[i] += w
        
            putative_particles[i] = np.array(putative_particles[i], dtype=np.int64)
            cluster_sizes[i] = np.array(cluster_sizes[i])
            n_clusters[i] = np.array(n_clusters[i])
            old_weights[i] = np.array(old_weights[i])
        
        putative_weights = [old_weights[i] + self.prior(self.state.n_obs, cluster_sizes[i] + 1, n_clusters[i]) for i in range(n_probs)]
        sur_LL = [self.surrogate.post_predictive(new_obs, cluster_sizes[i], summary_stats[i]) for i in range(n_probs)]
        
        # Choose a subproblem
        if n_probs == 1:
            p = jnp.zeros((), dtype=jnp.int32)
            putative_particles = putative_particles[0]
            putative_weights = putative_weights[0]
            sur_LL = sur_LL[0]
        else:
            test_weights = [putative_weights[i] + sur_LL[i] - jax.scipy.special.logsumexp(old_weights[i], axis=-1) for i in range(n_probs)]
            singleton_weight = jax.scipy.special.logsumexp(jnp.concatenate([ test_weights[i][cluster_sizes[i]==0] for i in range(len(test_weights)) ]))
            test_weights = jnp.array([ jax.scipy.special.logsumexp(test_weights[i][cluster_sizes[i]>0]) for i in range(len(test_weights)) ] + [singleton_weight])
            test_weights -= jax.scipy.special.logsumexp(test_weights)
            
            # Check which subproblems are expected to be selected in the next resample
            # Could work with selection probability instead
            n_expected = jnp.exp(test_weights[:-1])*(self.max_evals if self.max_evals>0 else self.max_particles)
            if jnp.sum(n_expected >= 1) <= 1:
                # Only one subproblem selected
                p = jnp.argmax(test_weights[:-1])
                putative_particles = putative_particles[p]
                putative_weights = putative_weights[p]
                sur_LL = sur_LL[p]
            else:
                # More than one subproblem selected
                ps = jnp.where(jnp.exp(test_weights[:-1])*(self.max_evals if self.max_evals>0 else self.max_particles) >= 1)[0]
                p = jnp.concatenate([jnp.full(len(putative_weights[p_i]), p_i) for p_i in ps])
                putative_particles = np.concatenate([putative_particles[p_i] for p_i in ps], axis=-1)
                putative_weights = jnp.concatenate([putative_weights[p_i] - jax.scipy.special.logsumexp(putative_weights[p_i]) for p_i in ps])
                sur_LL = jnp.concatenate([sur_LL[p_i] for p_i in ps])
                
                if verbose:
                    if self.callback is not None:
                        self.callback(self.state, highlight=self.state.n_obs)
                    else:
                        print(f"\nMore than one subproblem for new point: {self.state.data[self.state.n_obs]}, problem weights {test_weights[ps]}")
                        self.summary(print_cluster_data=True, problems=ps)   
                    
        putative_weights += sur_LL
        
        if self.threshold is not None:
            # Discard assignments below threshold
            keep_ids = sur_LL > sur_LL[-1] - self.threshold
            sur_LL = sur_LL[keep_ids]
            putative_weights = putative_weights[keep_ids]
            putative_particles = putative_particles[:, keep_ids]
            if p.size > 1:
                p = p[keep_ids]
            
        if len(jnp.unique(putative_particles[1])) > max(self.max_evals, self.max_particles):
            # Resample max_evals particles if number of unique new assignments exceeds max_evals

            if p.size==1 and verbose:
                resampled = True
                print("Resampling:")
                self.summary(print_cluster_data=True, problems=[p])
                
            rng, resample_rng = jax.random.split(rng)
            new_particle_ids, putative_weights = self.resample_inner(resample_rng, putative_weights, max(self.max_evals, self.max_particles), state=self.state)
            putative_particles = putative_particles[:, new_particle_ids]
            sur_LL = sur_LL[new_particle_ids]
            if p.size > 1:
                p = p[new_particle_ids]
            
        if self.max_evals > 0:
            # Reweight according to model
            putative_weights -= sur_LL
            new_clusters = [self.state.clusters[cluster].add(self.state.n_obs) for cluster in putative_particles[1]]
            self.compute_scores(update_rng, new_clusters)
            update = jnp.array([ self.state.score_cache[hash(new_cluster)] - self.state.score_cache[old_cluster_id] for new_cluster, old_cluster_id in zip(new_clusters, putative_particles[1]) ])
            putative_weights += update
            
            if putative_weights.shape[0] > self.max_particles:
                # Resample
                if p.size==1 and verbose and not resampled:
                    resampled = True
                    print("Resampling:")
                    self.summary(print_cluster_data=True, problems=[p])
                    
                resampled = True
                rng, resample_rng = jax.random.split(rng)
                new_particle_ids, putative_weights = self.resample(resample_rng, putative_weights, self.max_particles)
                putative_particles = putative_particles[:, new_particle_ids]
                if p.size > 1:
                    p = p[new_particle_ids]                    
                    if len(jnp.unique(p)) == 1:
                        p = p[0]
                    
        empty_hash = self.state.ClusterClass([]).hash
        if (putative_particles[1] == empty_hash).all():
            # If new observation is in a cluster on its own on all particles, add it to a new subproblem
            self.state.add_subproblem(self.state.n_obs)
            p = len(self.state.particles) - 1
            
        elif p.size == 1:
            # Update particle set on chosen subproblem            
            self.state.update_particle_set(p, putative_particles[0], putative_weights - jnp.max(putative_weights), putative_particles[1], self.state.n_obs)
            if verbose and resampled:
                print("After resample:")
                self.summary(print_cluster_data=True, problems=[p])
        
        elif len(jnp.unique(p)) == 2:
            # Compute all combinations of chosen subproblems and merge
            ps = jnp.unique(p)
            n_pairs = sum(p==ps[0])*len(self.state.particles[ps[1]]) + sum(p==ps[1])*len(self.state.particles[ps[0]])
            pairings = np.zeros((n_pairs, 3), dtype=np.int64)
            pair_weights = np.zeros((n_pairs, ))

            i = 0            
            empty_hash = self.state.ClusterClass([]).hash
            singleton_assignments = (putative_particles[1] == empty_hash)
            if sum(singleton_assignments) > 1:
                pass
                        
            for w_i, (p_i, c_i) in zip(putative_weights[p==ps[0]], putative_particles[:, p==ps[0]].transpose()):
                for p_j in range(len(self.state.particles[ps[1]])):
                    pairings[i] = np.array([[p_i, p_j, c_i]])
                    pair_weights[i] = w_i + self.state.weights[ps[1]][p_j]
                    i += 1
                    
            for w_j, (p_j, c_j) in zip(putative_weights[p==ps[1]], putative_particles[:, p==ps[1]].transpose()):
                for p_i in range(len(self.state.particles[ps[0]])):
                    # First check if this combination has already been counted - only possible if new observation is assigned to empty cluster
                    if c_i == empty_hash:
                        duplicate_id = jnp.where((pairings == np.array([[p_i, p_j, c_j]])).all(axis=1))[0]
                        if len(duplicate_id) == 1:
                            pair_weights[duplicate_id] = jax.scipy.special.logsumexp(jnp.concatenate([ pair_weights[duplicate_id], (w_j + self.state.weights[ps[0]][p_i]).reshape((1,)) ]))
                            pass
                    pairings[i] = np.array([[p_i, p_j, c_j]])
                    pair_weights[i] = w_j + self.state.weights[ps[0]][p_i]
                    i += 1
            
            pairings = pairings[:i]
            pair_weights = pair_weights[:i]
            
            if i > self.max_particles:
                # Resample again to bring new subproblem to correct size
                rng, resample_rng = jax.random.split(rng)
                new_particle_ids, weights = self.resample(resample_rng, pair_weights, self.max_particles)
                self.state.update_and_merge_particle_set(ps, pairings[new_particle_ids, 0:2], weights, pairings[new_particle_ids, 2], self.state.n_obs)
            else:
                self.state.update_and_merge_particle_set(ps, pairings[:, 0:2], pair_weights, pairings[:, 2], self.state.n_obs)
                
            p = len(self.state.particles) - 1
            
            if verbose:
                if self.callback is not None:
                    self.callback(self.state, highlight=self.state.n_obs)
                else:
                    print("Merged problem:")
                    self.summary(print_cluster_data=True, problems=[p])
        else:
            # More than two subproblems - could handle with a multinomial resample (although this has high variance)
            raise ValueError("More than two subproblems selected.")

        
        if self.split_interval is not None and len(jnp.concatenate([self.state.clusters[cl].ids for cl in self.state.particles[p][0]])) > self.split_interval:
            # Check whether the current subproblem can be split
            old_summary = self.summary(print_cluster_data=True, problems=[p], print_summary=False)
            n_split = self.state.split_problem(p)
            
            if verbose:
                problems = [len(self.state.particles) - i - 1 for i in range(n_split)]
                if n_split > 1:
                    print("\nProblem to split:")
                    print(old_summary)
                    print("\nSplit problem:")
                    self.summary(print_cluster_data=True, problems=problems)
                

    def cluster(self, rng, steps=None, callback_interval=5, verbose=False):
        """
        Cluster the dataset
        """
        
        if steps is None:
            steps = self.state.data.shape[0] - 1
        
        self.state.n_obs += 1
        rng, score_rng = jax.random.split(rng)
        if self.score_fn is not None:
            self.compute_scores(score_rng, [frozenset({1})])
            
        for _ in (pbar := tqdm(range(1,min(steps, self.state.data.shape[0] - 1)))):
            rng, update_rng = jax.random.split(rng)
            self.state.n_obs += 1
            new_obs = self.state.data[self.state.n_obs]

            self.update_step(update_rng, new_obs, verbose)            
            pbar.set_postfix({"Subproblems":f"{len(self.state.particles)}"})

            if self.callback and callback_interval>0 and self.state.n_obs%callback_interval==0:
                self.callback(self.state)
            
        print()
        

    def summary(self, print_cluster_data = False, problems=None, max_print=10, min_problem_size=1, print_summary=True):
        """
        Summarise particle set
        """
        summary_text = ""
        if problems is None:
            problems = range(len(self.state.particles))
            n_points = [sum([self.state.clusters[c].size for c in self.state.particles[p][0]]) for p in problems]
            summary_text += f"{len(problems)} subproblems of sizes {sorted(n_points, reverse=True)}"
            
        for p in problems:
            n_points = sum([self.state.clusters[c].size for c in self.state.particles[p][0]])
            if n_points >= min_problem_size:
                summary_text += f"\nSubproblem {p}: {len(self.state.particles[p])} particles, {n_points} points"
                for i, (particle, weight) in enumerate(sorted(zip(self.state.particles[p], self.state.weights[p]), key=lambda c: c[1], reverse=True)):
                    if i>=max_print:
                        summary_text += f"\n\t... {len(self.state.particles[p])-max_print} particles omitted"
                        break
                    exp_weight = jnp.exp(weight - jax.scipy.special.logsumexp(jnp.array(self.state.weights[p])))
                    clusters = sorted(particle, key=lambda c: self.state.clusters[c].size, reverse=True)
                    summary_text += f"\n\tParticle {i}, weight {exp_weight:.2g}({weight:.3g}), {len(clusters)} clusters, {[self.state.clusters[c].size for c in clusters]}"
                    if print_cluster_data:
                        for c in clusters:
                            summary_text += f"\n\t\tLL:{self.state.score_cache[c]:.2g}, " if self.max_evals>0 else ""
                            summary_text += str(self.state.retrieve_cluster_data(c))
                        summary_text += "\n"
                summary_text += "\n"
        
        if print_summary:
            print(summary_text)
            
        return summary_text


def plot_particles_2D(state, subprob=None, n_plots=5, fig_scale=3, highlight=None):
    """
    Plot particles with highest weights
    """
    subprob = 0 if len(state.particles)==1 else subprob
    if subprob is not None:
        n_plots = min(n_plots, len(state.particles[subprob]))
        ncols = min(5, n_plots)
        nrows = 1 + (n_plots-1)//ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_scale*ncols,fig_scale*nrows))
        for i, (particle, weight ) in enumerate(sorted(zip(state.particles[subprob], state.weights[subprob]), key=lambda c: c[1], reverse=True)[:n_plots]):
            subfig = axes[int(i//ncols), i%ncols] if nrows>1 else axes[i] if n_plots>1 else axes
            weight = jnp.exp(weight - jax.scipy.special.logsumexp(jnp.array(state.weights[subprob])))
            clusters = sorted([state.retrieve_cluster_data(cluster_hash) for cluster_hash in particle], key=lambda c: c.shape[0], reverse=True)
    
            if state.n_obs < state.data.shape[0]:
                unobserved_pts = state.data[(state.n_obs+1):]
                subfig.plot(unobserved_pts[:,0], unobserved_pts[:,1], 'x', color='black', alpha=0.25, markersize=2)
                
            for c in clusters:
                if c.shape[0] > 1:
                    subfig.plot(c[:, 0], c[:, 1], 'o', markersize=3)
                else:
                    subfig.plot(c[:, 0], c[:, 1], 'ks', markersize=3, alpha=0.75)
                    
            if highlight is not None:
                subfig.plot(state.data[highlight,0], state.data[highlight,1], 'o', markersize=6, color='black', markerfacecolor='none')

            subfig.set_title(f'Particle {i+1}/{len(state.particles[subprob])}, weight {weight:.2g}')
            subfig.set_aspect('equal')
            subfig.set_xticks([])
            subfig.set_yticks([])
    
        for i in range(n_plots,nrows*ncols):
            subfig = axes[int(i//ncols), i%ncols] if nrows>1 else axes[i] if n_plots>1 else axes
            subfig.set_xticks([])
            subfig.set_yticks([])          
    
        fig.tight_layout()
        plt.show()
        
    else:
        ncols = min(5, n_plots)
        nrows = 1 + (n_plots-1)//ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_scale*ncols,fig_scale*nrows))
        markers = ['o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X', 0, 2, 4, 5, 6, 7, 8, 9, 10, 11]  
        for subprob in range(len(state.particles)):
            for i, (particle, weight ) in enumerate(sorted(zip(state.particles[subprob], state.weights[subprob]), key=lambda c: c[1], reverse=True)[:n_plots]):
                subfig = axes[int(i//ncols), i%ncols] if nrows>1 else axes[i] if n_plots>1 else axes
                weight = jnp.exp(weight - jax.scipy.special.logsumexp(jnp.array(state.weights[subprob])))
                clusters = sorted([state.retrieve_cluster_data(cluster_hash) for cluster_hash in particle], key=lambda c: c.shape[0], reverse=True)
        
                for c in clusters:
                    if c.shape[0] > 1:
                        subfig.plot(c[:, 0], c[:, 1], markers[subprob%len(markers)], markersize=3)
                    else:
                        subfig.plot(c[:, 0], c[:, 1], markers[subprob%len(markers)], markersize=3, alpha=0.75, color='black')                
                subfig.annotate(f'{weight:.2g}', (jnp.max(jnp.concatenate(clusters)[:,0]) + 0.1, jnp.mean(jnp.concatenate(clusters)[:,1])))

        for i in range(n_plots):
            subfig = axes[int(i//ncols), i%ncols] if nrows>1 else axes[i] if n_plots>1 else axes                    
            if state.n_obs < state.data.shape[0]:
                unobserved_pts = state.data[(state.n_obs+1):]
                subfig.plot(unobserved_pts[:,0], unobserved_pts[:,1], 'x', color='black', alpha=0.25, markersize=2)               

            if highlight is not None:
                subfig.plot(state.data[highlight,0], state.data[highlight,1], 'o', markersize=6, color='black', markerfacecolor='none')
            
            subfig.set_aspect('equal')
            subfig.set_xticks([])
            subfig.set_yticks([])
    
        for i in range(n_plots,nrows*ncols):
            subfig = axes[int(i//ncols), i%ncols] if nrows>1 else axes[i] if n_plots>1 else axes
            subfig.set_xticks([])
            subfig.set_yticks([])          
    
        fig.tight_layout()
        plt.show()

#====================== Resamplers ====================== 
 
def resample_multinomial(rng, weights, max_particles, **kwargs):
    """
    Simple multinomial resampling scheme.
    """
    w = jnp.exp(weights - jax.scipy.special.logsumexp(weights))
    resample_idx = jax.random.choice(rng, jnp.arange(len(weights)), (max_particles,), replace=True, p=w)
    unique_idx, counts = jnp.unique(resample_idx, return_counts=True)
    return unique_idx, jnp.log(counts) - jnp.log(max_particles)


def resample_stratified(rng, weights, max_particles, **kwargs):
    """
    Stratified resampling scheme of Carpenter et al. (1999).
    """
    w = jnp.exp(weights - jax.scipy.special.logsumexp(weights)) # can the rest be done in log space?
    k = jnp.sum(w)/max_particles
    w = jnp.concatenate([w,jnp.zeros((1,))])
    u = jax.random.uniform(rng, minval=0, maxval=k)
    resample_idx = []
    i = 0
    u -= w[i]
    while i<len(weights):
        if u<0:
            resample_idx.append(i)
            u = u + k
        else:
            i += 1
            u = u - w[i]
            
    unique_idx, counts = jnp.unique(jnp.array(resample_idx, dtype=jnp.int32), return_counts=True)
    return unique_idx, jnp.log(counts) - jnp.log(max_particles)


def resample_optimal(rng, weights, max_particles, **kwargs):
    """
    Optimal resampling scheme of Fearnhead and Clifford (2003) -
    automatically keeps the highest-weighted particles and uses stratified
    resampling to choose the rest.
    """
    w = jnp.exp(weights - jax.scipy.special.logsumexp(weights))
    bounds = [jnp.min(w),jnp.max(w)]
    while bounds[1] != bounds[0]:
        rng, k_rng = jax.random.split(rng)
        k = jax.random.uniform(k_rng, minval=bounds[0], maxval=bounds[1])
        p1 = w[w<=k]
        p2 = w[w>k]
        a = len(p2)
        b = jnp.sum(p1)
        if b/k + a <= max_particles:
            bounds[1] = jnp.max(p1)
        else:
            bounds[0] = jnp.min(p2)
            
    k = bounds[0]
    p1 = w[w<=k]
    p2 = w[w>k]
    a = len(p2)
    b = jnp.sum(p1)
    if b/k + a <= max_particles:
        c = (max_particles-a)/b
        
    elif (w > k).any():
        k = jnp.min(w[w>k])
        p1 = w[w<=k]
        p2 = w[w>k]
        a = len(p2)
        b = jnp.sum(p1)
        c = (max_particles-a)/b        
        
    else:
        c = max_particles
        
    resample_idx = jnp.where(w < 1/c)[0]
    keep_idx = jnp.where(w >= 1/c)[0]
    l = len(weights) - len(resample_idx)
    rng, resample_rng = jax.random.split(rng)
    p, w = resample_stratified(resample_rng, weights[resample_idx], max_particles-l)
    
    new_particles = jnp.concatenate([resample_idx[p], keep_idx])
    new_weights = jnp.concatenate([jnp.full_like(w, jnp.log(1/c)), weights[keep_idx]])
    
    return new_particles, new_weights


def resample_greedy(rng, weights, max_particles, **kwargs):
    """
    Deterministically chooses the top weighted particles.
    """
    idx = jnp.argsort(weights)
    return idx[-max_particles:], weights[idx[-max_particles:]]
