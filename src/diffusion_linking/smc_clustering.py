# Licensed under the MIT license.
from math import ceil, floor
import functools, collections
from unidecode import unidecode
import nltk
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from diffusion_linking.utils import batched_eval

#====================== Cluster types ====================== 

class Cluster():
    """
    Represents a cluster as a frozen set of datapoint ids.
    """
    def __init__(self, data_ids, dim):
        self.data = frozenset(data_ids)
        self.dim = dim
        self.size = len(data_ids)
    
    @property
    def ids(self):
        # Convert to numpy array for easier retrieval of datapoints
        return np.fromiter(self.data, dtype=np.int64)
    
    @property
    def hash(self):
        return hash(self.data)
    
    @property
    def summary(self):
        return []
    
    def add(self, data_id):
        # This is only used for computing the combined hash - could be computed directly in future
        return self.data.union({data_id})
    
    def merge_point(self, data_id, data):
        return Cluster(self.data.union({data_id}), self.dim)


class GaussianCluster(Cluster):
    def __init__(self, data_ids, dim=2, Sx=None, Sxx=None, data=None):
        super().__init__(data_ids, dim)
        
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
    counts = np.zeros((26,), dtype=np.int32)
    for string in strings:
        string = unidecode(string).lower()
        count_dict = collections.Counter(string)
        for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
            counts[i] += count_dict[char]
        
    return counts

class WordCluster(Cluster):
    def __init__(self, data_ids, dim=26, counts=None, data=None):
        super().__init__(data_ids, dim)
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
    def __init__(self, data_ids, dim=None, n=2, counts=None, data=None):
        super().__init__(data_ids, dim)
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
        return NgramCluster(self.data.union({data_id}), self.dim, self.n, counts=new_counts) 

class BigramCluster(NgramCluster):
    def __init__(self, data_ids, dim=None, counts=None, data=None):
        super().__init__(data_ids, dim, 2, counts, data)
        
class TrigramCluster(NgramCluster):
    def __init__(self, data_ids, dim=None, counts=None, data=None):
        super().__init__(data_ids, dim, 3, counts, data)


#====================== Mixture models ====================== 
    
class DPMixtureModel():
    def __init__(self, alpha):
        self.alpha = alpha
        
    def prior(self, n_obs, cluster_size):
        # Prior probability of a clustering, based on cluster size and hyperparameter alpha        
        return jnp.where(cluster_size > 1, jnp.log(cluster_size/(n_obs + self.alpha)), cluster_size*jnp.log(self.alpha/(n_obs + self.alpha)))
    
    def post_predictive(self, new_obs, cluster_size, cluster_summary):
        raise NotImplementedError
        
 
class GaussianMixture(DPMixtureModel):
    def __init__(self, alpha, a, b, mu, lam):
        super().__init__(alpha)       
        
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
    
    
class BagOfWordsMixture(DPMixtureModel):
    def __init__(self, alpha, alpha_0):
        super().__init__(alpha)       
        self.alpha_0 = alpha_0
            
    @functools.partial(jax.jit, static_argnums = (0))
    @functools.partial(jax.vmap, in_axes = (None, None, 0))
    def _post_predictive(self, x, counts):
        p = (counts + self.alpha_0) / (jnp.sum(counts) + jnp.sum(self.alpha_0))        
        return jax.scipy.stats.multinomial.logpmf(x, jnp.sum(x), p)
    
    def post_predictive(self, x, n, summary):
        batch_size = 2**int(jnp.log2(n.shape[0]).item())
        return batched_eval(self._post_predictive, batch_size, (1,2), get_counts(x), summary)
    
class NgramMixture(DPMixtureModel):
    def __init__(self, alpha, alpha_0, n=2, prior_counts=None):
        super().__init__(alpha)       
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
    def __init__(self, alpha, alpha_0, prior_counts=None):
        super().__init__(alpha, alpha_0, 2, prior_counts) 

class TrigramMixture(NgramMixture):
    def __init__(self, alpha, alpha_0, prior_counts=None):
        super().__init__(alpha, alpha_0, 3, prior_counts)

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
        self.clusters = {empty_cluster.hash: empty_cluster,
                         initial_cluster.hash: initial_cluster
                         }
        self.particles = [set({initial_cluster.hash})]
        self.weights = [0]        
        self.score_cache = {list(self.clusters.keys())[0]: 0}
        self.n_obs = 0
        
        
    def get_descendants(self, particle_id):
        # Compute possible descendants of a particle, retrieve summary statistics and weights
        old_cluster_ids = list(self.particles[particle_id]) + [self.ClusterClass([]).hash]
        old_clusters = [self.clusters[cluster_hash] for cluster_hash in self.particles[particle_id]] + [self.ClusterClass([])]
        
        n = [cluster.size for cluster in old_clusters]
        summary = [cluster.summary for cluster in old_clusters]
        
        return [[particle_id]*len(old_clusters), old_cluster_ids], n, summary, [self.weights[particle_id]]*len(old_clusters)
    
 
    def merge_point(self, particle_id, datapoint_id, cluster_hash):
        # Add a datapoint to its assigned cluster on a given particle        
        new_hash = hash(self.clusters[cluster_hash].add(datapoint_id)) # this could be computed directly
        self.particles[particle_id].add(new_hash)          
        
        if new_hash not in self.clusters:
            # create new cluster
            self.clusters[new_hash] = self.clusters[cluster_hash].merge_point(datapoint_id, self.data[datapoint_id])
            
        # delete old cluster from particle
        if cluster_hash in self.particles[particle_id]:
            self.particles[particle_id].remove(cluster_hash)      
            
        return new_hash
    
    
    def update_particle_set(self, keep_ids, weights, cluster_assignments, datapoint_id):
        # Update the particle set with assignments for the new datapoint
        self.particles = [set(self.particles[i]) for i in keep_ids]
        self.weights = weights
        
        for i, cluster_hash in enumerate(cluster_assignments):
            self.merge_point(i, datapoint_id, cluster_hash)
            
        # delete the clusters that are no longer in use
        old_clusters = set(self.clusters.keys()) - set.union(*self.particles) - set({self.ClusterClass([]).hash})
        for old_cluster in old_clusters:
            del self.clusters[old_cluster]

            
    def retrieve_cluster_data(self, cluster_hash):
        # Return the datapoints in a given cluster
        return self.data[self.clusters[cluster_hash].ids]
            
         
class SMCClusterer:
    def __init__(self, data, score_fn, max_evals, max_particles, surrogate, ClusterClass, resample_fn, resample_inner=None, callback = None):
        self.score_fn = score_fn
        self.surrogate = surrogate 
        
        self.max_evals = max_evals
        self.max_particles = max_particles
        self.resample = resample_fn
        if resample_inner is not None:
            self.resample_inner = resample_inner
        else:
            self.resample_inner = resample_fn
            
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

    
    def update_step(self, rng, new_obs):
        # Update particle set given new observation
        
        putative_particles = [[],[]]
        cluster_sizes = []
        summary_stats = []
        old_weights = []
        
        rng, update_rng, move_rng = jax.random.split(rng, 3)
        
        for i in range(len(self.state.weights)):
            p, n, s, w = self.state.get_descendants(i)
            putative_particles[0] += p[0]
            putative_particles[1] += p[1]
            cluster_sizes += n
            summary_stats += s
            old_weights += w
        
        putative_particles = np.array(putative_particles, dtype=np.int64)
        cluster_sizes = np.array(cluster_sizes)
        old_weights = np.array(old_weights)
        
        putative_weights = old_weights + self.surrogate.prior(self.state.n_obs, cluster_sizes + 1)
        
        if len(jnp.unique(putative_particles[1])) > self.max_evals:
            GMM_LL = self.surrogate.post_predictive(new_obs, cluster_sizes, summary_stats)
            putative_weights += GMM_LL
            
            rng, resample_rng = jax.random.split(rng)
            new_particle_ids, putative_weights = self.resample_inner(resample_rng, putative_weights, self.max_evals if self.max_evals>0 else self.max_particles, state=self.state)
            putative_particles = putative_particles[:, new_particle_ids]
            
            if self.max_evals>0:
                putative_weights -= GMM_LL[new_particle_ids]
        
        if self.max_evals>0:
            new_clusters = [self.state.clusters[cluster].add(self.state.n_obs) for cluster in putative_particles[1]]
            self.compute_scores(update_rng, new_clusters)
            putative_weights += jnp.array([ self.state.score_cache[hash(new_cluster)] - self.state.score_cache[old_cluster_id] for new_cluster, old_cluster_id in zip(new_clusters, putative_particles[1]) ])
            if putative_weights.shape[0] > self.max_particles:
                rng, resample_rng = jax.random.split(rng)
                new_particle_ids, putative_weights = self.resample(resample_rng, putative_weights, self.max_particles)
                putative_particles = putative_particles[:, new_particle_ids]
                
                
        self.state.update_particle_set(putative_particles[0], putative_weights - jnp.max(putative_weights), putative_particles[1], self.state.n_obs)


    def cluster(self, rng, plots=0):
        # Cluster the dataset
        
        self.state.n_obs += 1
        rng, score_rng = jax.random.split(rng)
        if self.score_fn is not None:
            self.compute_scores(score_rng, [frozenset({1})])
            
        for _ in (pbar := tqdm(range(1,self.state.data.shape[0]-1))):
            rng, update_rng = jax.random.split(rng)
            self.state.n_obs += 1
            new_obs = self.state.data[self.state.n_obs]

            self.update_step(update_rng, new_obs)
            
            pbar.set_postfix({"Particles":f"{len(self.state.particles)}"})

            if self.callback and self.state.n_obs%5==0:
                self.callback(self.state)
            
        print()
        

    def summary(self, print_cluster_data = False):
        # Print out summary of particle set
        
        for i, (particle, weight) in enumerate(sorted(zip(self.state.particles, self.state.weights), key=lambda c: c[1], reverse=True)):
            weight = jnp.exp(weight)/jnp.sum(jnp.exp(jnp.array(self.state.weights)))
            clusters = sorted([self.state.retrieve_cluster_data(cluster_hash) for cluster_hash in particle], key=lambda c: len(c), reverse=True)
            print(f"Particle {i}, weight {weight:.2g}, {len(clusters)} clusters, {sum([len(c) for c in clusters])} points, {[len(c) for c in clusters]}")           
            if print_cluster_data:
                for c in clusters:
                    print(c)
                print()


def plot_particles_2D(state, n_plots=5, fig_scale=3):
    # Plot particles with highest weights
    
    n_plots = min(n_plots, len(state.particles))
    ncols = min(5, n_plots)
    nrows = 1 + (n_plots-1)//ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_scale*ncols,fig_scale*nrows))
    for i, (particle, weight ) in enumerate(sorted(zip(state.particles, state.weights), key=lambda c: c[1], reverse=True)[:n_plots]):
        subfig = axes[int(i//ncols), i%ncols] if nrows>1 else axes[i] if n_plots>1 else axes
        weight = jnp.exp(weight)/jnp.sum(jnp.exp(jnp.array(state.weights)))
        clusters = sorted([state.retrieve_cluster_data(cluster_hash) for cluster_hash in particle], key=lambda c: c.shape[0], reverse=True)

        if state.n_obs < state.data.shape[0]:
            unobserved_pts = state.data[(state.n_obs+1):]
            subfig.plot(unobserved_pts[:,0], unobserved_pts[:,1], 'x', color='black', alpha=0.25, markersize=2)
            
        for c in clusters:
            if c.shape[0] > 1:
                subfig.plot(c[:, 0], c[:, 1], 'o', markersize=3)
            else:
                subfig.plot(c[:, 0], c[:, 1], 'ks', markersize=3, alpha=0.75)
        subfig.set_title(f'Particle {i+1}/{len(state.particles)}, weight {weight:.2g}')
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
    return unique_idx, [jnp.log(count/max_particles) for count in counts]


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
    return unique_idx, jnp.log(counts/max_particles)


def resample_optimal(rng, weights, max_particles, **kwargs):
    """
    Optimal resampling scheme of Fearnhead and Clifford (2003) -
    automatically keeps the highest-weighted particles and uses stratified
    resampling to choose the rest.
    """
    w = jnp.exp(weights - jax.scipy.special.logsumexp(weights)) # can the rest be done in log space?
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


def split_resample(rng, weights, max_particles, state):
    """
    Resamples assignments from each particle separately.
    """
    rng = jax.random.split(rng, len(state.particles))
    n_resample = floor(max_particles/len(state.particles))
    new_weights = []
    new_particles = []
    startpt = 0
    for i in range(len(state.particles)):
        endpt = startpt + len(state.particles[i]) + 1

        if n_resample < endpt - startpt:
            w_tot = jax.scipy.special.logsumexp(weights[startpt:endpt])
            idx, w = resample_stratified(rng[i], weights[startpt:endpt], n_resample)
            w += w_tot - jax.scipy.special.logsumexp(w)
            
            new_weights.append(w)
            new_particles.append(idx + startpt)
            
        else:
            new_weights.append(weights[startpt:endpt])
            new_particles.append(jnp.arange(startpt, endpt, dtype=jnp.int32))
        
        startpt = endpt
    return jnp.concatenate(new_particles), jnp.concatenate(new_weights)
        
