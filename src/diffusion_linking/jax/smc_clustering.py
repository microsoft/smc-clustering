# Licensed under the MIT license.
from math import ceil
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

class Cluster():
    """
    Represents a cluster as a frozen set of datapoint ids.
    """
    def __init__(self, data_ids, size=0):
        self.data = frozenset(data_ids)
        self.size = size
    
    @property
    def ids(self):
        # Convert to numpy array for easier retrieval of datapoints
        return np.fromiter(self.data, dtype=np.int64)
    
    @property
    def hash(self):
        return hash(self.data)
    
    def add(self, data_id):
        return self.data.union({data_id})


class SMCClustererState():
    """
    Represents particles as sets of cluster hashes, with a shared lookup table 
    for the ids of the datapoints in each cluster.
    """
    def __init__(self, data):
        self.data = data
        
        empty_cluster = Cluster([])
        initial_cluster = Cluster([1], 1)
        self.clusters = {empty_cluster.hash: empty_cluster,
                         initial_cluster.hash: initial_cluster
                         }
        self.particles = [set({initial_cluster.hash})]
        self.weights = [0]
        
        
    def add(self, particle_id, datapoint_id, cluster_hash):
        # Add a datapoint to its assigned cluster on a given particle
        
        new_hash = hash(self.clusters[cluster_hash].add(datapoint_id)) # this could be computed directly
        self.particles[particle_id].add(new_hash)          
        
        if new_hash not in self.clusters:
            # create new cluster
            data_ids = self.clusters[cluster_hash].data.union([datapoint_id])            
            size = self.clusters[cluster_hash].size + 1            
            self.clusters[new_hash] = Cluster(data_ids, size)
            
        # delete old cluster from particle
        if cluster_hash in self.particles[particle_id]:
            self.particles[particle_id].remove(cluster_hash)      
            
        return new_hash
    
    
    def update_particle_set(self, keep_ids, weights, cluster_assignments, datapoint_id):
        # Update the particle set with assignments for the new datapoint
        self.particles = [set(self.particles[i]) for i in keep_ids]
        self.weights = weights
        
        for i, cluster_hash in enumerate(cluster_assignments):
            self.add(i, datapoint_id, cluster_hash)
            
        # delete the clusters that are no longer in use
        old_clusters = set(self.clusters.keys()) - set.union(*self.particles) - set({Cluster([]).hash})
        for old_cluster in old_clusters:
            del self.clusters[old_cluster]

            
    def retrieve_cluster_data(self, cluster_hash):
        # Return the datapoints in a given cluster
        return self.data[self.clusters[cluster_hash].ids]
            

class SMCClusterer:
    def __init__(self, data, score_fn, max_particles, alpha, resample_fn, moves = 0, labels = None, batch_shape = (16,16)):
        self.data_dim = data[0].shape[-1]
        self.score_fn = score_fn
        self.max_particles = max_particles
        self.alpha = alpha
        self.score_cache = {}
        self.state = SMCClustererState(jnp.array([jnp.full((1,self.data_dim), float('nan'))] + data).squeeze(1))
        self.n_obs = 0
        
        self.batch_shape = batch_shape
        
        self.resample = resample_fn

    def compute_scores(self, rng, clusters, force_recompute=False):
        """
        For a list of clusters, compute the score for each cluster.

        We use a cache to avoid recomputing scores for clusters that have already been computed.

        The score function is assumed to take a batch of data and a batch of masks,
        and return a batch of scores. The shape of the data is (batch_size, max_cluster_size, dim),
        the shape of the mask is (batch_size, max_cluster_size), and the shape of the scores is (batch_size,).
        """
        # remove the score from the cache if we're forcing a recompute
        if force_recompute:
            [self.score_cache.pop(hash(cluster), None) for cluster in clusters]
            
        compute_clusters = [ cluster for cluster in clusters if hash(cluster) not in self.score_cache]        
        if len(compute_clusters)==0:
            return
        # split clusters into batches and pad last batch (avoids recompilations)
        n_batches = ceil(len(compute_clusters)/self.batch_shape[0])
        compute_clusters = [ compute_clusters[(self.batch_shape[0]*i):min(self.batch_shape[0]*(i+1), len(compute_clusters))] for i in range(n_batches) ]        
        
        compute_clusters[-1] += [frozenset({0})] * (self.batch_shape[0] - len(compute_clusters[-1]))
        
        for cluster_batch in compute_clusters:
            max_size = self.batch_shape[1] + max([0,ceil((max([len(c) for c in cluster_batch])-self.batch_shape[1])/8)])*8
            # prepare the data and masks
            data, masks, hashes = [], [], []
            for cluster in cluster_batch:
    
                hashes.append(hash(cluster))
                size = len(cluster)
    
                # prepare masks
                mask = jnp.concat([jnp.ones((size,)), jnp.zeros((max_size - size,))])
                masks.append(mask)
    
                # append nans onto data to make it the same size
                cluster_data = self.state.data[np.fromiter(cluster, dtype=np.int64)]
                padding = jnp.full((max_size - size, cluster_data.shape[1]), float('nan'))
                data.append(jnp.concat([cluster_data, padding], axis=0))

                
            if len(cluster_batch)<self.batch_shape[0]:
                data += [jnp.full((max_size, self.data_dim), float('nan'))] * (self.batch_shape[0] - len(compute_clusters[-1]))
                masks += [ jnp.zeros((max_size,)) ] * (self.batch_shape[0] - len(compute_clusters[-1]))
            
            if len(data) > 0:
                scores = self.score_fn(rng, jnp.stack(data), jnp.stack(masks))
    
                for score, hash_ in zip(scores, hashes):
                    self.score_cache[hash_] = score

            
    def prior(self, cluster_size):
        # Prior probability of a clustering, based on cluster size and hyperparameter alpha
        
        return jnp.log(cluster_size/(self.n_obs + self.alpha)) if cluster_size>1 else jnp.log(self.alpha/(self.n_obs + self.alpha))
        
    def update_particle(self, rng, new_obs, particle_id, weight):
        # Compute possible descendants (and their weights) of a particle given new observation
        
        old_cluster_ids = list(self.state.particles[particle_id]) + [Cluster([]).hash]
        new_clusters = [self.state.clusters[cluster].add(new_obs) for cluster in old_cluster_ids]
        
        self.compute_scores(rng, new_clusters)
        putative_weights = [ weight + self.prior(len(new_cluster)) - self.prior(self.state.clusters[old_cluster_id].size) + self.score_cache[hash(new_cluster)] - self.score_cache[old_cluster_id] for new_cluster, old_cluster_id in zip(new_clusters[:-1], old_cluster_ids[:-1]) ]
        putative_weights += [ weight + self.prior(1) + self.score_cache[hash(new_clusters[-1])] ]

        return [[particle_id]*len(old_cluster_ids), old_cluster_ids], putative_weights
            
    def update_step(self, rng):
        # Update particle set given new observation
        
        self.n_obs += 1
        putative_particles = [[],[]]
        putative_weights = []
        
        update_rng, move_rng = jax.random.split(rng)
        
        update_rng = jax.random.split(update_rng, self.max_particles)
        for i, (particle, weight) in enumerate(zip(self.state.particles, self.state.weights)):
            p, w = self.update_particle(update_rng[i], self.n_obs, i, weight)
            putative_particles[0] += p[0]
            putative_particles[1] += p[1]
            putative_weights += w
            
        putative_weights = jnp.array(putative_weights)
        
        if putative_weights.shape[0] > self.max_particles:
            rng, resample_rng = jax.random.split(rng)
            new_particle_ids, new_weights = self.resample(resample_rng, putative_weights, self.max_particles)
            keep_ids = [putative_particles[0][i] for i in new_particle_ids]
            cluster_assignments = [putative_particles[1][i] for i in new_particle_ids]
            self.state.update_particle_set(keep_ids, new_weights, cluster_assignments, self.n_obs)
                
        else:
            self.state.update_particle_set(putative_particles[0], putative_weights, putative_particles[1], self.n_obs)


    def cluster(self, rng, plots=0):
        # Cluster the dataset
        
        self.n_obs += 1
        rng, score_rng = jax.random.split(rng)
        self.compute_scores(score_rng, [frozenset({1})])
            
        for _ in (pbar := tqdm(range(1,self.state.data.shape[0]-1))):
            rng, update_rng = jax.random.split(rng)
            self.update_step(update_rng)
            
            pbar.set_postfix({"Particles":f"{len(self.state.particles)}"})

            if plots and self.n_obs%5==0:
                self.plot_particles(n_plots=plots)
            
        print()

    def summary(self):
        # Print out summary of particle set
        
        for i, (particle, weight) in enumerate(sorted(zip(self.state.particles, self.state.weights), key=lambda c: c[1], reverse=True)):
            weight = jnp.exp(weight)/jnp.sum(jnp.exp(jnp.array(self.state.weights)))
            clusters = sorted([self.state.retrieve_cluster_data(cluster_hash) for cluster_hash in particle], key=lambda c: c.shape[0], reverse=True)
            print(f"Particle {i}, weight {weight:.2g}, {len(clusters)} clusters, {sum([c.shape[0] for c in clusters])} points, {[c.shape[0] for c in clusters]}")           
                        
    def plot_particles(self, n_plots=5, fig_scale=3):
        # Plot particles with highest weights
        
        n_plots = min(n_plots, len(self.state.particles))
        ncols = min(5, n_plots)
        nrows = 1 + (n_plots-1)//ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_scale*ncols,fig_scale*nrows))
        for i, (particle, weight ) in enumerate(sorted(zip(self.state.particles, self.state.weights), key=lambda c: c[1], reverse=True)[:n_plots]):
            subfig = axes[int(i//ncols), i%ncols] if nrows>1 else axes[i] if n_plots>1 else axes
            weight = jnp.exp(weight)/jnp.sum(jnp.exp(jnp.array(self.state.weights)))
            clusters = sorted([self.state.retrieve_cluster_data(cluster_hash) for cluster_hash in particle], key=lambda c: c.shape[0], reverse=True)

            if self.n_obs < self.state.data.shape[0]:
                unobserved_pts = self.state.data[(self.n_obs+1):]
                subfig.plot(unobserved_pts[:,0], unobserved_pts[:,1], 'x', color='black', alpha=0.25, markersize=2)
                
            for c in clusters:
                if c.shape[0] > 1:
                    subfig.plot(c[:, 0], c[:, 1], 'o', markersize=3)
                else:
                    subfig.plot(c[:, 0], c[:, 1], 'ks', markersize=3, alpha=0.75)
            subfig.set_title(f'Particle {i+1}/{len(self.state.particles)}, weight {weight:.2g}')
            subfig.set_aspect('equal')
            subfig.set_xticks([])
            subfig.set_yticks([])

        for i in range(n_plots,nrows*ncols):
            subfig = axes[int(i//ncols), i%ncols] if nrows>1 else axes[i] if n_plots>1 else axes
            subfig.set_xticks([])
            subfig.set_yticks([])          

        fig.tight_layout()
        plt.show()
        
        
def resample_multinomial(rng, weights, max_particles):
    """
    Simple multinomial resampling scheme.
    """
    resample_idx = jax.random.choice(rng, jnp.arange(len(weights)), (max_particles,), replace=True, p=jnp.exp(weights)/jnp.sum(jnp.exp(weights)))
    unique_idx, counts = jnp.unique(resample_idx, return_counts=True)
    return unique_idx, [jnp.log(count/max_particles) for count in counts]


def resample_stratified(rng, weights, max_particles):
    """
    Stratified resampling scheme of Carpenter et al. (1999).
    """
    w = jnp.exp(weights)/jnp.sum(jnp.exp(weights))
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
            
    unique_idx, counts = jnp.unique(jnp.array(resample_idx), return_counts=True)
    return unique_idx, [jnp.log(count/max_particles) for count in counts]
  

def resample_optimal(rng, weights, max_particles):
    """
    Optimal resampling scheme of Fearnhead and Clifford (2003) -
    automatically keeps the highest-weighted particles and uses stratified
    resampling to choose the rest.
    """
    w = jnp.exp(weights)/jnp.sum(jnp.exp(weights))
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
    c = (max_particles-a)/b
    
    resample_idx = jnp.array([i for i in range(len(weights)) if w[i] < 1/c])
    l = len(weights) - len(resample_idx)
    rng, resample_rng = jax.random.split(rng)
    p, w = resample_stratified(resample_rng, jnp.array([weights[i] for i in resample_idx]), max_particles-l)
    
    new_particles = list(resample_idx[p]) + [i for i in range(len(weights)) if i not in resample_idx]
    new_weights = [jnp.log(1/c) for w_i in w] + [weights[i] for i in range(len(weights)) if i not in resample_idx]
    return jnp.array(new_particles), new_weights

              