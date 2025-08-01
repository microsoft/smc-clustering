# Licensed under the MIT license.
import collections
import functools

import jax
import jax.numpy as jnp
import nltk
import numpy as np
from unidecode import unidecode

from diffusion_linking.clustering import Cluster
from diffusion_linking.utils import batched_eval


# ====================== Surrogate models ======================
class Gaussian:
    """
    Gaussian model with normal-inverse-gamma prior on cluster parameters
    """

    def __init__(self, a, b, mu, lam):
        self.alpha_0 = a
        self.beta_0 = b
        self.mu_0 = mu
        self.lam_0 = lam

    @functools.partial(jax.jit, static_argnums=(0))
    @functools.partial(jax.vmap, in_axes=(None, None, 0, 0))
    def _post_predictive(self, x, n, summary):
        Sx = summary[0]
        Sxx = summary[1]
        alpha = self.alpha_0 + n / 2
        lam = self.lam_0 + n
        mu = (self.lam_0 + Sx) / (self.lam_0 + n)
        beta = self.beta_0 + jnp.where(n > 0, 1 / n, 0) * 1 / 2 * (
            Sxx * n - Sx**2 + self.lam_0 / lam * (Sx - self.mu_0 * n) ** 2
        )

        return jnp.sum(jax.scipy.stats.t.logpdf(x, df=2 * alpha, loc=mu, scale=beta * (lam + 1) / (alpha * lam)))

    def post_predictive(self, x, n, summary):
        batch_size = 2 ** int(jnp.log2(n.shape[0]).item())
        return batched_eval(self._post_predictive, batch_size, (1, 2), x, n, np.array(summary))

    def evidence(self, n, summary):
        Sx = summary[0]
        Sxx = summary[1]
        alpha = self.alpha_0 + n / 2
        lam = self.lam_0 + n
        beta = self.beta_0 + jnp.where(n > 0, 1 / n, 0) * 1 / 2 * (
            Sxx * n - Sx**2 + self.lam_0 / lam * (Sx - self.mu_0 * n) ** 2
        )

        return -jnp.sum(
            jax.scipy.special.gammaln(alpha)
            - jax.scipy.special.gammaln(self.alpha_0)
            + self.alpha_0 * jnp.log(self.beta_0)
            - alpha * jnp.log(beta)
            + 0.5 * jnp.log(self.lam_0)
            - 0.5 * jnp.log(lam)
            - n / 2 * jnp.log(2 * jnp.pi)
        )


def get_counts(strings):
    """
    Convert strings to ASCII and get character counts
    """
    counts = np.zeros((26,), dtype=np.int32)
    for string in strings:
        string = unidecode(string).lower()
        count_dict = collections.Counter(string)
        for i, char in enumerate("abcdefghijklmnopqrstuvwxyz"):
            counts[i] += count_dict[char]

    return counts


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


class BagOfWords:
    """
    Bag-of-words model with Dirichlet prior on frequencies
    """

    def __init__(self, alpha_0):
        self.alpha_0 = alpha_0

    @functools.partial(jax.jit, static_argnums=(0))
    @functools.partial(jax.vmap, in_axes=(None, None, 0, 0))
    def _post_predictive(self, x, n, counts):
        alpha_n = counts + self.alpha_0
        sum_alpha = jnp.sum(alpha_n)
        return (
            jax.scipy.special.gammaln(n + 1)
            - jnp.sum(jax.scipy.special.gammaln(x + 1))
            + jax.scipy.special.gammaln(sum_alpha)
            - jax.scipy.special.gammaln(sum_alpha + n)
            + jnp.sum(jax.scipy.special.gammaln(x + alpha_n) - jax.scipy.special.gammaln(alpha_n))
        )

    def post_predictive(self, x, n, summary):
        batch_size = 2 ** int(jnp.log2(n.shape[0]).item())
        return batched_eval(self._post_predictive, batch_size, (1, 2), get_counts(x), n, summary)

    def evidence(self, n, summary):
        sum_alpha = summary.shape[0] * self.alpha_0
        return (
            jax.scipy.special.gammaln(n + 1)
            - jnp.sum(jax.scipy.special.gammaln(summary + 1))
            + jax.scipy.special.gammaln(sum_alpha)
            - jax.scipy.special.gammaln(sum_alpha + n)
            + jnp.sum(jax.scipy.special.gammaln(summary + self.alpha_0) - jax.scipy.special.gammaln(self.alpha_0))
        )


def get_ngrams(string, n):
    """
    Convert string to ASCII and get n-grams
    """
    return nltk.everygrams(" " * (n - 1) + unidecode(string.strip()).lower() + "E", max_len=n, min_len=n - 1)


def get_ngram_counts(strings, n=2):
    if type(strings[0]) is list:
        strings = sum(strings, start=[])

    if len(strings) == 1:
        ngrams = get_ngrams(strings[0], n)
        return collections.Counter(ngrams)

    else:
        ngrams = [get_ngrams(string, n) for string in strings if len(string.strip()) > 0]
        counts = collections.Counter(ngrams[0])
        for ns in ngrams[1:]:
            counts.update(ns)
        return counts


@jax.jit
@functools.partial(jax.vmap, in_axes=(None, 0, 0))
def dirichlet_categorical_logpmf(x, alphas, sum_alpha):
    return (
        jax.scipy.special.gammaln(sum_alpha)
        - jax.scipy.special.gammaln(sum_alpha + jnp.sum(x))
        + jnp.sum(jax.scipy.special.gammaln(x + alphas) - jax.scipy.special.gammaln(alphas))
    )


class Ngram:
    """
    N-gram model with Dirichlet prior on n-gram frequencies
    """

    def __init__(self, prior_scale, prior_counts, n=2):
        self.prior_scale = prior_scale
        self.n = n

        self.prior_counts = prior_counts
        self.V = len([key for key in prior_counts.keys() if len(key) == 1]) + 1
        if self.V < 2:
            raise ValueError("Count dictionary must contain unigram counts.")

    def post_predictive(self, obs, n, summary):
        batch_size = 2 ** int(jnp.log2(n.shape[0]).item())
        counts = get_ngram_counts(obs, self.n)
        histories = [h for h in counts.keys() if len(h) == (self.n - 1) and h[-1] != "E"]

        LL = np.zeros((len(summary)))
        for h in histories:
            continuations = [ngram for ngram in counts.keys() if ngram[:-1] == h and len(ngram) == self.n]
            x = jnp.array([counts[ngram] for ngram in continuations])
            alphas = jnp.array(
                [
                    [summary[i][ngram] + self.prior_scale * self.prior_counts[ngram] for ngram in continuations]
                    for i in range(n.shape[0])
                ]
            )
            sum_alphas = jnp.array(
                [
                    summary[i][h] + self.prior_scale * (self.prior_counts[h] + self.V * self.prior_counts["<UNK>"])
                    for i in range(n.shape[0])
                ]
            )

            if counts[h] == 1:
                # just one n-gram observation with this history, so equivalent to cheaper categorical pmf
                LL += jnp.log(alphas).flatten() - jnp.log(sum_alphas).flatten()
            else:
                LL += batched_eval(dirichlet_categorical_logpmf, batch_size, (1, 2), x, alphas, sum_alphas)

        return LL

    def evidence(self, n, summary):
        histories = [h for h in summary.keys() if len(h) == (self.n - 1) and h[-1] != "E"]
        LL = 0
        for h in histories:
            continuations = [ngram for ngram in summary.keys() if ngram[:-1] == h and len(ngram) == self.n]
            x = jnp.array([summary[ngram] for ngram in continuations])
            alphas = jnp.array([self.prior_scale * self.prior_counts[ngram] for ngram in continuations])
            sum_alphas = self.prior_scale * (self.prior_counts[h] + self.V * self.prior_counts["<UNK>"])

            if summary[h] == 1:
                LL += jnp.log(alphas) - jnp.log(sum_alphas)
            else:
                LL += dirichlet_categorical_logpmf(x, alphas[None, :], jnp.array([sum_alphas])[None, :]).flatten()

        return LL


class Bigram(Ngram):
    def __init__(self, prior_scale, prior_counts):
        super().__init__(prior_scale, prior_counts, 2)


class Trigram(Ngram):
    def __init__(self, prior_scale, prior_counts):
        super().__init__(prior_scale, prior_counts, 3)


# ====================== Cluster classes with sufficient statistics for surrogate models ======================


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


class WordCluster(Cluster):
    """
    Cluster subclass with summary statistics for a bag-of-words model
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
