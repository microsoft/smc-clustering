# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import collections
import collections.abc
import functools
import re
from typing import Any

import jax
import jax.numpy as jnp
import nltk
import numpy as np
import scipy
from unidecode import unidecode

from smc_clustering.clustering.cluster import Cluster
from smc_clustering.clustering.utils import batched_eval


# ====================== Surrogate models ======================
class Gaussian:
    """Gaussian model with normal-inverse-gamma prior on cluster parameters"""

    def __init__(self, a: float, b: float, mu: float, lam: float):
        self.alpha_0 = a
        self.beta_0 = b
        self.mu_0 = mu
        self.lam_0 = lam

    @functools.partial(jax.jit, static_argnums=(0))
    @functools.partial(jax.vmap, in_axes=(None, None, 0, 0))
    def _post_predictive(self, x: jax.Array, n: jax.Array, summary: jax.Array) -> jax.Array:
        Sx = summary[0]
        Sxx = summary[1]
        alpha = self.alpha_0 + n / 2
        lam = self.lam_0 + n
        mu = (self.lam_0 + Sx) / (self.lam_0 + n)
        beta = self.beta_0 + jnp.where(n > 0, 1 / n, 0) * 1 / 2 * (
            Sxx * n - Sx**2 + self.lam_0 / lam * (Sx - self.mu_0 * n) ** 2
        )

        return jnp.sum(
            jax.scipy.stats.t.logpdf(
                x, df=2 * alpha, loc=mu, scale=jnp.sqrt(beta * (lam + 1) / (alpha * lam))
            )
        )

    def post_predictive(self, x: np.ndarray, n: np.ndarray, summary: np.ndarray) -> np.ndarray:
        batch_size = 2 ** int(np.log2(n.shape[0]).item())
        return np.array(batched_eval(self._post_predictive, batch_size, (1, 2), x, n, np.array(summary)))

    @functools.partial(jax.jit, static_argnums=(0))
    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def _evidence(self, n: jax.Array, summary: jax.Array) -> jax.Array:
        Sx = summary[0]
        Sxx = summary[1]
        alpha = self.alpha_0 + n / 2
        lam = self.lam_0 + n
        beta = self.beta_0 + jnp.where(n > 0, 1 / n, 0) * 1 / 2 * (
            Sxx * n - Sx**2 + self.lam_0 / lam * (Sx - self.mu_0 * n) ** 2
        )

        return jnp.sum(
            jax.scipy.special.gammaln(alpha)
            - jax.scipy.special.gammaln(self.alpha_0)
            + self.alpha_0 * jnp.log(self.beta_0)
            - alpha * jnp.log(beta)
            + 0.5 * jnp.log(self.lam_0)
            - 0.5 * jnp.log(lam)
            - n / 2 * jnp.log(2 * jnp.pi)
        )

    def evidence(self, n: np.ndarray, summary: np.ndarray) -> np.ndarray:
        batch_size = 2 ** int(np.log2(n.shape[0]).item())
        return np.array(batched_eval(self._evidence, batch_size, (0, 1), n, np.array(summary)))


class Bernoulli:
    """Bernoulli model with beta prior on cluster parameters"""

    def __init__(self, a: float, b: float):
        self.alpha_0 = a
        self.beta_0 = b

    @functools.partial(jax.jit, static_argnums=(0))
    @functools.partial(jax.vmap, in_axes=(None, None, 0, 0))
    def _post_predictive(self, x: jax.Array, n: jax.Array, Sy: jax.Array) -> jax.Array:
        alpha = self.alpha_0 + Sy
        beta = self.beta_0 + n - Sy

        return jnp.sum(jnp.log(x * alpha + (1 - x) * beta) - jnp.log(alpha + beta))

    def post_predictive(self, x: np.ndarray, n: np.ndarray, summary: np.ndarray) -> np.ndarray:
        batch_size = 2 ** int(np.log2(n.shape[0]).item())
        return np.array(batched_eval(self._post_predictive, batch_size, (1, 2), x, n, np.array(summary)))

    @functools.partial(jax.jit, static_argnums=(0))
    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def _evidence(self, n: jax.Array, Sy: jax.Array) -> jax.Array:
        alpha = self.alpha_0 + Sy
        beta = self.beta_0 + n - Sy

        return jnp.sum(
            jax.scipy.special.gammaln(alpha)
            + jax.scipy.special.gammaln(beta)
            - jax.scipy.special.gammaln(self.alpha_0)
            - jax.scipy.special.gammaln(self.beta_0)
            - jax.scipy.special.gammaln(alpha + beta)
            + jax.scipy.special.gammaln(self.alpha_0 + self.beta_0)
        )

    def evidence(self, n: np.ndarray, summary: np.ndarray) -> np.ndarray:
        batch_size = 2 ** int(np.log2(n.shape[0]).item())
        return np.array(batched_eval(self._evidence, batch_size, (0, 1), n, np.array(summary)))


def get_counts(strings: list[str]) -> np.ndarray:
    """Convert strings to ASCII and get character counts"""
    counts = np.zeros((26,), dtype=np.int32)
    for string in strings:
        string = unidecode(string).lower()
        count_dict = collections.Counter(string)
        for i, char in enumerate("abcdefghijklmnopqrstuvwxyz"):
            counts[i] += count_dict[char]

    return counts


class CountDict(dict):
    """Dictionary with a default value. Does not insert new keys into the dictionary."""

    def __init__(self, default_val: Any, *args: Any, **kwargs: Any):  # noqa: ANN401
        super().__init__(*args, **kwargs)
        self.default_val = default_val

    def __getitem__(self, key: Any) -> Any:  # noqa: ANN401
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.default_val

    def copy(self) -> CountDict:
        return CountDict(self.default_val, super().copy())


class Multinomial:
    """Multinomial model with Dirichlet prior on frequencies"""

    def __init__(self, alpha_0: float):
        self.alpha_0 = alpha_0

    @functools.partial(jax.jit, static_argnums=(0))
    @functools.partial(jax.vmap, in_axes=(None, None, 0, 0))
    def _post_predictive(self, x: jax.Array, n: jax.Array, counts: jax.Array) -> jax.Array:
        alpha_n = counts + self.alpha_0
        sum_alpha = jnp.sum(alpha_n)
        return (
            jax.scipy.special.gammaln(n + 1)
            - jnp.sum(jax.scipy.special.gammaln(x + 1))
            + jax.scipy.special.gammaln(sum_alpha)
            - jax.scipy.special.gammaln(sum_alpha + jnp.sum(x))
            + jnp.sum(jax.scipy.special.gammaln(x + alpha_n) - jax.scipy.special.gammaln(alpha_n))
        )

    def post_predictive(self, x: np.ndarray, n: np.ndarray, summary: np.ndarray) -> np.ndarray:
        batch_size = 2 ** int(np.log2(n.shape[0]).item())
        return np.array(
            batched_eval(self._post_predictive, batch_size, (1, 2), x, n, np.array(summary)).flatten()
        )

    @functools.partial(jax.jit, static_argnums=(0))
    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def _evidence(self, n: jax.Array, summary: jax.Array) -> jax.Array:
        sum_alpha = summary.shape[0] * self.alpha_0
        return (
            jax.scipy.special.gammaln(n + 1)
            - jnp.sum(jax.scipy.special.gammaln(summary + 1))
            + jax.scipy.special.gammaln(sum_alpha)
            - jax.scipy.special.gammaln(sum_alpha + n)
            + jnp.sum(
                jax.scipy.special.gammaln(summary + self.alpha_0)
                - jax.scipy.special.gammaln(self.alpha_0)
            )
        )

    def evidence(self, n: np.ndarray, summary: np.ndarray) -> np.ndarray:
        batch_size = 2 ** int(np.log2(n.shape[0]).item())
        return np.array(batched_eval(self._evidence, batch_size, (0, 1), n, np.array(summary)))


def get_ngrams(string: str, n: int) -> collections.abc.Iterator[tuple[str, ...]]:
    """Convert string to ASCII and get n-grams"""
    string = re.sub(r"[^a-z0-9 \-]", "", unidecode(string.strip()).lower())
    return nltk.everygrams(" " * (n - 1) + string + "E", max_len=n, min_len=n - 1)


def get_ngram_counts(
    strings: list[str] | list[list[str]], n: int = 2
) -> collections.Counter[tuple[str, ...]]:
    if type(strings[0]) is list:
        strings = sum(strings, start=[])

    if len(strings) == 1:
        ngrams = get_ngrams(strings[0], n)
        return collections.Counter(ngrams)

    ngrams = [get_ngrams(string, n) for string in strings if len(string.strip()) > 0]
    counts = collections.Counter(ngrams[0])
    for ns in ngrams[1:]:
        counts.update(ns)
    return counts


@jax.jit
@functools.partial(jax.vmap, in_axes=(None, 0, 0))
def dirichlet_categorical_logpmf(x: jax.Array, alphas: jax.Array, sum_alpha: jax.Array) -> jax.Array:
    return (
        jax.scipy.special.gammaln(sum_alpha)
        - jax.scipy.special.gammaln(sum_alpha + jnp.sum(x))
        + jnp.sum(jax.scipy.special.gammaln(x + alphas) - jax.scipy.special.gammaln(alphas))
    )


def dirichlet_categorical_logpmf_numpy(
    x: np.ndarray, alphas: np.ndarray, sum_alpha: np.ndarray
) -> np.ndarray:
    return (
        scipy.special.gammaln(sum_alpha)
        - scipy.special.gammaln(sum_alpha + np.sum(x))
        + np.sum(scipy.special.gammaln(x[None, :] + alphas) - scipy.special.gammaln(alphas), axis=-1)
    )


class Ngram:
    """N-gram model with Dirichlet prior on n-gram frequencies"""

    def __init__(self, prior_scale: float, prior_counts: CountDict, n: int = 2):
        self.prior_scale = prior_scale
        self.n = n

        self.prior_counts = prior_counts
        self.V = len([key for key in prior_counts.keys() if len(key) == 1]) + 1
        if self.V < 2:
            raise ValueError("Count dictionary must contain unigram counts.")

    def post_predictive(
        self,
        obs: list[str] | list[list[str]],
        n: np.ndarray,
        summary: list[collections.Counter[tuple[str, ...]]],
    ) -> np.ndarray:
        batch_size = 2 ** int(np.log2(n.shape[0]).item())
        counts = get_ngram_counts(obs, self.n)
        histories = [h for h in counts.keys() if len(h) == (self.n - 1) and h[-1] != "E"]

        LL = np.zeros(len(summary))
        for h in histories:
            continuations = [
                ngram for ngram in counts.keys() if ngram[:-1] == h and len(ngram) == self.n
            ]
            x = np.array([counts[ngram] for ngram in continuations])
            alphas = np.array(
                [
                    [
                        summary[i][ngram] + self.prior_scale * self.prior_counts[ngram]
                        for ngram in continuations
                    ]
                    for i in range(n.shape[0])
                ]
            )
            sum_alphas = np.array(
                [
                    summary[i][h]
                    + self.prior_scale * (self.prior_counts[h] + self.V * self.prior_counts["<UNK>"])
                    for i in range(n.shape[0])
                ]
            )

            if counts[h] == 1:
                # just one n-gram observation with this history, so equivalent to cheaper categorical pmf
                LL += np.log(alphas).flatten() - np.log(sum_alphas).flatten()
            else:
                # LL += np.array(batched_eval(dirichlet_categorical_logpmf, batch_size, (1, 2), x, alphas, sum_alphas))
                LL += dirichlet_categorical_logpmf_numpy(x, alphas, sum_alphas)

        return LL

    def _evidence(self, n: np.intp, summary: collections.Counter[tuple[str, ...]]) -> float | np.ndarray:
        histories = [h for h in summary.keys() if len(h) == (self.n - 1) and h[-1] != "E"]
        LL = 0
        for h in histories:
            continuations = [
                ngram for ngram in summary.keys() if ngram[:-1] == h and len(ngram) == self.n
            ]
            x = np.array([summary[ngram] for ngram in continuations])
            alphas = np.array([self.prior_scale * self.prior_counts[ngram] for ngram in continuations])
            sum_alphas = self.prior_scale * (self.prior_counts[h] + self.V * self.prior_counts["<UNK>"])

            if summary[h] == 1:
                LL += np.log(alphas) - np.log(sum_alphas)
            else:
                LL += dirichlet_categorical_logpmf_numpy(
                    x, alphas[None, :], np.array([sum_alphas])[None, :]
                ).flatten()

        return LL

    def evidence(self, n: np.ndarray, summary: list[collections.Counter[tuple[str, ...]]]) -> np.ndarray:
        return np.array([self._evidence(n[i], summary[i]) for i in range(len(n))])


class Bigram(Ngram):
    def __init__(self, prior_scale: float, prior_counts: CountDict):
        super().__init__(prior_scale, prior_counts, 2)


class Trigram(Ngram):
    def __init__(self, prior_scale: float, prior_counts: CountDict):
        super().__init__(prior_scale, prior_counts, 3)


# ====================== Cluster classes with sufficient statistics for surrogate models ======================


class GaussianCluster(Cluster):
    """Cluster subclass with summary statistics for a Gaussian model"""

    def __init__(
        self,
        data_ids: frozenset[int],
        dim: int = 2,
        Sx: np.ndarray | None = None,
        Sxx: np.ndarray | None = None,
        data: np.ndarray | None = None,
    ):
        super().__init__(data_ids)

        self.dim = dim

        if Sx is not None:
            self.Sx = Sx
        elif data is not None:
            self.Sx = np.sum(data, axis=0) if len(data.shape) > 1 else data
        else:
            self.Sx = np.zeros((dim,))

        if Sxx is not None:
            self.Sxx = Sxx
        elif data is not None:
            self.Sxx = np.sum(data**2, axis=0) if len(data.shape) > 1 else data**2
        else:
            self.Sxx = np.zeros((dim,))

    @property
    def summary(self) -> list[np.ndarray]:
        return [self.Sx, self.Sxx]

    def merge_point(self, data_id: int, data: np.ndarray) -> GaussianCluster:
        data_ids = self.data.union({data_id})
        Sx = self.Sx + data
        Sxx = self.Sxx + data**2

        return GaussianCluster(data_ids, self.dim, Sx=Sx, Sxx=Sxx)


class BernoulliCluster(Cluster):
    """Cluster subclass with summary statistics for a Bernoulli model"""

    def __init__(
        self,
        data_ids: frozenset[int],
        dim: int = 1,
        Sy: np.ndarray | None = None,
        data: np.ndarray | None = None,
    ):
        super().__init__(data_ids)

        self.dim = dim

        if Sy is not None:
            self.Sy = Sy
        elif data is not None:
            self.Sy = np.sum(data, axis=0) if len(data.shape) > 1 else data
        else:
            self.Sy = np.zeros((dim,))

    @property
    def summary(self) -> np.ndarray:
        return self.Sy

    def merge_point(self, data_id: int, data: np.ndarray) -> BernoulliCluster:
        data_ids = self.data.union({data_id})
        Sy = self.Sy + data

        return BernoulliCluster(data_ids, self.dim, Sy=Sy)


class MultinomialCluster(Cluster):
    """Cluster subclass with summary statistics for a multinomial model"""

    def __init__(self, data_ids: frozenset[int], dim: int, data: np.ndarray | None = None):
        super().__init__(data_ids)
        self.dim = dim
        if data is not None:
            self.counts = np.sum(data, axis=0) if len(data.shape) > 1 else data
        else:
            self.counts = np.zeros((dim,))

    @property
    def summary(self) -> np.ndarray:
        return self.counts

    def merge_point(self, data_id: int, data: np.ndarray) -> MultinomialCluster:
        return MultinomialCluster(self.data.union({data_id}), self.dim, data=self.counts + data)


class WordCluster(Cluster):
    """Cluster subclass with summary statistics for a bag-of-words model"""

    def __init__(
        self,
        data_ids: frozenset[int],
        dim: int = 26,
        counts: np.ndarray | None = None,
        data: list[str] | None = None,
    ):
        super().__init__(data_ids)
        self.dim = dim
        if counts is not None:
            self.counts = counts
        elif data is not None:
            self.counts = get_counts(data)
        else:
            self.counts = np.zeros((dim,))

    @property
    def summary(self) -> np.ndarray:
        return self.counts

    def merge_point(self, data_id: int, data: list[str]) -> WordCluster:
        data_counts = get_counts(data)
        return WordCluster(self.data.union({data_id}), self.dim, counts=self.counts + data_counts)


class NgramCluster(Cluster):
    """Cluster subclass with summary statistics for an n-gram model"""

    def __init__(
        self,
        data_ids: frozenset[int],
        n: int = 2,
        counts: collections.Counter[tuple[str, ...]] | None = None,
        data: list[str] | None = None,
    ):
        super().__init__(data_ids)
        self.n = n
        if counts is not None:
            self.counts = counts
        elif data is not None:
            self.counts = get_ngram_counts(data, self.n)
        else:
            self.counts = collections.Counter()

    @property
    def summary(self) -> collections.Counter[tuple[str, ...]]:
        return self.counts

    def merge_point(self, data_id: int, data: list[str]) -> NgramCluster:
        new_counts = self.counts + get_ngram_counts(data, self.n)
        return NgramCluster(self.data.union({data_id}), self.n, counts=new_counts)


class BigramCluster(NgramCluster):
    def __init__(
        self,
        data_ids: frozenset[int],
        counts: collections.Counter[tuple[str, ...]] | None = None,
        data: list[str] | None = None,
    ):
        super().__init__(data_ids, 2, counts, data)


class TrigramCluster(NgramCluster):
    def __init__(
        self,
        data_ids: frozenset[int],
        counts: collections.Counter[tuple[str, ...]] | None = None,
        data: list[str] | None = None,
    ):
        super().__init__(data_ids, 3, counts, data)
