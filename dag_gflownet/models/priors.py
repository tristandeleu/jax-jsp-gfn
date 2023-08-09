import jax.numpy as jnp
import jax
import math

from abc import ABC, abstractmethod
from functools import partial


class BasePrior(ABC):
    @abstractmethod
    def __call__(self, masks):
        r"""Computes the log-prior \log P(G).
        
        Parameters
        ----------
        masks : jnp.ndarray, (num_variables, num_variables)
            The *transpose* of the adjacency matrix of the graph G. Note
            that the adjacency matrix is transposed, meaning that the
            first dimension of `masks` corresponds to variables.

        Returns
        -------
        log_prior : jnp.ndarray, ()
            The (unnormalized) log-prior over graphs \log P(G)
        """
        pass


class UniformPrior(BasePrior):
    def __call__(self, masks):
        return jnp.array(0.)


class ErdosRenyiPrior(BasePrior):
    def __init__(self, num_variables, num_edges_per_node):
        self.num_variables = num_variables
        self.num_edges_per_node = num_edges_per_node

        self._p = 2. * num_edges_per_node / (num_variables - 1)

    def __call__(self, masks):
        num_parents = jnp.sum(masks, axis=0)
        log_priors = num_parents * math.log(self._p) \
            + (self.num_variables - num_parents - 1) * math.log1p(-self._p)
        return jnp.sum(log_priors)


class EdgesPrior(BasePrior):
    def __init__(self, num_variables, num_edges_per_node):
        self.num_variables = num_variables
        self.num_edges_per_node = num_edges_per_node

        self._p = 2. * num_edges_per_node / (num_variables - 1)
        self._log_beta = math.log(self._p) - math.log1p(-self._p)

    def __call__(self, masks):
        num_parents = jnp.sum(masks, axis=0)
        return jnp.sum(num_parents * self._log_beta)
