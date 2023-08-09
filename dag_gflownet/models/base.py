import jax.numpy as jnp
import jax
import math

from abc import ABC, abstractmethod
from collections import namedtuple
from functools import partial

from dag_gflownet.models.priors import UniformPrior

_LOG_2PI = math.log(2 * math.pi)

NormalParams = namedtuple('NormalParams', ['loc', 'scale'])
MaskedDistribution = namedtuple('MaskedDistribution', ['dist', 'mask'])

class BaseModel(ABC):
    def __init__(self, num_variables, prior_graph=UniformPrior()):
        self.num_variables = num_variables
        self.prior_graph = prior_graph

    @abstractmethod
    def posterior_parameters(self, features, adjacencies):
        r"""Compute the parameters of the posterior over parameters.

        The posterior approximation over parameters is P_F(\theta_i | G, s_f),
        i.e. this is the forward transition probability of generating \theta
        from G, given that we stopped adding edges (hence conditioning on s_f).
        Typically, this distribution will be a Normal distribution; in that case
        this function returns the means and covariance matrices.

        This function returns the parameters for each \theta_i, meaning that
        typically `axis=1` of the returned values will correspond to each
        variable `i` (with `axis=0` being the batch dimension).

        Parameters
        ----------
        features : jraph.GraphsTuple
            The features for each graph G in the batch. These features consist
            of the features returned by the "GNN + self-attention" network. Note
            that the padding has been removed for `features.nodes` (which have
            shape `(batch_size, num_variables, feature_size_nodes)`) and for
            `features.globals` (which have shape `(batch_size, feature_size_globals)`),
            but not for `features.edges`.

        adjacencies : jnp.ndarray, (batch_size, num_variables, num_variables)
            The adjacency matrices for each graph G in the batch.

        Returns
        -------
        parameters : Any (e.g., `NormalParams`)
            The parameters of the posterior approximation, for each graph G in
            the batch. For example, if the posterior approximation is a Normal
            distribution with full covariance (see `LingaussFullModel`), then
            it returns means of shape `(batch_size, num_vars, num_vars)` (one
            vector of size `(num_vars,)` for each \theta_i, of which there are
            `num_vars` for each graph G in the batch) and covariance matrices
            of shape `(batch_size, num_vars, num_vars, num_vars)`.
        """
        pass

    def mask_parameters(self, adjacencies):
        pass

    def masked_posterior_parameters(self, features, adjacencies):
        return MaskedDistribution(
            dist=self.posterior_parameters(features, adjacencies),
            mask=self.mask_parameters(adjacencies)
        )

    @abstractmethod
    def sample(self, key, masked_dists, num_samples):
        r"""Sample parameters \theta from a masked distribution.

        This function samples multiple parameters \theta from the masked
        distributions giving the parameters of P_F(\theta_i | G, s_f).

        Parameters
        ----------
        key : jax.random.PRNGKey
            The random key to sample the parameters \theta

        masked_dists : `MaskedDistribution` instance
            The parameters of P_F(\theta_i | G, s_f). This distribution must
            be masked based on the adjacency matrix of G. Note that `axis=1`
            in the objects of `masked_dists` correspond to the dimension of
            the variables in each G of the batch; concretely, this means that
            `masked_dist.mask` is of shape `(batch_size, num_vars, num_vars)`,
            where the second dimension corresponds to the variables in each
            graph (they are adjacency matrices transposed).

        num_samples : int
            The number of samples \theta for each graph G.

        Returns
        -------
        samples : Any (e.g., jnp.ndarray, (batch_size, num_samples, num_vars, num_vars))
            The samples of P_F(\theta_i | G, s_f). They can be any object,
            depending on the observation model. For example, for a linear
            Gaussian model, it will be a jnp.ndarray instance of shape
            `(batch_size, num_samples, num_vars, num_vars)`. Note that `axis=2`
            in any object returned here correspond to \theta_i for each graph G.
        """
        pass

    @abstractmethod
    def sample_obs_and_log_probs(self, key, data, theta, parents):
        r"""Sample an observation from P(X_i | X_rest, \theta_i, Pa(X_i)), and
        return the corresponding log-probabilities.

        Parameters
        ----------
        key : jax.random.PRNGKey
            The random key to sample the observations X_i.

        data : jnp.ndarray, (num_observations, num_variables)
            The data for conditioning on the variables X_rest other than X_i.

        theta : Any (e.g., jnp.ndarray, (num_variables,))
            The parameters \theta_i for a single graph G, and a single
            conditional probability distribution for X_i. For example, if the
            observation model is linear-Gaussian, then `theta` is a jnp.ndarray
            of shape `(num_variables,)`.

        parents : jnp.ndarray, (num_variables,)
            The binary-valued vector corresponding to the parents Pa(X_i) of
            the variable X_i in the graph G. This corresponds to the column
            "i" of the adjacency matrix of G.

        Returns
        -------
        sample : jnp.ndarray, (num_observations,)
            The samples of X_i, one for each observation in "data".

        log_probs : jnp.ndarray, (num_observations,)
            The value of \log P(X_i | X_rest, \theta_i, Pa(X_i)) for each sample.
        """
        pass

    @abstractmethod
    def log_likelihood(self, dataset, thetas, masks):
        r"""Computes the log-likelihood \log P(D | \theta, G).

        Parameters
        ----------
        dataset : Dataset, (num_observations, num_variables)
            The dataset object containing the `data` and `interventions` to
            evaluate the log-likelihood over. Both of the arrays in this object
            have size (num_observations, num_variables)

        thetas : Any (e.g., jnp.ndarray, (num_vars, num_vars))
            The parameters \thetas for a single graph G. For example, if
            the observation model is linear-Gaussian, then `thetas` is a
            jnp.ndarray of shape `(num_vars, num_vars)`, where the first
            dimension corresponds to variables.

        masks : jnp.ndarray, (num_variables, num_variables)
            The *transpose* of the adjacency matrix of the graph G. Note
            that the adjacency matrix is transposed, meaning that the
            first dimension of `masks` corresponds to variables.

        Returns
        -------
        log_likelihood : jnp.ndarray, ()
            The log-likelihood \log P(D | \theta, G).
        """
        pass

    @abstractmethod
    def log_prior_theta(self, thetas, masks):
        r"""Computes the log-prior \log P(\theta | G).

        Parameters
        ----------
        thetas : Any (e.g., jnp.ndarray, (num_vars, num_vars))
            The parameters \thetas for a single graph G. For example, if
            the observation model is linear-Gaussian, then `thetas` is a
            jnp.ndarray of shape `(num_vars, num_vars)`, where the first
            dimension corresponds to variables.

        masks : jnp.ndarray, (num_variables, num_variables)
            The *transpose* of the adjacency matrix of the graph G. Note
            that the adjacency matrix is transposed, meaning that the
            first dimension of `masks` corresponds to variables.

        Returns
        -------
        log_prior : jnp.ndarray, ()
            The log-prior over parameters \log P(\theta | G).       
        """
        pass

    def log_joint(self, masks, thetas, dataset, normalization):
        return (self.log_likelihood(dataset, thetas, masks) * normalization
            + self.log_prior_theta(thetas, masks)
            + self.prior_graph(masks))


class GaussianModel(BaseModel):
    def __init__(
            self,
            num_variables,
            prior=NormalParams(loc=0., scale=1.),
            obs_scale=math.sqrt(0.1),
            prior_graph=UniformPrior()
        ):
        super().__init__(num_variables, prior_graph=prior_graph)
        self.prior = prior
        self.obs_scale = obs_scale

    @abstractmethod
    def model(self, theta, data, mask):
        pass

    @abstractmethod
    def log_prob(self, theta, masked_dist):
        pass

    def sample_obs_and_log_probs(self, key, data, theta, parents):
        mean = self.model(theta, data, parents)
        epsilon = jax.random.normal(key, mean.shape)
        samples = mean + self.obs_scale * epsilon
        log_probs = -0.5 * (epsilon ** 2
            + _LOG_2PI + 2 * math.log(self.obs_scale))
        return (samples, log_probs)

    def log_likelihood(self, dataset, thetas, masks):
        @partial(jax.vmap, in_axes=(0, 1, 0), out_axes=1)
        def _log_likelihood(theta, y, mask):
            mean = self.model(theta, dataset.data, mask)
            diff = (y - mean) / self.obs_scale
            const = _LOG_2PI + 2 * math.log(self.obs_scale)
            return -0.5 * (diff ** 2 + const)

        log_likelihood = _log_likelihood(thetas, dataset.data, masks)
        return jnp.sum(log_likelihood * (1. - dataset.interventions))

    def log_prior_theta(self, thetas, masks):
        @partial(jax.vmap, in_axes=(0, 0))
        def _log_prior(theta, mask):
            diff = (theta - self.prior.loc) / self.prior.scale
            return -0.5 * jnp.sum(mask * (diff ** 2
                + _LOG_2PI + 2 * jnp.log(self.prior.scale)))
        return jnp.sum(_log_prior(thetas, masks))


class LinearGaussianModel(GaussianModel):
    def model(self, theta, data, mask):
        return jnp.dot(data, theta * mask)

    def mask_parameters(self, adjacencies):
        # Leading dim is the variables
        return jnp.swapaxes(adjacencies, -1, -2)
