import jax.numpy as jnp
import haiku as hk
import jax
import math

from functools import partial

from dag_gflownet.models.base import LinearGaussianModel, NormalParams, _LOG_2PI
from dag_gflownet.models.priors import UniformPrior


class LingaussDiagModel(LinearGaussianModel):
    def __init__(
            self,
            num_variables,
            hidden_sizes,
            prior=NormalParams(loc=0., scale=1.),
            obs_scale=math.sqrt(0.1),
            prior_graph=UniformPrior()
        ):
        super().__init__(
            num_variables,
            prior=prior,
            obs_scale=obs_scale,
            prior_graph=prior_graph
        )
        self.hidden_sizes = tuple(hidden_sizes)

    def posterior_parameters(self, features, adjacencies):
        global_features = jnp.repeat(features.globals[:, None], self.num_variables, axis=1)
        inputs = jnp.concatenate((features.nodes, global_features), axis=2)
        params = hk.nets.MLP(
            self.hidden_sizes,
            activate_final=True,
            name='edges'
        )(inputs)
        params = hk.Linear(
            2 * self.num_variables,
            w_init=hk.initializers.Constant(0.),
            b_init=hk.initializers.Constant(0.)
        )(params)
        params = params.reshape(-1, self.num_variables, 2 * self.num_variables)

        # Split the parameters into loc and scale
        locs, scales = jnp.split(params, 2, axis=-1)

        # Mask the parameters, based on the adjacencies
        masks = jnp.swapaxes(adjacencies, -1, -2)
        return NormalParams(
            loc=locs * masks,
            scale=jax.nn.softplus(scales * masks)
        )

    def log_prob(self, theta, masked_dist):
        diff = (theta - masked_dist.dist.loc) / masked_dist.dist.scale
        return -0.5 * jnp.sum(masked_dist.mask * (diff ** 2 + _LOG_2PI
            + 2 * jnp.log(masked_dist.dist.scale)), axis=-1)

    def sample(self, key, masked_dists, num_samples):
        shape = masked_dists.mask.shape
        epsilons = jax.random.normal(key, shape=(num_samples,) + shape)

        @partial(jax.vmap, in_axes=(1, 0))
        def _sample(epsilon, masked_dist):
            # Sample from a Normal distribution
            samples = masked_dist.dist.loc + epsilon * masked_dist.dist.scale
            return samples * masked_dist.mask  # Mask the samples

        return _sample(epsilons, masked_dists)
