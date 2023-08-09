import jax.numpy as jnp
import optax
import jax

from functools import partial
from collections import namedtuple
from jax import grad, random, jit, tree_util, lax

from dag_gflownet.nets.gnn.gflownet import gflownet
from dag_gflownet.utils.gflownet import uniform_log_policy, sub_trajectory_balance_loss
from dag_gflownet.utils.jnp_utils import batch_random_choice


DAGGFlowNetParameters = namedtuple('DAGGFlowNetParameters', ['online', 'target'])
DAGGFlowNetState = namedtuple('DAGGFlowNetState', ['optimizer', 'key', 'steps'])
GFNState = namedtuple('GFNState', ['thetas', 'log_pi', 'log_p_theta', 'scores', 'diffs'])


class DAGGFlowNet:
    def __init__(self, model, network=None, delta=1., num_samples=1, update_target_every=0, dataset_size=1, batch_size=None):
        if network is None:
            network = gflownet(model.masked_posterior_parameters)

        self.model = model
        self.network = network
        self.delta = delta
        self.num_samples = num_samples
        self.update_target_every = update_target_every
        self.dataset_size = dataset_size
        self.batch_size = batch_size

        self._optimizer = None

    def get_state(self, params, key, graphs, masks, dataset, num_samples, norm1, norm2):
        # Compute the forward transition probability
        log_pi, dist = self.network.apply(params, graphs, masks, norm1)

        # Sample the parameters of the model
        thetas = self.model.sample(key, dist, num_samples)

        def diff_grads(theta, dist, dataset):
            score = self.model.log_joint(dist.mask, theta, dataset, norm2)

            log_p_theta = self.model.log_prob(theta, dist)
            log_p_theta = jnp.sum(log_p_theta)  # Sum over variables

            return (score - log_p_theta, score)

        v_diff_grads = jax.grad(diff_grads, has_aux=True)        
        v_diff_grads = jax.vmap(v_diff_grads, in_axes=(0, None, None))  # vmapping over thetas
        v_diff_grads = jax.vmap(v_diff_grads, in_axes=(0, 0, None))  # vmapping over batch
        diffs, scores = v_diff_grads(thetas, dist, dataset)

        # Compute the log-probabilities of the parameters. Use stop-gradient
        # to incorporate information about sub-trajectories of length 2.
        v_log_prob = jax.vmap(self.model.log_prob, in_axes=(0, None))  # vmapping over thetas
        v_log_prob = jax.vmap(v_log_prob, in_axes=(0, 0))  # vmapping over batch
        log_p_theta = v_log_prob(lax.stop_gradient(thetas), dist)

        return GFNState(
            thetas=thetas,
            log_pi=log_pi,
            log_p_theta=jnp.sum(log_p_theta, axis=2),  # Sum the log-probabilities over the variables
            scores=scores,
            diffs=jnp.sum(diffs ** 2, axis=(1, 2, 3)) / (
                self.num_samples * jnp.sum(dist.mask, axis=(1, 2))),
        )

    def loss(self, params, target_params, key, samples, dataset, normalization):
        @partial(jax.vmap, in_axes=0)
        def _loss(state_t, state_tp1, actions, num_edges):
            # Compute the delta-scores. Use stop-gradient to incorporate
            # information about sub-trajectories of length 2. This comes
            # from the derivative of log P_F(theta | G) wrt. theta is the
            # same as the derivative of log R(G, theta) wrt. theta.
            delta_scores = state_tp1.scores[:, None] - state_t.scores
            delta_scores = lax.stop_gradient(delta_scores)

            # Compute the sub-trajectory balance loss
            loss, logs = sub_trajectory_balance_loss(
                state_t.log_pi, state_tp1.log_pi,
                state_t.log_p_theta, state_tp1.log_p_theta,
                actions,
                delta_scores,
                num_edges,
                normalization=self.dataset_size,
                delta=self.delta
            )

            # Add penalty for sub-trajectories of length 2 (in differential form)
            loss = loss + 0.5 * (state_t.diffs + state_tp1.diffs)

            return (loss, logs)

        subkey1, subkey2, subkey3 = random.split(key, 3)

        # Sample a batch of data
        if self.batch_size is not None:
            indices = jax.random.choice(subkey1, self.dataset_size,
                shape=(self.batch_size,), replace=False)
            batch = jax.tree_util.tree_map(lambda x: x[indices], dataset)
        else:
            batch = dataset
        
        norm = (normalization / self.batch_size
            if self.batch_size is not None else jnp.array(1.))

        # Compute the states
        state_t = self.get_state(params, subkey2,
            samples['graph'], samples['mask'], batch, self.num_samples, normalization, norm)
        if self.update_target_every == 0:
            target_params = params
        state_tp1 = self.get_state(target_params, subkey3,
            samples['next_graph'], samples['next_mask'], batch, self.num_samples, normalization, norm)

        outputs = _loss(state_t, state_tp1,
            samples['actions'], samples['num_edges'])
        loss, logs = tree_util.tree_map(partial(jnp.mean, axis=0), outputs)
        logs.update({
            'post_theta/log_prob': state_t.log_p_theta,
            'error': outputs[1]['error'],  # Leave "error" unchanged
        })
        return (loss, logs)

    @partial(jit, static_argnums=(0,))
    def act(self, params, key, observations, epsilon, normalization):
        masks = observations['mask'].astype(jnp.float32)
        graphs = observations['graph']
        batch_size = masks.shape[0]
        key, subkey1, subkey2 = random.split(key, 3)

        # Get the GFlowNet policy
        log_pi, _ = self.network.apply(params, graphs, masks, normalization)

        # Get uniform policy
        log_uniform = uniform_log_policy(masks)

        # Mixture of GFlowNet policy and uniform policy
        is_exploration = random.bernoulli(
            subkey1, p=1. - epsilon, shape=(batch_size, 1))
        log_pi = jnp.where(is_exploration, log_uniform, log_pi)

        # Sample actions
        actions = batch_random_choice(subkey2, jnp.exp(log_pi), masks)

        logs = {
            'is_exploration': is_exploration.astype(jnp.int32),
        }
        return (actions, key, logs)

    @partial(jit, static_argnums=(0, 5))
    def act_and_params(self, params, key, observations, dataset, num_samples):
        masks = observations['mask'].astype(jnp.float32)
        graphs = observations['graph']
        key, subkey1, subkey2 = random.split(key, 3)

        normalization = jnp.array(dataset.data.shape[0])
        norm = jnp.array(1.)
        state = self.get_state(params, subkey1, graphs, masks, dataset, num_samples, normalization, norm)

        # Sample actions
        actions = batch_random_choice(subkey2, jnp.exp(state.log_pi), masks)

        logs = {}
        return (actions, key, state, logs)

    @partial(jit, static_argnums=(0,))
    def step(self, params, state, samples, dataset, normalization):
        key, subkey = random.split(state.key)
        grads, logs = grad(self.loss, has_aux=True)(params.online, params.target, subkey, samples, dataset, normalization)

        # Update the online params
        updates, opt_state = self.optimizer.update(grads, state.optimizer, params.online)
        state = DAGGFlowNetState(optimizer=opt_state, key=key, steps=state.steps + 1)
        online_params = optax.apply_updates(params.online, updates)
        if self.update_target_every > 0:
            target_params = optax.periodic_update(
                online_params,
                params.target,
                state.steps,
                self.update_target_every
            )
        else:
            target_params = params.target
        params = DAGGFlowNetParameters(online=online_params, target=target_params)

        return (params, state, logs)

    def init(self, key, optimizer, graph, mask):
        key, subkey = random.split(key)
        # Set the optimizer
        self._optimizer = optax.chain(optimizer, optax.zero_nans())
        online_params = self.network.init(subkey, graph, mask, jnp.array(1.))
        target_params = online_params if (self.update_target_every > 0) else None
        params = DAGGFlowNetParameters(online=online_params, target=target_params)
        state = DAGGFlowNetState(
            optimizer=self.optimizer.init(online_params),
            key=key,
            steps=jnp.array(0),
        )
        return (params, state)

    @property
    def optimizer(self):
        if self._optimizer is None:
            raise RuntimeError('The optimizer is not defined. To train the '
                               'GFlowNet, you must call `DAGGFlowNet.init` first.')
        return self._optimizer
