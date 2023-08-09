import numpy as np
import jax.numpy as jnp
import optax

from tqdm.auto import trange
from jax import nn

from dag_gflownet.utils.jraph_utils import to_graphs_tuple


MASKED_VALUE = -jnp.inf


def mask_logits(logits, masks):
    return jnp.where(masks, logits, MASKED_VALUE)


def sub_trajectory_balance_loss(
        log_pi_t,
        log_pi_tp1,
        log_p_theta_t,
        log_p_theta_tp1,
        actions,
        delta_scores,
        num_edges,
        normalization=1.,
        delta=1.
    ):
    r"""Sub-Trajectory Balance loss.

    This function computes the sub-trajectory balance loss, over
    sub-trajectories of length 3. This loss function is given by:

    $$ L(\theta; s_{t}, s_{t+1}) = \left[\log\frac{
        R(G_{t+1}, \theta_{t+1})P_{B}(G_{t} \mid G_{t+1})P_{\phi}(\theta_{t} \mid s_{t})}{
        R(G_{t}, \theta_{t})P_{\phi}(G_{t+1} \mid G_{t})P_{\phi}(\theta_{t+1} \mid s_{t+1})
    }\right]^{2} $$

    In practice, to avoid gradient explosion, we use the Huber loss instead
    of the L2-loss (the L2-loss can be emulated with a large value of delta).

    Parameters
    ----------
    log_pi_t : jnp.DeviceArray
        The log-probabilities $\log P_{\theta}(s' \mid s_{t})$, for all the
        next states $s'$, including the terminal state $s_{f}$. This array
        has size `(N ** 2 + 1,)`, where `N` is the number of variables in a graph.

    log_pi_tp1 : jnp.DeviceArray
        The log-probabilities $\log P_{\theta}(s' \mid s_{t+1})$, for all the
        next states $s'$, including the terminal state $s_{f}$. This array
        has size `(N ** 2 + 1,)`, where `N` is the number of variables in a graph.

    log_p_theta_t : jnp.DeviceArray
        The log-probabilities $\log P_{\phi}(\theta_{t} \mid s_{t})$ for the
        sampled parameters of the Bayesian Network whose graph is given by
        $s_{t}$. This array has size `(num_samples,)`.

    log_p_theta_tp1 : jnp.DeviceArray
        The log-probabilities $\log P_{\phi}(\theta_{t+1} \mid s_{t+1})$ for
        the sampled parameters of the Bayesian Network whose graph is given by
        $s_{t+1}$. This array has size `(num_samples,)`.

    actions : jnp.DeviceArray
        The actions taken to go from state $s_{t}$ to state $s_{t+1}$. This
        array has size `(1,)`.

    delta_scores : jnp.DeviceArray
        The delta-scores between state $s_{t}$ and state $s_{t+1}$, given by
        $\log R(s_{t+1}) - \log R(s_{t})$. This array has size `(num_samples, num_samples)`.

    num_edges : jnp.DeviceArray
        The number of edges in $s_{t}$. This array has size `(1,)`.

    normalization : float (default: 1.)
        The normalization constant for the error term.

    delta : float (default: 1.)
        The value of delta for the Huber loss.

    Returns
    -------
    loss : jnp.DeviceArray
        The sub-trajectory balance loss averaged over a batch of samples.

    logs : dict
        Additional information for logging purposes.
    """
    # Compute the forward log-probabilities
    log_pF = log_pi_t[actions[0]]

    # Compute the backward log-probabilities
    log_pB = -jnp.log1p(num_edges[0])

    # Compute the forward log-probability of terminating from s_{t}
    # with parameters \theta_{t}
    log_psf_t = log_pi_t[-1] + log_p_theta_t

    # Compute the forward log-probability of terminating from s_{t+1}
    # with parameters \theta_{t+1}
    log_psf_tp1 = log_pi_tp1[-1] + log_p_theta_tp1[:, None]

    error = delta_scores + log_pB + log_psf_t - log_pF - log_psf_tp1
    normalized_error = error / normalization
    loss = jnp.mean(optax.huber_loss(normalized_error, delta=delta))

    logs = {
        'error': error,
        'loss': loss,
    }
    return (loss, logs)


def log_policy(logits, stop, masks):
    masks = masks.reshape(logits.shape)
    masked_logits = mask_logits(logits, masks)
    can_continue = jnp.any(masks, axis=-1, keepdims=True)

    logp_continue = (nn.log_sigmoid(-stop)
        + nn.log_softmax(masked_logits, axis=-1))
    logp_stop = nn.log_sigmoid(stop)

    # In case there is no valid action other than stop
    logp_continue = jnp.where(can_continue, logp_continue, MASKED_VALUE)
    logp_stop = logp_stop * can_continue

    return jnp.concatenate((logp_continue, logp_stop), axis=-1)


def uniform_log_policy(masks):
    masks = masks.reshape(masks.shape[0], -1)
    num_edges = jnp.sum(masks, axis=-1, keepdims=True)

    logp_stop = -jnp.log1p(num_edges)
    logp_continue = mask_logits(logp_stop, masks)

    return jnp.concatenate((logp_continue, logp_stop), axis=-1)


def posterior_estimate(
        gflownet,
        params,
        env,
        key,
        dataset,
        num_samples=1000,
        num_samples_thetas=1,
        verbose=True,
        **kwargs
    ):
    """Get the posterior estimate of DAG-GFlowNet as a collection of graphs
    sampled from the GFlowNet.

    Parameters
    ----------
    gflownet : `DAGGFlowNet` instance
        Instance of a DAG-GFlowNet.

    params : dict
        Parameters of the neural network for DAG-GFlowNet. This must be a dict
        that can be accepted by the Haiku model in the `DAGGFlowNet` instance.

    env : `GFlowNetDAGEnv` instance
        Instance of the environment.

    key : jax.random.PRNGKey
        Random key for sampling from DAG-GFlowNet.

    dataset :
        The training dataset.

    num_samples : int (default: 1000)
        The number of samples in the posterior approximation.

    num_samples_thetas : int (default: 1)
        The number of samples of parameters for each graph.

    verbose : bool
        If True, display a progress bar for the sampling process.

    Returns
    -------
    posterior : np.ndarray instance
        The posterior approximation, given as a collection of adjacency matrices
        from graphs sampled with the posterior approximation. This array has
        size `(B, N, N)`, where `B` is the number of sample graphs in the
        posterior approximation, and `N` is the number of variables in a graph.

    logs : dict
        Additional information for logging purposes.
    """
    orders, thetas, scores = [], [], []
    observations = env.reset()
    with trange(num_samples, disable=(not verbose), **kwargs) as pbar:
        while len(orders) < num_samples:
            order = observations['order']
            observations['graph'] = to_graphs_tuple(observations['adjacency'])
            actions, key, state, _ = gflownet.act_and_params(
                params,
                key,
                observations,
                dataset,
                num_samples=num_samples_thetas
            )
            observations, _, dones, _ = env.step(np.asarray(actions))

            orders.extend([order[i] for i, done in enumerate(dones) if done])
            thetas.extend([state.thetas[i] for i, done in enumerate(dones) if done])
            scores.extend([state.scores[i] for i, done in enumerate(dones) if done])

            pbar.update(min(num_samples - pbar.n, np.sum(dones).item()))

    orders = np.stack(orders[:num_samples], axis=0)
    thetas = np.stack(thetas[:num_samples], axis=0)
    scores = np.stack(scores[:num_samples], axis=0)

    logs = {
        'orders': orders,
        'thetas': thetas,
        'scores': scores,
    }
    return ((orders >= 0).astype(np.int_), logs)
