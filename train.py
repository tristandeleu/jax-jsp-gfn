import jax.numpy as jnp
import numpy as np
import optax
import jax
import wandb

from pathlib import Path
from tqdm import trange
from numpy.random import default_rng

from dag_gflownet.env import GFlowNetDAGEnv
from dag_gflownet.gflownet import DAGGFlowNet
from dag_gflownet.utils.replay_buffer import ReplayBuffer
from dag_gflownet.utils.factories import get_model, get_model_prior
from dag_gflownet.utils.gflownet import posterior_estimate
from dag_gflownet.utils.jraph_utils import to_graphs_tuple
from dag_gflownet.utils.data import load_artifact_continuous


def main(args):
    api = wandb.Api()

    rng = default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)

    # Get the artifact from wandb
    artifact = api.artifact(args.artifact)
    artifact_dir = Path(artifact.download()) / f'{args.seed:02d}'

    if args.seed not in artifact.metadata['seeds']:
        raise ValueError(f'The seed `{args.seed}` is not in the list of seeds '
            f'for artifact `{args.artifact}`: {artifact.metadata["seeds"]}')

    # Load data & graph
    train, valid, graph = load_artifact_continuous(artifact_dir)
    train_jnp = jax.tree_util.tree_map(jnp.asarray, train)

    # Create the environment
    env = GFlowNetDAGEnv(
        num_envs=args.num_envs,
        num_variables=train.data.shape[1],
        max_parents=args.max_parents
    )

    # Create the replay buffer
    replay = ReplayBuffer(
        args.replay_capacity,
        num_variables=env.num_variables,
    )

    # Create the model
    if 'obs_noise' not in artifact.metadata['cpd_kwargs']:
        if args.obs_scale is None:
            raise ValueError('The obs_noise is not defined in the artifact, '
                'therefore is must be set as a command argument `--obs_scale`.')
        obs_scale = args.obs_scale
    else:
        obs_scale = artifact.metadata['cpd_kwargs']['obs_noise']
    prior_graph = get_model_prior(args.prior, artifact.metadata, args)
    model = get_model(args.model, prior_graph, train_jnp, obs_scale)

    # Create the GFlowNet & initialize parameters
    gflownet = DAGGFlowNet(
        model=model,
        delta=args.delta,
        num_samples=args.params_num_samples,
        update_target_every=args.update_target_every,
        dataset_size=train_jnp.data.shape[0],
        batch_size=args.batch_size_data,
    )

    optimizer = optax.adam(args.lr)
    params, state = gflownet.init(
        subkey,
        optimizer,
        replay.dummy['graph'],
        replay.dummy['mask']
    )

    exploration_schedule = jax.jit(optax.linear_schedule(
        init_value=jnp.array(0.),
        end_value=jnp.array(1. - args.min_exploration),
        transition_steps=args.num_iterations // 2,
        transition_begin=args.prefill,
    ))

    # Training loop
    indices = None
    observations = env.reset()
    normalization = jnp.array(train_jnp.data.shape[0])

    with trange(args.prefill + args.num_iterations, desc='Training') as pbar:
        for iteration in pbar:
            # Sample actions, execute them, and save transitions in the replay buffer
            epsilon = exploration_schedule(iteration)
            observations['graph'] = to_graphs_tuple(observations['adjacency'])
            actions, key, logs = gflownet.act(params.online, key, observations, epsilon, normalization)
            next_observations, delta_scores, dones, _ = env.step(np.asarray(actions))
            indices = replay.add(
                observations,
                actions,
                logs['is_exploration'],
                next_observations,
                delta_scores,
                dones,
                prev_indices=indices
            )
            observations = next_observations

            if iteration >= args.prefill:
                # Update the parameters of the GFlowNet
                samples = replay.sample(batch_size=args.batch_size, rng=rng)
                params, state, logs = gflownet.step(params, state, samples, train_jnp, normalization)

                pbar.set_postfix(loss=f"{logs['loss']:.2f}")

    # Evaluate the posterior estimate
    posterior, logs = posterior_estimate(
        gflownet,
        params.online,
        env,
        key,
        train_jnp,
        num_samples=args.num_samples_posterior,
        desc='Sampling from posterior'
    )


if __name__ == '__main__':
    from argparse import ArgumentParser
    import json
    import math

    parser = ArgumentParser(description='JSP-GFN for Strucure Learning.')

    # Environment
    environment = parser.add_argument_group('Environment')
    environment.add_argument('--num_envs', type=int, default=8,
        help='Number of parallel environments (default: %(default)s)')
    environment.add_argument('--prior', type=str, default='uniform',
        choices=['uniform', 'erdos_renyi', 'edge', 'fair'],
        help='Prior over graphs (default: %(default)s)')
    environment.add_argument('--max_parents', type=int, default=None,
        help='Maximum number of parents')

    # Data
    data = parser.add_argument_group('Data')
    data.add_argument('--artifact', type=str, required=True,
        help='Path to the artifact for input data in Wandb')
    data.add_argument('--obs_scale', type=float, default=math.sqrt(0.1),
        help='Scale of the observation noise (default: %(default)s)')

    # Model
    model = parser.add_argument_group('Model')
    model.add_argument('--model', type=str, default='lingauss_diag',
        choices=['lingauss_diag', 'lingauss_full', 'mlp_gauss'],
        help='Type of model (default: %(default)s)')

    # Optimization
    optimization = parser.add_argument_group('Optimization')
    optimization.add_argument('--lr', type=float, default=1e-5,
        help='Learning rate (default: %(default)s)')
    optimization.add_argument('--delta', type=float, default=1.,
        help='Value of delta for Huber loss (default: %(default)s)')
    optimization.add_argument('--batch_size', type=int, default=32,
        help='Batch size (default: %(default)s)')
    optimization.add_argument('--num_iterations', type=int, default=100_000,
        help='Number of iterations (default: %(default)s)')
    optimization.add_argument('--params_num_samples', type=int, default=1,
        help='Number of samples of model parameters to compute the loss (default: %(default)s)')
    optimization.add_argument('--update_target_every', type=int, default=0,
        help='Frequency of update for the target network (0 = no target network)')
    optimization.add_argument('--batch_size_data', type=int, default=None,
        help='Batch size for the data (default: %(default)s)')

    # Replay buffer
    replay = parser.add_argument_group('Replay Buffer')
    replay.add_argument('--replay_capacity', type=int, default=100_000,
        help='Capacity of the replay buffer (default: %(default)s)')
    replay.add_argument('--prefill', type=int, default=1000,
        help='Number of iterations with a random policy to prefill '
             'the replay buffer (default: %(default)s)')
    
    # Exploration
    exploration = parser.add_argument_group('Exploration')
    exploration.add_argument('--min_exploration', type=float, default=0.1,
        help='Minimum value of epsilon-exploration (default: %(default)s)')
    
    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--num_samples_posterior', type=int, default=1000,
        help='Number of samples for the posterior estimate (default: %(default)s)')
    misc.add_argument('--seed', type=int, default=0,
        help='Random seed (default: %(default)s)')

    args = parser.parse_args()

    main(args)
