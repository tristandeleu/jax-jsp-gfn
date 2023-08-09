import jax.numpy as jnp
import haiku as hk
import jraph
import math

from jax import lax, nn
from collections import namedtuple

from dag_gflownet.utils.gflownet import log_policy
from dag_gflownet.utils.jraph_utils import edge_features_to_dense


def gflownet(posterior_parameters_model):
    @hk.without_apply_rng
    @hk.transform
    def _gflownet(graphs, masks, normalization):
        batch_size, num_variables = masks.shape[:2]
        edge_masks = jnp.ones(graphs.edges.shape, dtype=jnp.float32)

        # Embedding of the nodes & edges
        node_embeddings = hk.Embed(num_variables, embed_dim=128)
        edge_embedding = hk.get_parameter('edge_embed', shape=(1, 128),
            init=hk.initializers.TruncatedNormal())

        graphs = graphs._replace(
            nodes=node_embeddings(graphs.nodes),
            edges=jnp.repeat(edge_embedding, graphs.edges.shape[0], axis=0),
            globals=jnp.zeros((graphs.n_node.shape[0], 1)),
        )

        # Define graph network updates
        @jraph.concatenated_args
        def update_node_fn(features):
            return hk.nets.MLP([128, 128], name='node')(features)

        @jraph.concatenated_args
        def update_edge_fn(features):
            return hk.nets.MLP([128, 128], name='edge')(features)

        @jraph.concatenated_args
        def update_global_fn(features):
            return hk.nets.MLP([128, 128], name='global')(features)

        graph_net = jraph.GraphNetwork(
            update_edge_fn=update_edge_fn,
            update_node_fn=update_node_fn,
            update_global_fn=update_global_fn,
        )
        features = graph_net(graphs)

        # Reshape the node features, and project into keys, queries & values
        node_features = features.nodes[:batch_size * num_variables]
        node_features = node_features.reshape(batch_size, num_variables, -1)
        node_features = hk.Linear(128 * 3, name='projection')(node_features)
        queries, keys, values = jnp.split(node_features, 3, axis=2)

        # Self-attention layer
        node_features = hk.MultiHeadAttention(
            num_heads=4,
            key_size=32,
            w_init_scale=2.
        )(queries, keys, values)

        # Replace the node & global features
        features = features._replace(
            nodes=hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True
            )(node_features),
            globals=hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True
            )(features.globals[:batch_size])
        )

        senders = hk.nets.MLP([128, 128], name='senders')(features.nodes)
        receivers = hk.nets.MLP([128, 128], name='receivers')(features.nodes)

        logits = lax.batch_matmul(senders, receivers.transpose(0, 2, 1))
        logits = logits.reshape(batch_size, -1)
        stop = hk.nets.MLP([128, 1], name='stop')(features.globals)

        with hk.experimental.name_scope('posterior_parameters'):
            adjacencies = edge_features_to_dense(features, edge_masks, num_variables)
            adjacencies = adjacencies[:batch_size]
            post_parameters = posterior_parameters_model(features, adjacencies)

        return (log_policy(logits * normalization, stop * normalization, masks), post_parameters)

    return _gflownet
