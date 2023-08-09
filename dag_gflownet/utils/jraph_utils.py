import numpy as np
import jax.numpy as jnp
import jraph


def to_graphs_tuple(adjacencies, pad=True):
    num_graphs, num_variables = adjacencies.shape[:2]
    n_node = np.full((num_graphs,), num_variables, dtype=np.int_)

    counts, senders, receivers = np.nonzero(adjacencies)
    n_edge = np.bincount(counts, minlength=num_graphs)

    # Node features: node indices
    nodes = np.tile(np.arange(num_variables), num_graphs)
    edges = np.ones_like(senders)

    graphs_tuple =  jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders + counts * num_variables,
        receivers=receivers + counts * num_variables,
        globals=None,
        n_node=n_node,
        n_edge=n_edge,
    )
    if pad:
        graphs_tuple = pad_graph_to_nearest_power_of_two(graphs_tuple)
    return graphs_tuple


def _nearest_bigger_power_of_two(x):
    y = 2
    while y < x:
        y *= 2
    return y


def pad_graph_to_nearest_power_of_two(graphs_tuple):
    # Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py
    # Add 1 since we need at least one padding node for pad_with_graphs.
    pad_nodes_to = _nearest_bigger_power_of_two(np.sum(graphs_tuple.n_node)) + 1
    pad_edges_to = _nearest_bigger_power_of_two(np.sum(graphs_tuple.n_edge))

    # Add 1 since we need at least one padding graph for pad_with_graphs.
    # We do not pad to nearest power of two because the batch size is fixed.
    pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
    return jraph.pad_with_graphs(
        graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to)


def edge_features_to_dense(graphs, features, num_variables):
    # Get the batch indices
    batch_indices = jnp.arange(graphs.n_edge.shape[0])
    batch_indices = jnp.repeat(batch_indices, graphs.n_edge,
        axis=0, total_repeat_length=graphs.edges.shape[0])

    # Remove the offset to senders & receivers
    offset = batch_indices * num_variables
    senders = graphs.senders - offset
    receivers = graphs.receivers - offset

    # Transform the features into a dense array
    shape = (graphs.n_node.shape[0], num_variables, num_variables) + features.shape[1:]
    dense_array = jnp.zeros(shape, dtype=features.dtype)
    return dense_array.at[batch_indices, senders, receivers].add(features)
