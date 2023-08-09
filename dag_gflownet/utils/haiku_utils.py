import jax.numpy as jnp
import haiku as hk
import jax

ACTIVATIONS = {
    'sigmoid': jax.nn.sigmoid,
    'tanh': jax.nn.tanh,
    'relu': jax.nn.relu,
    'leakyrelu': jax.nn.leaky_relu
}

def create_mlp(hidden_sizes, activation='relu', **kwargs):
    @hk.without_apply_rng
    @hk.transform
    def _mlp(inputs):
        outputs = hk.nets.MLP(
            hidden_sizes,
            activation=ACTIVATIONS[activation],
            with_bias=True,
            activate_final=False,
            name='mlp',
            **kwargs
        )(inputs)
        if outputs.shape[-1] == 1:
            outputs = jnp.squeeze(outputs, axis=-1)
        return outputs
    return _mlp

def get_first_weights(params):
    params = hk.data_structures.filter(
        lambda module_name, name, _: (module_name == 'mlp/~/linear_0') and (name == 'w'),
        params
    )
    return params['mlp/~/linear_0']['w']
