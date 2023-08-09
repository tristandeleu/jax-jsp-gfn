from dag_gflownet.models import LingaussDiagModel, LingaussFullModel, MLPGaussianModel
from dag_gflownet.models import priors as model_priors


def get_model_prior(name, metadata, args):
    if name == 'uniform':
        prior = model_priors.UniformPrior()
    elif name == 'erdos_renyi':
        prior = model_priors.EdgesPrior(
            num_variables=metadata['num_variables'],
            num_edges_per_node=metadata['num_edges_per_node']
        )
    else:
        raise NotImplementedError(f'Unknown prior: {name}')

    return prior


def get_model(name, prior_graph, dataset, obs_scale):
    num_variables = dataset.data.shape[1]

    if name == 'lingauss_diag':
        model = LingaussDiagModel(
            num_variables=num_variables,
            hidden_sizes=(128,),
            obs_scale=obs_scale,
            prior_graph=prior_graph
        )
    elif name == 'lingauss_full':
        model = LingaussFullModel(
            num_variables=num_variables,
            hidden_sizes=(128,),
            obs_scale=obs_scale,
            prior_graph=prior_graph
        )
    elif name == 'mlp_gauss':
        model = MLPGaussianModel(
            num_variables=num_variables,
            hidden_sizes=(128,),
            obs_scale=obs_scale,
            prior_graph=prior_graph,
            min_scale=0.
        )
    else:
        raise NotImplementedError(f'Unknown model: {name}')

    return model
