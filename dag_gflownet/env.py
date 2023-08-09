import numpy as np
import gym

from copy import deepcopy
from gym.spaces import Dict, Box, Discrete


class GFlowNetDAGEnv(gym.vector.VectorEnv):
    def __init__(
            self,
            num_envs,
            num_variables,
            max_parents=None,
        ):
        """GFlowNet environment for learning a distribution over DAGs.

        Parameters
        ----------
        num_envs : int
            Number of parallel environments, or equivalently the number of
            parallel trajectories to sample.
        
        num_variables : int
            Number of variables in the graphs.

        max_parents : int, optional
            Maximum number of parents for each node in the DAG. If None, then
            there is no constraint on the maximum number of parents.
        """
        self.num_variables = num_variables
        self._state = None
        self.max_parents = max_parents or self.num_variables

        shape = (self.num_variables, self.num_variables)
        max_edges = self.num_variables * (self.num_variables - 1) // 2
        observation_space = Dict({
            'adjacency': Box(low=0., high=1., shape=shape, dtype=np.int_),
            'mask': Box(low=0., high=1., shape=shape, dtype=np.int_),
            'num_edges': Discrete(max_edges),
            'score': Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float_),
            'order': Box(low=-1, high=max_edges, shape=shape, dtype=np.int_)
        })
        action_space = Discrete(self.num_variables ** 2 + 1)
        super().__init__(num_envs, observation_space, action_space)

    def reset(self):
        shape = (self.num_envs, self.num_variables, self.num_variables)
        closure_T = np.eye(self.num_variables, dtype=np.bool_)
        self._closure_T = np.tile(closure_T, (self.num_envs, 1, 1))
        self._state = {
            'adjacency': np.zeros(shape, dtype=np.int_),
            'mask': 1 - self._closure_T,
            'num_edges': np.zeros((self.num_envs,), dtype=np.int_),
            'score': np.zeros((self.num_envs,), dtype=np.float_),
            'order': np.full(shape, -1, dtype=np.int_)
        }
        return deepcopy(self._state)

    def step(self, actions):
        sources, targets = divmod(actions, self.num_variables)
        dones = (sources == self.num_variables)
        sources, targets = sources[~dones], targets[~dones]

        # Make sure that all the actions are valid
        if not np.all(self._state['mask'][~dones, sources, targets]):
            raise ValueError('Some actions are invalid: either the edge to be '
                             'added is already in the DAG, or adding this edge '
                             'would lead to a cycle.')

        # Update the adjacency matrices
        self._state['adjacency'][~dones, sources, targets] = 1
        self._state['adjacency'][dones] = 0

        # Update transitive closure of transpose
        source_rows = np.expand_dims(self._closure_T[~dones, sources, :], axis=1)
        target_cols = np.expand_dims(self._closure_T[~dones, :, targets], axis=2)
        self._closure_T[~dones] |= np.logical_and(source_rows, target_cols)  # Outer product
        self._closure_T[dones] = np.eye(self.num_variables, dtype=np.bool_)

        # Update the masks
        self._state['mask'] = 1 - (self._state['adjacency'] + self._closure_T)

        # Update the masks (maximum number of parents)
        num_parents = np.sum(self._state['adjacency'], axis=1, keepdims=True)
        self._state['mask'] *= (num_parents < self.max_parents)

        # Update the order
        self._state['order'][~dones, sources, targets] = self._state['num_edges'][~dones]
        self._state['order'][dones] = -1

        # Update the number of edges
        self._state['num_edges'] += 1
        self._state['num_edges'][dones] = 0

        delta_scores = np.zeros((self.num_envs,), dtype=np.float_)

        # Update the scores. The scores returned by the environments are scores
        # relative to the empty graph: score(G) - score(G_0).
        self._state['score'] += delta_scores
        self._state['score'][dones] = 0

        return (deepcopy(self._state), delta_scores, dones, {})

    def close_extras(self, **kwargs):
        pass
