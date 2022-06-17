import torch

import torch.nn as nn

from optlearn.nets import masking
from optlearn.nets import evrp_encoder_decoder


class vrpEncoderDecoderSolver(nn.Module, masking.evrpMasker):
    def __init__(self, model_parameters):
        super(vrpEncoderDecoderSolver, self).__init__()

        self._neural_net = evrp_encoder_decoder.vrpEncoderDecoderModel(model_parameters)

    def _get_num_problems(self, problem_parameters):
        """ Get the number of problems """

        num_problems = problem_parameters.get("num_problems")
        if num_problems is None:
            raise ValueError("The number of problems must be specified")

        return num_problems

    def _set_num_problems(self, problem_parameters):
        """ Set the number of problems  """

        num_problems = self._get_num_problems(problem_parameters)
        self._num_problems = num_problems

        return None

    def _get_num_customers(self, graph_parameters):
        """ Get the number of customers """

        num_customers = graph_parameters.get("num_customers")
        if num_customers is None:
            raise ValueError("Number of customers must not be None!")

        return num_customers

    def _set_num_customers(self, graph_parameters):
        """ Set the number of customers """

        num_customers = self._get_num_customers(graph_parameters)
        self._num_customers = num_customers

        return None

    def _get_num_stations(self, graph_parameters):
        """ Get the number of stations """

        num_stations = graph_parameters.get("num_stations")
        if num_stations is None:
            num_stations = 0

        return num_stations

    def _set_num_stations(self, graph_parameters):
        """ Set the number of stations """

        num_stations = self._get_num_stations(graph_parameters)
        self._num_stations = num_stations

        return None

    def _set_num_nodes(self):
        """ Set the number of nodes """

        self._num_nodes = self._num_customers + self._num_stations + 1

        return None
        
    def _reset_problem_parameters(self, problem_parameters):

        self._set_num_problems(problem_parameters)

        return None
        
    def _compute_placeholder_embedding(self):

        placeholder_embedding = torch.randn(
            (self._num_problems, self._neural_net._encoder._initial_embedding_dim)
        )
        
        return placeholder_embedding
        
    def _step(self, problem_features, problem_parameters, output_mask):

        self._reset_problem_parameters(problem_parameters)

        initial_embedding = self._compute_placeholder_embedding()
        final_embedding = self._compute_placeholder_embedding()
        
        node_probabilities = self._neural_net(
            problem_features,
            initial_embedding,
            final_embedding,
            None,
        )

        return node_probabilities

    def _clear_stored_embeddings(self):

        del(self._neural_net._node_embeddings)
        del(self._neural_net._graph_embedding)

        return None

    def forward(self, problem_features, problem_parameters):

        self._reset_problem_parameters(problem_parameters)

        node_probabilities = self._step(problem_features, problem_parameters, None)
        
        self._clear_stored_embeddings()

        return node_probabilities
