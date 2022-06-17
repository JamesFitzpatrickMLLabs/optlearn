import torch

import torch.nn as nn

import numpy as np

from optlearn.nets import masking
from optlearn.nets import charging
from optlearn.nets import attention
from optlearn.nets import routing_utils
from optlearn.nets import truncated_normal


class vrpEncoderModel(nn.Module, masking.evrpMasker):
    def __init__(self, model_parameters):
        super(vrpEncoderModel, self).__init__()
         
        self._set_model_parameters(model_parameters)
        self._build_neural_network_model()

        return None

    def _get_initial_embedding_dim(self, model_parameters):
        """ Get the initial embedding dimensions """

        initial_embedding_dim = model_parameters.get("initial_embedding_dim")
        if initial_embedding_dim is None:
            raise ValueError("Initial embedding dimensions must not be None")

        return initial_embedding_dim

    def _set_initial_embedding_dim(self, model_parameters):
        """ Set the initial embedding dimensions """

        initial_embedding_dim = self._get_initial_embedding_dim(model_parameters)
        self._initial_embedding_dim = initial_embedding_dim

        return None

    def _get_hidden_layer_dim(self, model_parameters):
        """ Get the hidden layer dimensions """

        hidden_layer_dim = model_parameters.get("hidden_layer_dim")
        if hidden_layer_dim is None:
            raise ValueError("Hidden layer dimensions must not be None")

        return hidden_layer_dim

    def _set_hidden_layer_dim(self, model_parameters):
        """ Set the hidden layer dimensions """

        hidden_layer_dim = self._get_hidden_layer_dim(model_parameters)
        self._hidden_layer_dim = hidden_layer_dim

        return None

    def _get_query_dim(self, model_parameters):
        """ Get the query dimensions """

        query_dim = model_parameters.get("query_dim")
        if query_dim is None:
            raise ValueError("Query dimensions must not be None")

        return query_dim

    def _set_query_dim(self, model_parameters):
        """ Set the query dimensions """

        query_dim = self._get_query_dim(model_parameters)
        self._query_dim = query_dim

        return None

    def _get_key_dim(self, model_parameters):
        """ Get the key dimensions """

        key_dim = model_parameters.get("key_dim")
        if key_dim is None:
            raise ValueError("Key dimensions must not be None")

        return key_dim

    def _set_key_dim(self, model_parameters):
        """ Set the key dimensions """

        key_dim = self._get_key_dim(model_parameters)
        self._key_dim = key_dim

        return None

    def _get_value_dim(self, model_parameters):
        """ Get the value dimensions """

        value_dim = model_parameters.get("value_dim")
        if value_dim is None:
            raise ValueError("Value dimensions must not be None")

        return value_dim

    def _set_value_dim(self, model_parameters):
        """ Set the value dimensions """

        value_dim = self._get_value_dim(model_parameters)
        self._value_dim = value_dim

        return None

    def _get_encoding_layers_num(self, model_parameters):
        """ Get the number of encoding layers """

        encoding_layers_num = model_parameters.get("encoding_layers_num")
        if encoding_layers_num is None:
            raise ValueError("Number of encoding layers must not be None")

        return encoding_layers_num

    def _set_encoding_layers_num(self, model_parameters):
        """ Set the number of encoding layers """

        encoding_layers_num = self._get_encoding_layers_num(model_parameters)
        self._encoding_layers_num = encoding_layers_num

        return None
    
    def _set_model_parameters(self, model_parameters):
        """ Get the model parameters """

        self._set_initial_embedding_dim(model_parameters)
        self._set_hidden_layer_dim(model_parameters)
        self._set_query_dim(model_parameters)
        self._set_key_dim(model_parameters)
        self._set_value_dim(model_parameters)
        self._set_encoding_layers_num(model_parameters)

        return None

    def _build_neural_network_model(self):
        """ Build the neural network model """

        self._node_embedder = attention.transformerEncoder(
            self._initial_embedding_dim,
            self._query_dim,
            self._key_dim,
            self._value_dim,
            self._hidden_layer_dim,
            self._encoding_layers_num,
        )
        self._graph_embedder = attention.graphMeanEmbedder()

        return None

    def _embed_graph(self, problem_features):
        """ Compute the static embeddings for the problems """

        node_embeddings = self._node_embedder(
            problem_features.get("customer_features"),
            problem_features.get("depot_features"),
            problem_features.get("station_features"),
            return_dict=True,
        )
        graph_embedding = self._graph_embedder(node_embeddings)

        return node_embeddings, graph_embedding

    def forward(self, problem_features):

        node_embeddings, graph_embedding = self._embed_graph(problem_features)

        return node_embeddings, graph_embedding


class vrpDecoderModel(nn.Module, masking.evrpMasker):
    def __init__(self, model_parameters):
        super(vrpDecoderModel, self).__init__()
         
        self._set_model_parameters(model_parameters)
        self._build_neural_network_model()

        return None

    def _get_initial_embedding_dim(self, model_parameters):
        """ Get the initial embedding dimensions """

        initial_embedding_dim = model_parameters.get("initial_embedding_dim")
        if initial_embedding_dim is None:
            raise ValueError("Initial embedding dimensions must not be None")

        return initial_embedding_dim

    def _set_initial_embedding_dim(self, model_parameters):
        """ Set the initial embedding dimensions """

        initial_embedding_dim = self._get_initial_embedding_dim(model_parameters)
        self._initial_embedding_dim = initial_embedding_dim

        return None

    def _get_hidden_layer_dim(self, model_parameters):
        """ Get the hidden layer dimensions """

        hidden_layer_dim = model_parameters.get("hidden_layer_dim")
        if hidden_layer_dim is None:
            raise ValueError("Hidden layer dimensions must not be None")

        return hidden_layer_dim

    def _set_hidden_layer_dim(self, model_parameters):
        """ Set the hidden layer dimensions """

        hidden_layer_dim = self._get_hidden_layer_dim(model_parameters)
        self._hidden_layer_dim = hidden_layer_dim

        return None

    def _get_query_dim(self, model_parameters):
        """ Get the query dimensions """

        query_dim = model_parameters.get("query_dim")
        if query_dim is None:
            raise ValueError("Query dimensions must not be None")

        return query_dim

    def _set_query_dim(self, model_parameters):
        """ Set the query dimensions """

        query_dim = self._get_query_dim(model_parameters)
        self._query_dim = query_dim

        return None

    def _get_key_dim(self, model_parameters):
        """ Get the key dimensions """

        key_dim = model_parameters.get("key_dim")
        if key_dim is None:
            raise ValueError("Key dimensions must not be None")

        return key_dim

    def _set_key_dim(self, model_parameters):
        """ Set the key dimensions """

        key_dim = self._get_key_dim(model_parameters)
        self._key_dim = key_dim

        return None

    def _get_value_dim(self, model_parameters):
        """ Get the value dimensions """

        value_dim = model_parameters.get("value_dim")
        if value_dim is None:
            raise ValueError("Value dimensions must not be None")

        return value_dim

    def _set_value_dim(self, model_parameters):
        """ Set the value dimensions """

        value_dim = self._get_value_dim(model_parameters)
        self._value_dim = value_dim

        return None

    def _get_encoding_layers_num(self, model_parameters):
        """ Get the number of encoding layers """

        encoding_layers_num = model_parameters.get("encoding_layers_num")
        if encoding_layers_num is None:
            raise ValueError("Number of encoding layers must not be None")

        return encoding_layers_num

    def _set_encoding_layers_num(self, model_parameters):
        """ Set the number of encoding layers """

        encoding_layers_num = self._get_encoding_layers_num(model_parameters)
        self._encoding_layers_num = encoding_layers_num

        return None
    
    def _set_model_parameters(self, model_parameters):
        """ Get the model parameters """

        self._set_initial_embedding_dim(model_parameters)
        self._set_hidden_layer_dim(model_parameters)
        self._set_query_dim(model_parameters)
        self._set_key_dim(model_parameters)
        self._set_value_dim(model_parameters)
        self._set_encoding_layers_num(model_parameters)

        return None

    def _build_neural_network_model(self):
        """ Build the neural network model """

        self.context_decoder = attention.multiHeadedContextNodeEncoder(
            self._initial_embedding_dim * 3,
            self._query_dim,
            self._key_dim,
            self._value_dim,
        )
        self.single_headed_decoder = attention.singleHeadedLogitDecoder(
            self._initial_embedding_dim,
        )
        self.normal_parameter_decoder = attention.linearNodeEncoder(
            self._initial_embedding_dim + 4, 2
        )

        return None

    def _concatenate_node_embeddings(self, node_embeddings):

        if node_embeddings.get("station_embedding") is not None:
            node_embeddings = torch.cat([
                node_embeddings.get("depot_embedding"),
                node_embeddings.get("customer_embedding"),
                node_embeddings.get("station_embedding")
            ], -2)
        else:
            node_embeddings = torch.cat([
                node_embeddings.get("depot_embedding"),
                node_embeddings.get("customer_embedding"),
            ], -2)

        return node_embeddings

    def _encode_context(self, node_embeddings, graph_embedding, initial_embedding, final_embedding): 

        context_embedding = torch.cat([
            graph_embedding,
            initial_embedding,
            final_embedding,
        ], -1)
        node_embeddings = self._concatenate_node_embeddings(node_embeddings)
        context_embedding = self.context_decoder(context_embedding, node_embeddings)

        return context_embedding

    def _compute_node_probabilities(self, context_embedding, node_embeddings, output_mask=None):

        node_embeddings = self._concatenate_node_embeddings(node_embeddings)
        node_probabilities = self.single_headed_decoder(
            context_embedding,
            node_embeddings,
            output_mask,
        )

        return node_probabilities

    def forward(self, node_embeddings, graph_embedding, initial_embedding, final_embedding, output_mask):
        context_embedding = self._encode_context(
            node_embeddings,
            graph_embedding,
            initial_embedding,
            final_embedding
        )
        node_probabilities = self._compute_node_probabilities(
            context_embedding,
            node_embeddings,
            output_mask
        )

        return node_probabilities

    
class vrpEncoderDecoderModel(nn.Module, masking.evrpMasker):
    def __init__(self, model_parameters):
        super(vrpEncoderDecoderModel, self).__init__()
         
        self._set_encoder(model_parameters)
        self._set_decoder(model_parameters)

        return None

    def _set_encoder(self, model_parameters):

        self._encoder = vrpEncoderModel(model_parameters)

        return None

    def _set_decoder(self, model_parameters):

        self._decoder = vrpDecoderModel(model_parameters)

        return None

    def _store_node_embeddings(self, node_embeddings):

        self._node_embeddings = node_embeddings

        return None


    def _store_graph_embedding(self, graph_embedding):

        self._graph_embedding = graph_embedding

        return None

    def _encode(self, problem_features):

        node_embeddings, graph_embedding = self._encoder(problem_features)
        self._store_node_embeddings(node_embeddings)
        self._store_graph_embedding(graph_embedding)

        return None

    def _decode(self, initial_embedding, final_embedding, output_mask):

        node_probabilities = self._decoder(
            self._node_embeddings,
            self._graph_embedding,
            initial_embedding,
            final_embedding,
            output_mask,
        )

        return node_probabilities

    def forward(self, problem_features, initial_embedding, final_embedding, output_mask):

        if not hasattr(self, "_node_embeddings"):
            self._encode(problem_features)
        node_probabilities = self._decode(
            initial_embedding,
            final_embedding,
            output_mask,
        )

        return node_probabilities        
    

class vrpEncoderDecoder(nn.Module, masking.evrpMasker):
    def __init__(self, model_parameters):
        super(vrpEncoderDecoder, self).__init__()
         
        self._set_model_parameters(model_parameters)
        self._build_neural_network_model()

        return None

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
        
    def _reset_num_parameters(self, problem_parameters, graph_parameters):

        self._set_num_problems(problem_parameters)
        self._set_num_customers(graph_parameters)
        self._set_num_stations(graph_parameters)
        self._set_num_nodes()

        return None

    def _get_graph_parameters(self):
        """ Get the graph parameters """

        graph_parameters = {
            "num_customers": self._num_customers,
            "num_stations": self._num_stations,
            "num_nodes": self._num_nodes,
        }

        return graph_parameters

    def _reset_current_parameters(self, customer_features, station_features=None):
        """ Reset the current problem parameters """

        self._reset_num_parameters(customer_features, station_features)

        return None

    def _get_vehicle_capacity_parameter(self, vehicle_parameters):
        """ Get the vehicle carrying capacity parameter and default it if not found """

        vehicle_capacity = vehicle_parameters.get("vehicle_capacity")
        if vehicle_capacity is None:
            vehicle_capacity = 0

        return vehicle_capacity

    def _set_vehicle_capacity(self, vehicle_parameters):
        """ Set the carrying capacity of the vehicle """

        vehicle_capacity = self._get_vehicle_capacity_parameter(vehicle_parameters)
        self._vehicle_capacity = vehicle_capacity

        return None

    def _get_battery_capacity_parameter(self, vehicle_parameters):
        """ Get the battery capacity parameter and default it if not found """

        battery_capacity = vehicle_parameters.get("battery_capacity")
        if battery_capacity is None:
            battery_capacity = None

        return battery_capacity

    def _set_battery_capacity(self, vehicle_parameters):
        """ Set the battery capacity of the vehicle """

        battery_capacity = self._get_battery_capacity_parameter(vehicle_parameters)
        self._battery_capacity = battery_capacity

        return None

    def _get_time_limit_parameter(self, vehicle_parameters):
        """ Get the time limit parameter and default it if not found """

        time_limit = vehicle_parameters.get("time_limit")
        if time_limit is None:
            time_limit = 0

        return time_limit

    def _set_time_limit(self, vehicle_parameters):
        """ Set the time limit for a route """

        time_limit = self._get_time_limit_parameter(vehicle_parameters)
        self._time_limit = time_limit

        return None

    def _get_energy_consumption_rate_parameter(self, vehicle_parameters):
        """ Get the energy consumption rate parameter and default it if not found """

        energy_consumption_rate = vehicle_parameters.get("energy_consumption_rate")
        if energy_consumption_rate is None:
            energy_consumption_rate = 0.25
            
        return energy_consumption_rate

    def _set_energy_consumption_rate(self, vehicle_parameters):
        """ Set the energy consumption rate parameter """

        energy_consumption_rate = self._get_energy_consumption_rate_parameter(vehicle_parameters)
        self._energy_consumption_rate = energy_consumption_rate

        return None

    def _get_average_velocity_parameter(self, vehicle_parameters):
        """ Get the average velocity parameter and default it if not found """

        average_velocity = vehicle_parameters.get("average_velocity")
        if average_velocity is None:
            average_velocity = 40.0

        return average_velocity

    def _set_average_velocity(self, vehicle_parameters):
        """ Set the average velocity parameter """

        average_velocity = self._get_average_velocity_parameter(vehicle_parameters)
        self._average_velocity = average_velocity

        return None

    def _set_vehicle_parameters(self, vehicle_parameters):
        """ Set the parameters for the vehicles """

        self._set_vehicle_capacity(vehicle_parameters)
        self._set_battery_capacity(vehicle_parameters)
        self._set_time_limit(vehicle_parameters)
        self._set_energy_consumption_rate(vehicle_parameters)
        self._set_average_velocity(vehicle_parameters)

        return None

    def _get_vehicle_parameters(self):

        vehicle_parameters = {
            "vehicle_capacity": self._vehicle_capacity,
            
            "battery_capacity": self._battery_capacity,
            "time_limit": self._time_limit,
        }

        return None

    def _reset_arrays(self, node_coordinates):
        """ Reset the weight arrays """
        
        self._weight_tensor = self._build_weight_tensor(node_coordinates)
        self._energy_tensor = self._build_energy_tensor(self._weight_tensor)
        self._time_tensor = self._build_time_tensor(self._weight_tensor)
        self._service_time_tensor = self._build_service_time_tensor(self._weight_tensor)
        self._taba_weight_tensor = self._build_taba_weight_tensor(self._weight_tensor)
        self._taba_energy_tensor = self._build_taba_energy_tensor(self._taba_weight_tensor)
        self._taba_time_tensor = self._build_taba_time_tensor(self._taba_weight_tensor)
        self._taba_service_time_tensor = self._build_taba_service_time_tensor(self._taba_weight_tensor)
        self._tac_weight_tensor = self._build_tac_weight_tensor(self._weight_tensor)
        self._tac_energy_tensor = self._build_tac_energy_tensor(self._tac_weight_tensor)
        self._tac_time_tensor = self._build_tac_time_tensor(self._tac_weight_tensor)
        self._tac_service_time_tensor = self._build_tac_service_time_tensor(self._tac_weight_tensor)
        
        return None

    def _get_initial_embedding_dim(self, model_parameters):
        """ Get the initial embedding dimensions """

        initial_embedding_dim = model_parameters.get("initial_embedding_dim")
        if initial_embedding_dim is None:
            raise ValueError("Initial embedding dimensions must not be None")

        return initial_embedding_dim

    def _set_initial_embedding_dim(self, model_parameters):
        """ Set the initial embedding dimensions """

        initial_embedding_dim = self._get_initial_embedding_dim(model_parameters)
        self._initial_embedding_dim = initial_embedding_dim

        return None

    def _get_hidden_layer_dim(self, model_parameters):
        """ Get the hidden layer dimensions """

        hidden_layer_dim = model_parameters.get("hidden_layer_dim")
        if hidden_layer_dim is None:
            raise ValueError("Hidden layer dimensions must not be None")

        return hidden_layer_dim

    def _set_hidden_layer_dim(self, model_parameters):
        """ Set the hidden layer dimensions """

        hidden_layer_dim = self._get_hidden_layer_dim(model_parameters)
        self._hidden_layer_dim = hidden_layer_dim

        return None

    def _get_query_dim(self, model_parameters):
        """ Get the query dimensions """

        query_dim = model_parameters.get("query_dim")
        if query_dim is None:
            raise ValueError("Query dimensions must not be None")

        return query_dim

    def _set_query_dim(self, model_parameters):
        """ Set the query dimensions """

        query_dim = self._get_query_dim(model_parameters)
        self._query_dim = query_dim

        return None

    def _get_key_dim(self, model_parameters):
        """ Get the key dimensions """

        key_dim = model_parameters.get("key_dim")
        if key_dim is None:
            raise ValueError("Key dimensions must not be None")

        return key_dim

    def _set_key_dim(self, model_parameters):
        """ Set the key dimensions """

        key_dim = self._get_key_dim(model_parameters)
        self._key_dim = key_dim

        return None

    def _get_value_dim(self, model_parameters):
        """ Get the value dimensions """

        value_dim = model_parameters.get("value_dim")
        if value_dim is None:
            raise ValueError("Value dimensions must not be None")

        return value_dim

    def _set_value_dim(self, model_parameters):
        """ Set the value dimensions """

        value_dim = self._get_value_dim(model_parameters)
        self._value_dim = value_dim

        return None

    def _get_encoding_layers_num(self, model_parameters):
        """ Get the number of encoding layers """

        encoding_layers_num = model_parameters.get("encoding_layers_num")
        if encoding_layers_num is None:
            raise ValueError("Number of encoding layers must not be None")

        return encoding_layers_num

    def _set_encoding_layers_num(self, model_parameters):
        """ Set the number of encoding layers """

        encoding_layers_num = self._get_encoding_layers_num(model_parameters)
        self._encoding_layers_num = encoding_layers_num

        return None
    
    def _set_model_parameters(self, model_parameters):
        """ Get the model parameters """

        self._set_initial_embedding_dim(model_parameters)
        self._set_hidden_layer_dim(model_parameters)
        self._set_query_dim(model_parameters)
        self._set_key_dim(model_parameters)
        self._set_value_dim(model_parameters)
        self._set_encoding_layers_num(model_parameters)

        return None

    def _build_neural_network_model(self):
        """ Build the neural network model """

        self._encoder = attention.transformerEmbedder(
            self._initial_embedding_dim,
            self._query_dim,
            self._key_dim,
            self._value_dim,
            self._hidden_layer_dim,
            self._encoding_layers_num,
        )        
        self.context_decoder = attention.multiHeadedContextNodeEncoder(
            self._initial_embedding_dim * 3,
            self._query_dim,
            self._key_dim,
            self._value_dim,
        )
        self.single_headed_decoder = attention.singleHeadedLogitDecoder(
            self._initial_embedding_dim,
        )
        self.normal_parameter_decoder = attention.linearNodeEncoder(
            self._initial_embedding_dim + 4, 2
        )

        return None
    
    def _build_weight_tensor(self, node_coordinates):
        """ Build the weight tensor """
        
        weight_tensor = routing_utils.compute_weight_tensor_batch(node_coordinates)
        
        return weight_tensor
    
    def _build_energy_tensor(self, weight_tensor):
        
        energy_tensor = routing_utils.compute_energy_tensor(
            weight_tensor, self._energy_consumption_rate
        )
        
        return energy_tensor
    
    def _build_time_tensor(self, weight_tensor):
        
        time_tensor = routing_utils.compute_time_tensor(
            weight_tensor, self._average_velocity
        )
        
        return time_tensor
    
    def _build_service_time_tensor(self, weight_tensor):
        
        service_time_tensor = routing_utils.compute_service_time_tensor(
            weight_tensor, self._num_customers, self._average_velocity
        )
        
        return service_time_tensor
    
    def _build_taba_weight_tensor(self, weight_tensor):
        
        taba_weight_tensor = routing_utils.compute_taba_tensor(weight_tensor)
        
        return taba_weight_tensor
    
    def _build_taba_energy_tensor(self, taba_weight_tensor):
        
        taba_energy_tensor = routing_utils.compute_taba_energy_tensor(
            taba_weight_tensor, self._energy_consumption_rate
        )
        
        return taba_energy_tensor
    
    def _build_taba_time_tensor(self, taba_weight_tensor):
        
        taba_time_tensor = routing_utils.compute_taba_time_tensor(
            taba_weight_tensor, self._average_velocity
        )
        
        return taba_time_tensor
    
    def _build_taba_service_time_tensor(self, taba_weight_tensor):
        
        taba_service_time_tensor = routing_utils.compute_taba_service_time_tensor(
            taba_weight_tensor, self._num_customers, self._average_velocity
        )
        
        return taba_service_time_tensor
    
    def _build_tac_weight_tensor(self, weight_tensor):
        
        tac_weight_tensor = routing_utils.compute_tac_tensor(
            weight_tensor, self._num_customers,
        )
        
        return tac_weight_tensor
    
    def _build_tac_energy_tensor(self, tac_weight_tensor):
        
        tac_energy_tensor = routing_utils.compute_tac_energy_tensor(
            tac_weight_tensor, self._energy_consumption_rate
        )
        
        return tac_energy_tensor
    
    def _build_tac_time_tensor(self, tac_weight_tensor):
        
        tac_time_tensor = routing_utils.compute_tac_time_tensor(
            tac_weight_tensor, self._average_velocity
        )
        
        return tac_time_tensor
    
    def _build_tac_service_time_tensor(self, tac_weight_tensor):
        
        tac_service_time_tensor = routing_utils.compute_tac_service_time_tensor(
            tac_weight_tensor, self._num_customers, self._average_velocity
        )
        
        return tac_service_time_tensor

    def toggle_sample_mode(self):
        """ Toggle the sample mode """

        if hasattr(self, "_sample_mode"):
            self._sample_mode = not(self._sample_mode)
        else:
            self._sample_mode = True

        if self._sample_mode:
            print("Sample mode enabled!")
        else:
            print("Sample mode disabled!")

        return None

    def _reset_parameters(self):
        """ Reset working parameters for checking feasibility """

        self._verbose = True
        self._time_remaining = torch.tensor([
            [self._time_limit] for num in range(self._num_problems)
        ]).float()
        self._energy_remaining = torch.tensor([
            [self._battery_capacity] for num in range(self._num_problems)
        ]).float()
        self._selection_probabilities = [
            [] for num in range(self._num_problems)
        ]
        self._waiting_probabilities = [
            [] for num in range(self._num_problems)
        ]
        self._visited_nodes = [
            [0] for num in range(self._num_problems)
        ]
        self._visited_depots = [
            [] for num in range(self._num_problems)
        ]
        self._visited_customers = [
            [] for num in range(self._num_problems)
        ]
        self._visited_stations = [
            [] for num in range(self._num_problems)
        ]
        self._last_nodes = torch.tensor([
            [0] for num in range(self._num_problems)
        ]).float()
        self._iterations_since_last_customer = torch.tensor([
            [0] for num in range(self._num_problems)
        ]).float()
        self._batch_indices = torch.tensor([
            [num for num in range(self._num_problems)]
        ]).int()
        self._solution_time = torch.tensor([
            [0.0] for num in range(self._num_problems)
        ]).float()
        self._variable_time = [
            [0] for num in range(self._num_problems)
        ]
        self._terminate_solution = [
            False for num in range(self._num_problems)
        ]
        
        return None

    def _was_last_node_depot(self):
        
        was_last_node_depot = (self._last_nodes == 0).flatten()
        
        return was_last_node_depot
    
    def _is_solution_complete(self):
        
        is_solution_complete = (self._are_all_customers_visited() * self._was_last_node_depot()).bool()
        
        return is_solution_complete
        
    def _are_all_solutions_complete(self):
        
        are_all_solutions_complete = (
            self._are_all_customers_visited() * self._was_last_node_depot()
        ).prod().bool()
        
        return are_all_solutions_complete

    def get_solution_time(self):
        
        solution_time = self._solution_time
        
        return solution_time
    
    def get_station_count(self):
        """ Count how often the stations were visited in the solution """
        
        station_count = torch.tensor([len(item) for item in self._visited_stations]).unsqueeze(-1)
        
        return station_count

    def get_depot_count(self):
        """ Count how often the depot was visited in the solution """

        depot_count = torch.tensor([len(item) for item in self._visited_depots]).unsqueeze(-1)
        
        return depot_count

        
    def get_solution_objective(self):
        
        station_count = self.get_station_count()
        depot_count = self.get_depot_count()
        solution_time = self.get_solution_time()
        solution_objective = station_count + solution_time + depot_count
        
        return solution_objective

    def _compute_minimum_station_duration(self):
        
        minimum_station_charge_time = charging.compute_minimum_station_charge_time(
            self._energy_tensor, 
            self._num_customers, 
            self._energy_remaining,
            self._station_features,
            self._batch_indices, 
            self._last_nodes,
        )
        
        return minimum_station_charge_time
    
    def _compute_maximum_station_duration(self):
        
        maximum_station_charge_time = charging.compute_maximum_station_charge_time(
            self._energy_tensor, 
            self._taba_time_tensor,
            self._num_customers, 
            self._energy_remaining, 
            self._time_remaining,
            self._station_features,
            self._batch_indices, 
            self._last_nodes,
        )
        
        return maximum_station_charge_time
        
    def _compute_minimum_depot_duration(self):
        
        minimum_depot_duration = torch.zeros((self._num_problems, 1))
        
        return minimum_depot_duration
    
    def _compute_maximum_depot_duration(self):
        
        maximum_depot_duration = torch.zeros((self._num_problems, 1))
        
        return maximum_depot_duration

    def _compute_minimum_customer_duration(self):
        
        minimum_customer_duration = torch.ones((self._num_problems, self._num_customers)) * 0.5
        
        return minimum_customer_duration
    
    def _compute_maximum_customer_duration(self):
        
        maximum_customer_duration = torch.ones((self._num_problems, self._num_customers)) * 0.5
        
        return maximum_customer_duration
    
    def _compute_minimum_node_duration(self):
        
        minimum_node_duration = torch.concat([
            self._compute_minimum_depot_duration(),
            self._compute_minimum_customer_duration(),
            self._compute_minimum_station_duration(),
        ], axis=1)
        
        return minimum_node_duration
    
    def _compute_maximum_node_duration(self):
        
        maximum_node_duration = torch.concat([
            self._compute_maximum_depot_duration(),
            self._compute_maximum_customer_duration(),
            self._compute_maximum_station_duration(),
        ], axis=1)
        
        return maximum_node_duration
    
    def _get_minimum_node_duration(self):
        
        minimum_node_duration = self._compute_minimum_node_duration()
        maximum_node_duration = self._compute_maximum_node_duration()
        
        mismatches = (maximum_node_duration < minimum_node_duration)
        minimum_node_duration[mismatches] = 0
        
        return minimum_node_duration
        
    def _get_maximum_node_duration(self):
        
        minimum_node_duration = self._compute_minimum_node_duration()
        maximum_node_duration = self._compute_maximum_node_duration()
        
        mismatches = (maximum_node_duration < minimum_node_duration)
        maximum_node_duration[mismatches] = 0
        
        return maximum_node_duration
    
    def _count_visited_nodes(self):

        if not hasattr(self, "_visited_nodes"):
            raise Exception("Forward parameters not yet set! Method should not yet be called!")
        visited_count = [len(item) for item in self._visited_nodes]

        return visited_count
    
    def _get_placeholder_initial_embedding(self, node_embeddings):
        
        initial_embedding = node_embeddings[:, 0]
        
        return initial_embedding
    
    def _get_placeholder_final_embedding(self, node_embeddings):
        
        final_embedding = node_embeddings[:, 0]
        
        return final_embedding
    
    def _get_visited_initial_embedding(self, node_embeddings):
        
        exponents = [torch.arange(0, len(item), 1) for item in self._visited_nodes]
        bases = [0.9 ** torch.ones(len(item)) for item in self._visited_nodes]
        factors = [base ** exponent for base, exponent in zip(bases, exponents)]

        batch_indices = [[num for mun in range(len(item))]
                         for num, item in enumerate(self._visited_nodes)]
        squashed_embeddings = [(
            node_embeddings[batch_indices[num], self._visited_nodes[num]] * factors[num].unsqueeze(-1)
        ).mean(0) for num in range(len(batch_indices))]
        visited_initial_embedding =  - torch.stack(squashed_embeddings, 0)
        
        return visited_initial_embedding
    
    def _get_initial_embedding(self, node_embeddings):

        if sum(self._count_visited_nodes()) == 0:
            initial_embedding = self._get_placeholder_initial_embedding(node_embeddings)
        else:
            initial_embedding = self._get_visited_initial_embedding(node_embeddings)

        return initial_embedding
     
    def _get_final_embedding(self, node_embeddings):

        if sum(self._count_visited_nodes()) == 0:
            final_embedding = self._get_placeholder_final_embedding(node_embeddings)
        else:
            final_nodes = self._get_last_visited_nodes()
            final_embedding = self._get_node_final_embedding(node_embeddings, final_nodes)

        return final_embedding

    def _build_distribution_features(self, node_embeddings, minimum_node_duration, maximum_node_duration):
        
        distribution_features = torch.concat([
            node_embeddings,
            minimum_node_duration.unsqueeze(-1),
            maximum_node_duration.unsqueeze(-1),
            self._energy_remaining.repeat(1, self._num_customers + self._num_stations + 1).unsqueeze(-1),
            self._time_remaining.repeat(1, self._num_customers + self._num_stations + 1).unsqueeze(-1),
        ], -1)
        
        return distribution_features
    
    def _build_distribution_parameters(self, distribution_features, minimum_node_duration, maximum_node_duration):
        
        distribution_parameters = self.normal_parameter_decoder(distribution_features)
        distribution_parameters = torch.pow(distribution_parameters, 2)
        
        return distribution_parameters
    
    def _build_sampling_distribution(self, distribution_parameters, minimum_node_duration, maximum_node_duration):
        
        mus, sigmas = distribution_parameters[..., 0], distribution_parameters[..., 1] + 5.0
        sampling_distribution = truncated_normal.TruncatedNormal(
            mus, 
            sigmas, 
            minimum_node_duration, 
            maximum_node_duration,
        )
        
        return sampling_distribution
    
    def _update_time_remaining(self, chosen_duration, problem_num):
        """ Update the time remaining for the current route for the given problem """
        
        self._time_remaining[problem_num, 0] -= float(chosen_duration)
        
        return None
    
    def _update_solution_time(self, chosen_duration, problem_num):
        """ Update the total solution duration for the given problem """
        
        self._solution_time[problem_num, 0] += float(chosen_duration)
        
        return None
    
    def _append_chosen_duration(self, chosen_duration, problem_num):
        """ Store the chosen duration value for the given problem """
        
        self._variable_time[problem_num].append(float(chosen_duration.data.numpy()))
        
        return None
    
    def _get_station_time_function(self, problem_num, node_index):
        """ Get the charging time function for the given station node for the given problem """
        
        if self._station_features[problem_num, int(node_index) - self._num_customers -1, 2] == 0.0:
            time_function = charging.piecewise_slow
        if self._station_features[problem_num, int(node_index) - self._num_customers -1, 2] == 0.5:
            time_function = charging.piecewise_normal
        if self._station_features[problem_num, int(node_index) - self._num_customers -1, 2] == 1.0:
            time_function = charging.piecewise_fast
            
        return time_function
    
    def _get_station_energy_function(self, problem_num, node_index):
        """ Get the charging energy fucntion for the given station node for the given problem """
        
        if self._station_features[problem_num, int(node_index) - self._num_customers -1, 2] == 0.0:
            energy_function = charging.piecewise_slow_energy
        if self._station_features[problem_num, int(node_index) - self._num_customers -1, 2] == 0.5:
            energy_function = charging.piecewise_normal_energy
        if self._station_features[problem_num, int(node_index) - self._num_customers -1, 2] == 1.0:
            energy_function = charging.piecewise_fast_energy
            
        return energy_function
    
    def _is_node_index_station_node(self, node_index):
        """ Check if the given node index is one for a station node """
        
        is_non_customer = node_index > self._num_customers 
        is_station = node_index <= self._num_customers + self._num_stations + 1
        is_station = is_non_customer and is_station
        
        return is_station
    
    def _compute_new_energy(self, energy_function, time_function, problem_num, chosen_duration):
        """ Compute the updated energy for the vehicle if charging for the chosen duration """
    
        initial_time = time_function(self._energy_remaining[problem_num, 0].data.numpy())
        final_time = initial_time + chosen_duration
        new_energy = energy_function(final_time)
        
        return new_energy
    
    def _update_energy_remaining(self, problem_num, node_index, new_energy):
        """ Update the remaining energy for the given problem vehicle to the given energy value """
        
        self._energy_remaining[problem_num, 0] = float(new_energy)
        
        return None

    def _store_node_probabilities(self, probabilities, ignore_probabilities=None):
        """ Store the node selection probability value(s). Ignore (don't store) those specificied. """

        if ignore_probabilities is None:
            ignore_probabilities = self._terminate_solution
        for num, (probability, ignore) in enumerate(zip(probabilities, ignore_probabilities)):
            if not ignore:
                self._selection_probabilities[num].append(probability)

        return None

    def _squish_probabilities(self, node_probabilities):
        """ Squish the probabilities for the stations if customers have not been seen in a while """

        squished_probabilities = torch.ones(node_probabilities.shape)
        squishing_factor = torch.exp(self._iterations_since_last_customer)
        customer_probabilities = node_probabilities[:, 1:self._num_customers + 1] * squishing_factor
        station_probabilities = node_probabilities[:, self._num_customers + 1:] / squishing_factor
        squished_probabilities[:, 1:self._num_customers + 1] = customer_probabilities 
        squished_probabilities[:, self._num_customers + 1:] = station_probabilities
        squished_probabilities = node_probabilities * squished_probabilities

        return squished_probabilities
        
    def _sample_node(self, node_probabilities, exponential_squishing=False):
        """ Select one of the nodes with a probability equal to its selection probability """

        if not exponential_squishing:
            node_index = node_probabilities.multinomial(1).squeeze()
        else:
            squished_probabilities = self._squish_probabilities(node_probabilities)
            node_index = squished_probabilities.multinomial(1).squeeze()
        probabilities = node_probabilities[[list(range(len(node_probabilities))), node_index]]
        self._store_node_probabilities(probabilities, self._are_all_customers_visited())
         
        return node_index

    def _snatch_node(self, node_probabilities, exponential_squishing=False):
        """ Select the next node by choosing the one with highest estimated probability """

        if not exponential_squishing:
            node_index = node_probabilities.argmax(-1)
        else:
            squished_probabilities = self._squish_probabilities(node_probabilities)
            node_index = squished_probabilities.argmax(-1)
        probabilities = node_probabilities[[list(range(len(node_probabilities))), node_index]]
        self._store_node_probabilities(probabilities, self._are_all_customers_visited())
         
        return node_index

    def _ignore_node_storage(self):
        """ Figure out if we don't need to store nodes for a solution anymore """

        are_all_customers_visited = self._are_all_customers_visited()
        last_nodes = self._last_nodes

        for num, (boolean, last_node) in enumerate(zip(are_all_customers_visited, last_nodes)):

            if boolean:
                if last_node == 0:
                    self._terminate_solution[num] = True
            
        return None
        
    def _store_node_indices(self, node_indices, ignore_node_indices=None):
        """ Store the selected node indices to form part of the solution """
        
        self._ignore_node_storage()
        ignore_node_indices = self._terminate_solution
        # ignore_node_indices =  [False] * len(ignore_node_indices)
        
        for num, node_index in enumerate(node_indices):
            if not ignore_node_indices[num]:
                if int(node_index) < self._num_customers + self._num_stations + 2:
                    travel_time = self._time_tensor[num, int(self._last_nodes[num, 0]), int(node_index)]
                    travel_energy = self._energy_tensor[num, int(self._last_nodes[num, 0]), int(node_index)]
                    self._time_remaining[num, 0] -= travel_time
                    self._solution_time[num, 0] += travel_time
                    self._energy_remaining[num, 0] -= travel_energy
                    self._visited_nodes[num].append(int(node_index))
                    self._last_nodes[num, 0] = int(node_index)     
                if int(node_index) == 0:
                    self._time_remaining[num, 0] = 10.0
                    self._energy_remaining[num, 0] = 16.0
                    self._visited_depots[num].append(0)
                if int(node_index) > 0 and int(node_index) <= self._num_customers:
                    self._visited_customers[num].append(int(node_index))
                    self._iterations_since_last_customer[num, 0] = 0
                else:
                    self._iterations_since_last_customer[num, 0] += 1
                if int(node_index) > self._num_customers:
                    self._visited_stations[num].append(int(node_index))
                
        return None
    
    def _store_variable_probabilities(self, probabilities, ignore_probabilities=None):

        if ignore_probabilities is None:
            ignore_probabilities = [False] * len(probabilities)
            ignore_probabilities = self._terminate_solution
        for num, (probability, ignore) in enumerate(zip(probabilities, ignore_probabilities)):
            if not ignore:
                self._waiting_probabilities[num].append(probability)

        return None    

    def _snatch_duration(self, node_indices, minimum_node_duration, maximum_node_duration):
        """ Select the highest probability duration to spend at the just-selected node """

        chosen_duration = maximum_node_duration
        self._update_time_and_energy(node_indices, chosen_duration)
        
        return chosen_duration

    def _sample_duration(self, node_indices, minimum_node_duration, maximum_node_duration):
        """ Sample a duration value according to the current truncated distribution """

        chosen_duration = (maximum_node_duration - minimum_node_duration) / 2
        chosen_duration = chosen_duration + minimum_node_duration
        self._update_time_and_energy(node_indices, chosen_duration)
        
        return chosen_duration

    def _update_time_and_energy(self, node_indices, chosen_duration):
        """ Update the time remaining and energy left in the vehicle """


        ignore_updates = self._terminate_solution
        
        for batch_num, node_index in enumerate(node_indices):
            duration_value = chosen_duration[batch_num, node_index]
            self._append_chosen_duration(duration_value, batch_num)
            self._update_time_remaining(duration_value, batch_num)
            self._update_solution_time(duration_value, batch_num)
            if self._is_node_index_station_node(int(node_index)):
                energy_function = self._get_station_energy_function(batch_num, int(node_index))
                time_function = self._get_station_time_function(batch_num, int(node_index))
                new_energy = self._compute_new_energy(
                    energy_function,
                    time_function,
                    batch_num,
                    duration_value.data.numpy()
                )
                self._update_energy_remaining(batch_num, node_index, new_energy)

        return None

    def _prepare_for_inference(self, problem_features, problem_parameters, graph_parameters, vehicle_parameters):
        """ Prepare for inference """

        self._reset_parameters()
        self._reset_arrays(
            problem_features.get("node_coordinates")
        )
        ##################### TEMPORARY HACK ############################
        self._station_features = problem_features.get("station_features")
        ##################### TEMPORARY HACK ############################
        self._reset_num_parameters(problem_parameters, graph_parameters)
        self._set_vehicle_parameters(vehicle_parameters)

        
        return None

    def _get_unvisited_customers(self):
        """ Get the unvisited customers """

        unvisited_customers = [
            [
                num for num in range(1, self._num_customers + 1)
                if num not in self._visited_customers[problem_num] 
            ] for problem_num in range(self._num_problems)
        ]

        return unvisited_customers

    def _embed_graph(self, problem_features):
        """ Compute the static embeddings for the problems """

        node_embeddings, graph_embedding = self._encoder(
            problem_features.get("depot_features"),
            problem_features.get("customer_features"),
            problem_features.get("station_features"),
        )

        return node_embeddings, graph_embedding

    def _step(self, node_embeddings, graph_embedding, problem_features, print_step=False):
        """ Select one node and the amount of time spent at that node """
        
        minimum_node_duration = self._get_minimum_node_duration()
        maximum_node_duration = self._get_maximum_node_duration()
        initial_embedding = self._get_initial_embedding(node_embeddings)
        final_embedding = self._get_placeholder_final_embedding(node_embeddings)        
        context_embedding = torch.cat(
            (graph_embedding, initial_embedding, final_embedding), -1
        )
        final_context_embedding = self.context_decoder(context_embedding, node_embeddings)
        output_mask = self._build_output_mask()

        ############################ HACK #######################################
        
        for problem_num in range(self._num_problems):
            if self._are_all_customers_visited()[problem_num] == True:
                output_mask[problem_num, 0] = False

        ############################ HACK #######################################
        ############################ HACK #######################################

        unvisited_customers = self._get_unvisited_customers()
        for problem_num in range(self._num_problems):
            unvisited_set = unvisited_customers[problem_num]
            tac_energy_values = self._tac_energy_tensor[problem_num, self._num_customers + 1:]
            station_indices = [[station for num in range(len(unvisited_set))
                                for station in range(self._num_stations)]]
            stuff = []
            for item in station_indices:
                stuff += item
            unvisited_indices = unvisited_set * self._num_stations
            tac_energy_values = tac_energy_values[stuff, unvisited_indices]
            if self._visited_nodes[problem_num][-1] == 0 and len(unvisited_indices) > 0:
                tac_energy_values = tac_energy_values.min(0)[0]
                for station_num in range(self._num_stations):
                    station_index = self._num_customers + 1 + station_num
                    minimum_charge_value = tac_energy_values[station_num]
                    if minimum_charge_value > self._battery_capacity:
                        output_mask[problem_num, station_index] = True
                    else:
                        if problem_features["station_features"][problem_num][station_num][2] == 0.0:
                            charge_function = charging.charge_time_slow
                        elif problem_features["station_features"][problem_num][station_num][2] == 0.5:
                            charge_function = charging.charge_time_normal
                        else:
                            charge_function = charging.charge_time_fast
                        start_charge = self._battery_capacity
                        start_charge -= float(self._energy_tensor[problem_num, station_index, 0])
                        start_charge = np.array([start_charge])
                        end_charge = np.array([float(minimum_charge_value)])
                        new_min_duration = charge_function(start_charge, end_charge)
                        current_min_duration = minimum_node_duration[problem_num, station_index].data
                        update_duration = max([new_min_duration, current_min_duration.numpy()])
                        minimum_node_duration[problem_num, station_index] = float(update_duration)

            if len(self._visited_nodes[problem_num]) >= 2:
                if self._visited_nodes[problem_num][-2] == 0:
                    if self._visited_nodes[problem_num][-1] >= self._num_customers + 1:
                        for station_num in range(self._num_stations):
                            station_index = self._num_customers + 1 + station_num
                            output_mask[problem_num, station_index] = True
                        output_mask[problem_num, 0] = True
                            
        ############################ HACK #######################################
        
        # if print_step:
        #     print("Output mask: ", output_mask)        
        node_probabilities = self.single_headed_decoder(
            final_context_embedding, node_embeddings, output_mask
        )
        # if print_step:
            # print("Node probabilities: ", node_probabilities)
        if self._sample_mode:
            node_indices = self._sample_node(node_probabilities)
            if len(node_indices.shape) == 0:
                node_indices = torch.tensor([int(node_indices)])
        else:
            node_indices = self._snatch_node(node_probabilities)
        # if print_step:
            # print("Chosen node: ", node_indices)
        self._store_node_indices(node_indices, self._are_all_customers_visited())                    
        if self._sample_mode:
            chosen_duration = self._sample_duration(
                node_indices,
                minimum_node_duration,
                maximum_node_duration,
            )
        else:
            chosen_duration = self._snatch_duration(
                node_indices,
                minimum_node_duration,
                maximum_node_duration,
            )
            
        return node_probabilities

    def _toggle_verbosity(self):
        """ Toggle verbosity """

        if hasattr(self, "_verbose"):
            self._verbose = not(self._verbose)
        else:
            verbose = False

        return None

    def _build_node_string_basic(self, problem_num):
        """ Print the nodes of the problem """

        depot_string  = "0"
        customer_string = "".join([str(num + 1) for num in range(self._num_customers)])
        station_string = "".join([str(num + 1 + self._num_customers)
                                   for num in range(self._num_stations)])
        node_string = f"||{depot_string}||{customer_string}||{station_string}||"

        return node_string
        
        
    def forward(self, problem_features, problem_parameters, graph_parameters, vehicle_parameters):
        
        self._prepare_for_inference(
            problem_features, problem_parameters, graph_parameters, vehicle_parameters
        )

        node_embeddings, graph_embedding = self._embed_graph(problem_features) 
        while not self._are_all_solutions_complete():
            # if any([len(item) > 5 * self._num_customers for item in self._visited_nodes]):
            #     return (False, None)
                    
            node_probabilities = self._step(
                node_embeddings, graph_embedding, problem_features, self._verbose
            )
            

        selection_sums = torch.stack([
            torch.log(torch.stack(item)).sum() for item in self._selection_probabilities
        ])
            
        return (True, selection_sums) 
