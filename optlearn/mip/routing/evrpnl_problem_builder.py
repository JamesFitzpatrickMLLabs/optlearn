import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from optlearn.plotting import plot_utils
from optlearn.mip.routing import problem_builder
from optlearn.mip.routing import piecewise_builder


class evrpnlProblemBuilder(problem_builder.basicProblemBuilder, piecewise_builder.pwlBuilder):

    def __init__(self, solver_package, is_directed=True, pwl_model="sos"):

        problem_builder.basicProblemBuilder.__init__(
            self,
            solver_package=solver_package,
            problem_type="evrpnl",
            is_directed=is_directed,
            is_multi=False
        )
        piecewise_builder.pwlBuilder.__init__(
            self,
            solver_package=solver_package,
        )

        self.pwl_model = pwl_model
        
        self.initialise_problem()
        self.initialise_storage()
        self.assume_hyperparameters()
        self.warn_hyperparmaters()

    def initialise_storage(self):
        """ Initialise the graphs that store the variables """

        self._initialise_arc_travel_variable_graph()
        self._initialise_arc_time_variable_graph()
        self._initialise_arc_energy_variable_graph()        
        self._initialise_node_time_variable_graph()
        self._initialise_node_energy_variable_graph()
        self._initialise_station_node_arrival_energy_variable_graph()
        self._initialise_station_node_departure_energy_variable_graph()
        self._initialise_station_node_arrival_time_variable_graph()
        self._initialise_station_node_departure_time_variable_graph()
        self._initialise_station_node_duration_variable_graph()
        self._initialise_station_node_arrival_breakpoint_graph()
        self._initialise_station_node_departure_breakpoint_graph()
        self._initialise_station_node_arrival_coefficient_graph()
        self._initialise_station_node_departure_coefficient_graph()

        return None

    def _build_node_variables(self, graph):
        """ Build all of the node formulation variables """

        self.build_arc_travel_variables(graph)
        self.build_node_time_variables(graph)
        self.build_node_energy_variables(graph)
        self.build_station_node_arrival_energy_variables(graph)
        self.build_station_node_departure_energy_variables(graph)
        self.build_station_node_arrival_time_variables(graph)
        self.build_station_node_departure_time_variables(graph)
        self.build_station_node_duration_variables(graph)
        self.build_station_nodes_arrival_pwl_variables(graph)
        self.build_station_nodes_departure_pwl_variables(graph)

        return None

    def _build_arc_variables(self, graph):
        """ Build all of the arc formulation variables """

        self.build_arc_travel_variables(graph)
        self.build_arc_time_variables(graph)
        self.build_arc_energy_variables(graph)
        self.build_station_node_arrival_energy_variables(graph)
        self.build_station_node_departure_energy_variables(graph)
        self.build_station_node_arrival_time_variables(graph)
        self.build_station_node_departure_time_variables(graph)
        self.build_station_node_duration_variables(graph)
        self.build_station_nodes_arrival_pwl_variables(graph)
        self.build_station_nodes_departure_pwl_variables(graph)

        return None
        
    def _build_node_constraints(self, graph):
        """ Build all of the node formulation constraints """

        self.build_customer_visiting_constraints_directed(graph)
        self.build_station_visiting_constraints_directed(graph)
        self.build_flow_constraints_directed(graph)
        
        self.build_station_nodes_arrival_pwl_constraints(graph)
        self.build_station_nodes_departure_pwl_constraints(graph)
        
        self.build_station_nodes_duration_constraints(graph)
        self.build_energy_arrival_departure_constraints(graph)
        self.build_customer_energy_node_tracking_constraints(graph)
        self.build_station_energy_node_tracking_constraints(graph)
        self.build_depot_energy_node_tracking_constraints(graph)
        self.build_station_departure_energy_node_reset_constraints(graph)
        self.build_depot_energy_initialisation_constraints(graph)
        self.build_customer_nodes_time_tracking_constraints(graph)
        self.build_station_nodes_time_tracking_constraints(graph)
        self.build_depot_nodes_return_constraints(graph)

        return None

    def _build_arc_constraints(self, graph):
        """ Build all of the arc formulation constraints """

        self.build_customer_visiting_constraints_directed(graph)
        self.build_station_visiting_constraints_directed(graph)
        self.build_flow_constraints_directed(graph)
        
        self.build_station_nodes_arrival_pwl_constraints(graph)
        self.build_station_nodes_departure_pwl_constraints(graph)
        
        self.build_station_nodes_duration_constraints(graph)
        self.build_energy_arrival_departure_constraints(graph)
        self.build_customer_energy_arcs_tracking_constraints(graph)
        self.build_energy_arcs_upper_bound_constraints(graph)
        self.build_station_arrival_energy_arcs_tracking_constraints(graph)
        self.build_station_departure_energy_arcs_tracking_constraints(graph)
        self.build_depot_energy_arcs_tracking_constraints(graph)
        self.build_customer_time_arcs_return_constraints(graph)        
        self.build_customer_time_arcs_tracking_constraints(graph)
        self.build_station_time_arcs_tracking_constraints(graph)        
        self.build_station_time_arcs_return_constraints(graph)
        self.build_energy_arcs_return_constraints(graph)
        self.build_time_arcs_return_constraints(graph)
        self.build_time_arcs_leave_constraints(graph)
        self.build_time_arcs_zero_constraints(graph)
        self.build_stations_clone_prime_constraints(graph)
        self.build_energy_valid_ineqaulities(graph)
        
        return None
        
    def _build_objective(self, graph):
        """ Build the objective function """

        self.set_time_objective(graph)

        return None

    def build_node_problem(self, graph):
        """ Build the node fomrulation problem """

        self._build_node_variables(graph)
        self._build_node_constraints(graph)
        self._build_objective(graph)

        return None

    def build_arc_problem(self, graph):
        """ Build the arc fomrulation problem """

        self._build_arc_variables(graph)
        self._build_arc_constraints(graph)
        self._build_objective(graph)

        return None
    
    def assume_hyperparameters(self):
        """ Assume some of the hyperparmaeters """

        self.maximum_travel_time = 10
        self.energy_consumption_rate = 0.125
        self.battery_energy_capacity = 16

    def check_hyperparameters(self):
        """ Make sure everything is set up correctly """

        pass

    def warn_hyperparmaters(self):
        """ Warn the user about the hyper-parameters """

        print(f"WARNING! Attribute maximum_travel_time set to {self.maximum_travel_time}")
        print(f"WARNING! Attribute energy_consumption_rate set to {self.energy_consumption_rate}")
        print(f"WARNING! Attribute battery_energy_capacity set to {self.battery_energy_capacity}")
        
    def _initialise_arc_time_variable_graph(self):
        """ Initialise the graph used to store the arc time variables """

        self.arc_time_graph = nx.DiGraph()
        
    def _initialise_arc_energy_variable_graph(self):
        """ Initialise the graph used to store the arc energy variables """

        self.arc_energy_graph = nx.DiGraph()
        
    def _initialise_node_energy_variable_graph(self):
        """ Initialise the graph used to store the node energy variables """

        self.energy_graph = nx.Graph()

    def _initialise_station_node_arrival_energy_variable_graph(self):
        """ Initialise the graph used to store the station node energy variables """

        self.station_arrival_energy_graph = nx.Graph()

    def _initialise_station_node_departure_energy_variable_graph(self):
        """ Initialise the graph used to store the station node energy variables """

        self.station_departure_energy_graph = nx.Graph()
        
    def _initialise_station_node_arrival_time_variable_graph(self):
        """ Initialise the graph used to store the station node time variables """

        self.station_arrival_time_graph = nx.Graph()
        
    def _initialise_station_node_departure_time_variable_graph(self):
        """ Initialise the graph used to store the station node time variables """

        self.station_departure_time_graph = nx.Graph()

    def _initialise_station_node_duration_variable_graph(self):
        """ Initialise the graph used to store the station node duration variables """

        self.station_duration_graph = nx.Graph()

    def _initialise_station_node_arrival_breakpoint_graph(self):
        """ Initialise the graph used to store the station node arrival breakpoint variables """

        self.station_node_arrival_breakpoint_graph = nx.Graph()
        
    def _initialise_station_node_departure_breakpoint_graph(self):
        """ Initialise the graph used to store the station node departure breakpoint variables """

        self.station_node_departure_breakpoint_graph = nx.Graph()
        
    def _initialise_station_node_arrival_coefficient_graph(self):
        """ Initialise the graph used to store the station node arrival coefficient variables """

        self.station_node_arrival_coefficient_graph = nx.Graph()
        
    def _initialise_station_node_departure_coefficient_graph(self):
        """ Initialise the graph used to store the station node departure coefficient variables """

        self.station_node_departure_coefficient_graph = nx.Graph()
        
        
    def name_node_energy_variable(self, node):
        """ Name the given node variable for tracking the energy """

        variable_name = f"y_{node}"

        return variable_name

    def build_node_time_variables(self, graph):
        """ Build the node time variables, using the store maximum travel time """

        time_limit = self.get_maximum_travel_time()
        
        for node in self.get_nodes(graph):
            self.build_node_time_variable(node, time_limit)

        return None

    def build_node_energy_variable(self, node, energy_capacity):
        """ Build the variable for tracking the energy at the given node """
        
        variable_name = self.name_node_energy_variable(node)
        variable = self.set_continuous_variable(
            lower_bound=0,
            upper_bound=energy_capacity,
            name=variable_name,
        )
        self.store_node_energy_variable(node, variable)
        
        return None

    def build_node_energy_variables(self, graph):
        """ Build fuel tracking variables for all the nodes """

        energy_capacity = self.get_battery_energy_capacity()
        
        for node in self.get_nodes(graph):
            self.build_node_energy_variable(node, energy_capacity)

        return None

    def store_node_energy_variable(self, node, energy_variable):
        """ Store the node energy variable for the given node """

        self.energy_graph.add_nodes_from([node], variable=energy_variable)

        return None

    def name_arc_energy_variable(self, first_node, second_node):
        """ Name the given arc variable for tracking the energy """

        variable_name = f"y_{first_node},{second_node}"

        return variable_name

    def build_arc_energy_variable(self, first_node, second_node, energy_capacity):
        """ Build the variable for tracking the energy at the given arc """
        
        variable_name = self.name_arc_energy_variable(first_node, second_node)
        variable = self.set_continuous_variable(
            lower_bound=0,
            upper_bound=energy_capacity,
            name=variable_name,
        )
        self.store_arc_energy_variable(first_node, second_node, variable)
        
        return None

    def build_arc_energy_variables(self, graph):
        """ Build energy tracking variables for all the arcs """

        energy_capacity = self.get_battery_energy_capacity()
        
        for edge in graph.edges:
            self.build_arc_energy_variable(edge[0], edge[1], energy_capacity)

        return None
    
    def store_arc_energy_variable(self, first_node, second_node, energy_variable):
        """ Store the arc energy variable for the given arc """

        self.arc_energy_graph.add_edges_from([(first_node, second_node)], variable=energy_variable)

        return None
    
    def name_arc_time_variable(self, first_node, second_node):
        """ Name the given arc variable for tracking the time """

        variable_name = f"t_{first_node},{second_node}"

        return variable_name

    def build_arc_time_variable(self, first_node, second_node, time_limit):
        """ Build the variable for tracking the time at the given arc """

        variable_name = self.name_arc_time_variable(first_node, second_node)
        variable = self.set_continuous_variable(
            lower_bound=0,
            upper_bound=time_limit,
            name=variable_name,
        )
        self.store_arc_time_variable(first_node, second_node, variable)
        
        return None

    def build_arc_time_variables(self, graph):
        """ Build time tracking variables for all the arcs """

        time_limit = self.get_maximum_travel_time()
        
        for edge in graph.edges:
            self.build_arc_time_variable(edge[0], edge[1], time_limit)

        return None
    
    def store_arc_time_variable(self, first_node, second_node, time_variable):
        """ Store the arc time variable for the given arc """

        self.arc_time_graph.add_edges_from([(first_node, second_node)], variable=time_variable)

        return None
    
    def name_station_node_arrival_energy_variable(self, node):
        """ Name the given station node variable for tracking the energy """

        variable_name = f"q_{node}"

        return variable_name

    def build_station_node_arrival_energy_variable(self, node, energy_capacity):
        """ Build the variable for tracking the energy at the given station node """
        
        variable_name = self.name_station_node_arrival_energy_variable(node)
        variable = self.set_continuous_variable(
            lower_bound=0,
            upper_bound=energy_capacity,
            name=variable_name,
        )
        self.store_station_node_arrival_energy_variable(node, variable)
        
        return None

    def build_station_node_arrival_energy_variables(self, graph):
        """ Build fuel tracking variables for all the station nodes """

        energy_capacity = self.get_battery_energy_capacity()
        
        for station_node in self.get_stations(graph):
            self.build_station_node_arrival_energy_variable(station_node, energy_capacity)

        return None

    def store_station_node_arrival_energy_variable(self, station_node, energy_variable):
        """ Store the station node energy variable for the given node """

        self.station_arrival_energy_graph.add_nodes_from([station_node], variable=energy_variable)

        return None

    def name_station_node_departure_energy_variable(self, node):
        """ Name the given station node variable for tracking the energy """

        variable_name = f"o_{node}"

        return variable_name

    def build_station_node_departure_energy_variable(self, node, energy_capacity):
        """ Build the variable for tracking the energy at the given station node """
        
        variable_name = self.name_station_node_departure_energy_variable(node)
        variable = self.set_continuous_variable(
            lower_bound=0,
            upper_bound=energy_capacity,
            name=variable_name,
        )
        self.store_station_node_departure_energy_variable(node, variable)
        
        return None

    def build_station_node_departure_energy_variables(self, graph):
        """ Build fuel tracking variables for all the station nodes """

        energy_capacity = self.get_battery_energy_capacity()
        
        for station_node in self.get_stations(graph):
            self.build_station_node_departure_energy_variable(station_node, energy_capacity)

        return None

    def store_station_node_departure_energy_variable(self, station_node, energy_variable):
        """ Store the station node energy variable for the given node """

        self.station_departure_energy_graph.add_nodes_from([station_node], variable=energy_variable)

        return None

    def name_station_node_arrival_time_variable(self, node):
        """ Name the given station node variable for tracking the time """

        variable_name = f"s_{node}"

        return variable_name

    def build_station_node_arrival_time_variable(self, node, time_limit):
        """ Build the variable for tracking the time at the given station node """
        
        variable_name = self.name_station_node_arrival_time_variable(node)
        variable = self.set_continuous_variable(
            lower_bound=0,
            upper_bound=time_limit,
            name=variable_name,
        )
        self.store_station_node_arrival_time_variable(node, variable)
        
        return None

    def build_station_node_arrival_time_variables(self, graph):
        """ Build time tracking variables for all the station nodes """

        time_limit = self.get_maximum_travel_time()
        
        for station_node in self.get_stations(graph):
            self.build_station_node_arrival_time_variable(station_node, time_limit)

        return None

    def store_station_node_arrival_time_variable(self, station_node, time_variable):
        """ Store the station node time variable for the given node """

        self.station_arrival_time_graph.add_nodes_from([station_node], variable=time_variable)

        return None
    
    def name_station_node_departure_time_variable(self, node):
        """ Name the given station node variable for tracking the time """

        variable_name = f"d_{node}"

        return variable_name

    def build_station_node_departure_time_variable(self, node, time_limit):
        """ Build the variable for tracking the energy at the given station node """
        
        variable_name = self.name_station_node_departure_time_variable(node)
        variable = self.set_continuous_variable(
            lower_bound=0,
            upper_bound=time_limit,
            name=variable_name,
        )
        self.store_station_node_departure_time_variable(node, variable)
        
        return None

    def build_station_node_departure_time_variables(self, graph):
        """ Build time tracking variables for all the station nodes """

        time_limit = self.get_maximum_travel_time()
        
        for station_node in self.get_stations(graph):
            self.build_station_node_departure_time_variable(station_node, time_limit)

        return None

    def store_station_node_departure_time_variable(self, station_node, time_variable):
        """ Store the station node energy variable for the given node """

        self.station_departure_time_graph.add_nodes_from([station_node], variable=time_variable)

        return None

    def name_station_node_duration_variable(self, node):
        """ Name the given station node variable for tracking the charging duration """

        variable_name = f"^_{node}"

        return variable_name

    def build_station_node_duration_variable(self, node, time_limit):
        """ Build the variable for tracking the charging duration at the given station node """
        
        variable_name = self.name_station_node_duration_variable(node)
        variable = self.set_continuous_variable(
            lower_bound=0,
            upper_bound=time_limit,
            name=variable_name,
        )
        self.store_station_node_duration_variable(node, variable)
        
        return None

    def build_station_node_duration_variables(self, graph):
        """ Build charging duration tracking variables for all the station nodes """

        time_limit = self.get_maximum_travel_time()
        
        for station_node in self.get_stations(graph):
            self.build_station_node_duration_variable(station_node, time_limit)

        return None

    def store_station_node_duration_variable(self, station_node, duration_variable):
        """ Store the station node charging duration variable for the given node """

        self.station_duration_graph.add_nodes_from([station_node], variable=duration_variable)

        return None

    def build_station_node_arrival_lambda_pwl_variables(self, graph, station_node):
        """ Build energy and charge tracking variables for arrival at the given station """

        self.reset_breakpoints(graph, station_node)
        
        _ = self.build_variables_lambda(
            lambda_stem="a_",
            binary_stem="z_",
            binary_storage = self.station_node_arrival_breakpoint_graph,
            lambda_storage = self.station_node_arrival_coefficient_graph,
            base_index=station_node,
        )
        
        return None

    def build_station_node_arrival_delta_pwl_variables(self, graph, station_node):
        """ Build energy and charge tracking variables for arrival at the given station """

        self.reset_breakpoints(graph, station_node)
        
        _ = self.build_variables_delta(
            delta_stem="a_",
            binary_stem="z_",
            binary_storage = self.station_node_arrival_breakpoint_graph,
            delta_storage = self.station_node_arrival_coefficient_graph,
            base_index=station_node,
        )
        
        return None
    
    def build_station_nodes_arrival_pwl_variables(self, graph):
        """ Build all tracking variables for arrival at all station nodes """

        if self.pwl_model == "lambda" or self.pwl_model == "sos":
            variable_builder = self.build_station_node_arrival_lambda_pwl_variables
        elif self.pwl_model == "delta":
            variable_builder = self.build_station_node_arrival_delta_pwl_variables
        
        for station_node in self.get_stations(graph):
            variable_builder(graph, station_node)

        return None
    
    def build_station_node_departure_lambda_pwl_variables(self, graph, station_node):
        """ Build energy and charge tracking variables for departure from at the given station """

        self.reset_breakpoints(graph, station_node)
        
        _ = self.build_variables_lambda(
            lambda_stem="h_",
            binary_stem="w_",
            binary_storage = self.station_node_departure_breakpoint_graph,
            lambda_storage = self.station_node_departure_coefficient_graph,
            base_index=station_node,
        )
        
        return None
    
    def build_station_node_departure_delta_pwl_variables(self, graph, station_node):
        """ Build energy and charge tracking variables for departure from at the given station """

        self.reset_breakpoints(graph, station_node)
        
        _ = self.build_variables_delta(
            delta_stem="h_",
            binary_stem="w_",
            binary_storage = self.station_node_departure_breakpoint_graph,
            delta_storage = self.station_node_departure_coefficient_graph,
            base_index=station_node,
        )
        
        return None
    
    def build_station_nodes_departure_pwl_variables(self, graph):
        """ Build all tracking variables for departure from all station nodes """

        if self.pwl_model == "lambda" or self.pwl_model == "sos":
            variable_builder = self.build_station_node_departure_lambda_pwl_variables
        elif self.pwl_model == "delta":
            variable_builder = self.build_station_node_departure_delta_pwl_variables
        
        for station_node in self.get_stations(graph):
            variable_builder(graph, station_node)

        return None
    
    def store_station_node_departure_coefficient_variable(self, station_node, coefficient, variable):
        """ Store the station node charging duration variable for the given node """

        attribute_dict = {f"variable_{coefficient}":variable} 
        self.station_node_departure_coefficient_graph.add_nodes_from([(station_node, attribute_dict)])

        return None

    def set_maximum_charging_time(self, graph):
        """ Set the maximum charging time """

        station_nodes = self.get_stations(graph)
        maximum_charging_times = [max(self.get_time_breakpoints(graph, station_node))
                                  for station_node in station_nodes]
        maximum_charging_time = max(maximum_charging_times)
        self.maximum_charging_time = maximum_charging_time

        return None

    def get_maximum_charging_time(self):
        """ Get the maximum charging time """

        maximum_charging_time = self.maximum_charging_time

        return maximum_charging_time
        
    def get_maximum_travel_time(self):
        """ Get the maximum travel time """

        maximum_travel_time = self.maximum_travel_time

        return maximum_travel_time

    def get_energy_consumption_rate(self):
        """ Get the energy consumption rate """

        energy_consumption_rate = self.energy_consumption_rate

        return energy_consumption_rate

    def get_average_velocity(self, graph):
        """ Get the average velocity of a vehicle in the given graph. Assume homogeneity. """

        first_vehicle_info = graph.graph["fleet"]["vehicle_0"]
        average_velocity = first_vehicle_info["speed_factor"]

        return average_velocity

    def distance_to_duration(self, distance, speed):
        """ Convert the given travel distance to a duration. Assume metric units. """
        
        duration = distance / speed

        return duration

    def distances_to_durations(self, distances, speed):
        """ Convert the given travel distance to a duration. Assume metric units. """
        
        durations =  [distance / speed for distance in distances]

        return distances

    def get_travel_time(self, graph, first_node, second_node):
        """ Get the travel time between the two given nodes """

        velocity = self.get_average_velocity(graph)
        weight = graph[first_node][second_node]["weight"]
        time = self.distance_to_duration(weight, velocity)

        return time
    
    def get_node_service_time(self, graph, node):
        """ Get the service time for a given node """

        service_time = graph.nodes[node]["service_time"]

        return service_time

    def get_node_service_times(self, graph, nodes):
        """ Get the service times for the given nodes """

        service_times = [self.get_node_service_time(graph, node) for node in nodes]
        
        return service_times
    
    def get_battery_energy_capacity(self):
        """ Get the battery energy capacity """

        battery_energy_capacity = self.battery_energy_capacity

        return battery_energy_capacity

    def get_energy_consumption(self, graph, first_node, second_node):
        """ Get the energy consumption between the two given nodes """

        energy_consumption_rate = self.get_energy_consumption_rate()
        weight = graph[first_node][second_node]["weight"]
        energy_consumption = weight * energy_consumption_rate

        return energy_consumption

    def get_energy_consumptions(self, graph, tuples):
        """ Get the energy consumption between the given nodes """

        energy_consumptions = [self.get_energy_consumption(graph, arc[0], arc[1]) for arc in tuples]

        return energy_consumptions
    
    def get_travel_times(self, graph, tuples):
        """ Get the travel times between the given nodes """

        travel_times = [self.get_travel_time(graph, arc[0], arc[1]) for arc in tuples]

        return travel_times
    
    def get_charging_functions(self, graph):
        """ Get the types of charging function that are specified """

        charging_functions = graph.graph["fleet"]["functions_0"]["charging_functions"]

        return charging_functions

    def get_charging_station_type(self, graph, station_node):
        """ Get the charging station type for the given station node """

        charging_station_type = graph.nodes[station_node]["cs_type"]

        return charging_station_type

    def get_charging_function(self, graph, station_node):
        """ Get the charging function for the given station node """

        charging_functions = self.get_charging_functions(graph)
        charging_station_type = self.get_charging_station_type(graph, station_node)
        charging_function = charging_functions[charging_station_type]["breakpoints"]

        return charging_function

    def get_energy_breakpoints(self, graph, station_node):
        """ Get the energy breakpoints for the charging function at the given node """

        charging_function = self.get_charging_function(graph, station_node)
        energy_breakpoints = [item["battery_level"] for item in charging_function]
        energy_breakpoints = np.sort(energy_breakpoints).tolist()
        # Only in the files are the energies stored in Watts.
        energy_breakpoints = [item / 1000 for item in energy_breakpoints]

        return energy_breakpoints

    def get_time_breakpoints(self, graph, station_node):
        """ Get the time breakpoints for the charging function at the given node """
        
        charging_function = self.get_charging_function(graph, station_node)
        time_breakpoints = [item["charging_time"] for item in charging_function]
        time_breakpoints = np.sort(time_breakpoints).tolist()
        
        return time_breakpoints

    def reset_breakpoints(self, graph, station_node):
        """ Reset the current stored breakpoints  """

        # On reflection, this is a terrible way of doing it
        time_coefficients = self.get_time_breakpoints(graph, station_node)
        energy_coefficients = self.get_energy_breakpoints(graph, station_node)
        self.set_horizontal_breakpoints(time_coefficients)
        self.set_vertical_breakpoints(energy_coefficients)

        return None
    
    def get_node_energy_variable_from_storage(self, node):
        """ Get the variable from the storage dict using a node as a key """

        variable = self.energy_graph.nodes[node]["variable"]
        
        return variable


    def get_node_energy_variables_from_storage(self, nodes):
        """ Get the variable from the storage dict using a node as a key """

        variables = [self.energy_graph.nodes[node]["variable"] for node in nodes]
        
        return variables
    
    def get_arc_time_variable_from_storage(self, tuple):
        """ Get the variable from the storage dict using a tuple (edge or arc) as a key """

        variable = self.arc_time_graph[tuple[0]][tuple[1]]["variable"]

        return variable
    
    def get_arc_energy_variable_from_storage(self, tuple):
        """ Get the variable from the storage dict using a tuple (edge or arc) as a key """

        variable = self.arc_energy_graph[tuple[0]][tuple[1]]["variable"]

        return variable
    
    def get_arc_time_variables_from_storage(self, tuples):
        """ Get the variables from the storage dict using tuples (edges or arcs) as keys """

        variables = [self.arc_time_graph[tuple[0]][tuple[1]]["variable"] for tuple in tuples]

        return variables
    
    def get_arc_energy_variables_from_storage(self, tuples):
        """ Get the variables from the storage dict using tuples (edges or arcs) as keys """

        variables = [self.arc_energy_graph[tuple[0]][tuple[1]]["variable"] for tuple in tuples]

        return variables
    
    def get_station_node_arrival_energy_variable_from_storage(self, station_node):
        """ Get the variable from the storage dict using a node as a key """

        variable = self.station_arrival_energy_graph.nodes[station_node]["variable"]
        
        return variable

    def get_station_node_departure_energy_variable_from_storage(self, station_node):
        """ Get the variable from the storage dict using a node as a key """

        variable = self.station_departure_energy_graph.nodes[station_node]["variable"]
        
        return variable

    def get_station_node_arrival_time_variable_from_storage(self, station_node):
        """ Get the variable from the storage dict using a node as a key """

        variable = self.station_arrival_time_graph.nodes[station_node]["variable"]
        
        return variable

    def get_station_node_departure_time_variable_from_storage(self, station_node):
        """ Get the variable from the storage dict using a node as a key """

        variable = self.station_departure_time_graph.nodes[station_node]["variable"]
        
        return variable
    
    def get_station_node_duration_variable_from_storage(self, station_node):
        """ Get the variable from the storage dict using a node as a key """

        variable = self.station_duration_graph.nodes[station_node]["variable"]
        
        return variable

    def get_station_node_arrival_breakpoint_variable_from_storage(self, station_node, breakpoint):
        """ Get the variable from the storage dict using a node and breakpoint as keys """

        node = self.station_node_arrival_breakpoint_graph.nodes[station_node]
        variable = node[breakpoint]
        
        return variable

    def get_station_node_departure_breakpoint_variable_from_storage(self, station_node, breakpoint):
        """ Get the variable from the storage dict using a node and breakpoint as keys """

        node = self.station_node_departure_breakpoint_graph.nodes[station_node]
        variable = node[breakpoint]
        
        return variable

    def get_station_node_arrival_coefficient_variable_from_storage(self, station_node, coefficient):
        """ Get the variable from the storage dict using a node and breakpoint as keys """

        node = self.station_node_arrival_coefficient_graph.nodes[station_node]
        variable = node[coefficient]
        
        return variable

    def get_station_node_departure_coefficient_variable_from_storage(self, station_node, coefficient):
        """ Get the variable from the storage dict using a node and breakpoint as keys """

        node = self.station_node_departure_coefficient_graph.nodes[station_node]
        variable = node[coefficient]
        
        return variable

    def set_time_objective(self, graph):
        """ Set a time-measured objective function """

        travel_arcs = list(graph.edges)
        station_nodes = self.get_stations(graph)
        
        travel_variables = self.get_travel_variables_from_storage(travel_arcs)
        travel_times = [self.get_travel_time(graph, arc[0], arc[1]) for arc in travel_arcs]
        travel_time_sum = [time * variable
                           for (time, variable) in zip(travel_times, travel_variables)]
        travel_time_sum = self.sum_items(travel_time_sum)
        duration_sum = [self.get_station_node_duration_variable_from_storage(node)
                        for node in station_nodes]
        duration_sum = self.sum_items(duration_sum) + travel_time_sum
        _ = self.set_objective(duration_sum)
        
        return None
    
    def build_customer_energy_node_tracking_constraint(self, graph, first_node, customer_node):
        """ Build a constraint tracking the energy moving from any node to a customer node """

        travel_variable = self.get_travel_variable_from_storage((first_node, customer_node))
        first_node_energy = self.get_node_energy_variable_from_storage(first_node)
        customer_node_energy = self.get_node_energy_variable_from_storage(customer_node)

        battery_capacity = self.get_battery_energy_capacity()
        energy_consumption = self.get_energy_consumption(graph, first_node, customer_node)

        lhs = energy_consumption * travel_variable - (1 - travel_variable) * battery_capacity
        mid = first_node_energy - customer_node_energy
        rhs = energy_consumption * travel_variable + (1 - travel_variable) * battery_capacity

        first_name = f"Lower energy constraint for nodes {first_node} and {customer_node}"
        second_name = f"Upper energy constraint for nodes {first_node} and {customer_node}"
        _ = self.set_constraint(lhs, mid, "<=", first_name)
        _ = self.set_constraint(mid, rhs, "<=", second_name)

        return None

    def build_customer_energy_node_tracking_constraints(self, graph):
        """ Build all node to customer node energy tracking constraints """

        nodes = self.get_nodes(graph)
        customer_nodes = self.get_customers(graph)

        for node in nodes:
            for customer_node in customer_nodes:
                if node != customer_node:
                    self.build_customer_energy_node_tracking_constraint(graph, node, customer_node)

        return None

    def build_station_energy_node_tracking_constraint(self, graph, first_node, station_node):
        """ Build a constraint tracking the energy moving from any node to a station node """

        travel_variable = self.get_travel_variable_from_storage((first_node, station_node))
        first_node_energy = self.get_node_energy_variable_from_storage(first_node)
        arrival_energy = self.get_station_node_arrival_energy_variable_from_storage(station_node)
        battery_capacity = self.get_battery_energy_capacity()
        energy_consumption = self.get_energy_consumption(graph, first_node, station_node)

        lhs = energy_consumption * travel_variable - (1 - travel_variable) * battery_capacity
        mid = first_node_energy - arrival_energy
        rhs = energy_consumption * travel_variable + (1 - travel_variable) * battery_capacity

        first_name = f"Lower energy constraint for nodes {first_node} and {station_node}"
        second_name = f"Upper energy constraint for nodes {first_node} and {station_node}"
        _ = self.set_constraint(lhs, mid, "<=", first_name)
        _ = self.set_constraint(mid, rhs, "<=", second_name)

        return None

    def build_station_energy_node_tracking_constraints(self, graph):
        """ Build all node to station node energy tracking constraints """

        nodes = self.get_nodes(graph)
        station_nodes = self.get_stations(graph)

        for node in nodes:
            for station_node in station_nodes:
                if node != station_node:
                    self.build_station_energy_node_tracking_constraint(graph, node, station_node)

        return None

    def build_depot_energy_node_tracking_constraint(self, graph, other_node, depot_node):
        """ Make sure that there is enough energy to get back to the given depot node """

        node_energy_variable = self.get_node_energy_variable_from_storage(other_node)
        energy_consumption = self.get_energy_consumption(graph, other_node, depot_node)
        travel_variable = self.get_travel_variable_from_storage((other_node, depot_node))

        lhs = node_energy_variable
        rhs = energy_consumption * travel_variable
        name = f"Depot energy constraint between {other_node} and {depot_node}"
        _ = self.set_constraint(lhs, rhs, ">=", name)

        return None

    def build_depot_energy_node_tracking_constraints(self, graph):
        """ Make sure that there is enough energy to get back to the given depot node """

        nodes = self.get_nodes(graph)
        depot_nodes = self.get_depots(graph)
        for node in nodes:
            for depot_node in depot_nodes:
                if node != depot_node:
                    self.build_depot_energy_node_tracking_constraint(graph, node, depot_node)

        return None

    def build_station_departure_energy_node_reset_constraint(self, graph, node):
        """ Reset the general energy variable for the given station node to the departure energy """

        energy_variable = self.get_node_energy_variable_from_storage(node)
        departure_variable = self.get_station_node_departure_energy_variable_from_storage(node)

        lhs = energy_variable
        rhs = departure_variable
        name = f"Departure energy reset constraint for station {node}"
        _ = self.set_constraint(lhs, rhs, "==", name)

        return None

    def build_station_departure_energy_node_reset_constraints(self, graph):
        """g Reset the general energy variable for all station nodes to the departure energy """

        station_nodes = self.get_stations(graph)

        for station_node in station_nodes:
            self.build_station_departure_energy_node_reset_constraint(graph, station_node)
        
        return None

    def build_depot_energy_initialisation_constraint(self, graph, depot_node):
        """ Make sure the battery is fully charged leaving the given depot node """

        energy_variable = self.get_node_energy_variable_from_storage(depot_node)
        battery_capacity = self.get_battery_energy_capacity()

        lhs = energy_variable
        rhs = battery_capacity
        name = f"Depot energy initialisation constraint for depot {depot_node}"
        _ = self.set_constraint(lhs, rhs, "==", name)

        return None

    def build_depot_energy_initialisation_constraints(self, graph):
        """ Make sure battery is fully charged leaving any depot node """

        depot_nodes = self.get_depots(graph)

        for depot_node in depot_nodes:
            self.build_depot_energy_initialisation_constraint(graph, depot_node)
        
        return None

    def build_energy_arrival_departure_constraint(self, graph, node):
        """ Make sure the departure energy for the station is never less than the arrival energy """

        arrival_energy = self.get_station_node_arrival_energy_variable_from_storage(node)
        departure_energy = self.get_station_node_departure_energy_variable_from_storage(node)

        lhs = arrival_energy
        rhs = departure_energy
        name = f"Arrival/departure node energy constraint for station {node}"
        _ = self.set_constraint(lhs, rhs, "<=", name)

        return None

    def build_energy_arrival_departure_constraints(self, graph):
        """ Make sure the departure energy for a station is never less than the arrival energy """

        station_nodes = self.get_stations(graph)

        for station_node in station_nodes:
            self.build_energy_arrival_departure_constraint(graph, station_node)
        
        return None

    def build_station_node_arrival_lambda_pwl_constraints(self, graph, station_node):
        """ Build station node arrival constraints for the lambda PWL model """

        self.reset_breakpoints(graph, station_node)
        
        incident_arcs = self.get_outward_incident_arcs(graph, station_node)
        travel_variables = self.get_travel_variables_from_storage(incident_arcs)
        energy_variable = self.get_station_node_arrival_energy_variable_from_storage(station_node)
        time_variable = self.get_station_node_arrival_time_variable_from_storage(station_node)

        self.build_constraints_lambda(
            lambda_storage=self.station_node_arrival_coefficient_graph,
            binary_storage=self.station_node_arrival_breakpoint_graph,
            equality_rhs=self.sum_items(travel_variables),
            base_index=station_node,
            is_convex=False,
        )
        self.build_vertical_lambda_combination_constraint(
            lambda_storage=self.station_node_arrival_coefficient_graph,
            rhs_variable=energy_variable,
            base_index=station_node
        )
        self.build_horizontal_lambda_combination_constraint(
            lambda_storage=self.station_node_arrival_coefficient_graph,
            rhs_variable=time_variable,
            base_index=station_node
        )

        return None

    def build_station_node_arrival_sos_pwl_constraints(self, graph, station_node):
        """ Build station node arrival constraints for the lambda PWL model """

        self.reset_breakpoints(graph, station_node)
        
        incident_arcs = self.get_outward_incident_arcs(graph, station_node)
        travel_variables = self.get_travel_variables_from_storage(incident_arcs)
        energy_variable = self.get_station_node_arrival_energy_variable_from_storage(station_node)
        time_variable = self.get_station_node_arrival_time_variable_from_storage(station_node)

        self.build_constraints_lambda(
            lambda_storage=self.station_node_arrival_coefficient_graph,
            binary_storage=self.station_node_arrival_breakpoint_graph,
            equality_rhs=self.sum_items(travel_variables),
            base_index=station_node,
            is_convex=False,
        )
        self.build_vertical_lambda_combination_constraint(
            lambda_storage=self.station_node_arrival_coefficient_graph,
            rhs_variable=energy_variable,
            base_index=station_node
        )
        self.build_horizontal_lambda_combination_constraint(
            lambda_storage=self.station_node_arrival_coefficient_graph,
            rhs_variable=time_variable,
            base_index=station_node
        )

        return None    
    
    def build_station_node_arrival_delta_pwl_constraints(self, graph, station_node):
        """ Build station node arrival constraints for the delta PWL model """

        self.reset_breakpoints(graph, station_node)
        
        incident_arcs = self.get_outward_incident_arcs(graph, station_node)
        travel_variables = self.get_travel_variables_from_storage(incident_arcs)
        energy_variable = self.get_station_node_arrival_energy_variable_from_storage(station_node)
        time_variable = self.get_station_node_arrival_time_variable_from_storage(station_node)

        self.build_constraints_delta(
            delta_storage=self.station_node_arrival_coefficient_graph,
            binary_storage=self.station_node_arrival_breakpoint_graph,
            equality_rhs=time_variable,
            base_index=station_node,
            is_convex=False,
        )
        self.build_vertical_delta_combination_constraint(
            delta_storage=self.station_node_arrival_coefficient_graph,
            rhs_variable=energy_variable,
            base_index=station_node
        )

        return None
    
    def build_station_nodes_arrival_pwl_constraints(self, graph):
        """ Build station node arrival constraints for the PWL model for all station nodes """

        if self.pwl_model == "lambda":
            constraint_builder = self.build_station_node_arrival_lambda_pwl_constraints
        elif self.pwl_model == "sos":
            constraint_builder = self.build_station_node_arrival_sos_pwl_constraints
        elif self.pwl_model == "delta":
            constraint_builder = self.build_station_node_arrival_delta_pwl_constraints

        station_nodes = self.get_stations(graph)
        for station_node in station_nodes:
            constraint_builder(graph, station_node)

        return None
    
    def build_station_node_departure_lambda_pwl_constraints(self, graph, station_node):
        """ Build station node departure constraints for the lambda PWL model """

        self.reset_breakpoints(graph, station_node)
        
        incident_arcs = self.get_outward_incident_arcs(graph, station_node)
        travel_variables = self.get_travel_variables_from_storage(incident_arcs)
        energy_variable = self.get_station_node_departure_energy_variable_from_storage(station_node)
        time_variable = self.get_station_node_departure_time_variable_from_storage(station_node)

        self.build_constraints_lambda(
            lambda_storage=self.station_node_departure_coefficient_graph,
            binary_storage=self.station_node_departure_breakpoint_graph,
            equality_rhs=self.sum_items(travel_variables),
            base_index=station_node,
            is_convex=False,
        )
        self.build_vertical_lambda_combination_constraint(
            lambda_storage=self.station_node_departure_coefficient_graph,
            rhs_variable=energy_variable,
            base_index=station_node
        )
        self.build_horizontal_lambda_combination_constraint(
            lambda_storage=self.station_node_departure_coefficient_graph,
            rhs_variable=time_variable,
            base_index=station_node
        )

        return None

    def build_station_node_departure_sos_pwl_constraints(self, graph, station_node):
        """ Build station node departure constraints for the lambda PWL model """

        self.reset_breakpoints(graph, station_node)
        
        incident_arcs = self.get_outward_incident_arcs(graph, station_node)
        travel_variables = self.get_travel_variables_from_storage(incident_arcs)
        energy_variable = self.get_station_node_departure_energy_variable_from_storage(station_node)
        time_variable = self.get_station_node_departure_time_variable_from_storage(station_node)

        self.build_constraints_lambda(
            lambda_storage=self.station_node_departure_coefficient_graph,
            binary_storage=self.station_node_departure_breakpoint_graph,
            equality_rhs=self.sum_items(travel_variables),
            base_index=station_node,
            is_convex=True,
        )
        self.build_sos_lambda_constraint(
            lambda_storage=self.station_node_departure_coefficient_graph,
            base_index=station_node
        )
        self.build_vertical_lambda_combination_constraint(
            lambda_storage=self.station_node_departure_coefficient_graph,
            rhs_variable=energy_variable,
            base_index=station_node
        )
        self.build_horizontal_lambda_combination_constraint(
            lambda_storage=self.station_node_departure_coefficient_graph,
            rhs_variable=time_variable,
            base_index=station_node
        )

        return None
    
    def build_station_node_departure_delta_pwl_constraints(self, graph, station_node):
        """ Build station node departure constraints for the delta PWL model """

        self.reset_breakpoints(graph, station_node)
        
        incident_arcs = self.get_outward_incident_arcs(graph, station_node)
        travel_variables = self.get_travel_variables_from_storage(incident_arcs)
        energy_variable = self.get_station_node_departure_energy_variable_from_storage(station_node)
        time_variable = self.get_station_node_departure_time_variable_from_storage(station_node)

        self.build_constraints_delta(
            delta_storage=self.station_node_departure_coefficient_graph,
            binary_storage=self.station_node_departure_breakpoint_graph,
            equality_rhs=time_variable,
            base_index=station_node,
            is_convex=False,
        )
        self.build_vertical_delta_combination_constraint(
            delta_storage=self.station_node_departure_coefficient_graph,
            rhs_variable=energy_variable,
            base_index=station_node
        )

        return None
    
    def build_station_nodes_departure_pwl_constraints(self, graph):
        """ Build station node departure constraints for the PWL model for all station nodes """

        if self.pwl_model == "lambda":
            constraint_builder = self.build_station_node_departure_lambda_pwl_constraints
        elif self.pwl_model == "sos":
            constraint_builder = self.build_station_node_departure_sos_pwl_constraints
        elif self.pwl_model == "delta":
            constraint_builder = self.build_station_node_departure_delta_pwl_constraints

        station_nodes = self.get_stations(graph)
        for station_node in station_nodes:
            constraint_builder(graph, station_node)

        return None
    
    def build_station_node_duration_constraint(self, graph, station_node):
        """ Set the duration as difference between arrival and depature times for the station """

        arrival_time = self.get_station_node_arrival_time_variable_from_storage(station_node)
        departure_time = self.get_station_node_departure_time_variable_from_storage(station_node)
        duration = self.get_station_node_duration_variable_from_storage(station_node)
        
        lhs = duration
        rhs = departure_time - arrival_time
        name = f"Charging duration constraint for station {station_node}"
        _ = self.set_constraint(lhs, rhs, "==", name)

        return None

    def build_station_nodes_duration_constraints(self, graph):
        """ Set the duration constraints for all station nodes """

        station_nodes = self.get_stations(graph)

        for station_node in station_nodes:
            self.build_station_node_duration_constraint(graph, station_node)
        
        return None

    def build_customer_node_time_tracking_constraint(self, graph, other_node, customer_node):
        """ Track the time of arrival between the given node and customer node """

        other_node_time = self.get_node_time_variable_from_storage(other_node)
        customer_node_time = self.get_node_time_variable_from_storage(customer_node)
        travel_variable = self.get_travel_variable_from_storage((other_node, customer_node))

        travel_time = self.get_travel_time(graph, other_node, customer_node)
        maximum_travel_time = self.get_maximum_travel_time()
        customer_service_time = self.get_node_service_time(graph, customer_node)

        lhs = other_node_time + (travel_time + customer_service_time) * travel_variable
        lhs = lhs - maximum_travel_time * (1 - travel_variable)
        rhs = customer_node_time
        constraint_name = f"Customer time tracking constraint between {other_node} and {customer_node}"
        _ = self.set_constraint(lhs, rhs, "<=", name=constraint_name)
        
        return None

    def build_customer_nodes_time_tracking_constraints(self, graph):
        """ Track the time of arrival between all nodes and customer nodes """

        other_nodes = self.get_nodes(graph)
        customer_nodes = self.get_customers(graph)

        for other_node in other_nodes:
            for customer_node in customer_nodes:
                if other_node != customer_node:
                    _ = self.build_customer_node_time_tracking_constraint(
                        graph,
                        other_node,
                        customer_node
                    )
        return None
    
    def build_station_node_time_tracking_constraint(self, graph, other_node, station_node):
        """ Track the time of arrival between the given node and station node """

        if not hasattr(self, "maximum_charging_time"):
            self.set_maximum_charging_time(graph)
        
        other_node_time = self.get_node_time_variable_from_storage(other_node)
        station_node_time = self.get_node_time_variable_from_storage(station_node)
        travel_variable = self.get_travel_variable_from_storage((other_node, station_node))
        duration_variable = self.get_station_node_duration_variable_from_storage(station_node)

        travel_time = self.get_travel_time(graph, other_node, station_node)
        maximum_travel_time = self.get_maximum_travel_time()
        maximum_charging_time = self.get_maximum_charging_time()
        
        lhs = other_node_time + duration_variable + travel_time * travel_variable
        lhs = lhs - (maximum_charging_time + maximum_travel_time) * (1 - travel_variable)
        rhs = station_node_time
        constraint_name = f"Station time tracking constraint between {other_node} and {station_node}"
        _ = self.set_constraint(lhs, rhs, "<=", name=constraint_name)
        
        return None

    def build_station_nodes_time_tracking_constraints(self, graph):
        """ Track the time of arrival between all nodes and station nodes """

        other_nodes = self.get_nodes(graph)
        station_nodes = self.get_stations(graph)

        for other_node in other_nodes:
            for station_node in station_nodes:
                if other_node != station_node:
                    _ = self.build_station_node_time_tracking_constraint(
                        graph,
                        other_node,
                        station_node
                    )
        return None

    def build_depot_node_return_constraint(self, graph, other_node, depot_node):
        """ Build constraint to make sure that we can return to the given depot in time """

        time_variable = self.get_node_time_variable_from_storage(other_node)

        travel_time = self.get_travel_time(graph, other_node, depot_node)
        maximum_travel_time = self.get_maximum_travel_time()

        lhs = time_variable + travel_time
        rhs = maximum_travel_time
        constraint_name = f"Return constraint between node {other_node} and depot {depot_node}"
        _ = self.set_constraint(lhs, rhs, "<=", constraint_name)

        return None

    def build_depot_nodes_return_constraints(self, graph):
        """ Build a node return constraint for all node and depots """

        other_nodes = self.get_nodes(graph)
        depot_nodes = self.get_depots(graph)

        for other_node in other_nodes:
            for depot_node in depot_nodes:
                if other_node != depot_node:
                    _ = self.build_depot_node_return_constraint(graph, other_node, depot_node)

        return None

    def build_depot_energy_arc_tracking_constraint(self, graph, depot_node, other_node):
        """ Build a constraint tracking the energy along  the arc leaving the given depot """

        arc_energy_variable = self.get_arc_energy_variable_from_storage((depot_node, other_node))
        travel_variable = self.get_travel_variable_from_storage((depot_node, other_node))
        energy_capacity = self.get_battery_energy_capacity()

        lhs = arc_energy_variable
        rhs = energy_capacity * travel_variable
        constraint_name = "Depot energy tracking variable for arc ({depot_node},{other_node})"
        _ = self.set_constraint(lhs, rhs, "==", constraint_name)
        
        return None

    def build_depot_energy_arcs_tracking_constraints(self, graph):
        """ Build a constraint tracking the energy along all arcs leaving all depots """

        depot_nodes = self.get_depots(graph)
        non_depot_nodes = self.get_nodes_without_depots(graph)

        for depot_node in depot_nodes:
            for other_node in non_depot_nodes:
                if depot_node != other_node:
                    self.build_depot_energy_arc_tracking_constraint(graph, depot_node, other_node)
    
        return None

    def build_customer_energy_arc_tracking_constraint(self, graph, customer_node):
        """ Build a constraint tracking the energy along arcs going through the customer """

        inward_arcs = self.get_inward_incident_arcs(graph, customer_node)
        outward_arcs = self.get_outward_incident_arcs(graph, customer_node)

        travel_variables = self.get_travel_variables_from_storage(inward_arcs)
        inward_energy_variables = self.get_arc_energy_variables_from_storage(inward_arcs)
        outward_energy_variables = self.get_arc_energy_variables_from_storage(outward_arcs)
        inward_consumptions = self.get_energy_consumptions(graph, inward_arcs)

        lhs = [energy * variable
               for (energy, variable) in zip(inward_consumptions, travel_variables)]
        lhs = - self.sum_items(lhs) + self.sum_items(inward_energy_variables)
        rhs = self.sum_items(outward_energy_variables)
        constraint_name = f"Arc energy tracking constraint for customer {customer_node}"
        _ = self.set_constraint(lhs, rhs, "==", constraint_name)
        
        return None

    def build_customer_energy_arcs_tracking_constraints(self, graph):
        """ Build constraints tracking energy along the arcs going through all customers """
        
        customer_nodes = self.get_customers(graph)
        for customer_node in customer_nodes:
            self.build_customer_energy_arc_tracking_constraint(graph, customer_node)

        return None
        
    def build_station_arrival_energy_arc_tracking_constraint(self, graph, station_node):
        """ Build a constraint tracking the energy along arcs going through the station """

        inward_arcs = self.get_inward_incident_arcs(graph, station_node)

        travel_variables = self.get_travel_variables_from_storage(inward_arcs)
        inward_energy_variables = self.get_arc_energy_variables_from_storage(inward_arcs)
        energy_variable = self.get_station_node_arrival_energy_variable_from_storage(station_node)
        inward_consumptions = self.get_energy_consumptions(graph, inward_arcs)

        lhs = [energy * variable
               for (energy, variable) in zip(inward_consumptions, travel_variables)]
        lhs = - self.sum_items(lhs) + self.sum_items(inward_energy_variables)
        rhs = energy_variable
        constraint_name = f"Arc arival energy tracking constraint for station {station_node}"
        _ = self.set_constraint(lhs, rhs, "==", constraint_name)
        
        return None

    def build_station_arrival_energy_arcs_tracking_constraints(self, graph):
        """ Build constraints tracking energy along the arcs going through all stations """
        
        station_nodes = self.get_stations(graph)
        for station_node in station_nodes:
            self.build_station_arrival_energy_arc_tracking_constraint(graph, station_node)

        return None

    def build_station_departure_energy_arc_tracking_constraint(self, graph, station_node):
        """ Build a constraint tracking the energy along arcs going through the station """

        outward_arcs = self.get_outward_incident_arcs(graph, station_node)

        outward_energy_variables = self.get_arc_energy_variables_from_storage(outward_arcs)
        energy_variable = self.get_station_node_departure_energy_variable_from_storage(station_node)

        lhs = self.sum_items(outward_energy_variables)
        rhs = energy_variable
        constraint_name = f"Arc departure energy tracking constraint for station {station_node}"
        _ = self.set_constraint(lhs, rhs, "==", constraint_name)
        
        return None

    def build_station_departure_energy_arcs_tracking_constraints(self, graph):
        """ Build constraints tracking energy along the arcs going through all stations """
        
        station_nodes = self.get_stations(graph)
        for station_node in station_nodes:
            self.build_station_departure_energy_arc_tracking_constraint(graph, station_node)

        return None

    def build_energy_arc_upper_bound_constraint(self, graph, first_node, second_node):
        """ Build a constraint upper-bounding the given energy arc variable """

        depot_nodes = self.get_depots(graph)
        station_nodes = self.get_stations(graph)
        inward_nodes = station_nodes + depot_nodes
        inward_nodes = [node for node in inward_nodes if node != first_node]
        inward_arcs = [(node, first_node) for node in inward_nodes]
        
        energy_variable = self.get_arc_energy_variable_from_storage((first_node, second_node))
        travel_variable = self.get_travel_variable_from_storage((first_node, second_node))

        energy_capacity = self.get_battery_energy_capacity()
        energy_consumptions = self.get_energy_consumptions(graph, inward_arcs)
        minimum_consumption = min(energy_consumptions)
        if first_node in depot_nodes or first_node in station_nodes:
            minimum_consumption = 0

        lhs = energy_variable
        rhs = (energy_capacity - minimum_consumption) * travel_variable
        constraint_name = f"Energy upper-bounding constraint for arc ({first_node},{second_node})"
        _ = self.set_constraint(lhs, rhs, "<=", constraint_name)
        
        return None

    def build_energy_arcs_upper_bound_constraints(self, graph):
        """ Build a constraint upper-bounding the all energy arc variables """

        depot_nodes = self.get_depots(graph)
        
        for edge in graph.edges:
            self.build_energy_arc_upper_bound_constraint(graph, edge[0], edge[1])
            
        return None

    def build_energy_arc_return_constraint(self, graph, other_node, depot_node):
        """ Make sure there is always enough energy to return if the vehicle comes home """

        travel_variable = self.get_travel_variable_from_storage((other_node, depot_node))
        energy_variable = self.get_arc_energy_variable_from_storage((other_node, depot_node))

        consumption = self.get_energy_consumption(graph, other_node, depot_node)

        lhs = energy_variable
        rhs = consumption * travel_variable
        constraint_name = f"Depot energy arc return constraint for {other_node} and {depot_node}"
        _ = self.set_constraint(lhs, rhs, ">=", constraint_name)

        return None

    def build_energy_arcs_return_constraints(self, graph):
        """ Make sure there is always enough energy to return if the vehicle comes home """

        other_nodes = self.get_nodes(graph)
        depot_nodes = self.get_depots(graph)

        for other_node in other_nodes:
            for depot_node in depot_nodes:
                if other_node != depot_node:
                    self.build_energy_arc_return_constraint(graph, other_node, depot_node)

    def build_time_arc_return_constraint(self, graph, other_node, depot_node):
        """ Make sure there is always enough time to return if the vehicle comes home """

        travel_variable = self.get_travel_variable_from_storage((other_node, depot_node))
        time_variable = self.get_arc_time_variable_from_storage((other_node, depot_node))

        travel_time = self.get_travel_time(graph, other_node, depot_node)

        lhs = time_variable
        rhs = travel_time * travel_variable
        constraint_name = f"Depot time arc return constraint for {other_node} and {depot_node}"
        _ = self.set_constraint(lhs, rhs, ">=", constraint_name)

        return None

    def build_time_arc_leave_constraint(self, graph, other_node, depot_node):
        """ Make sure the time is always zero for leaving arcs  """

        time_variable = self.get_arc_time_variable_from_storage((depot_node, other_node))

        lhs = time_variable
        rhs = 0
        constraint_name = f"Depot time arc leaving constraint for {other_node} and {depot_node}"
        _ = self.set_constraint(lhs, rhs, "==", constraint_name)

        return None

    def build_time_arc_zero_constraint(self, graph, first_node, second_node):
        """ Make sure the time is always zero for if we don't traverse the arc """

        time_variable = self.get_arc_time_variable_from_storage((first_node, second_node))
        travel_variable = self.get_travel_variable_from_storage((first_node, second_node))

        time_limit = self.get_maximum_travel_time()
        
        lhs = time_variable
        rhs = travel_variable * time_limit
        constraint_name = f"Time arc zeroing constraint for {first_node} and {second_node}"
        _ = self.set_constraint(lhs, rhs, "<=", constraint_name)

        return None
    
    def build_time_arcs_return_constraints(self, graph):
        """ Make sure there is always enough time to return if the vehicle comes home """

        other_nodes = self.get_nodes(graph)
        depot_nodes = self.get_depots(graph)

        for other_node in other_nodes:
            for depot_node in depot_nodes:
                if other_node != depot_node:
                    self.build_time_arc_return_constraint(graph, other_node, depot_node)

    def build_time_arcs_leave_constraints(self, graph):
        """ Make sure there time is always zero when leaving along any arc from a depot """

        other_nodes = self.get_nodes(graph)
        depot_nodes = self.get_depots(graph)

        for other_node in other_nodes:
            for depot_node in depot_nodes:
                if other_node != depot_node:
                    self.build_time_arc_leave_constraint(graph, other_node, depot_node)

    def build_time_arcs_zero_constraints(self, graph):
        """ Make sure there time is always zero when an arc is not travelled """

        first_nodes = self.get_nodes(graph)
        second_nodes = self.get_depots(graph)

        for first_node in first_nodes:
            for second_node in second_nodes:
                if first_node != second_node:
                    self.build_time_arc_zero_constraint(graph, first_node, second_node)

                    
    def build_customer_time_arc_tracking_constraint(self, graph, customer_node):
        """ Build a constraint tracking the time along arcs going through the customer """

        inward_arcs = self.get_inward_incident_arcs(graph, customer_node)
        outward_arcs = self.get_outward_incident_arcs(graph, customer_node)

        travel_variables = self.get_travel_variables_from_storage(inward_arcs)
        inward_time_variables = self.get_arc_time_variables_from_storage(inward_arcs)
        outward_time_variables = self.get_arc_time_variables_from_storage(outward_arcs)

        travel_times = self.get_travel_times(graph, inward_arcs)
        service_time = self.get_node_service_time(graph, customer_node)

        lhs = [(travel_time + service_time) for travel_time in travel_times]
        lhs = [item * variable for (item, variable) in zip(lhs, travel_variables)]
        lhs = self.sum_items(lhs) + self.sum_items(inward_time_variables)
        rhs = self.sum_items(outward_time_variables)
        constraint_name = f"Customer arc time constraint for customer {customer_node}"
        _ = self.set_constraint(lhs, rhs, "==", constraint_name)
        
        return None


    def build_customer_time_arcs_tracking_constraints(self, graph):
        """ Build constraints tracking time along the arcs going through all customers """
        
        customer_nodes = self.get_customers(graph)
        for customer_node in customer_nodes:
            self.build_customer_time_arc_tracking_constraint(graph, customer_node)

        return None

    def build_station_time_arc_tracking_constraint(self, graph, station_node):
        """ Build a constraint tracking the time along arcs going through the station """

        inward_arcs = self.get_inward_incident_arcs(graph, station_node)
        outward_arcs = self.get_outward_incident_arcs(graph, station_node)

        travel_variables = self.get_travel_variables_from_storage(inward_arcs)
        inward_time_variables = self.get_arc_time_variables_from_storage(inward_arcs)
        outward_time_variables = self.get_arc_time_variables_from_storage(outward_arcs)

        travel_times = self.get_travel_times(graph, inward_arcs)
        charging_duration = self.get_station_node_duration_variable_from_storage(station_node)

        lhs = [time * variable for (time, variable) in zip(travel_times, travel_variables)]
        lhs = self.sum_items(lhs) + self.sum_items(inward_time_variables) + charging_duration
        rhs = self.sum_items(outward_time_variables)
        constraint_name = f"Station arc time constraint for customer {station_node}"
        _ = self.set_constraint(lhs, rhs, "==", constraint_name)
        
        return None


    def build_station_time_arcs_tracking_constraints(self, graph):
        """ Build constraints tracking time along the arcs going through all stations """
        
        station_nodes = self.get_stations(graph)
        for station_node in station_nodes:
            self.build_station_time_arc_tracking_constraint(graph, station_node)

        return None

    def build_customer_time_arc_return_constraint(self, graph, customer_node, other_node):
        """ Build a constraint making sure that the vehicle has enough time to return """

        travel_variable = self.get_travel_variable_from_storage((other_node, customer_node))
        time_variable = self.get_arc_time_variable_from_storage((other_node, customer_node))

        travel_time = self.get_travel_time(graph, other_node, customer_node)
        service_time = self.get_node_service_time(graph, customer_node)
        time_limit = self.get_maximum_travel_time()
        
        for depot_node in self.get_depots(graph):
            
            return_time = self.get_travel_time(graph, customer_node, depot_node)
            lhs = time_variable
            rhs = (time_limit - travel_time - service_time - return_time) * travel_variable
            constraint_name = f"Customer return constraint for customer {customer_node}"
            constraint_name = constraint_name + f" and depot {depot_node}"
            _ = self.set_constraint(lhs, rhs, "<=", constraint_name)

            
        return None
    
    def build_customer_time_arcs_return_constraints(self, graph):
        """ Build constraints making sure that the vehicle has enough time to return """

        customer_nodes = self.get_customers(graph)
        other_nodes = self.get_nodes(graph)

        for customer_node in customer_nodes:
            for other_node in other_nodes:
                if customer_node != other_node:
                    self.build_customer_time_arc_return_constraint(graph, customer_node, other_node)

        return None

    def get_minimum_needed_station_charge(self, graph, station_node):
        """ Get the minimum needed charge to get the vechicle to """

    def build_station_time_arc_return_constraint(self, graph, station_node, other_node):
        """ Build a constraint making sure that the vehicle has enough time to return """

        travel_variable = self.get_travel_variable_from_storage((other_node, station_node))
        time_variable = self.get_arc_time_variable_from_storage((other_node, station_node))

        travel_time = self.get_travel_time(graph, other_node, station_node)
        # duration = self.get_minimum_needed_station_charge(graph, station_node)
        duration = 0
        # REPLACE ABOVE WITH FUNCTIONAL EXPRESSION
        time_limit = self.get_maximum_travel_time()
        
        for depot_node in self.get_depots(graph):
            
            return_time = self.get_travel_time(graph, station_node, depot_node)
            lhs = time_variable
            rhs = (time_limit - travel_time - duration - return_time) * travel_variable
            constraint_name = f"Station return constraint for station {station_node}"
            constraint_name = constraint_name + f" and depot {depot_node}"
            _ = self.set_constraint(lhs, rhs, "<=", constraint_name)
            
        return None

    def build_station_time_arcs_return_constraints(self, graph):
        """ Build constraints making sure that the vehicle has enough time to return """

        station_nodes = self.get_stations(graph)
        other_nodes = self.get_nodes(graph)

        for station_node in station_nodes:
            for other_node in other_nodes:
                if station_node != other_node:
                    self.build_station_time_arc_return_constraint(graph, station_node, other_node)

        return None

    def get_station_clones(self, graph, station_node):
        """ Get the clone nodes for the given station """

        nodes = self.get_stations(graph)
        primes = [graph.nodes[node].get("prime_node") for node in nodes]
        clones = [node for (node, prime) in zip(nodes, primes) if prime == station_node]

        return clones

    def build_station_clone_prime_constraints(self, graph, station_node):
        """ Build constraints preventing travel between station clones """

        station_clones = self.get_station_clones(graph, station_node)
        for station_clone in station_clones:
            lhs, rhs = self.get_travel_variable_from_storage((station_node, station_clone)), 0
            constraint_name = f"Clone-prime constraint for {station_node} and {station_clone}"
            _ = self.set_constraint(lhs, rhs, "==", constraint_name)
            lhs = self.get_travel_variable_from_storage((station_clone, station_node))
            constraint_name = f"Clone-prime constraint for {station_clone} and {station_node}"
            _ = self.set_constraint(lhs, rhs, "==", constraint_name)
            for other_clone in station_clones:
                if other_clone != station_clone:
                    lhs = self.get_travel_variable_from_storage((station_clone, other_clone))
                    constraint_name = f"Clone constraint for {station_clone} and {other_clone}"
                    _ = self.set_constraint(lhs, rhs, "==", constraint_name)
                    lhs = self.get_travel_variable_from_storage((other_clone, station_clone))
                    constraint_name = f"Clone constraint for {other_clone} and {station_clone}"
                    _ = self.set_constraint(lhs, rhs, "==", constraint_name)

        return None

    def build_energy_valid_inequality(self, graph, first_node, second_node):
        """ Build a valid inequality between the given nodes to tighten formulation """

        arc_energy_variable = self.get_arc_energy_variable_from_storage((first_node, second_node))
        travel_variable = self.get_travel_variable_from_storage((first_node, second_node))
        energy_consumption = self.get_energy_consumption(graph, first_node, second_node)

        nodes = self.get_stations(graph) + self.get_depots(graph)
        if second_node not in nodes:
            tuples = [(second_node, other_node) for other_node in nodes]
            consumptions = self.get_energy_consumptions(graph, tuples)
            minimum_consumption = min(consumptions)
        else:
            minimum_consumption = 0
        lhs = arc_energy_variable
        rhs = (energy_consumption + minimum_consumption) * travel_variable
        constraint_name = f"Valid energy inequality for nodes {first_node} and {second_node}"
        _ = self.set_constraint(lhs, rhs, ">=", constraint_name)

        return None

    def build_energy_valid_ineqaulities(self, graph):
        """ Build all energy valid inequalities for the given problem """

        non_depot_nodes = self.get_nodes_without_depots(graph)
        other_nodes = self.get_nodes(graph)

        for first_node in non_depot_nodes:
            for second_node in other_nodes:
                if first_node != second_node:
                    self.build_energy_valid_inequality(graph, first_node, second_node)

        return None
        
    def build_stations_clone_prime_constraints(self, graph):
        """ Build all constraints preventing travel between station clones """

        station_nodes = self.get_stations(graph)
        for station_node in station_nodes:
            self.build_station_clone_prime_constraints(graph, station_node)

        return None

    def insert_station_clone_travel_variables(self, graph, station_node):
        """ Insert the travel variables associated with the given station clone """

        inward_incident_edges = self.get_inward_incident_arcs(graph, station_node)
        outward_incident_edges = self.get_outward_incident_arcs(graph, station_node)
        for inward_edge in inward_incident_edges:
            print(station_node, inward_edge, "inward")
            self.build_arc_travel_variable(inward_edge[0], inward_edge[1])
        for outward_edge in outward_incident_edges:
            print(station_node, outward_edge, "outward")
            self.build_arc_travel_variable(outward_edge[0], outward_edge[1])

        return None

    def insert_station_clone_time_node_variable(self, graph, station_node):
        """ Insert the node time variable for the station clone """

        time_limit = self.get_maximum_travel_time()
        self.build_node_time_variable(node, time_limit)
            
        return None

    def insert_station_clone_arc_time_variables(self, graph, station_node):
        """ Insert time tracking variables for all the arcs of the station clone """

        time_limit = self.get_maximum_travel_time()
        inward_incident_edges = self.get_inward_incident_arcs(graph, station_node)
        outward_incident_edges = self.get_outward_incident_arcs(graph, station_node)
        for inward_edge in inward_incident_edges:
            self.build_arc_time_variable(inward_edge[0], inward_edge[1], time_limit)
        for outward_edge in outward_incident_edges:
            self.build_arc_time_variable(outward_edge[0], outward_edge[1], time_limit)

        return None

    def insert_station_clone_node_energy_variable(self, graph, station_node):
        """ Insert the node energy variable for the station clone """

        energy_capacity = self.get_battery_energy_capacity()
        
        self.build_node_energy_variable(station_node, energy_capacity)

        return None

    def insert_station_clone_arc_energy_variables(self, graph, station_node):
        """ Insert energy tracking variables for all the arcs of the station clone """

        energy_capacity = self.get_battery_energy_capacity()
        inward_incident_edges = self.get_inward_incident_arcs(graph, station_node)
        outward_incident_edges = self.get_outward_incident_arcs(graph, station_node)
        for inward_edge in inward_incident_edges:
            self.build_arc_energy_variable(inward_edge[0], inward_edge[1], energy_capacity)
        for outward_edge in outward_incident_edges:
            self.build_arc_energy_variable(outward_edge[0], outward_edge[1], energy_capacity)

        return None

    def insert_station_clone_node_arrival_energy_variable(self, graph, station_node):
        """ Insert a fuel tracking variable for the given station clone """

        energy_capacity = self.get_battery_energy_capacity()
        self.build_station_node_arrival_energy_variable(station_node, energy_capacity)

        return None

    def insert_station_clone_node_departure_energy_variable(self, graph, station_node):
        """ Insert a fuel tracking variable for the given station clone """

        energy_capacity = self.get_battery_energy_capacity()
        self.build_station_node_departure_energy_variable(station_node, energy_capacity)

        return None

    def insert_station_clone_node_arrival_time_variable(self, graph, station_node):
        """ Insert a time tracking variable for the given station clone """

        time_limit = self.get_maximum_travel_time()
        self.build_station_node_arrival_time_variable(station_node, time_limit)

        return None

    def insert_station_clone_node_departure_time_variable(self, graph, station_node):
        """ Insert a time tracking variable for the given station clone """

        time_limit = self.get_maximum_travel_time()
        self.build_station_node_departure_time_variable(station_node, time_limit)

        return None

    def insert_station_clone_node_duration_variable(self, graph, station_node):
        """ Insert a charging duration tracking variable for the given station clone """

        time_limit = self.get_maximum_travel_time()
        self.build_station_node_duration_variable(station_node, time_limit)

        return None

    def insert_station_clone_node_arrival_pwl_variables(self, graph, station_node):
        """ Insert all tracking variables for arrival at the given station clone """

        if self.pwl_model == "lambda" or self.pwl_model == "sos":
            variable_builder = self.build_station_node_arrival_lambda_pwl_variables
        elif self.pwl_model == "delta":
            variable_builder = self.build_station_node_arrival_delta_pwl_variables
        
        _ = variable_builder(graph, station_node)

        return None

    def insert_station_clone_node_departure_pwl_variables(self, graph, station_node):
        """ Insert all tracking variables for departure from the given station clone """

        if self.pwl_model == "lambda" or self.pwl_model == "sos":
            variable_builder = self.build_station_node_departure_lambda_pwl_variables
        elif self.pwl_model == "delta":
            variable_builder = self.build_station_node_departure_delta_pwl_variables
        
        _ = variable_builder(graph, station_node)

        return None
    
    def insert_station_clone_node_variables(self, graph, station_node):
        """ Insert the variables associated with the station clone """

        _ = self.insert_station_clone_travel_variables(graph, station_node)
        _ = self.insert_station_clone_node_time_variable(graph, station_node)
        _ = self.insert_station_clone_node_energy_variable(graph, station_node)
        _ = self.insert_station_clone_node_arrival_energy_variable(graph, station_node)
        _ = self.insert_station_clone_node_departure_energy_variable(graph, station_node)
        _ = self.insert_station_clone_node_arrival_time_variable(graph, station_node)
        _ = self.insert_station_clone_node_departure_time_variable(graph, station_node)
        _ = self.insert_station_clone_node_duration_variable(graph, station_node)
        _ = self.insert_station_clone_node_arrival_pwl_variables(graph, station_node)
        _ = self.insert_station_clone_node_departure_pwl_variables(graph, station_node)
        
        return None

    def insert_station_clones_node_variables(self, graph, station_nodes):
        """ Insert the variables associated with the station clones """

        for station_node in station_nodes:
            self.insert_station_clone_node_variables(graph, station_nodes)
        
        return None
    
    def insert_station_clone_arc_variables(self, graph, station_node):
        """ Insert the variables associated with the station clone """

        _ = self.insert_station_clone_travel_variables(graph, station_node)
        _ = self.insert_station_clone_arc_time_variables(graph, station_node)
        _ = self.insert_station_clone_arc_energy_variables(graph, station_node)
        _ = self.insert_station_clone_node_arrival_energy_variable(graph, station_node)
        _ = self.insert_station_clone_node_departure_energy_variable(graph, station_node)
        _ = self.insert_station_clone_node_arrival_time_variable(graph, station_node)
        _ = self.insert_station_clone_node_departure_time_variable(graph, station_node)
        _ = self.insert_station_clone_node_duration_variable(graph, station_node)
        _ = self.insert_station_clone_node_arrival_pwl_variables(graph, station_node)
        _ = self.insert_station_clone_node_departure_pwl_variables(graph, station_node)
        
        return None

    def insert_station_clones_arc_variables(self, graph, station_nodes):
        """ Insert the variables associated with the station clones """

        for station_node in station_nodes:
            self.insert_station_clone_arc_variables(graph, station_nodes)
        
        return None

    def insert_station_clones_node_constraints(self, graph):
        """ Insert the node constraints associated with the given station clone """

        self.build_customer_visiting_constraints_directed(graph)
        self.build_station_visiting_constraints_directed(graph)
        self.build_flow_constraints_directed(graph)
        self.build_station_nodes_arrival_pwl_constraints(graph)
        self.build_station_nodes_departure_pwl_constraints(graph)
        self.build_station_nodes_duration_constraints(graph)
        self.build_energy_arrival_departure_constraints(graph)
        self.build_customer_energy_node_tracking_constraints(graph)
        self.build_station_energy_node_tracking_constraints(graph)
        self.build_depot_energy_node_tracking_constraints(graph)
        self.build_station_departure_energy_node_reset_constraints(graph)
        self.build_depot_energy_initialisation_constraints(graph)
        self.build_customer_nodes_time_tracking_constraints(graph)
        self.build_station_nodes_time_tracking_constraints(graph)
        self.build_depot_nodes_return_constraints(graph)

        return None

        _ = self._build_node_constraints(graph)

        return None        
        
    def insert_station_clone_arc_constraints(self, graph, station_node):
        """ Insert the arc constraints associated with the given station clone """

        _ = self._build_node_constraints(graph)

        return None
    
    def insert_station_clones_node(self, graph, station_nodes):
        """ Insert the variables and constraints associated with the station clone (node-based) """

        _ = self.insert_station_clones_node_variables(graph, station_nodes)
        _ = self.insert_station_clones_node_constraints(graph)

        return None

    def insert_station_clones_arc(self, graph, station_nodes):
        """ Insert the variables and constraints associated with the station clone (arc-based) """

        _ = self.insert_station_clones_arc_variables(graph, station_nodes)
        _ = self.insert_station_clones_arc_constraints(graph)

        return None        
            
    def build_plot_axis(self):
        """ Initialise a new axis """

        figure, axis = plt.subplots()

        return figure, axis

    def compute_reachability_radius(self, graph):
        """ Compute the reachability radius """

        reachability_radius = self.get_battery_energy_capacity()
        reachability_radius /=  self.get_energy_consumption_rate()

        return reachability_radius

    def compute_distance_euclidean(self, first_coordinate, second_coordinate):
        """ Compute the Euclidean distance between the given pair of coordinates """

        first_part = (first_coordinate[0] - second_coordinate[0]) ** 2
        second_part = (first_coordinate[1] - second_coordinate[1]) ** 2
        distance = (first_part + second_part) ** 0.5

        return distance
        
    def check_if_depot_reachable(self, graph, depot_node, other_node):
        """ Check if the given node is strictly depot-reachable """

        depot_coordinate = graph.graph["coord_dict"][other_node]
        other_coordinate = graph.graph["coord_dict"][other_node]
        reachability_radius = self.compute_reachability_radius(graph)
        distance = self.compute_distance_euclidean(other_coordinate, depot_coordinate)
        is_reachable = distance <= reachability_radius

        return is_reachable

    def get_depot_reachable_stations(self, graph, depot_node):
        """ Get the stations that are depot-reachable for the given depot """

        station_nodes = self.get_stations(graph)
        is_reachables = [self.check_if_depot_reachable(graph, depot_node, station_node)
                         for station_node in station_nodes]
        reachability_pairs = zip(station_nodes, is_reachables)
        reachable_stations = [node for (node, is_reachable) in reachability_pairs if is_reachable]

        return reachable_stations

    def plot_depot_reachability_radius(self, graph, depot_node, axis):
        """ Plot the reachability radius circle for the given depot """

        depot_coordinate = graph.graph["coord_dict"][depot_node]
        reachability_radius = self.compute_reachability_radius(graph)
        plot_utils.plot_circle(depot_coordinate, "blue", 1, reachability_radius/2, axis)
        plot_utils.plot_circle(depot_coordinate, "blue", 1, reachability_radius, axis)

        return None

    def plot_strict_depot_reachability_radius(self, graph, depot_node, axis):
        """ Plot the reachability radius circle for the given depot """

        depot_coordinate = graph.graph["coord_dict"][depot_node]
        reachability_radius = self.compute_reachability_radius(graph)
        plot_utils.plot_circle(depot_coordinate, "blue", 1, reachability_radius/2, axis)

        return None
    
    def plot_depots_reachability_radius(self, graph, axis):
        """ Plot the reachability radius circle for the given depot """

        depot_nodes = self.get_depots(graph)
        for depot_node in depot_nodes:
            self.plot_depot_reachability_radius(graph, depot_node, axis)
        
        return None
    
    def plot_strict_depots_reachability_radius(self, graph, axis):
        """ Plot the reachability radius circle for the given depot """

        depot_nodes = self.get_depots(graph)
        for depot_node in depot_nodes:
            self.plot_strict_depot_reachability_radius(graph, depot_node, axis)
        
        return None

    def plot_station_reachability_radius(self, graph, station_node, axis):
        """ Plot the reachability radius circle for the given station """

        station_coordinate = graph.graph["coord_dict"][station_node]
        reachability_radius = self.compute_reachability_radius(graph)
        plot_utils.plot_circle(station_coordinate, "green", 1, reachability_radius/2, axis)
        plot_utils.plot_circle(station_coordinate, "green", 1, reachability_radius, axis)

        return None

    def plot_strict_station_reachability_radius(self, graph, station_node, axis):
        """ Plot the reachability radius circle for the given station """

        station_coordinate = graph.graph["coord_dict"][station_node]
        reachability_radius = self.compute_reachability_radius(graph)
        plot_utils.plot_circle(station_coordinate, "green", 1, reachability_radius/2, axis)

        return None
    
    def plot_stations_reachability_radius(self, graph, axis):
        """ Plot the reachability radius circle for the given sttaion """

        station_nodes = self.get_stations(graph)
        for station_node in station_nodes:
            self.plot_station_reachability_radius(graph, station_node, axis)
        
        return None

    def plot_strict_stations_reachability_radius(self, graph, axis):
        """ Plot the reachability radius circle for the given sttaion """

        station_nodes = self.get_stations(graph)
        depot_nodes = self.get_depots(graph)
        for depot_node in depot_nodes:
            station_nodes = self.get_depot_reachable_stations(graph, depot_node)
            for station_node in station_nodes:
                self.plot_strict_station_reachability_radius(graph, station_node, axis)
        
        return None

    def plot_reachabilities_radii(self, graph, axis):
        """ Plot the reachability radius of all recharge nodes """

        _ = self.plot_depots_reachability_radius(graph, axis)
        _ = self.plot_stations_reachability_radius(graph, axis)

        return None

    def plot_strict_reachabilities_radii(self, graph, axis):
        """ Plot the reachability radius of all recharge nodes """

        _ = self.plot_strict_depots_reachability_radius(graph, axis)
        _ = self.plot_strict_stations_reachability_radius(graph, axis)

        return None

    def plot_problem(self, graph, show=True, strict=False):
        """ Plot the nodes and the reachability radii of the problem """

        figure, axis = self.build_plot_axis()
        _ = self.plot_nodes(graph)
        if strict:
            _ = self.plot_strict_reachabilities_radii(graph, axis)
        else:
            _ = self.plot_reachabilities_radii(graph, axis)
        if show:
            plot_utils.show_plots()

        return None
