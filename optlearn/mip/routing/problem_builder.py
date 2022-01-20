import copy

import numpy as np
import networkx as nx

from optlearn.plotting import plot_utils
from optlearn.graph import process_utils
from optlearn.mip import mip_wrapper


class basicProblemBuilder(mip_wrapper.mipWrapper):

    def __init__(self, solver_package, problem_type, is_directed=True,
                 is_multi=False, verbose=False, very_verbose=False):

        self.solver_package = solver_package
        self.problem_type = problem_type
        self.is_directed = is_directed
        self.is_multi = is_multi
        self.verbose = verbose
        self.very_verbose = very_verbose

        self._initialise_solver_functions()

    def toggle_verbosity(self):
        """ Toggle the verbosity setting """

        self.verbose = not(self.verbose)
        if not self.verbose:
            self.very_verbose = False

    def toggle_verybosity(self):
        """ Toggle the very verbose setting """

        self.very_verbose = not(self.very_verbose)
        if self.very_verbose:
            self.verbose = True

    def _initialise_arc_travel_variable_graph(self):
        """ Initialise the graph used to store the arc travel variables """

        if not self.is_directed and not(self.is_multi):
            self.travel_graph = nx.Graph()
        elif not self.is_directed and self.is_multi:
            self.travel_graph = nx.MultiGraph()
        elif self.is_directed and not(self.is_multi):
            self.travel_graph = nx.DiGraph()
        elif self.is_directed and self.is_multi:
            self.travel_graph = nx.MultiDiGraph()
        else:
            print("Cannot understand the graph type!")
            

    def _initialise_node_time_variable_graph(self):
        """ Initialise the graph used to store the node time variables """

        self.time_graph = nx.Graph()

    def _initialise_node_fuel_variable_graph(self):
        """ Initialise the graph used to store the node fuel variables """

        self.fuel_graph = nx.Graph()
        
    def name_arc_travel_variable(self, first_node, second_node):
        """ Name the arc travel variable between the given nodes """

        variable_name = f"x_{first_node},{second_node}"

        return variable_name

    def name_arc_travel_variable_multi(self, first_node, second_node, station_node=None):
        """ Name the arc travel variable between the nodes, possibly through a station """

        if station_node is not None:
            variable_name = f"x_{first_node},{second_node},{station_node}"
        else:
            variable_name = f"x_{first_node},{second_node},0"
            
        return variable_name

    def build_arc_travel_variable(self, first_node, second_node):
        """ Build an arc travel variable between the given nodes """

        variable_name = self.name_arc_travel_variable(first_node, second_node)
        variable = self.set_binary_variable(name=variable_name)
        self.store_arc_travel_variable(first_node, second_node, variable)

        return None

    def build_arc_travel_variable_multi(self, first_node, second_node, station_node=None):
        """ Build an arc travel variable between the given nodes """

        station_node = station_node or 0
        variable_name = self.name_arc_travel_variable_multi(first_node, second_node, station_node)
        variable = self.set_binary_variable(name=variable_name)
        self.store_arc_travel_variable_multi(first_node, second_node, station_node, variable)

        return None
        
    def build_arc_travel_variables(self, graph):
        """ Build an arc travel variable between all nodes in the given graph """

        for arc in graph.edges:
            self.build_arc_travel_variable(*arc)
        self.travel_graph.graph["node_types"] = graph.graph["node_types"]

        return None

    def build_arc_travel_variables_multi(self, graph):
        """ Build an arc travel variable between all nodes in the given graph """

        stations = self.get_stations(graph)
        for arc in graph.edges:
            self.build_arc_travel_variable_multi(*arc, None)
            for station in stations:
                if arc[0] not in stations and arc[1] not in stations:
                    self.build_arc_travel_variable_multi(*arc, station)
        self.travel_graph.graph["node_types"] = graph.graph["node_types"]

        return None

    def store_arc_travel_variable(self, first_node, second_node, arc_travel_variable):
        """ Store the given arc in a graph for quick connectedness checking """

        storage_tuple = tuple([first_node, second_node])
        self.travel_graph.add_edges_from([storage_tuple], variable=arc_travel_variable)

        return None

    def store_arc_travel_variable_multi(self, first_node, second_node, station_node, variable):
        """ Store the given arc in a graph for quick connectedness checking """

        station_node = station_node or 0
        storage_tuple = tuple([first_node, second_node, station_node])
        self.travel_graph.add_edges_from([storage_tuple], variable=variable)

        return None
    
    def name_node_time_variable(self, node):
        """ Name the given node variable for tracking the time """

        variable_name = f"t_{node}"

        return variable_name

    def build_node_time_variable(self, node, time_limit):
        """ Build the variable for tracking the time at the given node """
        
        variable_name = self.name_node_time_variable(node)
        variable = self.set_continuous_variable(
            lower_bound=0,
            upper_bound=time_limit,
            name=variable_name,
        )
        self.store_node_time_variable(node, variable)
        
        return None

    def build_node_time_variables(self, graph, time_limit):
        """ Build time tracking variables for all the nodes """

        for node in self.get_nodes(graph):
            self.build_node_time_variable(node, time_limit)

        return None

    def store_node_time_variable(self, node, time_variable):
        """ Store the node time variable for the given node """

        self.time_graph.add_nodes_from([node], variable=time_variable)

        return None

    def name_node_fuel_variable(self, node):
        """ Name the given node variable for tracking the fuel """

        variable_name = f"f_{node}"

        return variable_name

    def build_node_fuel_variable(self, node, fuel_capacity):
        """ Build the variable for tracking the fuel at the given node """
        
        variable_name = self.name_node_fuel_variable(node)
        variable = self.set_continuous_variable(
            lower_bound=0,
            upper_bound=fuel_capacity,
            name=variable_name,
        )
        self.store_node_fuel_variable(node, variable)
        
        return None

    def build_node_fuel_variables(self, graph, fuel_capacity):
        """ Build fuel tracking variables for all the nodes """

        for node in self.get_nodes(graph):
            self.build_node_fuel_variable(node, fuel_capacity)

        return None

    def store_node_fuel_variable(self, node, fuel_variable):
        """ Store the node fuel variable for the given node """

        self.fuel_graph.add_nodes_from([node], variable=fuel_variable)

        return None
    
    def get_customers(self, graph):
        """ Find the customer set for the given graph """

        customers = process_utils.get_customer_nodes_from_metadata(
            graph=graph,
        )

        return customers

    def get_stations(self, graph):
        """ Find the station set for the given graph """

        stations = process_utils.get_station_nodes_from_metadata(
            graph=graph,
        )

        return stations

    def get_depots(self, graph):
        """ Find the depot set for the given graph """

        depots = process_utils.get_depot_nodes_from_metadata(
            graph=graph,
        )

        return depots

    def get_nodes(self, graph):
        """ Find all of the nodes for the given graph """

        nodes = list(graph.nodes)

        return nodes

    def get_nodes_without_depots(self, graph):
        """ Get all the nodes, except the depot nodes """

        all_nodes = self.get_nodes(graph)
        depot_nodes = self.get_depots(graph)
        all_nodes_without_depots = [node for node in all_nodes if node not in depot_nodes]

        return all_nodes_without_depots

    def get_num_vehicles(self, graph):
        """ get the number of vehicles permitted to be used in a solution """

        num_vehicles = graph.graph["num_vehicles"]

        return num_vehicles

    def get_time_limit(self, graph):
        """ Get the time limit for travel in the given graph """

        time_limit = graph.graph["fleet"]["vehicle_0"]["max_travel_time"]

        return time_limit

    def get_incident_edges(self, graph, node):
        """ Find all of the edges incident with a given node (undirected) """

        # edges = list(graph.edges(node))
        edges = self.get_incident_edges_multi(graph, node)

        return edges

    def get_incident_edges_multi(self, graph, node):
        """ Find all of the edges incident with a given node (undirected) """

        edges = list(self.travel_graph.edges)
        edges = [edge for edge in edges if node in edge]

        return edges
    
    def get_inward_incident_arcs(self, graph, node):
        """ Find all of the inward arcs incident with the given node (directed) """

        # arcs = list(self.travel_graph.in_edges(node))
        arcs = self.get_inward_incident_arcs_multi(graph, node)
        
        return arcs

    def get_inward_incident_arcs_multi(self, graph, node):
        """ Find all of the inward arcs incident with the given node (directed) """

        edges = self.get_incident_edges_multi(self.travel_graph, node)
        inward_incident_arcs = [edge for edge in edges if edge[1] == node]

        return inward_incident_arcs
    
    def get_outward_incident_arcs(self, graph, node):
        """ Find all of the outward arcs incident with the given node (directed) """

        # arcs = list(self.travel_graph.out_edges(node))
        arcs = self.get_outward_incident_arcs_multi(graph, node)

        return arcs

    def get_outward_incident_arcs_multi(self, graph, node):
        """ Find all of the outward arcs incident with the given node (directed) """

        edges = self.get_incident_edges_multi(self.travel_graph, node)
        outward_incident_arcs = [edge for edge in edges if edge[0] == node]

        return outward_incident_arcs

    def get_incident_arcs(self, graph, node):
        """ Find all incident with the given node, inward or outward (directed) """

        inward_arcs = self.get_inward_incident_arcs(graph, node)
        outward_arcs = self.get_outward_incident_arcs(graph, node)
        arcs = set(inward_arcs + outward_arcs)

        return arcs
    
    def get_travel_variable_from_storage(self, tuple):
        """ Get the variable from the storage dict using a tuple (edge or arc) as a key """

        variable = self.travel_graph[tuple[0]][tuple[1]]["variable"]

        return variable

    def get_travel_variable_from_storage_multi(self, triple):
        """ Get the variable from the storage dict using a tuple (edge or arc) as a key """

        variable = self.travel_graph[triple[0]][triple[1]][triple[2]]["variable"]

        return variable

    def get_node_time_variable_from_storage(self, node):
        """ Get the variable from the storage dict using a nodes as a key """

        variable = self.time_graph.nodes[node]["variable"]
        
        return variable

    def get_node_fuel_variable_from_storage(self, node):
        """ Get the variable from the storage dict using a node as a key """

        variable = self.fuel_graph.nodes[node]["variable"]
        
        return variable
    
    def get_travel_variables_from_storage(self, tuples):
        """ Get the variables from the storage dict using tuples (edges or arcs) as keys """

        variables = [self.travel_graph[tuple[0]][tuple[1]]["variable"] for tuple in tuples]

        return variables

    def get_travel_variables_from_storage_multi(self, triples):
        """ Get the variables from the storage dict using triples (edges or arcs) as keys """

        variables = [self.get_travel_variable_from_storage_multi(triple) for triple in triples]

        return variables
    
    def get_node_time_variables_from_storage(self, nodes):
        """ Get the variables from the storage dict using nodes as keys """

        variables = [self.time_graph.nodes[node]["variable"] for node in nodes]
        
        return variables

    def get_node_fuel_variables_from_storage(self, nodes):
        """ Get the variables from the storage dict using nodes as keys """

        variables = [self.fuel_graph.nodes[node]["variable"] for node in nodes]
        
        return variables
    
    def build_customer_visiting_constraint_undirected(self, graph, customer_node):
        """ Make sure the given customer is visited exactly once """

        incident_edges = self.get_incident_edges(graph, customer_node)
        variables = self.get_travel_variables_from_storage(incident_edges)
        variable_sum = self.sum_items(variables)
        name = f"Customer-visiting constraint for node {customer_node}"
        constraint = self.set_constraint(variable_sum, 2, "==", name=name)

        return None

    def build_customer_visiting_constraint_directed(self, graph, customer_node):
        """ Make sure the given customer is visited exactly once """

        incident_arcs = self.get_outward_incident_arcs(graph, customer_node)
        variables = self.get_travel_variables_from_storage(incident_arcs)
        variable_sum = self.sum_items(variables)
        name = f"Customer-visiting constraint for node {customer_node}"
        constraint = self.set_constraint(variable_sum, 1, "==", name=name)
        
        return None

    def build_customer_visiting_constraint_directed_multi(self, graph, customer_node):
        """ Make sure the given customer is visited exactly once """

        incident_arcs = self.get_outward_incident_arcs(graph, customer_node)
        variables = self.get_travel_variables_from_storage_multi(incident_arcs)
        variable_sum = self.sum_items(variables)
        name = f"Customer-visiting constraint for node {customer_node}"
        constraint = self.set_constraint(variable_sum, 1, "==", name=name)
        
        return None
    
    def build_customer_visiting_constraints_undirected(self, graph):
        """ Make sure each customer is visited exactly once """

        if self.very_verbose:
            print("Setting constraints to ensure each customer is visited exactly once!")
        customer_nodes = self.get_customers(graph)
        for customer_node in customer_nodes:
            self.build_customer_visiting_constraint_undirected(graph, customer_node)

        return None

    def build_customer_visiting_constraints_directed(self, graph):
        """ Make sure each customer is visited exactly once """

        if self.very_verbose:
            print("Setting constraints to ensure each customer is visited exactly once!")
        customer_nodes = self.get_customers(graph)
        for customer_node in customer_nodes:
            self.build_customer_visiting_constraint_directed(graph, customer_node)

        return None

    def build_customer_visiting_constraints_directed_multi(self, graph):
        """ Make sure each customer is visited exactly once """

        if self.very_verbose:
            print("Setting constraints to ensure each customer is visited exactly once!")
        customer_nodes = self.get_customers(graph)
        for customer_node in customer_nodes:
            self.build_customer_visiting_constraint_directed_multi(graph, customer_node)

        return None
    
    def build_station_visiting_constraint_undirected(self, graph, station_node):
        """ Make sure the given station is visited at most once """

        incident_edges = self.get_incident_edges(graph, station_node)
        variables = self.get_travel_variables_from_storage(incident_edges)
        variable_sum = self.sum_items(variables)
        name = f"Station-visiting constraint for node {station_node}"
        constraint = self.set_constraint(variable_sum, 2, "<=", name=name)

        return None

    def build_station_visiting_constraint_directed(self, graph, station_node):
        """ Make sure the given station is visited at most once """

        incident_arcs = self.get_outward_incident_arcs(graph, station_node)
        variables = self.get_travel_variables_from_storage(incident_arcs)
        variable_sum = self.sum_items(variables)
        name = f"Station-visiting constraint for node {station_node}"
        constraint = self.set_constraint(variable_sum, 1, "<=", name=name)
        
        return None

    def build_station_visiting_constraint_directed_multi(self, graph, station_node):
        """ Make sure the given station is visited at most once """

        incident_arcs = self.get_outward_incident_arcs(graph, station_node)
        variables = self.get_travel_variables_from_storage_multi(incident_arcs)
        variable_sum = self.sum_items(variables)
        name = f"Station-visiting constraint for node {station_node}"
        constraint = self.set_constraint(variable_sum, 1, "<=", name=name)
        
        return None
    
    def build_station_visiting_constraints_undirected(self, graph):
        """ Make sure each station is visited at most once """

        if self.very_verbose:
            print("Setting constraints to ensure each station is visited at most once!")
        station_nodes = self.get_stations(graph)
        for station_node in station_nodes:
            self.build_station_visiting_constraint_undirected(graph, station_node)

        return None

    def build_station_visiting_constraints_directed(self, graph):
        """ Make sure each station is visited at most once """

        if self.very_verbose:
            print("Setting constraints to ensure each station is visited at most once!")
        station_nodes = self.get_stations(graph)
        for station_node in station_nodes:
            self.build_station_visiting_constraint_directed(graph, station_node)

        return None

    def build_station_visiting_constraints_directed_multi(self, graph):
        """ Make sure each station is visited at most once """

        if self.very_verbose:
            print("Setting constraints to ensure each station is visited at most once!")
        station_nodes = self.get_stations(graph)
        for station_node in station_nodes:
            self.build_station_visiting_constraint_directed_multi(graph, station_node)

        return None
    
    def build_flow_constraint_directed(self, graph, node):
        """ Build a flow constraint for the given node """

        inward_incident_arcs = self.get_inward_incident_arcs(graph, node)
        outward_incident_arcs = self.get_outward_incident_arcs(graph, node)
        inward_variables = self.get_travel_variables_from_storage(inward_incident_arcs)
        outward_variables = self.get_travel_variables_from_storage(outward_incident_arcs)
        inward_sum = self.sum_items(inward_variables)
        outward_sum = self.sum_items(outward_variables)
        name = f"Flow constraint for node {node}"
        self.set_constraint(inward_sum, outward_sum, "==", name)
        
        return None

    def build_flow_constraint_directed_multi(self, graph, node):
        """ Build a flow constraint for the given node """

        inward_incident_arcs = self.get_inward_incident_arcs(graph, node)
        outward_incident_arcs = self.get_outward_incident_arcs(graph, node)
        inward_variables = self.get_travel_variables_from_storage_multi(inward_incident_arcs)
        outward_variables = self.get_travel_variables_from_storage_multi(outward_incident_arcs)
        inward_sum = self.sum_items(inward_variables)
        outward_sum = self.sum_items(outward_variables)
        name = f"Flow constraint for node {node}"
        self.set_constraint(inward_sum, outward_sum, "==", name)
        
        return None
    
    def build_flow_constraints_directed(self, graph):
        """ Build flow constraints for the nodes in the given graph """

        nodes = self.get_nodes(graph)
        for node in nodes:
            self.build_flow_constraint_directed(graph, node)
        
        return None

    def build_flow_constraints_directed_multi(self, graph):
        """ Build flow constraints for the nodes in the given graph """

        nodes = self.get_nodes(graph)
        for node in nodes:
            self.build_flow_constraint_directed_multi(graph, node)
        
        return None
    
    def build_depot_entry_constraint_directed(self, graph, depot_node, num_vehicles):
        """ Build a constraint restricting the number of vehicles entering the given depot """

        inward_arcs = self.get_inward_incident_arcs(graph, depot_node)
        inward_variables = self.get_travel_variables_from_storage(inward_arcs)
        inward_sum = self.sum_items(inward_variables)
        name = f"Depot entering constraint for depot {depot_node}"
        self.set_constraint(inward_sum, num_vehicles, "<=", name=name)

        return None

    def build_depot_entry_constraint_directed_multi(self, graph, depot_node, num_vehicles):
        """ Build a constraint restricting the number of vehicles entering the given depot """

        inward_arcs = self.get_inward_incident_arcs(graph, depot_node)
        inward_variables = self.get_travel_variables_from_storage_multi(inward_arcs)
        inward_sum = self.sum_items(inward_variables)
        name = f"Depot entering constraint for depot {depot_node}"
        self.set_constraint(inward_sum, num_vehicles, "<=", name=name)

        return None
    
    def build_depot_exit_constraint_directed(self, graph, depot_node, num_vehicles):
        """ Build a constraint restricting the number of vehicles exiting the given depot """

        outward_arcs = self.get_outward_incident_arcs(graph, depot_node)
        outward_variables = self.get_travel_variables_from_storage(outward_arcs)
        outward_sum = self.sum_items(outward_variables)
        name = f"Depot exiting constraint for depot {depot_node}"
        self.set_constraint(outward_sum, num_vehicles, "<=", name=name)

        return None

    def build_depot_exit_constraint_directed_multi(self, graph, depot_node, num_vehicles):
        """ Build a constraint restricting the number of vehicles exiting the given depot """

        outward_arcs = self.get_outward_incident_arcs(graph, depot_node)
        outward_variables = self.get_travel_variables_from_storage_multi(outward_arcs)
        outward_sum = self.sum_items(outward_variables)
        name = f"Depot exiting constraint for depot {depot_node}"
        self.set_constraint(outward_sum, num_vehicles, "<=", name=name)

        return None
    
    def build_depot_flow_constraints_directed(self, graph, depot_node, num_vehicles):
        """ Build the entering and exiting flow constraints for the given depot """

        self.build_depot_entry_constraint_directed(graph, depot_node, num_vehicles)
        self.build_depot_exit_constraint_directed(graph, depot_node, num_vehicles)

        return None

    def build_depot_flow_constraints_directed_multi(self, graph, depot_node, num_vehicles):
        """ Build the entering and exiting flow constraints for the given depot """

        self.build_depot_entry_constraint_directed_multi(graph, depot_node, num_vehicles)
        self.build_depot_exit_constraint_directed_multi(graph, depot_node, num_vehicles)

        return None
    
    def build_depot_uniform_flow_constraints_directed(self, graph):
        """ For all depot nodes, build the same flow constraints. Caution advised! """

        depot_nodes = self.get_depots(graph)
        num_vehicles = self.get_num_vehicles(graph)
        for depot_node in depot_nodes:
            self.build_depot_flow_constraints_directed(graph, depot_node, num_vehicles)

    def build_depot_uniform_flow_constraints_directed_multi(self, graph):
        """ For all depot nodes, build the same flow constraints. Caution advised! """

        depot_nodes = self.get_depots(graph)
        num_vehicles = self.get_num_vehicles(graph)
        for depot_node in depot_nodes:
            self.build_depot_flow_constraints_directed_multi(graph, depot_node, num_vehicles)
            
    def fix_travel_variable(self, tuple, fixing_value=None):
        """ Fix the travel variabe to the given value """

        fixing_value = fixing_value or 0
        travel_variable = self.get_travel_variable_from_storage(tuple)
        constraint_name = f"fixing constraint for direct arc ({tuple[0]},{tuple[1]})"
        _ = self.set_constraint(travel_variable, fixing_value, "==", constraint_name)

        return None

    def fix_travel_variables(self, tuples, fixing_values=None):
        """ Fix the travel variabe to the given value """

        fixing_values = fixing_values or [0] * len(tuples)
        for tuple, fixing_value in zip(tuples, fixing_values):
            _ = self.fix_travel_variable(tuple, fixing_value)

        return None
        
    def set_distance_objective(self, graph):
        """ Set an objective which minimises the distance travelled """

        tuple_weight_pairs = nx.get_edge_attributes(graph, "weight")
        variables = self.get_travel_variables_from_storage(tuple_weight_pairs.keys())
        weights = tuple_weight_pairs.values()
        terms = [variable * weight for (variable, weight) in zip(variables, weights)]
        expression = self.sum_items(terms)
        self.set_objective(expression)

    def process_solution(self, solution):
        """ Process the given solution to bget rid of computational issues """

        solution = np.round(solution, 5)
        solution = solution.tolist()

        return solution

    def get_travel_solutions(self):
        """ Get all identified solutions for the travel variables only """

        travel_variables = self.get_travel_variables_from_storage(self.travel_graph.edges)
        solutions = self.get_solutions()
        travel_solutions = [self.get_solution_values(solution, travel_variables)
                            for solution in solutions]
        
        return travel_solutions

    def get_all_travel_edges(self):
        """ Get the travel edges of the graph """

        travel_edges = self.travel_graph.edges

        return travel_edges

    def get_all_travel_variables(self):
        """ Get all the travel variables of the problem """

        travel_edges = self.get_all_travel_edges()
        travel_variables = self.get_travel_variables_from_storage(travel_edges)

        return travel_variables
        
    def get_travel_solution(self):
        """ Get the solution only for the travel variables """

        travel_variables = self.get_all_travel_variables()
        travel_solution = self.get_variable_values(travel_variables)
        travel_solution = self.process_solution(travel_solution)
        
        return travel_solution        

    def get_time_solution(self):
        """ Get the solution only for the time variables """

        time_variables = self.get_node_time_variables_from_storage(self.time_graph.nodes)
        time_solution = self.get_variable_values(time_variables)
        time_solution = self.process_solution(time_solution)
        
        return time_solution

    def get_fuel_solution(self):
        """ Get the solution only for the fuel variables """

        fuel_variables = self.get_node_fuel_variables_from_storage(self.fuel_graph.nodes)
        fuel_solution = self.get_variable_values(fuel_variables)
        fuel_solution = self.process_solution(fuel_solution)

        return fuel_solution

    def get_energy_solution(self):
        """ Get the solution only for the energy variables """

        energy_variables = self.get_node_energy_variables_from_storage(self.energy_graph.nodes)
        energy_solution = self.get_variable_values(energy_variables)
        energy_solution = self.process_solution(energy_solution)

        return energy_solution
    
    def build_solution_support_graph(self):
        """ Build a support grah for the given solution """
        
        self.support_graph = type(self.travel_graph)()
        arcs = self.travel_graph.edges
        solution = self.get_travel_solution()
        arcs = [arc for (arc, value) in zip(arcs, solution) if value > 0]
        solution = [value for value in solution if value > 0]
        tuples = [(*arc, {"value": value}) for (arc, value) in zip(arcs, solution)]
        
        self.support_graph.add_edges_from(tuples)

        return None

    def get_solution_cycles(self):
        """ Get the cycle edges from the support graph for all cycles """

        if not hasattr(self, "support_graph"):
            self.build_solution_support_graph()
        
        cycles = []
        
        copy_graph = copy.deepcopy(self.support_graph)
        while len(copy_graph.edges) != 0:
            cycle = nx.find_cycle(copy_graph)
            cycles.append(cycle)
            copy_graph.remove_edges_from(cycle)

        return cycles

    def cycle_to_tour(self, cycle):
        """ Convert a set of cycle edges to a tour """

        encountered, tour = set(), []
        for (a, b) in cycle:
            if a not in encountered:
                tour.append(a)
                encountered.add(a)
            if b not in encountered:
                tour.append(b)
                encountered.add(b)
        tour = tour + [tour[0]]
                
        return tour

    def get_solution_tours(self):
        """ Get the solution tours """

        self.build_solution_support_graph()
        cycles = self.get_solution_cycles()
        tours = [self.cycle_to_tour(cycle) for cycle in cycles]

        return tours

    def plot_customer_nodes(self, graph):
        """ Plot the customer nodes """

        customers = self.get_customers(graph)
        plot_utils.plot_nodes(graph, "*", "red", 1, 10, 12, customers)

        return None

    def get_prime_stations(self, graph):
        """ Get the prime stations only """

        stations = self.get_stations(graph)
        stations = [station
                    for station in stations if graph.nodes[station].get("prime_node") is None]

        return stations

    def plot_station_nodes(self, graph, offset=None):
        """ Plot the station nodes """

        stations = self.get_prime_stations(graph)
        plot_utils.plot_nodes(graph, "^", "green", 1, 10, 12, stations)

        return None

    def plot_depot_nodes(self, graph):
        """ Plot the depot nodes """

        depots = self.get_depots(graph)
        plot_utils.plot_nodes(graph, "^", "blue", 1, 10, 12, depots)

        return None

    def plot_nodes(self, graph):
        """ Plot the nodes """

        _ = self.plot_customer_nodes(graph)
        _ = self.plot_station_nodes(graph)
        _ = self.plot_depot_nodes(graph)

        return None

    def plot_fuel_and_time_status(self, graph, offset=None):
        """ Plot the fuel status at each node """

        solution_tours = self.get_solution_tours()
        if hasattr(self, "fuel_graph"):
            fuel_solution = self.get_fuel_solution()
        if hasattr(self, "energy_graph"):
            fuel_solution = self.get_energy_solution()            
        time_solution = self.get_time_solution()
        for (node, fuel, time) in zip(graph.nodes, fuel_solution, time_solution):
            if any([node in tour for tour in solution_tours]):
                point = graph.graph["coord_dict"][node]
                label = f"({fuel}, {time})"
                plot_utils.plot_point_text(point, label, 12, "purple", offset=offset)

        return None

    def plot_cycle_edges(self, graph):
        """ Plot the (directed) edges for the solution """

        cycles = self.get_solution_cycles()
        for cycle in cycles:
            plot_utils.plot_arcs(graph, arcs=cycle)

        return None
        
    def plot_solution(self, graph, offset=None, show=True):
        """ Plot the solution """

        solution_cycles = self.get_solution_cycles()
        
        self.plot_customer_nodes(graph)
        self.plot_station_nodes(graph)
        self.plot_depot_nodes(graph)
        self.plot_cycle_edges(graph)
        self.plot_fuel_and_time_status(graph, offset=offset)
        if show:
            plot_utils.show_plots()
