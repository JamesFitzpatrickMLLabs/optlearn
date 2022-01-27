import numpy as np
import networkx as nx

from frvcpy import solver

from optlearn import cc_solve

from optlearn.graph import graph_utils

from optlearn.heuristic import evrpnl_utils
from optlearn.heuristic import translate_evrpnl_graph

from optlearn.io.tsp import write_tsplib

from optlearn.feature.vrp import feature_utils



class heuristicSolver():

    def __init__(self, problem_graph=None, average_velocity=None, consumption_rate=None,
                 battery_capacity=None, time_limit=None, relabel=False):
        
        self.problem_graph = problem_graph
        self.average_velocity = average_velocity
        self.consumption_rate = consumption_rate
        self.battery_capacity = battery_capacity
        self.time_limit = time_limit
        self.relabel = relabel
        
        self._assume_average_velocity()
        self._assume_consumption_rate()
        self._assume_battery_capacity()
        self._assume_time_limit()

    def _assume_average_velocity(self):
        """ Assume the average velocity of the vehicles """

        if self.average_velocity is None:
            self.average_velocity = 40

        return None

    def _assume_consumption_rate(self):
        """ Assume the consumption rate of the vehicles """

        if self.consumption_rate is None:
            self.consumption_rate = 0.125

        return None

    def _assume_battery_capacity(self):
        """ Assume the battery capacity of the vehicle """

        if self.battery_capacity is None:
            self.battery_capacity = 16

        return None
    
    def _assume_time_limit(self):
        """ Assume the time limit for vehicle travel """

        if self.time_limit is None:
            self.time_limit = 10

        return None
    
    def set_average_velocity(self, average_velocity):
        """ Set the average velocity of the vehicle """

        self.average_velocity = average_velocity

        return None

    def set_consumption_rate(self, consumption_rate):
        """ Set the consumption rate of the vehicle """

        self.consumption_rate = consumption_rate

        return None

    def set_battery_capacity(self, battery_capacity):
        """ Set the battery capacity of the vehicle """

        self.battery_capacity = battery_capacity

        return None

    def set_time_limit(self, time_limit):
        """ Set the time limit for vehicle travel """

        self.time_limit = time_limit

        return None
    
    def get_average_velocity(self):
        """ Get the average velocity of the vehicle """

        average_velocity = self.average_velocity

        return average_velocity

    def get_consumption_rate(self):
        """ Get the consumption rate of the vehicle """

        consumption_rate = self.consumption_rate

        return consumption_rate

    def get_battery_capacity(self):
        """ Get the battery capacity of the vehicle """

        battery_capacity = self.battery_capacity

        return battery_capacity

    def get_time_limit(self):
        """ Get the time limit for vehicle travel """

        time_limit = self.time_limit

        return time_limit
        
    def get_depot_node(self):
        """ Get the depot node from the graph """

        depot_node = feature_utils.get_depot(self.problem_graph)

        return depot_node
        
    def construct_route_from_nodes(self, nodes):
        """ Using the given ordered nodes, insert the depot to form a route """

        depot_node = self.get_depot_node()
        route = [depot_node] + nodes + [depot_node]

        return route

    def construct_route_arcs(self, route):
        """ Get the arcs forming the given route """

        arcs = [(a, b) for (a, b) in zip(route[:-1], route[1:])]
        
        return arcs

    def get_route_weights(self, route):
        """ Get the weights of the route edges """

        route_arcs = self.construct_route_arcs(route)
        weights = graph_utils.get_edges_weights(self.problem_graph, route_arcs)

        return weights

    def compute_route_length(self, route):
        """ Get the length of the given route """

        route_length = sum(self.get_route_weights(route))
    
        return route_length

    def compute_route_travel_duration(self, route):
        """ Get the duration needed to travel the given route """

        average_velocity = self.get_average_velocity()
        travel_distance = self.compute_route_length(route) 
        travel_duration = travel_distance / average_velocity
        
        return travel_duration

    def get_nodes_service_times(self, nodes):
        """ Get the service times of the given nodes in the graph """

        service_times = translate_evrpnl_graph.get_nodes_service_times(
            self.problem_graph, nodes)

        return service_times

    def compute_route_service_duration(self, route):
        """ Compute the time needed to service the nodes in the route """
        
        service_times = self.get_nodes_service_times(route)
        service_duration = sum([item or 0 for item in service_times])
    
        return service_duration

    def compute_route_energy_consumption(self, route):
        """ Compute the energy needed to complete the route """

        consumption_rate = self.get_consumption_rate()
        travel_distance = self.compute_route_length(route)
        route_consumption = travel_distance * consumption_rate
    
        return route_consumption

    def compute_route_duration(self, route):
        """ Compute the duration of travel and service for the given route """
        
        travel_duration = self.compute_route_travel_duration(route)
        service_duration = self.compute_route_service_duration(route)
        route_duration = travel_duration + service_duration

        return route_duration

    def get_direct_route(self, node):
        """ Get the direct route from the depot to the node and back """

        depot_node = self.get_depot_node()
        direct_route = [depot_node, node, depot_node]

        return direct_route
    
    def get_direct_route_duration(self, node):
        """ Duration of the route directly to a node and back to the depot """

        direct_route = self.get_direct_route(node)
        direct_route_duration = self.compute_route_duration(route)

        return direct_route_duration

    def _check_temporary_tsp_filename(self):
        """ Make sure we specified where to write the TSP file """

        if not(hasattr(self, "temporary_tsp_filename")):
            raise ValueError("Must set temporary TSP filename!")

        return None

    def set_temporary_tsp_filename(self, temporary_tsp_filename):
        """ Set the filename to write the TSP file to for Concorde """

        self.temporary_tsp_filename = temporary_tsp_filename

        return None

    def get_temporary_tsp_filename(self):
        """ Get the filename to write the TSP file to for Concorde """

        temporary_tsp_filename = self.temporary_tsp_filename

        return temporary_tsp_filename

    def _construct_working_graph(self):
        """ Construct a working graph we can use to get a TSP tour """

        working_graph = graph_utils.clone_graph(self.problem_graph)
        self._working_graph = evrpnl_utils.remove_stations_from_graph(working_graph)

        return None

    def _write_temporary_tsp_file(self):
        """ Write a temporary TSP file  """
    
        write_tsplib.write_tsp_nodes(
            self.temporary_tsp_filename, self._working_graph.graph["coord_dict"])

        return None

    def _solve_temporary_tsp_problem(self):
        """ Use Concorde to solve the temporary TSP problem """

        working_nodes = list(self._working_graph.nodes)
        solution = cc_solve.solution_from_path(self.temporary_tsp_filename)
        customer_tsp_tour = [working_nodes[item] for item in solution.tour]

        return customer_tsp_tour
        
    def solve_customer_tsp_tour(self):
        """ Get the customer TSP tour for the given problem """

        self._check_temporary_tsp_filename()
        self._construct_working_graph()
        self._write_temporary_tsp_file()
        
        self.customer_tsp_tour = self._solve_temporary_tsp_problem()

        return None

    def translate_problem_instance(self):
        """ Translate the problem instance from a graph to a dictionary """

        self.translated_instance = translate_evrpnl_graph.graph_to_translated_instance(
            self.problem_graph, self.relabel)

        return None

    def set_frvcp_solver(self, route):
        """ Set the FRVCP solver with the given route """

        if not(hasattr(self, "translated_instance")):
            self.translate_problem_instance()

        self.solver = solver.Solver(
            self.translated_instance, route, self.translated_instance["max_q"])

        return None

    def solve_route(self):
        """ Solve for the current route solver """

        duration, feasible_route = self.solver.solve()

        return duration, feasible_route

    def charge_fixed_route(self, route):
        """ Use the FRVCPY heuristic to solve the fixed route charging problem """
        
        self.set_frvcp_solver(route)
        duration, feasible_route = self.solve_route()

        return duration, feasible_route

    def _initialise_time_feasible_route_storage(self):
        """ Set the time feasible route storage """

        self._time_feasible_routes = []

        return None

    def _initialise_time_feasible_duration_storage(self):
        """ Set the time feasible route duration storage """

        self._time_feasible_durations = []

        return None

    def _store_time_feasible_route(self, time_feasible_route):
        """ Store the time feasible route """

        self._time_feasible_routes.append(time_feasible_route)

        return None

    def _store_time_feasible_duration(self, time_feasible_duration):
        """ Store the time feasible route duration """
        
        self._time_feasible_durations.append(time_feasible_duration)

        return None

    def get_customer_node_index(self, node):
        """ Get the TSP tour index of the given customer node """

        node_index = self.customer_tsp_tour.index(node)

        return node_index

    def get_time_feasible_routes_from_node(self, node):
        """ Get the time feasible routes that start at the given customer node """

        time_limit = self.get_time_limit()
        node_index = self.get_customer_node_index(node)
        
        for num in range(node_index + 1, len(self.customer_tsp_tour) + 1):
            route = self.construct_route_from_nodes(self.customer_tsp_tour[node_index:num])
            route_duration = self.compute_route_duration(route)
            if route_duration < time_limit:
                self._store_time_feasible_route(route)
                self._store_time_feasible_duration(route_duration)
            else:
                return None
            
        return None

    def get_time_feasible_routes(self):
        """ Get the initial time (possibly energy-infeasible) feasible routes """

        self._initialise_time_feasible_duration_storage()
        self._initialise_time_feasible_route_storage()

        for node in self.customer_tsp_tour[1:]:
            self.get_time_feasible_routes_from_node(node)

        return None

    def _initialise_energy_feasible_route_storage(self):
        """ Set the energy feasible route storage """

        self._energy_feasible_routes = []

        return None

    def _initialise_energy_feasible_route_plan_storage(self):
        """ Set the energy feasible route plan storage """

        self._energy_feasible_route_plans = []

        return None
    
    def _initialise_energy_feasible_duration_storage(self):
        """ Set the energy feasible route duration storage """

        self._energy_feasible_durations = []

        return None

    def _store_energy_feasible_route(self, energy_feasible_route):
        """ Store the energy feasible route """

        self._energy_feasible_routes.append(energy_feasible_route)

        return None

    def _store_energy_feasible_route_plan(self, energy_feasible_route_plan):
        """ Store the energy feasible route plan """

        self._energy_feasible_route_plans.append(energy_feasible_route_plan)

        return None
    
    def _store_energy_feasible_duration(self, energy_feasible_duration):
        """ Store the energy feasible route duration """
        
        self._energy_feasible_durations.append(energy_feasible_duration)

        return None

    def get_route_from_route_plan(self, route_plan):
        """ Get a route from a route plan """

        route = [item[0] for item in route_plan]

        return route

    def get_energy_feasible_routes(self):
        """ From the pool of time feasible routes, compute energy feasible routes """

        time_limit = self.get_time_limit()
        self._initialise_energy_feasible_duration_storage()
        self._initialise_energy_feasible_route_storage()
        self._initialise_energy_feasible_route_plan_storage()
        
        for route in self._time_feasible_routes:
            route_duration, route_plan = self.charge_fixed_route(route)
            if route_duration <= time_limit:
                route = self.get_route_from_route_plan(route_plan)
                route_plan = self.prune_unused_stations_from_plan(route_plan)
                self._store_energy_feasible_route(route)
                self._store_energy_feasible_route_plan(route_plan)
                self._store_energy_feasible_duration(route_duration)

        return None

    def _initialise_heuristic_solution_graph(self):
        """ Initialise a heuristic solution graph with direct customer-depot edges """
        
        self._heuristic_graph = nx.DiGraph()
        self._heuristic_graph.add_nodes_from(self.customer_tsp_tour)
    
        return None

    def get_energy_feasible_triples(self):
        """ Get the energy feasible triples """

        energy_feasible_routes = self._energy_feasible_routes
        energy_feasible_route_plans = self._energy_feasible_route_plans
        energy_feasible_durations = self._energy_feasible_durations

        triples = zip(
            energy_feasible_routes,
            energy_feasible_route_plans,
            energy_feasible_durations,
        )

        return triples

    def remove_stations_from_route(self, route):
        """ Remove the stations from the given route """

        route = [item for item in route if item in self.customer_tsp_tour]

        return route

    def prune_unused_stations_from_plan(self, plan):
        """ Given a plan, if a station is not used, prune it. """

        plan = [item for item in plan if item[1] != 0]

        return plan

    def add_feasible_routes(self):
        """ Get the time and energy feasible routes and add their edges to the heuristic graph """

        for triple in self.get_energy_feasible_triples():
            route, route_plan, route_duration = triple
            route = self.remove_stations_from_route(route)
            start_index = self.customer_tsp_tour.index(route[1]) - 1
            self._heuristic_graph.add_edges_from([
                (self.customer_tsp_tour[start_index], route[-2])],
                                           weight=route_duration,
                                           plan=route_plan
            )
            
        return None

    def build_heuristic_solution_graph(self):
        """ Build the graph from which we will find heuristic solutions """


        self.solve_customer_tsp_tour()
        self.get_time_feasible_routes()
        self.get_energy_feasible_routes()
        self._initialise_heuristic_solution_graph()
        self.add_feasible_routes()
        
        return None

    def get_heuristic_path_edges(self, heuristic_path):
        """ Get the edges of a path in the heuristic graph """

        heuristic_path_edges = [(a, b) for (a, b) in zip(heuristic_path, heuristic_path[1:])]

        return heuristic_path_edges


    def compute_heuristic_path_plans(self, heuristic_path):
        """ Get the plans of a path in the heuristic graph """

        heuristic_path_edges = self.get_heuristic_path_edges(heuristic_path)
        heuristic_path_plans = [self._heuristic_graph[edge[0]][edge[1]]["plan"]
                                for edge in heuristic_path_edges]

        return heuristic_path_plans

    def compute_heuristic_path_routes(self, heuristic_path):
        """ Get the routes of a path in the heuristic_graph """

        heuristic_path_plans = self.compute_heuristic_path_plans(heuristic_path)
        heuristic_path_routes = [[item[0] for item in plan] for plan in heuristic_path_plans]

        return heuristic_path_routes

    def compute_heuristic_path_duration(self, heuristic_path):
        """ Get the duration of a path in the heuristic graph """

        heuristic_path_edges = self.get_heuristic_path_edges(heuristic_path)
        heuristic_path_durations = [self._heuristic_graph[edge[0]][edge[1]]["weight"]
                                    for edge in heuristic_path_edges]
        heuristic_path_duration = sum(heuristic_path_durations)
        number_of_customers = len(feature_utils.get_customers(self.problem_graph))
        heuristic_path_duration = heuristic_path_duration - 0.5 * number_of_customers 

        return heuristic_path_duration


    def compute_shortest_heuristic_path(self):
        """ Compute the shortest heuristic path, if it can be done """

        shortest_path = nx.shortest_path(
            self._heuristic_graph, 0, self.customer_tsp_tour[-1], "weight"
        )

        return shortest_path

    def get_shortest_heuristic_solution(self):
        """ Get the shortest heuristic solution from the heuristic_graph """

        shortest_path = self.compute_shortest_heuristic_path()
        shortest_path_edges = self.get_heuristic_path_edges(shortest_path)
        self.shortest_solution_plans = self.compute_heuristic_path_plans(shortest_path)
        self.shortest_solution_routes = self.compute_heuristic_path_routes(shortest_path)
        self.shortest_solution_duration = self.compute_heuristic_path_duration(shortest_path)
        
        return None

    def solve_problem(self):
        """ Construct and solve the given problem """

        self.build_heuristic_solution_graph()
        try:
            self.get_shortest_heuristic_solution()
        except Exception as exception:
            self.shortest_solution_plan = []
            self.shortest_solution_routes = []
            self.shortest_solution_duration = np.inf

        return None

    def get_shortest_solution_duration(self):
        """ Get the duration of the shortest solution """

        shortest_solution_duration = self.shortest_solution_duration

        return shortest_solution_duration

    def get_shortest_solution_plans(self):
        """ Get the plans of the shortest solution """

        shortest_solution_plans = self.shortest_solution_plans

        return shortest_solution_plans

    def get_shortest_solution_routes(self):
        """ Get the routes of the shortest solution """

        shortest_solution_routes = self.shortest_solution_routes

        return shortest_solution_routes

    def compute_solution_edges_from_routes(self, solution_routes):
        """ Given routes that form a solution, compute all the edges in them """

        solution_edges = [self.construct_route_arcs(route) for route in solution_routes]
        solution_edges = self.flatten_list_of_lists(solution_edges)

        return solution_edges
        
    def get_shortest_solution_edges(self):
        """ Get all the edges of the best overall solution """

        shortest_solution_routes = self.get_shortest_solution_routes()
        shortest_solution_edges = self.compute_solution_edges_from_routes(shortest_solution_routes)

        return shortest_solution_edges

    def compute_solution_station_usage_from_routes(self, solution_routes):
        """ Given the routes that form solution, count how many time each station was used """

        prime_stations = feature_utils.get_prime_stations(self.problem_graph)
        solution_edges = self.compute_solution_edges_from_routes(solution_routes)
        used_nodes, usage_counts = np.unique(solution_edges, return_counts=True)
        station_count_dict = {used_node: int(usage_count/2) for (used_node, usage_count)
                              in zip(used_nodes, usage_counts) if used_node in prime_stations}

        return station_count_dict

    def get_shortest_solution_station_usage(self):
        """ Get the station usage in the shortest solution """
        
        shortest_solution_routes = self.get_shortest_solution_routes()
        shortest_solution_station_count_dict = self.compute_solution_station_usage_from_routes(
            shortest_solution_routes)

        return shortest_solution_station_count_dict
        
    def compute_all_heuristic_paths(self):
        """ Get all the solutions that we can find in the heuristic graph """

        all_heuristic_paths = nx.simple_paths.all_simple_paths(
            self._heuristic_graph, self.get_depot_node(), self.customer_tsp_tour[-1])
        self._all_heuristic_paths = list(all_heuristic_paths)

        return None

    def compute_all_heuristic_path_durations(self):
        """ Get all the solutions that we can find in the heuristic graph """

        all_heuristic_path_durations = [self.compute_heuristic_path_duration(heuristic_path)
                                        for heuristic_path in self._all_heuristic_paths]
        
        self._all_heuristic_path_durations = all_heuristic_path_durations
        
        return None

    def compute_tenth_percentile_duration_cutoff(self):
        """ Compute the tenth percentile of the found paths durations """

        tenth_percentile_cutoff = np.percentile(self._all_heuristic_path_durations, 10)

        return tenth_percentile_cutoff

    def compute_dictionary_mean(self, dictionaries):
        """ Compute the mean of the value for each key over all dictionaries """

        keys = dictionaries[0].keys()
        mean_dictionary = {}
        for key in keys:
            values = []
            for dictionary in dictionaries:
                values.append(dictionary[key])
            mean_dictionary[key] = np.mean(values)

        return mean_dictionary

    def compute_tenth_percentile_station_usages(self):
        """ For every found solution below the tenth percentile, compute station usage """

        station_usages = []
        for path, duration in zip(self._all_heuristic_paths, self._all_heuristic_path_durations):
            tenth_percentile_cutoff = self.compute_tenth_percentile_duration_cutoff()
            if duration <= tenth_percentile_cutoff:
                routes = self.compute_heuristic_path_routes(path)
                station_usage = self.compute_solution_station_usage_from_routes(routes)
                station_usages.append(station_usage)                
        return station_usages

    def compute_mean_tenth_percentile_station_usages(self):
        """ Compute the mean station usage over the tenth percentile station usages """

        self.compute_all_heuristic_paths()
        self.compute_all_heuristic_path_durations()
        tenth_percentile_station_usages = self.compute_tenth_percentile_station_usages()
        mean_station_usages = self.compute_dictionary_mean(tenth_percentile_station_usages)

        return mean_station_usages
        
    def get_all_routes(self):
        """ Get all of the routes that we can find in the heuristic graph """

        all_edges = graph_utils.get_all_edges(self._heuristic_graph)
        all_plans = [self._heuristic_graph[edge[0]][edge[1]]["plan"] for edge in all_edges]
        all_routes = [self.get_route_from_route_plan(plan) for plan in all_plans]
    
        return all_routes

    def flatten_list_of_lists(self, list_of_lists):
        """ Flatten a list of lists """

        flattened_list = [item for sublist in list_of_lists for item in sublist]

        return flattened_list

    def get_all_used_nodes(self, return_counts=False):
        """ Get all of the used nodes in the routes of the heuristic_graph """

        all_routes = self.get_all_routes()
        all_nodes = self.flatten_list_of_lists(all_routes)
    
        return np.unique(all_nodes, return_counts=return_counts)

    def get_all_route_edges(self):
        """ Get all route edges """

        all_routes = self.get_all_routes()
        all_route_edges = [self.construct_route_arcs(route) for route in all_routes]
        all_route_edges = self.flatten_list_of_lists(all_route_edges)

        return all_route_edges

    def get_all_route_durations(self):
        """ Get the duration of every stored route """

        all_edges = graph_utils.get_all_edges(self._heuristic_graph)
        all_durations = [self._heuristic_graph[edge[0]][edge[1]]["weight"] for edge in all_edges]

        return all_durations

    def remove_depot_from_route(self, route):
        """ Remove the depot from each route """

        route = route[1:-1]

        return route

    def get_all_route_customer_counts(self):
        """ Get the number of customers in each route """

        all_routes = self.get_all_routes()
        all_routes = [self.remove_stations_from_route(route) for route in all_routes]
        all_routes = [self.remove_depot_from_route(route) for route in all_routes]
        all_customer_counts = [len(item) for item in all_routes]

        return all_customer_counts

    def count_edges_occurrences(self):
        """ Get a dictionary in which an edge is a key and the value is occurences """

        all_route_edges = self.get_all_route_edges()
        unique_edges, edge_counts = np.unique(all_route_edges, axis=0, return_counts=True)
        try:
            edge_counts = edge_counts / edge_counts.max()
        except Exception as exception:
            print(all_route_edges)
            raise exception
            
        count_dict = {tuple(edge): count for (edge, count) in zip(unique_edges, edge_counts)}

        return count_dict

    def get_all_routes_station_usage(self):
        """ Get the station usage among all routes """
        
        prime_stations = feature_utils.get_prime_stations(self.problem_graph)
        all_edges = self.get_all_route_edges()
        uniques, counts = np.unique(all_edges, return_counts=True)
        count_dict = {unique: int(count  / 2) for (unique, count) in zip(uniques, counts)
                      if unique in prime_stations}

        return count_dict
