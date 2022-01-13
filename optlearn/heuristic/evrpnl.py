import numpy as np
import networkx as nx

from frvcpy import solver

from optlearn import cc_solve

from optlearn.graph import graph_utils

from optlearn.heuristic import evrpnl_utils
from optlearn.heuristic import translate_evrpnl_graph

from optlearn.io.tsp import write_tsplib

from optlearn.feature.vrp import feature_utils


def construct_route(depot_node, other_nodes):
    """ Get the route for the given nodes """

    route = [depot_node] + other_nodes + [depot_node]

    return route


def construct_route_arcs(route):
    """ Get the arcs forming the given route """

    arcs = [(a, b) for (a, b) in zip(route[:-1], route[1:])]

    return arcs


def compute_route_length(graph, nodes):
    """ Get the length of the route with the given nodes in it """

    depot_node = feature_utils.get_depot(graph)
    route = construct_route(depot_node, nodes)
    route_arcs = construct_route_arcs(route)
    weights = graph_utils.get_edges_weights(graph, route_arcs)
    route_length = sum(weights)
    
    return route_length


def compute_route_travel_duration(graph, nodes, average_velocity=40):
    """ Get the duration needed to travel the given route """

    travel_duration = compute_route_length(graph, nodes) / average_velocity
    
    return travel_duration


def compute_route_service_duration(graph, nodes):
    """ Compute the time needed to service the nodes in the route """

    service_times = graph_utils.get_node_attributes(graph, nodes, "service_time")
    service_duration = sum([item or 0 for item in service_times])
    
    return service_duration


def compute_route_energy_consumption(graph, nodes, consumption_rate):
    """ Compute the energy needed to complete the route """
    
    depot_node = feature_utils.get_depot(graph)
    route = construct_route(depot_node, nodes)
    route_arcs = construct_route_arcs(route)
    weights = graph_utils.get_edges_weights(graph, route_arcs)
    route_consumption = sum(weights) * consumption_rate
    
    return route_consumption


def compute_route_duration(graph, nodes, average_velocity=40):
    """ Compute the duration of travel and service for the given route """

    travel_duration = compute_route_travel_duration(graph, nodes, average_velocity)
    service_duration = compute_route_service_duration(graph, nodes)
    route_duration = travel_duration + service_duration

    return route_duration
    

def get_direct_route_duration(graph, node, average_velocity=40):
    """ Duration of the route directly to a node and back to the depot """

    depot_node = feature_utils.get_depot(graph)
    route_duration = compute_route_duration(graph, [node], average_velocity)

    return route_duration


def get_time_feasible_routes(graph, node, customer_tsp_tour, average_velocity=40, time_limit=10):
    """ Given the customer TSP tour, find the time feasible ordered routes starting at the node """

    time_feasible_routes_nodes = []
    time_feasible_durations = []
    node_index = customer_tsp_tour.index(node)
    for num in range(node_index +1, len(customer_tsp_tour) + 1):
        route_duration = compute_route_duration(graph, customer_tsp_tour[node_index:num])
        if route_duration <= time_limit:
            time_feasible_routes_nodes.append(customer_tsp_tour[node_index:num])
            time_feasible_durations.append(route_duration)
        else:
            return time_feasible_durations, time_feasible_routes_nodes

    return time_feasible_durations, time_feasible_routes_nodes


def get_customer_tsp_tour(graph):
    """ Get the customer TSP tour for the given problem """

    working_graph = graph_utils.clone_graph(graph)
    working_graph = evrpnl_utils.remove_stations_from_graph(working_graph)
    working_nodes = np.array(list(working_graph.nodes))
    write_tsplib.write_tsp_nodes("/home/james/problem.tsp", working_graph.graph["coord_dict"])
    solution = cc_solve.solution_from_path("/home/james/problem.tsp") 
    customer_tsp_tour = [working_nodes[item] for item in solution.tour]

    return customer_tsp_tour


def charge_fixed_route(graph, route):
    """ Use the FRVCPY heuristic to solve the fixed route charging problem """

    translated_instance = translate_evrpnl_graph.graph_to_translated_instance(graph)
    frvcp_solver = solver.Solver(translated_instance, route, translated_instance["max_q"])
    duration, feasible_route = frvcp_solver.solve()

    return duration, feasible_route


def get_energy_feasible_routes(graph, routes_nodes, time_limit):
    """ Charge the fixed routes and keep them if they satisfy the time limit """

    triples = []
    depot_node = feature_utils.get_depot(graph)
    for route_nodes in routes_nodes:
        route = [depot_node] + route_nodes + [depot_node]
        duration, feasible_route = charge_fixed_route(graph, route)
        if duration <= time_limit:
            triples.append((duration, route_nodes, feasible_route))
        else:
            return triples
    return triples
            

def add_feasible_routes(graph, heuristic_graph, customer_tsp_tour, routes_nodes, time_limit):
    """ Get the time and energy feasible routes and add their edges to the heuristic graph """

    triples = get_energy_feasible_routes(graph, routes_nodes, time_limit)
    for triple in triples:
        duration, route_nodes, plan = triple
        start_index = customer_tsp_tour.index(route_nodes[0]) - 1
        heuristic_graph.add_edges_from([
            (customer_tsp_tour[start_index], route_nodes[-1])], weight=duration, plan=plan
        )
    return heuristic_graph


def initialise_heuristic_solution_graph(customer_tsp_tour, average_velocity=40, time_limit=10):
    """ Initialise a heuristic solution graph with direct customer-depot edges """

    heuristic_graph = nx.DiGraph()
    heuristic_graph.add_nodes_from(customer_tsp_tour)
    
    return heuristic_graph


def build_heuristic_solution_graph(graph, average_velocity=40, time_limit=10):
    """ Build the graph from which we will find heuristic solutions """

    customer_tsp_tour = get_customer_tsp_tour(graph)
    heuristic_graph = initialise_heuristic_solution_graph(
        customer_tsp_tour,
        average_velocity,
        time_limit
    )
    for (num, node) in enumerate(customer_tsp_tour[1:]):
        completion = round((num + 1) / (len(customer_tsp_tour) - 1) * 100, 3)
        print(f"Solving fixed route vehicle charging problems: {completion}%")
        time_feasible_durations, time_feasible_routes_nodes = get_time_feasible_routes(
            graph, node,
            customer_tsp_tour,
            average_velocity,
            time_limit,
        )
        heuristic_graph = add_feasible_routes(
            graph, heuristic_graph, customer_tsp_tour, time_feasible_routes_nodes, time_limit
            
        )
        
    return heuristic_graph


def get_shortest_heuristic_solution(heuristic_graph, customer_tsp_tour):
    """ Get the shortest heuristic solution from the heuristic_graph """

    shortest_path = nx.shortest_path(
        heuristic_graph, 0, customer_tsp_tour[-1], "weight"
    )
    shortest_path_edges = [(a, b) for (a, b) in zip(shortest_path, shortest_path[1:])]
    shortest_path_plan = [heuristic_graph[edge[0]][edge[1]]["plan"]
                          for edge in shortest_path_edges]

    return shortest_path-plan


from optlearn.io.vrp import vrp_utils
from optlearn.graph import process_utils
from optlearn.examples import load_solutions

# problem_graph = vrp_utils.read_evrpnl_problem("/home/james/Downloads/evrpnl/tc0c10s2cf1.xml")
problem_graph = load_solutions.read_problem_graph("/home/james/transfer/00a61420cc.pkl") 
problem_graph = process_utils.duplicate_stations_nonuniform(problem_graph, {2: 1})
customer_tsp_tour = get_customer_tsp_tour(problem_graph)
heuristic_graph = build_heuristic_solution_graph(problem_graph)
