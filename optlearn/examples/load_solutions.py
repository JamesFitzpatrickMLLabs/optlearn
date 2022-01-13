import os
import copy

import numpy as np
import networkx as nx

from optlearn.io.vrp import vrp_utils
from optlearn.io.graph import graph_io
from optlearn.mip.routing import evrpnl_problem_builder
from optlearn.graph import process_utils


def remove_clone_vertices(graph):
    """ Remove cloned station vertices from the graph """

    clone_stations = get_clone_stations(graph)
    graph.remove_nodes_from(clone_stations)

    return graph


def remove_clone_edges(graph):
    """ Remove cloned station edges from the graph """

    clone_stations = get_clone_stations(graph)
    clone_edges = [edge for edge in graph.edges
                   if edge[0] in clone_stations or edge[1] in clone_stations]
    graph.remove_edges_from(clone_edges)

    return graph


def remove_clones(graph):
    """ Remove clone items from the given edges """

    graph = remove_clone_vertices(graph)
    graph = remove_clone_edges(graph)

    return graph


def read_problem_graph(filename):
    """ Read a pickled graph """

    graph = graph_io.read_pickle(filename)
    graph = remove_clones(graph)
    
    return graph


def read_solution_graph(filename):
    """ Read a pickled graph """

    graph = graph_io.read_pickle(filename)

    return graph


def get_stations(graph):
    """ Get the stations of the graph """

    stations = graph.graph["node_types"]["station"]

    return stations


def get_depot(graph):
    """ Get the depot of the graph """

    depot = graph.graph["node_types"]["depot"][0]

    return depot


def get_customers(graph):
    """ Get the customers of the graph """

    customers = graph.graph["node_types"]["customer"]

    return customers


def build_support_graph(graph):
    """ Build a support graph from the solvals """

    support_graph = type(graph)()
    solvals = [graph[edge[0]][edge[1]]["solval"] for edge in graph.edges]
    support_graph.add_edges_from([(*edge, {"weight": val})
                                  for (edge, val) in zip (graph.edges, solvals) if val > 0])
    return support_graph


def get_solution_cycles(graph):
    """ Get the solution cycles """
    
    cycles = []
        
    support_graph = build_support_graph(graph)
    while len(support_graph.edges) != 0:
        cycle = nx.find_cycle(support_graph)
        cycles.append(cycle)
        support_graph.remove_edges_from(cycle)

    return cycles


def get_used_vertices(graph):
    """ Get the used vertices in the graph """

    vertices = []
    cycles = get_solution_cycles(graph)
    uniques = [np.unique(cycle).tolist() for cycle in cycles]
    for item in uniques:
        vertices += item
    uniques = np.unique(vertices)
    
    return uniques


def get_used_stations(graph):
    """ get the used stations in the graph """

    stations = get_stations(graph)
    used_vertices = get_used_vertices(graph)
    used_stations = [vertex for vertex in used_vertices if vertex in stations]

    return used_stations


def get_prime_stations(graph):
    """ Get the prime stations """

    stations = get_stations(graph)
    primes = [graph.nodes[station].get("prime_node") or station for station in stations]

    return primes


def get_clone_stations(graph):
    """ Get the clone stations """

    stations = get_stations(graph)
    primes = get_prime_stations(graph)
    clones = [station for station in stations if station not in primes]

    return clones


def get_used_prime_stations(graph):
    """ Get the used prime stations """

    stations = get_stations(graph)
    used_vertices = get_used_vertices(graph)
    used_stations = [vertex for vertex in used_vertices if vertex in stations]
    used_stations = [graph.nodes[station].get("prime_node") or station for station in used_stations]

    return used_stations


def get_vertex_coordinate(graph, vertex):
    """ Get the coordinate of the vertex """

    coordinate = np.array(graph.graph["coord_dict"][vertex])

    return coordinate

def get_vertices_coordinates(graph, vertices):
    """ Get the corodinates of the vertices """

    coordinates = [get_vertex_coordinate(graph, vertex) for vertex in vertices]

    return coordinates


def get_station_customer_distances(graph, station, customers):
    """ Get the distance from the station to each customer """ 

    station_coordinate = get_vertex_coordinate(graph, station)
    customer_coordinates = get_vertices_coordinates(graph, customers)

    distances = [vrp_utils.euclidean(station_coordinate, customer_coordinate)
                 for customer_coordinate in customer_coordinates]

    return distances


def get_strictly_reachable_customers(graph, station, reachability_radius):
    """ Get the set of customers that are reachable from the given station or depot """

    customers = get_customers(graph)
    dists = get_station_customer_distances(graph, station, customers)
    customers = [customer
                 for (customer, dist) in zip(customers, dists) if dist <= reachability_radius / 2]
    
    return customers


def get_nondepot_strictly_reachable_customers(graph, station, reachability_radius):
    """ Get the set of customers that are reachable from the given station (but not depot) """

    depot = get_depot(graph)
    station_reachable_customers = get_strictly_reachable_customers(
        graph,
        station,
        reachability_radius,
    )
    depot_reachable_customers = get_strictly_reachable_customers(
        graph,
        depot,
        reachability_radius,
    )
    nondepot_reachable_customers = [customer for customer in station_reachable_customers
                                    if customer not in depot_reachable_customers]
    
    return nondepot_reachable_customers


def get_unique_strictly_reachable_customers(graph, station, reachability_radius):
    """ Get the set of customers that only strictly reachable from the given station or depot """

    stations = [get_depot(graph)] + get_stations(graph)
    stations = [item for item in stations if item != station] 
    station_reachable_customers = get_strictly_reachable_customers(
        graph,
        station,
        reachability_radius,
    )
    other_reachable_customers = []
    for other_station in stations:
        current_reachable_customers = get_strictly_reachable_customers(
            graph,
            other_station,
            reachability_radius,
        )
        other_reachable_customers += current_reachable_customers
    unique_reachable_customers = [customer for customer in station_reachable_customers
                                  if customer not in other_reachable_customers]
    
    return unique_reachable_customers


def get_mean_customer_separation(graph, customers, reachability_radius):
    """ Get the mean separation of the customers """

    customer_coordinates = get_vertices_coordinates(graph, customers)
    if len(customer_coordinates) == 0:
        return 0
    separations = vrp_utils.compute_pairwise_distances(
        customer_coordinates,
        vrp_utils.euclidean,
        rounding=16
    )
    mean_separation = np.mean(separations)

    return mean_separation


def get_std_customer_separation(graph, customers, reachability_radius):
    """ Get the std separation of the customers """

    customer_coordinates = get_vertices_coordinates(graph, customers)
    if len(customer_coordinates) == 0:
        return 0
    separations = vrp_utils.compute_pairwise_distances(
        customer_coordinates,
        vrp_utils.euclidean,
        rounding=16
    )
    std_separation = np.std(separations)

    return std_separation


def get_depot_distance(graph, station, reachability_radius):
    """ Get the distance from the station to the depot """

    vertices = [get_depot(graph), station]
    mean_distance = get_mean_customer_separation(graph, vertices, reachability_radius)
    
    return mean_distance


def get_mean_station_distance(graph, station, customers, reachability_radius):
    """ Get the mean distances from the station to the customers """

    distances = get_station_customer_distances(graph, station, customers)
    mean_distance = np.mean(distances)
    
    return mean_distance


def get_std_station_distance(graph, station, customers, reachability_radius):
    """ Get the std distances from the station to the customers """

    distances = get_station_customer_distances(graph, station, customers)
    std_distance = np.std(distances)
    
    return std_distance


def get_reachability_mean_separation(graph, station, reachability_radius):
    """ Get the mean separation accounting for reachability """

    strictly_reachable_customers = get_strictly_reachable_customers(
        graph,
        station,
        reachability_radius,
    )
    mean_customer_separations = get_mean_customer_separation(
        graph,
        strictly_reachable_customers,
        reachability_radius,
    )

    return mean_customer_separations


def get_reachability_std_separation(graph, station, reachability_radius):
    """ Get the std separation accounting for reachability """

    strictly_reachable_customers = get_strictly_reachable_customers(
        graph,
        station,
        reachability_radius,
    )
    std_customer_separations = get_std_customer_separation(
        graph,
        strictly_reachable_customers,
        reachability_radius,
    )

    return std_customer_separations


def get_reachability_mean_distance(graph, station, reachability_radius):
    """ Get the mean distance accounting for reachability """

    strictly_reachable_customers = get_strictly_reachable_customers(
        graph,
        station,
        reachability_radius,
    )
    mean_customer_distances = get_mean_station_distance(
        graph,
        station,
        strictly_reachable_customers,
        reachability_radius
    )

    return mean_customer_distances


def get_reachability_std_distance(graph, station, reachability_radius):
    """ Get the std distance accounting for reachability """

    strictly_reachable_customers = get_strictly_reachable_customers(
        graph,
        station,
        reachability_radius,
    )
    std_customer_distances = get_std_station_distance(
        graph,
        station,
        strictly_reachable_customers,
        reachability_radius
    )

    return std_customer_distances


def get_strict_reachability_number(graph, station, reachability_radius):
    """ Get the number of customers within the reachability radius """
    
    customers = get_strictly_reachable_customers(
        graph,
        station,
        reachability_radius,
    )
    number_of_strictly_reachable_customers = len(customers)

    return number_of_strictly_reachable_customers


def get_strict_nondepot_reachability_number(graph, station, reachability_radius):
    """ Get the number of customers within the reachability radius """

    customers = get_nondepot_strictly_reachable_customers(
        graph,
        station,
        reachability_radius,
    )
    number_of_nondepot_strictly_reachable_customers = len(customers)
    
    return number_of_nondepot_strictly_reachable_customers


def get_strict_unique_reachability_number(graph, station, reachability_radius):
    """ Get the number of customers only within the reachability radius """

    customers = get_unique_strictly_reachable_customers(
        graph,
        station,
        reachability_radius,
    )
    number_of_unique_strictly_reachable_customers = len(customers)
    
    return number_of_unique_strictly_reachable_customers


def get_depot_cosine(graph, station):
    """ Get the cosine of the angle between the horizontal axis and the station vector """

    depot = get_depot(graph)
    depot_coordinate = get_vertex_coordinate(graph, depot)
    station_coordinate = get_vertex_coordinate(graph, station)

    difference = np.array(station_coordinate) - np.array(depot_coordinate)
    cosine = np.dot(difference, [1, 0]) / np.linalg.norm(difference)

    return cosine


def get_global_cosine(graph, station):
    """ Get the cosine of the angle between the horizontal axis and the station vector """

    depot = get_depot(graph)
    depot_coordinate = get_vertex_coordinate(graph, depot)
    station_coordinate = get_vertex_coordinate(graph, station)

    cosine = np.dot(station_coordinate, depot_coordinate)
    cosine /= (np.linalg.norm(station_coordinate) * np.linalg.norm(depot_coordinate))

    return cosine


def get_mean_coordinate(graph, vertices):
    """ Get the mean coordinate (centre) of the given vertices """

    coordinates = get_vertices_coordinates(graph, vertices)
    mean_coordinate = np.mean(coordinates, axis=0)

    return mean_coordinate

def get_mean_cosine(graph, station, vertices):
    """ Get the mean cosine between the given station axis and the vertices """



def get_reachability_mean_station_cosine(graph, station, reachability_radius):
    """ Get the mean station cosine of the reachable customers for the given station """

    strictly_reachable_customers = get_strictly_reachable_customers(
        graph,
        station,
        reachability_radius,
    )
    mean_reachable_coordinate = get_mean_coordinate(
        graph,
        strictly_reachable_customers,
    )
    station_coordinate = get_vertex_coordinate(graph, station)
    difference = mean_reachable_coordinate - station_coordinate
    cosine = np.dot(difference, [1, 0]) / np.linalg.norm(difference)

    return cosine


def get_nondepot_reachability_mean_station_cosine(graph, station, reachability_radius):
    """ Get the mean station cosine of the nondepot reachable customers for the given station """

    strictly_reachable_customers = get_nondepot_strictly_reachable_customers(
        graph,
        station,
        reachability_radius,
    )
    mean_reachable_coordinate = get_mean_coordinate(
        graph,
        strictly_reachable_customers,
    )
    station_coordinate = get_vertex_coordinate(graph, station)
    difference = mean_reachable_coordinate - station_coordinate
    cosine = np.dot(difference, [1, 0]) / np.linalg.norm(difference)

    return cosine


def get_unique_reachability_mean_station_cosine(graph, station, reachability_radius):
    """ Get the mean station cosine of the unique reachable customers for the given station """

    strictly_reachable_customers = get_unique_strictly_reachable_customers(
        graph,
        station,
        reachability_radius,
    )
    mean_reachable_coordinate = get_mean_coordinate(
        graph,
        strictly_reachable_customers,
    )
    station_coordinate = get_vertex_coordinate(graph, station)
    difference = mean_reachable_coordinate - station_coordinate
    cosine = np.dot(difference, [1, 0]) / np.linalg.norm(difference)

    return cosine


def get_reachability_std_station_cosine(graph, station, reachability_radius):
    """ Get the std station cosine of the reachable customers for the given station """

    strictly_reachable_customers = get_strictly_reachable_customers(
        graph,
        station,
        reachability_radius,
    )
    mean_reachable_coordinate = get_mean_coordinate(
        graph,
        strictly_reachable_customers,
    )
    station_coordinate = get_vertex_coordinate(graph, station)
    difference = mean_reachable_coordinate - station_coordinate
    cosine = np.dot(difference, [1, 0]) / np.linalg.norm(difference)

    return cosine


def get_nondepot_reachability_std_station_cosine(graph, station, reachability_radius):
    """ Get the std station cosine of the nondepot reachable customers for the given station """

    strictly_reachable_customers = get_nondepot_strictly_reachable_customers(
        graph,
        station,
        reachability_radius,
    )
    mean_reachable_coordinate = get_mean_coordinate(
        graph,
        strictly_reachable_customers,
    )
    station_coordinate = get_vertex_coordinate(graph, station)
    difference = mean_reachable_coordinate - station_coordinate
    cosine = np.dot(difference, [1, 0]) / np.linalg.norm(difference)

    return cosine


def get_unique_reachability_std_station_cosine(graph, station, reachability_radius):
    """ Get the std station cosine of the unique reachable customers for the given station """

    strictly_reachable_customers = get_unique_strictly_reachable_customers(
        graph,
        station,
        reachability_radius,
    )
    mean_reachable_coordinate = get_mean_coordinate(
        graph,
        strictly_reachable_customers,
    )
    station_coordinate = get_vertex_coordinate(graph, station)
    difference = mean_reachable_coordinate - station_coordinate
    cosine = np.dot(difference, [1, 0]) / np.linalg.norm(difference)

    return cosine


def get_station_type(graph, station):
    """ Get the station type """

    if graph.nodes[station]["cs_type"] == "slow":
        return 0
    if graph.nodes[station]["cs_type"] == "normal":
        return 0.5
    if graph.nodes[station]["cs_type"] == "fast":
        return 1
    raise Exception("No CS type!")


def compute_station_feature(graph, station, reachability_radius):
    """ Compute the feature vector for a goven station of the graph """

    feature = [
        get_reachability_mean_distance(graph, station, reachability_radius) / 128,
        get_reachability_std_distance(graph, station, reachability_radius) / 128,
        get_reachability_mean_separation(graph, station, reachability_radius) / 128,
        get_reachability_std_separation(graph, station, reachability_radius) / 128,
        get_strict_reachability_number(graph, station, reachability_radius) / 128,
        get_strict_nondepot_reachability_number(graph, station, reachability_radius) / 10,
        get_strict_unique_reachability_number(graph, station, reachability_radius) / 10,
        get_depot_distance(graph, station, reachability_radius) / 128,
        # get_depot_cosine(graph, station),
        # get_global_cosine(graph, station),
        # get_reachability_mean_station_cosine(graph, station, reachability_radius),
        # get_nondepot_reachability_mean_station_cosine(graph, station, reachability_radius),
        # get_unique_reachability_mean_station_cosine(graph, station, reachability_radius),
        # get_station_type(graph, station)
    ]

    return feature


from optlearn.feature.vrp import node_features


def compute_stations_features(graph, reachability_radius):
    """ Compute the station features for the stations in the graph """
    
    stations = np.unique(get_prime_stations(graph))
    features = [compute_station_feature(graph, station, reachability_radius)
                for station in stations]
    features = np.nan_to_num(features, -1)
    # features = [node_features.compute_station_strict_reachability_features(
    #     graph, station, reachability_radius) for station in stations]

    return features


def count_station_uses(graph, station):
    """ Using the solution graph, count how often a station is used """

    used_stations = get_used_prime_stations(graph)
    number_of_uses = used_stations.count(station)

    return number_of_uses


def compute_stations_labels(graph):
    """ Using the solution graph, count how often each station is used """

    stations = np.unique(get_prime_stations(graph))
    labels = [count_station_uses(graph, station) for station in stations]

    return labels


def get_problem_filenames(directory):
    """ Get all of the problem filenames in the directory """

    filenames = os.listdir(directory)
    filepaths = [os.path.join(directory, filename) for filename in filenames]
    filepaths = [item for item in filepaths if "pkl" in item]
    filepaths = [item for item in filepaths if "solution" not in item]
    
    return filepaths


def get_solution_filenames(directory):
    """ Get all of the solution filenames in the directory """

    filenames = os.listdir(directory)
    filepaths = [os.path.join(directory, filename) for filename in filenames]
    filepaths = [item for item in filepaths if "pkl" in item]
    filepaths = [item for item in filepaths if "solution" in item]
    
    return filepaths


def get_problem_solution_pairs(directory):
    """ Get the problema and solution filenames """

    problem_filenames = get_problem_filenames(directory)
    solution_filenames = [item.split(".")[0] + "_solution.pkl" for item in problem_filenames]
    pairs = list(zip(problem_filenames, solution_filenames))

    return pairs


solve_times, features = [], []

training_pairs = get_problem_solution_pairs("/home/james/transfer/")
for (problem_filename, solution_filename) in training_pairs:
    problem_graph = read_problem_graph(problem_filename)
    solution_graph = read_solution_graph(solution_filename)
    solve_times.append(solution_graph.graph["solve_time"])
    # features.append(np.array(compute_stations_features(problem_graph, 128)).max(axis=0))

    
# train_feature_set, train_label_set = [], []

# problem_solution_pairs = get_problem_solution_pairs("/home/james/transfer/")

# for (problem_filename, solution_filename) in problem_solution_pairs:
#     problem_graph = read_problem_graph(problem_filename)
#     problem_graph = node_features.process_strict_reachabilities(problem_graph, 128)
#     solution_graph = read_solution_graph(solution_filename)
#     features = np.array(compute_stations_features(problem_graph, 128))
#     # features = np.nan_to_num(features, -1)
#     labels = np.array(compute_stations_labels(solution_graph))
#     train_feature_set.append(features)
#     train_label_set.append(labels)

# train_feature_set = np.vstack(train_feature_set)
# train_label_set = np.concatenate(train_label_set)

# # problem_builder = evrpnl_problem_builder.evrpnlProblemBuilder(
# #     solver_package="xpress",
# #     problem_type="evrpnl",
# #     is_directed=True,
# #     pwl_model="delta",
# # )
# # problem_builder.build_arc_problem(solution_graph)
# # problem_builder.problem.solve()

# # problem_builder.plot_problem(solution_graph, show=False, strict=True)
# # problem_builder.plot_solution(solution_graph, show=True)


# from sklearn import svm
# from sklearn import linear_model
# from sklearn.neighbors import KNeighborsRegressor

# from optlearn.feature.vrp import node_features

# pois_reg = linear_model.PoissonRegressor()
# svm_reg = svm.SVR(kernel='poly', degree=8)
# # svm_reg = svm.SVR(kernel='rbf')
# knn_reg = KNeighborsRegressor(n_neighbors=4)

# pois_reg.fit(train_feature_set, train_label_set)
# svm_reg.fit(train_feature_set, train_label_set)
# knn_reg.fit(train_feature_set, train_label_set)

# import torch


# val_feature_set, val_label_set = [], []

# problem_solution_pairs = get_problem_solution_pairs("/home/james/problems/")

# for (problem_filename, solution_filename) in problem_solution_pairs:
#     print(problem_filename)
#     problem_graph = read_problem_graph(problem_filename)
#     problem_graph = node_features.process_strict_reachabilities(problem_graph, 128)
#     solution_graph = read_solution_graph(solution_filename)
#     features = np.array(compute_stations_features(problem_graph, 128))
#     labels = np.array(compute_stations_labels(solution_graph))
#     val_feature_set.append(features)
#     val_label_set.append(labels)

# val_feature_set = np.vstack(val_feature_set)
# val_label_set = np.concatenate(val_label_set)
# val_preds = np.round(svm_reg.predict(val_feature_set)).astype(int)

# svm_reg = svm.SVR(kernel='rbf')
# svm_reg.fit(decomposition, train_label_set)

# val_feature_set, val_label_set = [], []

# problem_solution_pairs = get_problem_solution_pairs("/home/james/problems/")

# for (problem_filename, solution_filename) in problem_solution_pairs:
#     print(problem_filename)
#     problem_graph = read_problem_graph(problem_filename)
#     problem_graph = node_features.process_strict_reachabilities(problem_graph, 128)
#     solution_graph = read_solution_graph(solution_filename)
#     features = np.array(compute_stations_features(problem_graph, 128))
#     labels = np.array(compute_stations_labels(solution_graph))
#     val_feature_set.append(features)
#     val_label_set.append(labels)

# val_feature_set = np.vstack(val_feature_set)
# val_label_set = np.concatenate(val_label_set)
# val_preds = np.round(svm_reg.predict(pca.transform(val_feature_set))).astype(int)


# # x = train_feature_set[:, 0]
# # y = train_feature_set[:, 1]
# x = decompsition[:, 0]
# y = decompsition[:, 1]

# colours = ["red", "purple", "green", "blue", "grey", "black", "pink"]
# markers = ["x", "^", ".", "*", "v", ","]

# for label in np.unique(train_label_set):
#     plt.scatter(
#         x[train_label_set==label],
#         y[train_label_set==label],
#         marker=markers[label],
#         color=colours[label],
#         s=25,
#     )
# plt.legend(np.unique(train_label_set))
# plt.grid()
# plt.show()
