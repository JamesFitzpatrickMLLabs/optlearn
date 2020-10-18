import time

import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

from optlearn import experiment_utils
from optlearn import graph_utils
from optlearn import io_utils
from optlearn import cc_solve

from optlearn.mst import mst_model
from optlearn.mip import mip_model


files = experiment_utils.list_files_recursive("/home/james/Data/MATHILDA/tsp/")

model = mst_model.mstSparsifier()

ones, twos, threes, fours, fives = [], [], [], [], []

optima, guesses, times = [], [], []


for num, file in enumerate(files):
    
    object = io_utils.optObject().read_from_file(file)
    graph = object.get_graph()
    graph = graph_utils.delete_self_weights(graph)
    graph = graph.to_undirected()
    
    solution = cc_solve.solution_from_path(file)
    edges = graph_utils.get_tour_edges(solution.tour)
    if np.min(solution.tour) < 1 and np.min(graph.nodes) > 0:
        edges += 1
    edges =  model.tuples_to_strings(edges)
    
    sparse_edges = model.run_sparsify(graph, 6)
    prune_edges = []
    for item in sparse_edges:
        prune_edges += list(item)
    prune_weights = [graph_utils.get_edge_weight(graph, *edge) for edge in prune_edges]
    prune_bunches = [(*edge, {"weight": weight}) for
                     (edge, weight) in zip(prune_edges, prune_weights)]
    
    g = nx.Graph()
    g.add_edges_from(prune_bunches)

    all_weights = []

    for mun, vertex in enumerate(graph.nodes):
        edges = [edge for edge in graph.edges if vertex in edge]
        weights = [9999999] * len(edges)
        for pum, edge in enumerate(edges):
            if edge in prune_edges:
                index = prune_edges.index(edge)
                weights[pum] = prune_weights[index]
        weights.insert(mun, 9999999)
        all_weights.append(weights)

    all_weights = np.array(all_weights)
    all_weights = np.maximum(all_weights, all_weights.transpose())

    object.write_edges_explicit("/home/james/temp.tsp", all_weights)

    new_solution = cc_solve.solution_from_path("/home/james/temp.tsp")

        
    # problem = mip_model.tspProblem(graph=g,
    #                                var_type="binary",
    #                                formulation="dantzig",
    #                                solver="xpress",
    #                                verbose=False,
    # )

    # problem.problem.setControl({"presolve": 0})
    # start = time.time()
    # problem.solve()
    # end = time.time()

    optimum = solution.optimal_value
    guess = new_solution.optimal_value
    
    plt.scatter(num, guess / optimum, marker="x", color="black")
    plt.pause(0.01)
    optima.append(optimum)
    guesses.append(guess)
    # times.append(end - start)

plt.savefig("/home/james/pruning.png")
