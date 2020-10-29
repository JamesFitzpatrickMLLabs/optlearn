import networkx as nx

import matplotlib.pyplot as plt

from optlearn import graph_utils


def plot_tour(object, tour, title="TSP Tour"):
    """ Plot the tour for a given object problem instance """

    path = graph_utils.append_last_node(tour)
    coords = object._problem.node_coords

    xs = [coords[key][0] for key in coords.keys()]
    ys = [coords[key][1] for key in coords.keys()]

    plt.scatter(xs, ys, marker="x", color="purple")
    
    for num in range(len(tour) - 1):
        xs = (coords[path[num]][0], coords[path[num+1]][0])
        ys = (coords[path[num]][1], coords[path[num+1]][1])
        plt.plot(xs, ys, 'k-', lw=2, color="green")


def plot_edges(object, edges, weights, title="TSP Edges"):
    """ Plot the given edges and their weights for a given object problem instance """

    coords = object._problem.node_coords

    xs = [coords[key][0] for key in coords.keys()]
    ys = [coords[key][1] for key in coords.keys()]

    plt.scatter(xs, ys, marker="x", color="purple")

    for a, b, k in zip(xs, ys, coords.keys()):
        plt.text(a, b, k
        )
    
    for (edge, weight) in zip(edges, weights):
        xs = (coords[edge[0]][0], coords[edge[1]][0])
        ys = (coords[edge[0]][1], coords[edge[1]][1])
        plt.plot(xs, ys, 'k-', lw=2, color="black", alpha=weight)


def plot_weighted_graph(graph, weight="weight", title="Random Weighted Graph Plot"):
    """ Plot the given graph, randomly positioning nodes """

    coords = nx.spring_layout(graph)

    xs = [coords[key][0] for key in coords.keys()]
    ys = [coords[key][1] for key in coords.keys()]

    plt.scatter(xs, ys, marker="x", color="purple")

    for a, b, k in zip(xs, ys, coords.keys()):
        plt.text(a, b, k)
    
    edges, weights = graph_utils.get_edges(graph), graph_utils.get_weights(graph, weight=weight)
    
    for (edge, weight) in zip(edges, weights):
        xs = (coords[edge[0]][0], coords[edge[1]][0])
        ys = (coords[edge[0]][1], coords[edge[1]][1])
        plt.plot(xs, ys, 'k-', lw=2, color="black", alpha=weight)
    

        
def plot_graph(graph, title="Random Graph Plot"):
    """ Plot the given graph, randomly positioning nodes """

    coords = nx.spring_layout(graph)

    xs = [coords[key][0] for key in coords.keys()]
    ys = [coords[key][1] for key in coords.keys()]

    plt.scatter(xs, ys, marker="x", color="purple")

    for a, b, k in zip(xs, ys, coords.keys()):
        plt.text(a, b, k)

    for edge in graph_utils.get_edges(graph):
        xs = (coords[edge[0]][0], coords[edge[1]][0])
        ys = (coords[edge[0]][1], coords[edge[1]][1])
        plt.plot(xs, ys, 'k-', lw=2, color="black")
