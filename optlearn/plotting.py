import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

from optlearn import graph_utils


def generate_coord_dict(graph):
    """ Generate a random coord_dict for plotting purposes """

    return nx.spring_layout(graph)


def check_indices(coord_dict, indices):
    """ Check that the coordinate indices match """

    coord_uniques = np.unique(list(coord_dict.keys()))
    vertex_uniques = np.unique(indices)

    missing_coord = [item for item in coord_uniques if item not in vertex_uniques]    
    missing_vertex = [item for item in vertex_uniques if item not in coord_uniques]

    if len(missing_coord) > 0 or len(missing_vertex) > 0:
        print("Vertices in coordinate dict not found in vertex set: {}".format(missing_coord))
        print("Vertices in vertex set not found in coordinate dict: {}".format(missing_vertex))
        raise ValueError("Vertex indices must match!")
    

def get_x_y_lists(coord_dict, indices=None):
    """ Get two lists of coordinates, one of x's and one of y's """


    if indices is None:
        inidices = coord_dict.keys()
    
    coords_x = [coord_dict[key][0] for key in indices]
    coords_y = [coord_dict[key][1] for key in indices]
    
    return  coords_x, coords_y


def plot_vertices(coord_dict, indices=None, withlabels=True):
    """ Plot the the vertices at the given coordinates """

    indices = coord_dict.keys()
    coords_x, coords_y = get_x_y_lists(coord_dict, indices=indices)
    plt.scatter(coords_x, coords_y, marker="x", color="purple", alpha=0.5)

    if withlabels:
        for a, b, k in zip(coords_x, coords_y, coord_dict.keys()):
            plt.text(a, b, k, color="purple")

def plot_tour(coord_dict, tour, withlabels=True):
    """ Plot the tour for a given object problem instance """

    check_indices(coord_dict, tour)
    
    appended_tour = graph_utils.append_last_node(tour)
    tour_x, tour_y = get_x_y_lists(coord_dict, appended_tour)

    plot_vertices(coord_dict, withlabels=withlabels)
    plt.plot(tour_x, tour_y, color="green")
    

def plot_edges(coord_dict, edges, weights=None, withlabels=True):
    """ Plot the given edges and their weights for a given object problem instance """

    check_indices(coord_dict, edges)

    if weights is None:
        weights = [1, ] * len(edges)
    
    plot_vertices(coord_dict, withlabels=withlabels)    
    
    for (edge, weight) in zip(edges, weights):
        if weight > 0:
            xs = (coord_dict[edge[0]][0], coord_dict[edge[1]][0])
            ys = (coord_dict[edge[0]][1], coord_dict[edge[1]][1])
            plt.plot(xs, ys, 'k-', lw=2, color="black", alpha=weight)


def plot_graph(graph, edges=None, weights=None, weight="weight", coord_dict=None, withlabels=True):
    """ Plot the given graph, plotting the given edges and weights (default all) """

    coord_dict = coord_dict or generate_coord_dict(graph)
    
    if edges is None:
        edges = graph_utils.get_edges(graph)
    if weights is None:
        weights = weights or graph_utils.get_weights(graph, weight=weight)
    
    check_indices(coord_dict, edges)
        
    plot_vertices(coord_dict, withlabels=withlabels)
    plot_edges(coord_dict, edges, weights)
    
        
