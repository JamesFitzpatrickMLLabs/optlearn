import networkx as nx

import matplotlib.pyplot as plt

from optlearn import graph_utils


def show_plots():
    """ Show the currently drawn plots """

    plt.show()

    return None


def generate_random_coord_dict(graph):
    """ Generate a random coordinate dictionary for plotting purposes """

    coord_dict =  nx.spring_layout(graph)

    return coord_dict


def check_coord_dict(graph):
    """ Check if there is a coordinate dictionary """

    if "coord_dict" not in graph.graph.keys():
        raise Exception("Graph must have coord_dict!")

    return None


def plot_point_text(point, label, fontsize, colour, offset=None, axis=None):
    """ Plot a string of text at the given position """

    if offset is not None:
        if type(offset) is int or type(offset) is float:
            point = (point[0] + offset, point[1] + offset)
        else:
            point = (point[0] + offset[0], point[1] + offset[1])
    if axis is None:
        plt.text(*point, label, fontsize=fontsize, color=colour)
    else:
        axis.text(*point, label, fontsize=fontsize, color=colour)

    return None
    

def plot_point(point, marker, colour, alpha, size, label=None, fontsize=None, offset=None, axis=None):
    """ Plot the given point with the associated arguments """

    if axis is None:
        plt.scatter(*point, marker=marker, color=colour, alpha=alpha, s=size)
    else:
        axis.scatter(*point, marker=marker, color=colour, alpha=alpha, s=size)
    if label is not None:
        plot_point_text(point, label, fontsize, colour, offset=offset, axis=axis)

    return None


def plot_circle(point, colour, alpha, radius, axis):
    """ Plot the given point with the associated arguments """

    circle = plt.Circle(point, color=colour, alpha=alpha, radius=radius, linestyle="--", fill=False)
    _ = axis.add_patch(circle)
    return None


def plot_points(points, marker, colour, alpha, size, labels=None, fontsizes=None, axis=None):
    """ Plot the given point with the associated arguments """

    if labels is None:
        labels, fontsizes = [None] * len(points), [None] * len(points)
    if fontsizes is None:
        fontsizes = [None] * len(points)
    for (point, label, fontsize) in zip(points, labels, fontsizes):
        plot_point(point, marker, colour, alpha, size, label, fontsize, axis=axis)

    return None


def plot_line(points, colour, thickness, style, alpha):
    """ Plot a line between the two points in Cartesian space """

    xs, ys = [item[0] for item in points], [item[1] for item in points]
    plt.plot(xs, ys, linestyle=style, lw=thickness, color=colour, alpha=alpha)

    return None
    

def plot_arrow(points, colour, thickness, style, alpha):
    """ Plot a line between the two points in Cartesian space """

    difference = (points[1][0] - points[0][0], points[1][1] - points[0][1])
    
    plt.arrow(*points[0], *difference, head_width=thickness / 70, head_length=thickness / 70,
              linewidth=thickness, color=colour, alpha=alpha, length_includes_head=True)

    return None


def plot_nodes(graph, marker="^", colour="red", alpha=1, size=10, fontsize=10, nodes=None, offset=None):
    """ Plot the nodes in the given graph. Must have a coordinate dict """

    check_coord_dict(graph)

    if nodes is None:
        nodes = graph.nodes
    points = [graph.graph["coord_dict"][node] for node in nodes] 
        
    for (point, node) in zip(points, nodes):
        plot_point(point, marker, colour, alpha, size, node, fontsize, offset=offset)
        
    return None


def plot_colocated_nodes(graph, nodes, markers, colours, alphas, sizes, fontsize):
    """ Plot the given nodes that are colocated, showing their label as a list """

    points = [graph.graph["coord_dict"][node] for node in nodes]
    labels = [f"{node}, " for (colour, node) in zip(colours, nodes)]
    label = "[" + "".join(labels)[:-2] + "]"

    for point, marker, colour, alpha, size in zip(points, markers, colours, alphas, sizes):
        plot_point(point, marker, colour, alpha, size)
    plot_point_text(points[0], label, fontsize, colour=None, offset=None)

    return None


def plot_edges(graph, colour="black", thickness=1, style="-", alpha="1", edges=None, weights=None):
    """ Plot the all of the specified edges for the given graph """

    check_coord_dict(graph)
    if edges is None:
        edges = graph.edges
    if weights is None:
        weights = [1] * len(edges)

    for (edge, weight) in zip(edges, weights):
        if weight >= 0:
            node_a, node_b = edge[0], edge[1]
            points = [graph.graph["coord_dict"][node_a], graph.graph["coord_dict"][node_b]]
            plot_line(points, colour, thickness, style, weight)

    return None


def plot_arcs(graph, colour="black", thickness=1, style="-", alpha="1", arcs=None, weights=None):
    """ Plot the all of the specified arcs for the given graph """

    check_coord_dict(graph)
    if arcs is None:
        arcs = graph.edges
    if weights is None:
        weights = [1] * len(arcs)

    for (arc, weight) in zip(arcs, weights):
        if weight >= 0:
            node_a, node_b = arc[0], arc[1]
            points = [graph.graph["coord_dict"][node_a], graph.graph["coord_dict"][node_b]]
            plot_arrow(points, colour, thickness, style, weight)

    return None    
