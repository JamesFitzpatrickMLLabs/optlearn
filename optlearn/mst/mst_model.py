import copy

import networkx as nx
import pandas as pd
import numpy as np

from optlearn import graph_utils


class edgeSparsifier():

    def copy_graph(self, graph):
        """ Copy the given graph """

        return copy.deepcopy(graph)

    def remove_edges(self, graph, edges):
        """ Remove the given edges from the graph """

        graph.remove_edges_from(edges)
        return graph
        
    def sparsify_once(self, graph, edge_extracter, weight="weight"):
        """ Sparsify the graph by removing edges identified by the extracter """

        edges = edge_extracter(graph, weight=weight)
        self.remove_edges(graph, edges)
        
        return edges

    def sparsify(self, graph, edge_extracter, iterations=0, weight="weight"):
        """ Sparsify for a given number of rounds, return removed edges """

        sparsified_edges = []

        self.check_weight_key(graph, weight=weight)
        graph = self.copy_graph(graph)
        
        for iteration in range(iterations):
            print("Iteration {} of {}".format(iteration + 1, iterations))
            edges = self.sparsify_once(graph, edge_extracter, weight=weight)
            sparsified_edges.append(edges)

        return sparsified_edges
    

class mstConstructor():
    
    def minimum_spanning_tree(self, graph, weight="weight", algorithm="prim"):
        """ Compute the minimum spanning tree graph """

        return nx.minimum_spanning_tree(graph, weight=weight)

    def minimum_spanning_tree_edges(self, graph, weight="weight"):
        """ Compute the minimum spanning tree graph """

        return self.minimum_spanning_tree(graph, weight=weight).edges
    
    def check_weight_key(self, graph, weight="weight"):
        """ Checks for the weight key in any edge's dictionary """

        a, b = list(graph.edges)[0]
        if weight not in graph[a][b].keys():
            raise ValueError("The key specifying the weights must match the one provided!")


class doubleTreeConstructor(mstConstructor):

    def construct_multigraph(self, graph):
        """ Construct a multigraph from a graph and double the edges """

        multigraph = nx.MultiGraph()
        multigraph.add_edges_from(graph.edges)
        multigraph.add_edges_from(graph.edges)
        return multigraph

    def get_eulerian_circuit(self, graph):
        """ Get the edges of an Eulerian circuit on the graph """

        edges =  nx.algorithms.eulerian_circuit(graph)
        return np.array(list(edges))

    def get_eulerian_shortcuts(self, eulerian_edges):
        """ Get a tour from a short-cutted eulerian circuit """

        return pd.unique(eulerian_edges.flatten())

    def get_doubletour(self, graph, reverse=False, roll=0):
        """ Get a doubletour for the given graph """

        tree = self.minimum_spanning_tree(graph)
        multigraph = self.construct_multigraph(tree)
        eulerian_edges = self.get_eulerian_circuit(multigraph)
        if reverse:
            eulerian_edges = eulerian_edges[::-1]
        eulerian_edges = np.roll(eulerian_edges, roll)
        return self.get_eulerian_shortcuts(eulerian_edges)

    def get_doubletour_from_tree(self, tree, reverse=False, roll=0):
        """ Get a doubletour for the given graph """

        multigraph = self.construct_multigraph(tree)
        eulerian_edges = self.get_eulerian_circuit(multigraph)
        if reverse:
            eulerian_edges = eulerian_edges[::-1]
        eulerian_edges = np.roll(eulerian_edges, roll)
        return self.get_eulerian_shortcuts(eulerian_edges)

    def get_doubletours(self, graph):
        """ Get forward and backward doubletours """

        doubletour_a = self.get_doubletour(graph, reverse=False)
        doubletour_b = self.get_doubletour(graph, reverse=True)

        return [doubletour_a, doubletour_b]

    def get_doubletours_from_tree(self, tree):
        """ Get forward and backward doubletours """

        doubletour_a = self.get_doubletour_from_tree(tree, reverse=False)
        doubletour_b = self.get_doubletour_from_tree(tree, reverse=True)

        return [doubletour_a, doubletour_b]

    
        
class mstSparsifier(edgeSparsifier, mstConstructor):

    def run_sparsify(self, graph, iterations=0, weight="weight"):
        """ Sparsify several times using the MST edges, returning all edges """

        return self.sparsify(
            graph=graph,
            edge_extracter=self.minimum_spanning_tree_edges,
            iterations=iterations,
            weight=weight
        )
    
    def tuple_to_string(self, item):
        """ Convert a tuple to a string """

        return "({},{})".format(*item)
    
    def tuples_to_strings(self, tuples):
        """ Convert a list of tuples to a list of strings """

        return [self.tuple_to_string(item) for item in tuples]

    def copy_graph_blank(self, graph):
        """ make a zero-weighted copy of the graph """

        copy_graph = nx.Graph()
        copy_graph.add_edges_from(graph.edges)
        nx.set_edge_attributes(copy_graph, 0, "weight")
        return copy_graph
    
    def fit_sparsify(self, graph, iterations, weight="weight"):
        """ Compute the sparsification features for the graph """
        
        sparse_graph = self.copy_graph_blank(graph)
        edges = self.run_sparsify(graph, iterations=iterations, weight=weight)
        for num, edge_set in enumerate(edges):
            weighted_edges = [edge + ((num + 1) / iterations, ) for edge in edge_set]
            sparse_graph.add_weighted_edges_from(weighted_edges)
        return graph_utils.get_weights(sparse_graph)


class doubleTreeSparsifier(edgeSparsifier, doubleTreeConstructor):

    def extract_doubletour_edges_from_tree(self, tree, weight="weight"):
        """ Extract the doubletour edges from the tree """

        tour = self.get_doubletour_from_tree(tree)
        return graph_utils.get_tour_edges(tour).tolist()

    def extract_doubletours_edges_from_tree(self, tree, weight="weight"):
        """ Extract the doubletours edges from the tree """

        tours = self.get_doubletours_from_tree(tree)
        edges_a = graph_utils.get_tour_edges(tours[0]).tolist()
        edges_b = graph_utils.get_tour_edges(tours[1]).tolist()
        return edges_a + edges_b

    def get_doubletour_edges(self, graph, weight="weight"):
        """ Extract the doubletour edges from the graph """

        tour = self.get_doubletour(graph)
        return graph_utils.get_tour_edges(tour).tolist()

    def get_doubletours_edges(self, graph, weight="weight"):
        """ Extract the doubletours edges from the graph """

        tours = self.get_doubletours(graph)
        edges_a = graph_utils.get_tour_edges(tours[0]).tolist()
        edges_b = graph_utils.get_tour_edges(tours[1]).tolist()
        return edges_a + edges_b

    def get_tree_and_doubletour_edges(self, graph, weight="weight"):
        """ Extract the doubletour and tree edges from the graph """

        tree = self.minimum_spanning_tree(graph)
        tree_edges = list(tree.edges)
        doubletour_edges = self.extract_doubletour_edges_from_tree(tree)
        return tree_edges + doubletour_edges
        
    def get_tree_and_doubletours_edges(self, graph, weight="weight"):
        """ Extract the doubletours and tree edges from the graph """

        tree = self.minimum_spanning_tree(graph)
        tree_edges = list(tree.edges)
        doubletours_edges = self.extract_doubletours_edges_from_tree(tree)
        return tree_edges + doubletours_edges
        return doubletours_edges
        
    def run_sparsify(self, graph, iterations=0, weight="weight"):
        """ Sparsify several times using the MST edges, returning all edges """

        return self.sparsify(
            graph=graph,
            edge_extracter=self.get_doubletours_edges,
            iterations=iterations,
            weight=weight
        )
