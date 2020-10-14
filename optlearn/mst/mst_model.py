import copy

import networkx as nx
import numpy as np


class mstSparsifier():

    def copy_graph(self, graph):
        """ Copy the given graph """

        return copy.deepcopy(graph)

    def minimum_spanning_tree(self, graph, weight="weight"):
        """ Compute the minimum spanning tree graph """

        return nx.minimum_spanning_tree(graph, weight=weight)
        
    def check_weight_key(self, graph, weight="weight"):
        """ Checks for the weight key in any edge's dictionary """

        a, b = list(graph.edges)[0]
        if weight not in graph[a][b].keys():
            raise ValueError("The key specifying the weights must match the one provided!")

    def remove_edges(self, graph, edges):
        """ Remove the given edges from the graph """

        graph.remove_edges_from(edges)
        return graph
        
    def sparsify(self, graph, weight="weight"):
        """ Sparsify the graph by removing the minimum spanning tree """

        edges = self.minimum_spanning_tree(graph, weight=weight).edges
        self.remove_edges(graph, edges)
        
        return edges

    def run_sparsify(self, graph, iterations=0, weight="weight"):
        """ Sparsify for a given number of rounds, return removed edges """

        sparsified_edges = []

        self.check_weight_key(graph, weight=weight)
        graph = self.copy_graph(graph)
        
        for iteration in range(iterations):
            edges = self.sparsify(graph, weight=weight)
            sparsified_edges.append(edges)

        return sparsified_edges

    def tuple_to_string(self, item):
        """ Convert a tuple to a string """

        return "({},{})".format(*item)
    
    def tuples_to_strings(self, tuples):
        """ Convert a list of tuples to a list of strings """

        return [self.tuple_to_string(item) for item in tuples]

    def fit_sparsify(self, graph, iterations, weight="weight"):
        """ Compute the sparsification features for the graph """
        
        return_array = np.zeros((len(graph.edges), 2))
        
        sparsified_edges = self.run_sparsify(graph, iterations, weight="weight")
        for num, item in enumerate(self.tuples_to_strings(graph.edges)):
            for set_num, sparsified_set in enumerate(sparsified_edges):
                sparsified_set = self.tuples_to_strings(sparsified_set)
                if item in sparsified_set:
                    return_array[num, 0] = 1
                    return_array[num, 1] = set_num

        return return_array.astype(int)
