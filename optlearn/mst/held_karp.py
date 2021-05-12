import numpy as np
import networkx as nx

from optlearn import graph_utils
from optlearn.fix import fix_utils
from optlearn.mst import mst_model


def assume_vertex(graph):
    """ If not vertex is set to special, always assume the same one """

    return list(graph.nodes)[0]


def find_special_pair(graph, vertex=None):
    """ Find the two smallest edges (special edges)  in the graph incident with the vertex """

    if vertex is None:
        vertex = assume_vertex(graph)

    edges = np.array([edge for edge in graph.edges if vertex in edge])
    incident_weights = graph_utils.get_edges_weights(graph, edges)
    indices = np.argpartition(incident_weights, 2)[:2]
    special_graph = nx.Graph()
    special_graph = fix_utils.migrate_edges(graph, special_graph, edges[indices])
    
    return special_graph


def build_storage_graph(graph, vertex=None):
    """ Build a graph to store the unspecial edges incident with the given vertex """

    storage_graph = nx.Graph()
    edges = np.array([edge for edge in graph.edges if vertex in edge])
    storage_graph = fix_utils.migrate_edges(graph, storage_graph, edges)
    graph.remove_edges_from(edges)
    
    return storage_graph
    

def build_onetree(unspecial_graph, special_graph):
    """ Compute the assumes edges incident to special vertex have been removed """

    model = mst_model.mstConstructor()
    tree = model.minimum_spanning_tree(unspecial_graph)
    tree = fix_utils.migrate_edges(special_graph, tree, special_graph.edges)

    return tree
    

class oneTreeConstructor():

    def __init__(self, graph, vertex=None):

        self.vertex = vertex
        self._ensure_vertex(graph)

    def _ensure_vertex(self, graph):
        """ Make sure some vertex is the special one """

        if self.vertex is None:
            self.vertex = assume_vertex(graph)

    def _identify_special_pair(self, graph, vertex=None):
        """ Get the special edge pair """

        if vertex is None:
            vertex = self.vertex
        
        return find_special_pair(graph, vertex=vertex)

    def _build_storage_graph(self, graph, vertex):

        if vertex is None:
            vertex = self.vertex

        return build_storage_graph(graph, vertex=vertex)

    def build_onetree(self, graph, vertex=None):

        if vertex is None:
            vertex = self.vertex
            print("No vertex, given, using {}!".format(self.vertex))

        special_graph = self._identify_special_pair(graph, vertex)
        storage_graph = self._build_storage_graph(graph, vertex)
        tree = build_onetree(graph, special_graph)
        graph = fix_utils.migrate_edges(storage_graph, graph, storage_graph.edges)
        return tree
        
