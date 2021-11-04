import numpy as np

from optlearn import graph_utils

from optlearn.mst import mst_model


class feasifierWrapper():

    def feasify_prune_vector(self, graph, y):
        """ Given the pruning vector, feasify it """

        return (self.compute_feasifier_vector(graph) + y).astype(bool)


class blankFeasifier(feasifierWrapper):

    def compute_feasifier_vector(self, graph):
        """ Compute the feasifier vector """

        return np.ones((len(graph.edges)))
    

class scratchFeasifier(feasifierWrapper):

    def __init__(self, stuff=None):

        self.stuff = None

    def _compute_nonzeros(self, graph):
        """ Compute the number of indices that should be nonzero """
        
        return len(graph.nodes) /  len(graph.edges) 

    def _fudge_feasifier_vector(self, graph):
        """ Compute a fudged feasifier vector """

        p = _compute_nonzeros(self, graph)
        return np.random.choice([0, 1], size=(m), p=[1-p, p])

    def compute_feasifier_vector(self, graph):
        """ Compute the feasifier vector """

        return self._fudge_feasifier_vector(graph)


class doubleTreeFeasifier(feasifierWrapper):

    def _compute_doubletree_tour(self, graph):
        """ Compute the doubletree tour """

        model = mst_model.doubleTreeConstructor()
        return model.get_doubletour(graph)

    def _compute_doubletree_edges(self, graph):
        """ Compute the doubletree edges """

        tour = self._compute_doubletree_tour(graph)
        is_symmetric = graph_utils.check_graph(graph) 
        return graph_utils.get_tour_edges(tour, is_symmetric)

    def compute_feasifier_vector(self, graph):
        """ Compute the feasifier vector """

        edges = self._compute_doubletree_edges(graph)
        return graph_utils.compute_indicator_vector(graph, edges)


class christofidesFeasifier(feasifierWrapper):

    def __init__(self, rounds=1):

        self.rounds = rounds

    def _compute_christofides_tour(self, graph):
        """ Compute the Christofides tour """

        model = mst_model.christofidesConstructor()
        return model.get_christofides_tour(graph)

    def _compute_christofides_edges(self, graph):
        """ Compute the Christofides edges """

        tour = self._compute_christofides_tour(graph)
        is_symmetric = graph_utils.check_graph(graph) 
        return graph_utils.get_tour_edges(tour, is_symmetric)

    def compute_feasifier_vector(self, graph):
        """ Compute the feasifier vector """

        edges = self._compute_christofides_edges(graph)
        return graph_utils.compute_indicator_vector(graph, edges)


_feasifiers = [
    None,
    scratchFeasifier,
    doubleTreeFeasifier,
    christofidesFeasifier,
]
