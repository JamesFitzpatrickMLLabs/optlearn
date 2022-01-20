from optlearn.graph import graph_utils

from optlearn.feature.vrp import feature_utils
from optlearn.feature.vrp import arc_feature_builder


class customerCustomerArcPruner():

    def __init__(self, classifier=None, function_names=None):

        self.classifier = classifier
        self.function_names = function_names

    def from_persister(self, persister):
        """ Load the classifier and metadata directly from a persister """

        self.classifier = persister.model
        self.function_names = persister.function_names

        return self

    def is_customer_customer_edge(self, graph, edge):
        """ Check if the given edge is a customer-customer edge """

        customers = feature_utils.get_customers(graph)
        if edge[0] in customers and edge[1] in customers:
            is_customer_customer_edge = True
        else:
            is_customer_customer_edge = False

        return is_customer_customer_edge
        
    def get_customer_customer_arcs(self, graph):
        """ Get the customer customer arcs in the stored graph """
    
        all_edges = graph_utils.get_all_edges(graph)
        edges = [edge for edge in all_edges if self.is_customer_customer_edge(graph, edge)]
        
        return edges

    def construct_edge_features(self, graph, reachability_radius, edges=None):
        """ Construct the features for the given edge """

        edges = edges or self.get_customer_customer_arcs(graph)
        feature_builder = arc_feature_builder.arcFeatureBuilder(self.function_names)
        feature_matrix = feature_builder.compute_feature_matrix(graph, edges, reachability_radius)

        return feature_matrix
        
    def classify_edges(self, graph, reachability_radius, edges=None):
        """ Classify the given edges in the graph """

        edges = edges or self.get_customer_customer_arcs(graph)        
        feature_matrix = self.construct_edge_features(graph, reachability_radius, edges=None)
        predictions = self.classifier.predict(feature_matrix)
        prediction_dict = {edge: value for (edge, value) in zip(edges, predictions)}
        
        return prediction_dict
    
        
