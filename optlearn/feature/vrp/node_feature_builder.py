import numpy as np

from optlearn.graph import graph_utils

from optlearn.feature.vrp import node_features


class nodeFeatureBuilder():
    def __init__(self, function_names):
        
        self.set_function_names(function_names)
        self.set_functions()

    def set_function_names(self, function_names):
        """ Set the names of the functions to be used """
        
        self.function_names = function_names

        return None

    def _get_defined_node_functions(self):
        """ Get the defined node functions """

        defined_functions = node_features.node_functions

        return defined_functions
    
    def _get_defined_node_functions_names(self):
        """ Get the names of the defined node functions """

        defined_functions = [item.__name__ for item in node_features.node_functions]

        return defined_functions

    def _get_defined_graph_functions(self):
        """ Get the defined graph functions """

        defined_functions = node_features.graph_functions

        return defined_functions

    def _get_defined_graph_functions_names(self):
        """ Get the names of the defined graph functions """

        defined_functions = [item.__name__ for item in node_features.graph_functions]

        return defined_functions

    def _get_all_defined_functions(self):
        """ Get all of the defined functions """

        node_functions = self._get_defined_node_functions()
        graph_functions = self._get_defined_graph_functions()
        defined_functions = node_functions + graph_functions

        return defined_functions
    
    def _get_all_defined_functions_names(self):
        """ Get all of the defined functions """

        node_functions = self._get_defined_node_functions_names()
        graph_functions = self._get_defined_graph_functions_names()
        defined_functions = node_functions + graph_functions

        return defined_functions

    def _get_all_defined_functions_dict(self):
        """ Get a dictionary of the defined functions """

        defined_functions = self._get_all_defined_functions()
        defined_functions_names = self._get_all_defined_functions_names()
        function_pairs = zip(defined_functions_names, defined_functions)
        defined_functions_dict = {
            function_name: function for (function_name, function) in function_pairs
        }        

        return defined_functions_dict
        
    def _check_function_names(self, function_names):
        """ Check if the supplied functions exist """

        defined_functions = self._get_all_defined_functions()
        for function_name in self.function_names:
            if function_name not in defined_functions:
                raise ValueError(f"Function {function_name} not found in node features!")

    def get_functions(self):
        """ Get the functions from the node features script """

        defined_functions_dict = self._get_all_defined_functions_dict()
        functions = [defined_functions_dict[item] for item in self.function_names]

        return functions

    def set_functions(self):
        """ Set the function names in the retrieved order """

        functions = self.get_functions()
        self._functions = functions

        return None
        
    def compute_node_feature(self, graph, node, reachability_radius, function):
        """ Compute a specific feature for the edge node the graph """

        if "reachability_radius" in function.__code__.co_varnames:
            feature = function(graph, node, reachability_radius)
        else:
            feature = function(graph, node)

        return feature

    def compute_node_features(self, graph, node, reachability_radius):
        """ Compute all specified features for the node in the graph """

        features = [self.compute_node_feature(graph, node, reachability_radius, function)
                    for function in self._functions]

        return features

    def compute_nodes_feature(self, graph, nodes, reachability_radius, function):
        """ Compute all features for all of the given nodes in the graph """

        feature_set = [self.compute_node_feature(graph, node, reachability_radius, function)
                       for node in nodes]

        return feature_set
    
    def compute_nodes_features(self, graph, nodes, reachability_radius):
        """ Compute all features for all of the given nodes in the graph """

        feature_set = [self.compute_node_features(graph, node, reachability_radius)
                       for node in nodes]

        return feature_set

    def compute_graph_feature(self, graph, nodes, reachability_radius, function):
        """ Compute all features for all of the given nodes in the graph """

        if "reachability_radius" in function.__code__.co_varnames:
            feature_dict = function(graph, reachability_radius)
        else:
            feature_dict = function(graph, reachability_radius)

        feature_set = [feature_dict[node] for node in nodes]
            
        return feature_set

    def _is_node_function(self, function_name):
        """ Check if the given function is a node function """

        node_functions = self._get_defined_node_functions_names()
        if function_name in node_functions:
            is_node_function = True
        else:
            is_node_function = False

        return is_node_function

    def _is_graph_function(self, function_name):
        """ Check if the given function is a node function """

        graph_functions = self._get_defined_graph_functions_names()
        if function_name in graph_functions:
            is_graph_function = True
        else:
            is_graph_function = False

        return is_graph_function

    def compute_feature_matrix(self, graph, nodes, reachability_radius):
        """ Compute the feature matrix for the given nodes """

        computed_features = []

        clone_graph = graph_utils.clone_graph(graph)
        clone_graph = node_features.process_reachabilities(clone_graph, reachability_radius)
        clone_graph = node_features.process_strict_reachabilities(clone_graph, reachability_radius)
        
        for function_name in self.function_names:
            function = self._functions[self.function_names.index(function_name)]
            if self._is_node_function(function_name):
                computed_features.append(
                    self.compute_nodes_feature(clone_graph, nodes, reachability_radius, function)
                )
            elif self._is_graph_function(function_name):
                computed_features.append(
                    self.compute_graph_feature(clone_graph, nodes, reachability_radius, function)
                )
            else:
                raise ValueError(f"Function name {function_name} not recognised!")

        computed_features = np.hstack(computed_features)

        return computed_features

        
