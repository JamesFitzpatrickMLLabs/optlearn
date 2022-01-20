import numpy as np

from optlearn.graph import graph_utils

from optlearn.feature.vrp import arc_features
from optlearn.feature.vrp import node_features


class arcFeatureBuilder():
    def __init__(self, function_names):
        
        self.set_function_names(function_names)
        self.set_functions()

    def set_function_names(self, function_names):
        """ Set the names of the functions to be used """
        
        self.function_names = function_names

        return None

    def _get_defined_arc_functions(self):
        """ Get the defined arc functions """

        defined_functions = arc_features.arc_functions

        return defined_functions
    
    def _get_defined_arc_functions_names(self):
        """ Get the names of the defined arc functions """

        defined_functions = [item.__name__ for item in arc_features.arc_functions]

        return defined_functions

    def _get_defined_graph_functions(self):
        """ Get the defined graph functions """

        defined_functions = arc_features.graph_functions

        return defined_functions

    def _get_defined_graph_functions_names(self):
        """ Get the names of the defined graph functions """

        defined_functions = [item.__name__ for item in arc_features.graph_functions]

        return defined_functions

    def _get_all_defined_functions(self):
        """ Get all of the defined functions """

        arc_functions = self._get_defined_arc_functions()
        graph_functions = self._get_defined_graph_functions()
        defined_functions = arc_functions + graph_functions

        return defined_functions
    
    def _get_all_defined_functions_names(self):
        """ Get all of the defined functions """

        arc_functions = self._get_defined_arc_functions_names()
        graph_functions = self._get_defined_graph_functions_names()
        defined_functions = arc_functions + graph_functions

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
                raise ValueError(f"Function {function_name} not found in arc features!")

    def get_functions(self):
        """ Get the functions from the arc features script """

        defined_functions_dict = self._get_all_defined_functions_dict()
        functions = [defined_functions_dict[item] for item in self.function_names]

        return functions

    def set_functions(self):
        """ Set the function names in the retrieved order """

        functions = self.get_functions()
        self._functions = functions

        return None
        
    def compute_arc_feature(self, graph, arc, reachability_radius, function):
        """ Compute a specific feature for the edge arc the graph """

        if "reachability_radius" in function.__code__.co_varnames:
            feature = function(graph, arc, reachability_radius)
        else:
            feature = function(graph, arc)

        return feature

    def compute_arc_features(self, graph, arc, reachability_radius):
        """ Compute all specified features for the arc in the graph """

        features = [self.compute_arc_feature(graph, arc, reachability_radius, function)
                    for function in self._functions]

        return features

    def compute_arcs_feature(self, graph, arcs, reachability_radius, function):
        """ Compute all features for all of the given arcs in the graph """

        feature_set = [self.compute_arc_feature(graph, arc, reachability_radius, function)
                       for arc in arcs]

        return feature_set
    
    def compute_arcs_features(self, graph, arcs, reachability_radius):
        """ Compute all features for all of the given arcs in the graph """

        feature_set = [self.compute_arc_features(graph, arc, reachability_radius)
                       for arc in arcs]

        return feature_set

    def compute_graph_feature(self, graph, arcs, reachability_radius, function):
        """ Compute all features for all of the given arcs in the graph """

        if "reachability_radius" in function.__code__.co_varnames:
            feature_dict = function(graph, reachability_radius)
        else:
            feature_dict = function(graph)

        feature_set = [feature_dict.get(arc) or -1 for arc in arcs]
            
        return feature_set

    def _is_arc_function(self, function_name):
        """ Check if the given function is a arc function """

        arc_functions = self._get_defined_arc_functions_names()
        if function_name in arc_functions:
            is_arc_function = True
        else:
            is_arc_function = False

        return is_arc_function

    def _is_graph_function(self, function_name):
        """ Check if the given function is a graph function """

        graph_functions = self._get_defined_graph_functions_names()
        if function_name in graph_functions:
            is_graph_function = True
        else:
            is_graph_function = False

        return is_graph_function

    def compute_feature_matrix(self, graph, arcs, reachability_radius):
        """ Compute the feature matrix for the given arcs """

        computed_features = []

        clone_graph = graph_utils.clone_graph(graph)
        clone_graph = node_features.process_reachabilities(clone_graph, reachability_radius)
        clone_graph = node_features.process_strict_reachabilities(clone_graph, reachability_radius)
        
        for function_name in self.function_names:
            function = self._functions[self.function_names.index(function_name)]
            if self._is_arc_function(function_name):
                computed_features.append(
                    self.compute_arcs_feature(clone_graph, arcs, reachability_radius, function)
                )
            elif self._is_graph_function(function_name):
                computed_features.append(
                    self.compute_graph_feature(clone_graph, arcs, reachability_radius, function)
                )
            else:
                raise ValueError(f"Function name {function_name} not recognised!")

        computed_features = np.stack(computed_features, axis=1)

        return computed_features
