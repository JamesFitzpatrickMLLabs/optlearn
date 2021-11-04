import numpy as np

from optlearn.feature import features


class buildFeatures():
    def __init__(self, function_names):
        self.function_names = function_names
        self.get_funcs()

    def set_function_names(self, function_names):
        """ Set the names of the functions to be used """
        
        self.function_names = function_names
        self.get_funcs()

    def get_funcs(self):
        """ Get the functions from graph_utils.py """
        
        self._funcs = [features.functions[item] for item in self.function_names]
        
    def compute_feature(self, graph, func):
        """ Compute a specific feature for the graph """

        print(func)
        return func(graph)

    def compute_features(self, graph):
        """ Compute the feature vector for the graph """
        
        data = [self.compute_feature(graph, func) for func in self._funcs]
        return np.stack(data, axis=1)


class handShaker():
    def __init__(self, function_dict, propagator_dict):

        self.function_dict = function_dict
        self.propagator_dict = propagator_dict

    def _check_keys(self):
        """ Check the keys (stages) match for the dictionaries """

        if set(list(self.function_dict.keys())) != set(list(self.propagator_dict.keys())):
            raise ValueError("Stages for propagator and classifiers should match!")

    def _check_nonempty(self):
        """ Make sure there is at least something being computed """

        lengths = [len(self.function_dict[key]) for key in self.function_dict.keys()]

        if any([length == 0 for length in lengths]):
            raise ValueError("At least one new feature should be in each step!")

    def _check_propagation(self, first_key, second_key):
        """ Check that the features requested can be propagated """

        return any([item not in self.function_dict[first_key]
                    for item in self.propagator_dict[second_key]])

    def _check_empty_primary(self):
        """ Make sure that the first propagator is empty """

        primary_key = sorted(list(self.propagator_dict.keys()))[0]

        if self.propagator_dict[primary_key] is not None:
            raise ValueError("Cannot propagate features at first step!")
    
    def _check_propagator(self):
        """ Check that in all cases the propagation can occur """

        sorted_keys = sorted(list(self.propagator_dict.keys()))
        _checks = []
        
        for first_key, second_key in zip(sorted_keys, sorted_keys[1:]):
            _checks.append(self._check_propagation(first_key, second_key))
        if any(_checks):
            raise ValueError("Propagation not possible with given architecture")

    def perform_handshake(self):
        """ Perform all the checks on the propagators """

        self._check_keys()
        self._check_nonempty()
        self._check_empty_primary()
        self._check_propagator()

        print("Hand-shook!")
