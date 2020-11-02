import numpy as np
import networkx as nx

from optlearn.feature import feature_utils


class modelPersister():

    def __init__(self, model=None, functions_names=None):
        
        self.model = model
        self.function_names = function_names

    def set_metadata(self, meta_dict):
        """ Set all the given metadata """

        for key in meta_dict:
            self.key = meta_dict[key]

    def set_function_names(self, function_names):
        """ Set the function names """

        self.function_names = function_names

    def set_model(self, model):
        """ Set the model """

        self.model = model

    def _check_model(self):
        """ Check if the model is given """

        if self.model is None:
            raise ValueError("No model set!")

    def _check_function_names(self):
        """ Check if the function names are given """

        if self.model is None:
            raise ValueError("No fucntion names set!")

    def _perform_checks(self):
        """ Perform the given checks """

        self._check_model()
        self._check_function_names()
        
    def _get_metadata(self, persist_dict):
        """ Get the metadata dictionary from the persist dictionary """

        return persist_dict["metadata"]

    def _get_function_names(self, meta_dict):
        """ Get the function names from the metadata dictioanry """

        return meta_dict["function_names"]

    def _get_model(self, persist_dict):
        """ Get the model from the persist dictionary """

        return persist_dict["model"]

    def _build_meta_dict(self):
        """ Build the metadata dictionary """

        return {
            "function_names": self.function_names
            }

    def _build_persist_dict(self):
        """ Build the metadata dictionary """

        return {
            "model": self.model,
            "metadata": self._build_meta_dict()
            }
        
    def save(self, fname):
        """ Dump the model, saving metadata too """

        self._perform_checks()

        persist_dict = self._build_persist_dict()
        joblib.dump(persist_dict, fname)
        
    def load(self, fname):
        """ Load a model, loading the model and metadata """

        persist_dict = joblib.load(fname)
        self.set_model(self._get_model(persist_dict))
        self.set_metadata(self._get_metadata(persist_dict))


        
class modelWrapper(feature_utils.buildFeatures, modelPersister):

    def __init__(self, model=None, function_names=None):

        self.model = model
        self.function_names = function_names


    def predict_vector(self, X):
        """ Make a prediction on a vector, outputs vector """

        return self.model.predict(X)

    def predict_graph(self, X):
        """ Make a prediction on a graph, outputs vector """

        X = self.compute_features(X)
        return self.predict_vector(X)

    def _detect_graph(self, object):
        """ Check if the object is a graph """

        is_graph = isinstance(object, nx.Graph)
        is_digraph = isinstance(object, nx.DiGraph)

        return bool(is_graph + is_digraph)

    def _detect_vector(self, object):
        """ Check if the given object is a vector of some kind """

        is_list = isinstance(object, list)
        is_array = isinstance(object, np.array)

        return bool(is_list + is_array)
    
    def predict(self, X):
        """ Make a prediction on the given object """

        if self._detect_graph(X):
            self._check_function_names()
            return self.predict_graph(X)
        if self._detect_vector(X):
            return self.predict_vector(X)
        raise ValueError("X is of unsupported type!")
