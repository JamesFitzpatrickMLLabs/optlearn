import torch
import joblib

import numpy as np
import networkx as nx

from optlearn.feature import feature_utils
from optlearn.fix import fix_model


class modelPersister():

    def __init__(self, model=None, function_names=None):
        
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
        self.set_function_names(self._get_function_names(self._get_metadata(persist_dict)))


        
class modelWrapper(feature_utils.buildFeatures, modelPersister):

    def __init__(self, model=None, function_names=None):

        self.model = model
        self.function_names = function_names
        self.get_funcs()
        self._set_device()

    def _build_prediction_graph(self, graph, y):
        """ Build a pruned graph using the predictions """

        new_graph = type(graph)()
        new_graph.graph = graph.graph
        new_graph.add_nodes_from(graph.nodes)
        edges = [edge for (edge, item) in zip(graph.edges, y) if item > 0.5]
        for edge in edges:
            new_graph.add_edge(*edge, **graph[edge[0]][edge[1]])
        return new_graph

    def _set_funcs(self):
        """ Get the function names """
    
    def _set_device(self):
        """ Set the device """

        if torch.cuda.is_available():
            self._device = torch.device("cuda:0")
            print("GPU detected!")
        else:
            self._device = torch.device("cpu")
            print("GPU not detected!")
    
    def _check_cuda(self, parameters):
        """ Check if the parmaeters are placed on the gpu """

        return parameters.device.type == "cuda"

    def _check_cudas(self):
        """ Check if the model is placed on the gpu, if it is a torch model """

        if hasattr(self.model, "parameters"):
            ret =  all([self._check_cuda(param) for param in self.model.parameters()])
            
        if ret and self._device == "cpu":
            raise ValueError("No GPU found, but the model needs to run on one!")

        return ret

    def _normalise(self, X):
        """ Perform softmax over the given array """

        return (X - X.min()) / (X.max() - X.min())
        
    def _predict_torch(self, X):
        """ Make a prediction using a torch model """

        if self._check_cudas():
            y = self.model(torch.tensor(X).to(self._device).float())
            return y.data.cpu().numpy()
        else:
            y = self.model(torch.tensor(X).float())
            return y.data.numpy()

    def predict_vector(self, X):
        """ Make a prediction on a vector, outputs vector """

        if hasattr(self.model, "parameters"):
            return self._predict_torch(X)
        else:
            return self.model.predict(X)

    def predict_proba_vector(self, X):
        """ Make a prediction on a vector, outputs vector """

        if hasattr(self.model, "parameters"):
            return self._predict_torch(X)
        else:
            if hasattr(self.model, "decision_function"):
                y = self.model.decision_function(X)
            else:
                y = self.model.predict_proba(X)[:,1]
        return self._normalise(y)
    
    def predict_proba_graph(self, graph):
        """ Make a prediction on a graph, outputs vector """

        X = self.compute_features(graph)
        return self.predict_proba_vector(X)
        
    def predict_graph(self, graph):
        """ Make a prediction on a graph, outputs vector """

        X = self.compute_features(graph)
        return self.predict_vector(X)

    def prune_graph(self, graph, threshold=None):
        """ Prune thes the given graph, returning a graph """

        X = self.compute_features(graph)
        if threshold is None:
            y = self.predict_vector(X)
        else:
            y = (self.predict_proba_vector(X) > threshold).astype(int)
        return self._build_prediction_graph(graph, y)

    def prune_graph_with_logic(self, original_graph):
        """ Prune the graph, re-adding edges using logical rules """

        pruned_graph = self.prune_graph(original_graph)
        # fixed_graph = fix_model.ensure_minimum_degree(original_graph, pruned_graph)
        # fixed_graph = fix_model.ensure_connected(original_graph, fixed_graph)
        return fix_model.ensure_doubletree(original_graph, fixed_graph)

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
        """ Make a prediction on the given object, returns vector """

        if self._detect_graph(X):
            self._check_function_names()
            return self.predict_graph(X)
        if self._detect_vector(X):
            return self.predict_vector(X)
        raise ValueError("X is of unsupported type!")
    
