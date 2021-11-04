import torch
import joblib

import numpy as np
import networkx as nx

from optlearn.feature import feature_utils
from optlearn.fix import feasifiers
from optlearn.fix import fix_model


class modelPersister():

    def __init__(self, model=None, function_names=None, threshold=None):
        
        self.model = model
        self.function_names = function_names
        self.threshold = threshold

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

    def set_threshold(self, threshold):
        """ Set the threshold """

        self.threshold = threshold
        
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

    def _get_threshold(self, persist_dict):
        """ Get the threshold from the persist dictionary """

        return persist_dict["threshold"]
    
    def _build_meta_dict(self):
        """ Build the metadata dictionary """

        return {
            "function_names": self.function_names
            }

    def _build_persist_dict(self):
        """ Build the metadata dictionary """

        return {
            "model": self.model,
            "metadata": self._build_meta_dict(),
            "threshold": self.threshold,
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
        self.set_threshold(self._get_threshold(persist_dict))
        
        
class modelWrapper(feature_utils.buildFeatures, modelPersister):

    def __init__(self, model=None, function_names=None, threshold=None, feasifier=None):

        self.model = model
        self.function_names = function_names
        self.threshold = threshold
        self.feasifier = feasifier
        
        if self.function_names is not None:
            self.get_funcs()
            
        self._set_device()

    def _check_feasifier(self):
        """ If there is no feasifier, default to doubltree """

        if self.feasifier is None:
            self.feasifier = feasifiers.doubleTreeFeasifier()

    def _build_prediction_graph(self, graph, y, threshold=None):
        """ Build a pruned graph using the predictions """

        if threshold is None:
            threshold = 0.5
        
        new_graph = type(graph)()
        new_graph.graph = graph.graph
        new_graph.add_nodes_from(graph.nodes)
        edges = [edge for (edge, item) in zip(graph.edges, y) if item > threshold]
        _ = [new_graph.add_edge(*edge, **graph[edge[0]][edge[1]]) for edge in edges]
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
        """ Perform a normalisation over the given array """

        return (X - X.min()) / (X.max() - X.min())

    def from_joblib(self, fname):
        """ Build a wrapper from a stored model joblib """

        self.load(fname)

        return self

    def _predict_proba_torch(self, X):
        """ Make a prediction using a torch model """

        if self._check_cudas():
            y = self.model(torch.tensor(X).to(self._device).float())
            return y.data.cpu().numpy()
        else:
            y = self.model(torch.tensor(X).float())
            return y.data.numpy()
    
    def _predict_torch(self, X, threshold=None):
        """ Make a prediction using a torch model """

        if threshold is None:
            return (self._predict_proba_torch(X) > 0.5).astype(int)
        else:
            return (self._predict_proba_torch(X) > threshold).astype(int)
        
    def predict_vector(self, X, threshold=None):
        """ Make a prediction on a vector, outputs vector """

        if hasattr(self.model, "parameters"):
            return self._predict_torch(X)
        else:
            if threshold is not None:
                return (self.predict_proba_vector(X) > threshold)
            else:
                return self.model.predict(X)

    def predict_proba_vector(self, X):
        """ Make a prediction on a vector, outputs vector """

        if hasattr(self.model, "parameters"):
            return self._predict_torch(X)
        else:
            if hasattr(self.model, "predict_proba"):
                y = self.model.predict_proba(X)[:,1]
            else:
                y = self._normalise(self.model.decision_function(X))
            return y

    def _compute_feasifier_vector(self, graph):
        """ Compute a tour for the given graph using the feasifier """

        return self.feasifier(graph)
    
    def predict_proba_graph(self, graph):
        """ Make a prediction on a graph, outputs vector """

        X = self.compute_features(graph)
        return self.predict_proba_vector(X)
        
    def predict_graph(self, graph, threshold=None):
        """ Make a prediction on a graph, outputs vector """

        X = self.compute_features(graph)
        return self.predict_vector(X, threshold=threshold)

    def prune_graph(self, graph, threshold=None, feasifier=None):
        """ Prune the given graph, returning a graph """

        threshold = threshold or self.threshold
        
        X = self.compute_features(graph)
        if threshold is None:
            y = self.predict_vector(X)
        else:
            y = (self.predict_vector(X, threshold)).astype(int)
        if feasifier is not None:
            y = np.clip(0, 1, feasifier.compute_feasifier_vector(graph) + y)
        return self._build_prediction_graph(graph, y)

    def prune_graph_with_logic(self, graph, threshold=None):
        """ Prune the graph, re-adding edges using a feasifier """

        self._check_feasifier()
        
        threshold = threshold or self.threshold
        
        return self.prune_graph(graph, threshold, self.feasifier)

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


def blank(stuff, threshold=None):

    return stuff


def rank(stuff, threshold=None):

    return stuff


class wrapperPersister():
    def __init__(self, model_dict=None, function_dict=None,
                 propagator_dict=None, feasifier_dict=None, threshold_dict=None):

        self.model_dict = model_dict
        self.function_dict = function_dict
        self.propagator_dict = propagator_dict
        self.feasifier_dict = feasifier_dict
        self.threshold_dict = threshold_dict

    def set_model(self, model, key):
        """ Set a model at the given key """

        self.model_dict[key] = model

    def set_functions(self, functions, key):
        """ Set a functions at the given key """

        self.function_dict[key] = functions

    def set_propagator(self, propagator, key):
        """ Set a propagator at the given key """

        self.propagator_dict[key] = propagator

    def set_feasifer(self, feasifier, key):
        """ Set a feasifier at the given key """

        self.feasifier_dict[key] = feasifier

    def set_threshold(self, threshold, key):
        """ Set a threshold at the given key """

        self.threshold_dict[key] = threshold

    def set_layer(self, model, functions, propagator, feasifer, threshold, key):
        """ Set a layer at the given key """

        self.set_model(model, key)
        self.set_functions(functions, key)
        self.set_propagator(propagator, key)
        self.set_feasifier(feasifier, key)
        self.set_threshold(threshold, key)

    def set_new_layer(self, model, functions, propagator, feasifer, threshold):
        """ Set the next layer """

        keys = sorted(list(self.model_dict.keys()))
        new_key = int(keys[-1] + 1)
        self.set_layer(model, functions, propagator, feasifer, threshold)

    def set_function_dict(self, function_dict):
        """ Set the function dict """

        self.function_dict = function_dict

    def set_model_dict(self, model_dict):
        """ Set the model dict """

        self.model_dict = model_dict

    def set_propagator_dict(self, propagator_dict):
        """ Set the propagator dict """

        self.propagator_dict = propagator_dict
        
    def set_threshold_dict(self, threshold_dict):
        """ Set the threshold dict """

        self.threshold_dict = threshold_dict

    def set_feasifier_dict(self, feasifier_dict):
        """ Set the feasifier dict """

        self.feasifier_dict = feasifier_dict
        
    def _perform_handshakes(self):
        """ Handshake the propagated features """

        handshaker = feature_utils.handShaker(function_dict=self.function_dict,
                                              propagator_dict=self.propagator_dict)
        handshaker.perform_handshake()
        
    def _check_models(self):
        """ Check if the models are given """

        if self.model_dict is None:
            raise ValueError("No models set!")

    def _check_functions(self):
        """ Check if the function names are given """

        if self.function_dict is None:
            raise ValueError("No fucntions set!")
        else:
            if len(self.function_dict.keys()) != len(self.model_dict.keys()):
                raise ValueError("There must be as many function sets as models")
            if list(self.function_dict.keys()) != list(self.model_dict.keys()):
                raise ValueError("The orderings of the models and functions must match!")

    def _check_propagators(self):
        """ Check that the propagators exist and make sense """

        if self.propagator_dict is None:
            raise ValueError("No propagators set!")
        else:
            self._perform_handshakes()

    def _check_feasifiers(self):
        """ Check that the feasifiers are given and make sense """

        if any([item not in feasifiers._feasifiers for item in self.feasifier_dict.values()]):
            raise ValueError("Feasifier not recognised!")
        
    def _check_thresholds(self):
        """ Check if the thresholds are given """

        if self.threshold_dict is None:
            raise ValueError("No thresholds set!")
        else:
            if any(item not in self.model_dict.keys() for item in self.function_dict.keys()):
                raise ValueError("Threshold keys must match model keys!")
            
    def _perform_checks(self):
        """ Perform the given checks """

        self._check_models()
        self._check_functions()
        self._check_propagators()
        self._check_feasifiers()
        self._check_thresholds()

    def _get_model_dict(self, persist_dict):
        """ Get the models from the persist dictionary """

        return persist_dict["model_dict"]

    def _get_function_dict(self, persist_dict):
        """ Get the functions from the persist dictionary """

        return persist_dict["function_dict"]
    
    def _get_propagator_dict(self, persist_dict):
        """ Get the propagators from the persist dictionary """

        return persist_dict["propagator_dict"]

    def _get_feasifier_dict(self, persist_dict):
        """ Get the feasifiers from the persist dictionary """

        return persist_dict["feasifier_dict"]

    
    def _get_threshold_dict(self, persist_dict):
        """ Get the thresholds from the persist dictionary """

        return persist_dict["threshold_dict"]
    
    def _build_persist_dict(self):
        """ Build the metadata dictionary """

        return {
            "model_dict": self.model_dict,
            "function_dict": self.function_dict,
            "propagator_dict": self.propagator_dict,
            "feasifier_dict": self.feasifier_dict,
            "threshold_dict": self.threshold_dict,
            }
        
    def save(self, fname):
        """ Dump the model, saving metadata too """

        self._perform_checks()

        persist_dict = self._build_persist_dict()
        joblib.dump(persist_dict, fname)
        
    def load(self, fname):
        """ Load a model, loading the model and metadata """

        persist_dict = joblib.load(fname)
        self.set_model_dict(self._get_model_dict(persist_dict))
        self.set_function_dict(self._get_function_dict(persist_dict))
        self.set_propagator_dict(self._get_propagator_dict(persist_dict))
        self.set_feasifier_dict(self._get_feasifier_dict(persist_dict))
        self.set_threshold_dict(self._get_threshold_dict(persist_dict))
    

class wrapperWrapper(feature_utils.buildFeatures, wrapperPersister):
    
    def __init__(self, models=None, function_sets=None, feasifiers=None):

        self.models = models
        self.feasifiers = feasifiers
        self.function_sets = function_sets
        self._wrappers = []

    def _add_model(self, model):
        """ Add a model to end of the the models dict """

        keys = sorted(list(self.model_dict.keys()))
        next_key = int(keys[-1] + 1)
        self.set_model(model, next_key)

    def _add_functions(self, functions):
        """ Add a function set to end of the the functions dict """

        keys = sorted(list(self.function_dict.keys()))
        next_key = int(keys[-1] + 1)
        self.set_functions(functions, next_key)

    def _add_propagator(self, propagator):
        """ Add a propagator to end of the the propagators dict """

        keys = sorted(list(self.propagator_dict.keys()))
        next_key = int(keys[-1] + 1)
        self.set_propagator(propagator, next_key)
        
    def _add_feasifier(self, feasifier):
        """ Add an feasifier to end of the feasifiers dict """
        
        keys = sorted(list(self.feasifier_dict.keys()))
        next_key = int(keys[-1] + 1)
        self.set_feasifier.append(feasifier, next_key)

    def _add_threshold(self, threshold):
        """ Add an threshold to end of the threshold dict """
        
        keys = sorted(list(self.threshold_dict.keys()))
        next_key = int(keys[-1] + 1)
        self.set_threshold.append(threshold, next_key)

    def _build_propagator_slicer(self, current_key, previous_key):
        """ Build a slicer for the feature array of the previous stage """

        return [self.function_dict[previous_key].index(item) for item in
                   self.propagator_dict[current_key]]
        
    def _propagate_features(self, X, y, current_key, previous_key):
        """ Propagate the previous features to the next stage """

        indices = self._build_propagator_slicer(current_key, previous_key)
        return  X[:, indices]
        
    def predict_proba_layer(self, model, X, z=None, y=None, threshold=None):
        """ Make predictions on the vectorised features of the given layer """

        if z is None:
            X = np.concatenate([X, z], axis=1)
        if y is None:
            X = np.concatenate([X, y], axis=1)

        return self.predict_proba_vector(X)

    def setup_stage(self, current_key):
        """ Setup the current stage """

        self.model = self.model_dict[current_key]
        self.threshold = self.threshold_dict[current_key]
        self.function_names = self.function_dict[current_key]

    def predict_stage(self, current_key, graph, z=None, y=None):
        """ Perform the predictions for a given stage """

        self.setup_stage(current_key)

        X = self.compute_features(graph)
        y = self.predict_proba_layer(self.model, X=X, z=z, y=y)
        X = X[(y > self.threshold).astype(bool)]
        z = z[(y > self.threshold).astype(bool)]

        return {"X": X, "z": z, "y": y}
        
