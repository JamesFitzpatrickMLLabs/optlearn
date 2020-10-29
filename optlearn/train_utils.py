import os
import joblib

import numpy as np
import networkx as nx

from optlearn import io_utils
from optlearn import cc_solve
from optlearn import features
from optlearn import graph_utils

from sklearn.model_selection import train_test_split


def problem_pairs_from_fnames(problem_fnames, solution_fnames=None):
    """ Build the problem solution pairs for computing features """

    solution_fnames = solution_fnames or [None, ] * len(problem_fnames)
    
    return {problem_fname: solution_fname for
            (problem_fname, solution_fname) in zip(problem_fnames, solution_fnames)}


def get_problem_name(fname):
    """ Get the fname of a problem """

    return os.path.basename(fname)


def get_pair_name(pair):
    """ Assign a name to a problem-solution pair """

    return get_problem_name(pair[0])


def get_name_stem(name):
    """ Get the stem of a problem name (without file extension) """

    return name.split(".")[0]
    

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
        
        self._funcs = [graph_utils.functions[item] for item in function_names]
        
    def compute_feature(self, graph, func):
        """ Compute a specific feature for the graph """
        
        return func(graph)

    def compute_features(self, graph):
        """ Compute the feature vector for the graph """
        
        data = [self.compute_feature(graph, func) for func in self.funcs]
        return np.stack(data, axis=1)

    
class createTrainingFeatures(buildFeatures):
    def __init__(self,
                 parent_directory,
                 function_names,
                 problem_pairs,
                 overrides=[],
                 verbose=False):
        self.parent_directory = parent_directory
        self.problem_pairs = problem_pairs
        self.function_names = function_names
        self.overrides = overrides
        self.verbose = verbose
        self.initialise_checklist()
        self.initialise_problem_dict()
        self.initialise_step_tracker()

    def build_empty_checkdict(self):
        """ Build an empty checklist dict """

        return {"problem": False, "solution": False}

    def initialise_checklist(self):
        """ Create a checklist of completed problems """

        check_dict = self.build_empty_checkdict()
        self._check_dict = {get_pair_name(pair): check_dict
                            for pair in self.problem_pairs}

    def initialise_problem_dict(self):
        """ Create a dictionary of problems to build features for """

        self._problem_dict = {get_pair_name(pair): pair for pair in self.problem_pairs}

    def initialise_step_tracker(self):
        """ Build dictionary to keep track of the steps performed in data creation """

        self._tracker = {}

    def make_verbose(self):
        """ Print some status updates with the data creation """

        self.verbose = True

    def make_silent(self):
        """  Print no statis updates with the data creation """

        self.verbose = False
        
    def set_parent_directory(self, parent_directory):
        """ Set the parent directory to which features will be saved """

        self.parent_directory = parent_directory    

    def set_problem_pairs(self, problem_pairs):
        """ Set the problem files to have features computed """

        self.problem_pairs = problem_pairs
        self.initialise_checklist()
        self.initialise_problem_dict()

    def set_overrides(self, override_fnames):
        """ Set the overrides for file creation, overwrite if they exist already """

        self.overrides = override_fnames

    def load_object(self, problem_fname):
        """ Load the problem file and store it as a hidden attribute """

        self._object = io_utils.optObject()
        self._object.read_problem_from_file(problem_fname)
        self._graph = self._object.get_graph()
        
    def compute_labels(self, problem_fname, solution_fname):
        """ Compute the labels for the graph edges """

        if solution_fname is not None:
            return self.compute_labels_from_solution(solution_fname)
        else:
            return self.compute_labels_from_problem(problem_fname)

    def compute_labels_from_problem(self, problem_fname):
        """ Using the problem file, compute the labels """
        
        tour = cc_solve.solution_from_path(problem_fname).tour
        edges = graph_utils.get_edges(self._graph)
        min_vertex = np.min(self._graph.nodes)
        order = len(self._graph.nodes)
        indices = [graph_utils.compute_vector_index_symmetric(edge, order, min_vertex)
                   for edge in edges]
        vector = np.zeros((len(self._graph.edges)))
        vector[indices] = 1
        return vector

    def compute_labels_from_solution(self, solution_fname):
        """ Using the solution file, read the solution in """

        tour = self._object.read_solution_from_file(solution_fname)
        edges = graph_utils.get_edges(self._graph)
        min_vertex = np.min(self._graph.nodes)
        order = len(self._graph.nodes)
        indices = [graph_utils.compute_vector_index_symmetric(edge, order, min_vertex)
                   for edge in edges]
        vector = np.zeros((len(self._graph.edges)))
        vector[indices] = 1
        return vector

    def write_to_npy(self, fname, data):
        """ Write the data to the given filename """

        np.save(fname, data)

    def check_parent_exists(self):
        """ Make sure the parent directory exists! """

        if not os.path.exists(self.parent_directory):
            raise FileExistsError("Parent Directory Not Found!")
    
    def check_directory_exists(self, directory):
        """ Check if the given directory exists at the parent """

        path = os.path.join(self.parent_directory, directory)
        return os.path.exists(path)

    def check_directories_exist(self):
        """ Check if the required directories exist at the parent """

        dirnames = [directory.split("_")[1] for directory in self.function_names]
        return [self.check_directory_exists(item) for item in dirnames]
        
    def build_directory(self, directory):
        """ Build the given direcotry at the parent node """
        
        path = os.path.join(self.parent_directory, directory)
        return os.mkdir(path)

    def build_directories(self, directories):
        """ Build the given directories at the parent node """

        for directory in directories:
            self.build_directory(directory)

    def check_solution_directory(self):
        """ Check if the solution directory exists at the parent """

        path = os.path.join(self.parent_directory, "solutions")
        return os.path.exists(path)
            
    def build_filename(self, directory, name):
        """ Build the filename for the given problem in the given directory """

        dirname = directory.split("_")[1]
        directory = os.path.join(self.parent_directory, dirname)
        return os.path.join(directory, name) + ".npy"

    def check_file_exists(self, filename):
        """ Check if the features for the given problem already exist """

        return os.path.exists(filename)

    def check_file_override(self, filename):
        """ Check if there is an override on the given file """

        return filename in self.overrides 
    
    def check_file_with_overrides(self, filename):
        """ Even if the features exists already, compute them if overridden """

        exists = self.check_file_exists(filename)
        override = self.check_file_override(filename)
        if exists and not override:
            return True
        else:
            return False

    def print_status(self, verbose=False):
        """ Print the step tracker nicely """

        if verbose:
            pass
        
        print("\nStatus:")
        for (key, value) in self._tracker.items():    
            print("{}: {}".format(key, value))
        print("\n")
    
    def directory_step(self):
        """ Check and build necessary directories first """

        self._tracker["Directory Status"] = "Checking/Building"
        self.check_parent_exists()
        exists = self.check_directories_exist()
        for (exist, directory) in zip(exists, self.function_names):
            if not exist:
                self.build_directory(directory.split("_")[1])
        exists = self.check_solution_directory()
        if not exists:
            self.build_directory("solutions")
        self._tracker["Directory Status"] = "Checked/Built"
                
    def feature_step(self, function_name, problem_fname, filename):
        """ Build the given feature for the given problem """

        self.load_object(problem_fname)
        data = features.functions[function_name](self._graph)
        if not self.check_file_with_overrides(filename):
            self.write_to_npy(filename, data)

    def label_step(self, problem_fname, solution_fname, name):
        """ Build the given labels for the the given problem """

        data = self.compute_labels(problem_fname, solution_fname)
        filename = os.path.join(self.parent_directory, "solutions")
        filename = os.path.join(filename, get_name_stem(name)) + ".npy"
        if not self.check_file_with_overrides(filename):
            self.write_to_npy(filename, data)

    def data_steps(self, name):
        """ Build all the features for the given problem """

        for function_name in self.function_names:
            problem_fname, solution_fname = self._problem_dict[name]
            namestem = get_name_stem(name)
            filename = self.build_filename(function_name, namestem)
            self.feature_step(function_name, problem_fname, filename)
            self._check_dict[name]["problem"] = True
        self.label_step(problem_fname, solution_fname, name)
        self._check_dict[name]["solution"] = True
            
    def data_create(self):
        """ Perform the data creation """

        self.directory_step()
        self.print_status(self.verbose)
        self._tracker["Features Status"] = "Checking/Building/Writing"
        for num, name in enumerate(self._problem_dict.keys()):
            self._tracker["Current Problem"] = "{}".format(name)
            self.data_steps(name)
            if self.verbose:
                length = len(self._problem_dict.keys())
                print("Problem {} of {} completed".format(num+1, length))
        self._tracker["Current Problem"] = "N/A"
        self._tracker["Features Status"] = "Checked/Built/Written"
        self.print_status(self.verbose)


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
        self._ocheck_function_names()
        
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


        
class modelWrapper(buildFeatures, modelPersister):

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


class dataMatcher():

    def __init__(self, feature_dirs, solution_dir, specific_stems=None):

        self.feature_dirs = feature_dirs
        self.solution_dir = solution_dir
        self.specific_stems = specific_stems
        self._specify_stems()

    def _get_file_stem(self, fname):
        """ Get the name stem of a file """

        basename = get_problem_name(fname)
        return get_name_stem(basename)

    def _get_file_stems(self, fnames):
        """ Get the name stem of files """

        return [self._get_file_stem(fname) for fname in fnames]

    def _get_fnames(self, directory):
        """ Get the absolute paths for the fnames in the given directory """

        fnames = os.listdir(directory)
        return [os.path.join(directory, fname) for fname in fnames]

    def _specify_stems(self):
        """ Get the specific stems if they were not specified already """

        if self.specific_stems is None:
            some_fnames = self._get_fnames(self.feature_dirs[0])
            self.specific_stems = self._get_file_stems(some_fnames)
            
    def _build_fnames_directory(self, directory):
        """ Build the fnames for a given directory """ 

        # return [(directory, stem) for stem in self.specific_stems]
        return [os.path.join(directory, stem) + ".npy" for stem in self.specific_stems]

    def build_fname_pairs(self):
        """ Build the pairs of feature fnames and solution fname """

        feature_fname_tuples = [
            self._build_fnames_directory(directory) for directory in self.feature_dirs
        ]
        feature_fname_tuples = list(zip(*feature_fname_tuples))
        solution_fnames = self._build_fnames_directory(self.solution_dir)
        return list(zip(feature_fname_tuples, solution_fnames))


    def build_fname_sets(self):
        """ Build the separate lists of feature fnames and solution fnames """

        feature_fname_tuples = [
            self._build_fnames_directory(directory) for directory in self.feature_dirs
        ]
        solution_fnames = self._build_fnames_directory(self.solution_dir)
        return feature_fname_tuples, solution_fnames

    
class dataLoader():

    def __init__(self, data_pairs, shuffle=True):

        self.data_pairs = data_pairs
        self.shuffle = shuffle

    def train_test_val_split(self, train=0.7, test=0.15, val=0.15):
        """ Generate the train, test and validation sets """

        self.train_pairs, self.test_pairs = train_test_split(self.data_pairs, train_size=train)
        ratio = test / (test + val)
        self.test_pairs, self.val_pairs = train_test_split(self.test_pairs, train_size=ratio)

    def load_features(self, feature_fnames):
        """ Load the features """

        features = [io_utils.load_npy_file(fname) for fname in feature_fnames]
        return np.stack(features, axis=1)

    def load_labels(self, label_fname):
        """ Load the labels """

        return io_utils.load_npy_file(label_fname)

    def load_pair(self, pair):
        """ Load the feature-label pair """

        features = self.load_features(pair[0])
        labels = self.load_labels(pair[1])
        return features, labels
        
    def load_training_features(self):
        """ Load the training pairs """

        return [
            self.load_features(pair[0]) for pair in self.train_pairs
        ]

    def load_training_labels(self):
        """ Load the training labels """

        return [
            self.load_labels(pair[1]) for pair in self.train_pairs
        ]

    def load_testing_features(self):
        """ Load the testing pairs """

        return [
            self.load_features(pair[0]) for pair in self.test_pairs
        ]

    def load_testing_labels(self):
        """ Load the testing labels """

        return [
            self.load_labels(pair[1]) for pair in self.test_pairs
        ]

    def load_validation_features(self):
        """ Load the validation pairs """

        return [
            self.load_features(pair[0]) for pair in self.val_pairs
        ]

    def load_validation_labels(self):
        """ Load the validation labels """

        return [
            self.load_labels(pair[1]) for pair in self.val_pairs
        ]


