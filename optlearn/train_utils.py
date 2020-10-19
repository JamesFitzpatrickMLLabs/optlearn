import os

import numpy as np

from optlearn import io_utils
from optlearn import cc_solve
from optlearn import features
from optlearn import graph_utils


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
        return np.vstack(data)

    
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
        
    def compute_labels(self, problem_fname, solution_fname):
        """ Compute the labels for the graph edges """

        if solution_fname is not None:
            return self.compute_labels_from_solution(solution_fname)
        else:
            return self.compute_labels_from_problem()

    def compute_labels_from_problem(self):
        """ Using the problem file, compute the labels """
        
        tour = cc_solve.solution_from_path(self._object.get_graph()).tour
        edges = graph_utils.get_edges(self._object.get_graph())
        return graph_utils.check_edges_in_tour(edges, tour)

    def compute_labels_from_solution(self, solution_fname):
        """ Using the solution file, read the solution in """

        tour = self._object.read_solution_from_file(solution_fname)
        edges = graph_utils.get_edges(self._object.get_graph())
        return graph_utils.check_edges_in_tour(edges, tour)

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

        return [self.check_directory_exists(item) for item in self.function_names]
        
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

        directory = os.path.join(self.parent_directory, directory)
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
                self.build_directory(directory)
        exists = self.check_solution_directory()
        if not exists:
            self.build_directory("solutions")
        self._tracker["Directory Status"] = "Checked/Built"
                
    def feature_step(self, function_name, problem_fname, filename):
        """ Build the given feature for the given problem """

        self.load_object(problem_fname)
        data = features.functions[function_name](self._object.get_graph())
        if not self.check_file_with_overrides(filename):
            self.write_to_npy(filename, data)

    def label_step(self, problem_fname, solution_fname, name):
        """ Build the given labels for the the given problem """

        data = self.compute_labels(problem_fname, solution_fname)
        filename = os.path.join(self.parent_directory, "solutions")
        print(filename)
        filename = os.path.join(filename, get_name_stem(name)) + ".npy"
        print(filename)
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
        for name in self._problem_dict.keys():
            self._tracker["Current Problem"] = "{}".format(name)
            self.data_steps(name)
        self._tracker["Current Problem"] = "N/A"
        self._tracker["Features Status"] = "Checked/Built/Written"
        self.print_status(self.verbose)
