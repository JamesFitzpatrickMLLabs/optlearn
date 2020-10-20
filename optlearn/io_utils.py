import tsplib95

import numpy as np

from optlearn import graph_utils


_write_fields = [
    "NAME",
    "TYPE",
    "DIMENSION",
]


def read_file(fname):
    """ Read in tsplib file format """

    return tsplib95.load(fname)


def write_file(optObject, fname):
    return None


def build_string(items):
    """ Build a string from a bunch of items, witn a newline at the end """

    empty = "{} " * len(items) 
    return empty[:-1].format(*items) + "\n"
    

def load_npz_data(path):
    """ Load a single npz traininf file """
    with np.load(path) as npz:
        return npz["X"], npz["y"]


def load_npz_datas_for_training(paths):
    """ Load many npz training files into X, y arrays """
    
    pairs = [[*load_npz_data(path)] for path in paths]
    X = np.vstack([pair[0] for pair in pairs])
    y = np.concatenate([pair[1] for pair in pairs])
    return X, y


def read_file_into_list(fname):
    """ Read a file into a list """

    with open(fname) as f:
        lines = f.readlines()
    return lines


def find_substring_in_stringlist(stringlist, substring):
    """ Find the first time a substring appears in a list of strings """

    for num, item in enumerate(stringlist):
        if substring in item:
            return num
    raise ValueError("Substring not found in the stringlist!")


def read_solution_from_file(fname):
    """ Read and parse a solution file """

    lines = read_file_into_list(fname)
    tour_index = find_substring_in_stringlist(lines, "TOUR_SECTION")
    minus_index = find_substring_in_stringlist(lines, "-1")
    lines = lines[tour_index+1:minus_index]
    return [int(item[:-1]) for item in lines]


class optObject():

    def read_problem_from_file(self, fname):
        self._problem = read_file(fname)
        return self

    def read_solution_from_file(self, fname, symmetric=True):
        tour = read_solution_from_file(fname)
        edges = graph_utils.get_tour_edges(tour, symmetric=symmetric)
        self._solution = edges
        return tour

    def get_graph(self):
        return self._problem.get_graph()

    def get_solution(self):
        return self._solution

    def get_dict(self):
        return self._problem.as_dict()

    def get_keyword_dict(self):
        return self._problem.as_keyword_dict()

    def write_edges_explicit(self, fname, edge_weight_groups):
        with open(fname, "w") as fid:
            for field in _write_fields:
                fid.write("{}: {}\n".format(field, self.get_keyword_dict()[field]))
            fid.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
            fid.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")            
            fid.write("{}\n".format("EDGE_WEIGHT_SECTION\n"))
            for edge_weight_group in edge_weight_groups:
                fid.write(build_string(edge_weight_group))
            fid.write("{}\n".format("EOF"))
