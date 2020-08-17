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
    

class optObject():

    def read_from_file(self, fname):
        self._object = read_file(fname)
        return self

    def get_graph(self):
        return self._object.get_graph()

    def get_dict(self):
        return self._object.as_dict()

    def get_keyword_dict(self):
        return self._object.as_keyword_dict()

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
