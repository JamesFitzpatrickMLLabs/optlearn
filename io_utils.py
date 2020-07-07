import tsplib95

import numpy as np

import features



_write_fields = [
    "NAME",
    "COMMENT",
    "TYPE",
    "DIMENSION",
    # "NODE_COORD_SECTION",
    "EDGE_WEIGHT_TYPE"
]


def read_file(fname):
    """ Read in tsplib file format """

    return tsplib95.load(fname)


def write_file(optObject, fname):
    return None


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

    def write_graph_adjacency(self, fname, edges, weights):
        with open(fname, "w") as fid:
            for field in _write_fields:
                fid.write("{}: {}\n".format(field, self.get_keyword_dict()[field]))
            fid.write("{}\n".format("NODE_COORD_SECTION"))
            node_dict = self.get_keyword_dict()["NODE_COORD_SECTION"]
            for key in node_dict.keys():
                a, b = node_dict[key]
                fid.write("{} {} {}\n".format(key, int(a), int(b)))
            fid.write("{}: {}\n".format("EDGE_DATA_FORMAT", "EDGE_LIST"))
            fid.write("{}\n".format("EDGE_DATA_SECTION"))
            for edge in edges:
                fid.write("{} {}\n".format(*edge))
            fid.write("-1\n")
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
