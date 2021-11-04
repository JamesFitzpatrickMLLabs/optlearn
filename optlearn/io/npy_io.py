import numpy as np


def load_npy_file(path):
    """ Load a single npy file """

    return np.load(path)


def load_npy_data(path):
    """ Load a single npy file (in a context manager) """

    with  np.load(path) as npy:
        return npy

def load_npz_data(path):
    """ Load a single npz training file """
    with load_npy_file(path) as npz:
        return npz["X"], npz["y"]


def load_npz_datas_for_training(paths):
    """ Load many npz training files into X, y arrays """
    
    pairs = [load_npz_data(path) for path in paths]
    X = np.vstack([pair[0] for pair in pairs])
    y = np.concatenate([pair[1] for pair in pairs])
    return X, y
