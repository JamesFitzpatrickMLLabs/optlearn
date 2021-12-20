from optlearn.io.npy import npy_io


def read_array_from_npy(filename):
    """ Read the numpy array from the given filename """

    array = npy_io.load_npy_file(filename)

    return array


def read_array_from_npz(filename):
    """ Read the numpy arrays from the given filename """

    array = npy_io.load_npz_file(filename)

    return array
