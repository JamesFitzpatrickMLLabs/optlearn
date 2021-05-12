import numpy as np

from optlearn import graph_utils

from optlearn.mip import mip_model


def get_root_relaxation(problem):
    """ Get the relaxed solution at the current root """

    problem.perform_relaxation()
    return problem.get_varvals()
