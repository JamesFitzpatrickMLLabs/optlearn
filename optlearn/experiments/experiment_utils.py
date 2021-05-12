import os

import numpy as np

from optlearn import io_utils

from optlearn.mip import mip_model
from optlearn.train import model_utils


def load_and_wrap_model(model_path):
    """ Load a trained model and wrap it to allow easy inference """

    persister = model_utils.modelPersister()
    persister.load(model_path)
    return model_utils.modelWrapper(model=persister.model,
                                    threshold=persister.threshold,
                                    function_names=persister.function_names) 


def load_problem(problem_path):
    """ Given the path to a tsp problem, load it as a networkx complete graph """

    object = io_utils.optObject().read_problem_from_file(problem_path)
    return object.get_graph()


def load_solution(solution_path):
    """ Load the solution at the given path """

    return io_utils.read_solution_from_file(solution_path)


def solve_problem_custom(graph, params=None):
    """ Solve the given problem using the specified params, if there are any """

    if params is None:
        params = {
            "solver": "xpress",
            "var_type": "binary",
            "verbose": False,
            }

    problem = mip_model.tspProblem(graph=graph, **params)
    problem.optimise()

    return problem

    
def check_problems(problem_dir):
    """ Make sure the files in the directory are all in TSPLIB format """

    try:
        files = [item.split(".")[1] for item in os.listdir(problem_dir)]
    except Exception as exception:
        print("Something went wrong, checking if all files in problem directory are valid!")
        raise exception
    checks = [item == "tsp" for item in files]
    if np.sum(checks) < len(checks):
        print("Found invalid files in problem_dir!")
        raise ValueError
