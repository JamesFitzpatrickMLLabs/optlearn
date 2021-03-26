import os
import copy
import json
import pprint
import argparse

import numpy as np

from pathlib import Path

from optlearn import cc_solve
from optlearn import io_utils
from optlearn import graph_utils

from optlearn.fix import fix_model
from optlearn.fix import fix_utils
from optlearn.io import write_tsplib
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

    object = io_utils.read_solution_from_file(solution_path)
    return object.get_solution()


def compute_solution_objective(graph, problem_path, solution_path=None):
    """ 
    Get the objective of the optimal solution to the given problem.
    If the solution file is provided, this is computed by computing
    the sum of the weights of the given solution edges. Some checking
    must be done to make sure the vertices match, so the networkx
    graph of the problem is needed for comparison. Otherwise, use 
    Concorde to compute the solution.
    """

    if solution_path is not None:
        solution = load_solution(solution_path)
        if np.min(solution) < np.min(graph.nodes):
            solution += np.min(graph.nodes) - np.min(solution)
        if np.min(graph.nodes) < np.min(solution):
            solution += np.min(solution) - np.min(graph.nodes)
        optimal_objective = graph_utils.compute_tour_length(graph, solution)
    else:
        optimal_objective = solve_for_objective(problem_path)

    return optimal_objective



def check_problems(problem_dir):
    """ Make sure the files in the directory are all in TSPLIB format """

    try:
        files = [item.split(".")[1] for item in os.listdir(problem_dir)]
    except Exception as exception:
        print("Something went wrong, checking if all files in problem direcotry are valid!")
        raise exception
    checks = [item == "tsp" for item in files]
    if np.sum(checks) < len(checks):
        print("Found invalid files in problem_dir!")
        raise ValueError


def match_problems_to_solutions(problem_dir, solution_dir=None):
    """ 
    If solutions are provided, match them to their corresponding
    TSPLIB problem file. This way the problem does not need to be
    solved,  making this much faster for larger problems. It is
    assumed that the structure of the problem filename is <name>.tsp
    and that the structure of the solution filename is <name>.opt.tour
    """

    problems = [os.path.join(problem_dir, item) for item in os.listdir(problem_dir)]
    
    if solution_dir is None:
        return [(problem, None) for problem in problems]
    else:
        solutions = [os.path.join(solution_dir, item) for item in os.listdir(solution_dir)]

    problem_stems = [os.path.basename(item).split(".")[0] for item in problems]
    solution_stems = [os.path.basename(item).split(".")[0] for item in solutions]
    
    pairs = [(problems[problem_stems.index(p_stem)], solutions[solution_stems.index(p_stem)])
             if p_stem in solution_stems else (problems[problem_stems.index(p_stem)], None) for
             p_stem in problem_stems]
    
    return pairs


def prune_graph(graph, wrapper):
    """ Prune the given graph, using the wrapper provided """

    y_pred = wrapper.predict_graph(graph)
    pruned_graph = type(graph)()
    pruned_graph.add_nodes_from(graph.nodes)
    retain_edges = np.array(list(graph.edges))[np.array(y_pred).astype(bool)]
    pruned_graph = fix_utils.migrate_edges(graph, pruned_graph, retain_edges)
    
    return pruned_graph


def ensure_feasible(original_graph, pruned_graph):
    """ Given a graph and its pruned counterpart, ensure the pruned one is feasible """

    return fix_model.ensure_doubletree(original_graph, pruned_graph, iterations=1)


def prune_graph_feasibly(graph, wrapper):
    """ Prune the given graph, using the wrapper provided, ensuring feasibility """
    
    pruned_graph = prune_graph(graph, wrapper)
    return ensure_feasible(graph, pruned_graph)
    

def solve_for_objective(problem_path):
    """ Solve the given problem for its objective value """

    solution = cc_solve.solution_from_path(problem_path)
    
    return solution.optimal_value


def write_pruned_graph_tsplib(problem_path, original_graph, pruned_graph):
    """ 
    Write the pruned graph as a complete TSPLIB problem (with very large weights 
    for the pruned egdes) to given problem path. This makes a copy of the original
    graph in order to minimise the time spent altering edge weights. 
    """
    
    writing_graph = copy.deepcopy(original_graph)
    ghost_edges = fix_utils.get_all_ghost_edges(original_graph, pruned_graph)
    writing_graph = graph_utils.largify_weights(writing_graph, ghost_edges)
    write_tsplib.write_tsp_weights(problem_path, writing_graph)

    return None


def evaluate_tsp_problem(wrapper, problem_path, temp_path, solution_path=None):
    """ 
    Given the tsp problem and solution path, evalate the optimality ratio
    using the sparsification classifier stored in the wrapper. Return the 
    optmality ratio and problem order
    """

    original_graph = load_problem(problem_path)
    original_objective = compute_solution_objective(original_graph, problem_path, solution_path)
    pruned_graph = prune_graph_feasibly(original_graph, wrapper)
    write_pruned_graph_tsplib(temp_path, original_graph, pruned_graph)
    pruned_objective = compute_solution_objective(original_graph, temp_path)
    
    optimality_ratio = pruned_objective / original_objective
    pruning_rate = 1 - len(pruned_graph.edges) / len(original_graph.edges)
    problem_order = len(original_graph.edges)

    return {"optimality_ratio": optimality_ratio,
            "pruning_rate": pruning_rate,
            "problem_order": problem_order
    }


def evaluate_tsp_problems(model_path, problem_dir, solution_dir, temp_path, results_path=None):
    """ 
    Evaluate the optimality ratios of the problems in the given directory.
    Calls Concorde for solving the instances, so that computations can be
    done as quickly as possible.
    """

    problem_orders, optimality_ratios = [], []
    
    try:
        check_problems(problem_dir)
    except Exception as exception:
        print("Something went wrong, checking if all files in problem dir are valid!")
        raise exception
    
    try:
        print("Attempting to find the problems in the specified directory!")
        if solution_dir is not None:
            print("Attempting to match solution files to problem files!")
        problem_pairs = match_problems_to_solutions(problem_dir, solution_dir)
    except Exception as exception:
        print("Error trying to locate files!")
        raise exception
    
    try:
        print("Loading and warpping trained model at {}!".format(model_path))
        wrapper = load_and_wrap_model(model_path)
    except Exception as exception:
        print("Could not load the model and wrap it!")
        raise exception

    results_dict = {}
    
    for num, (problem_path, solution_path) in enumerate(problem_pairs):
        print("Loading problem {} of {}".format(num+1, len(problem_pairs)))
        try:
            res = evaluate_tsp_problem(wrapper, problem_path, temp_path, solution_path)
            results_dict[os.path.basename(problem_path)] = res
        except Exception as exception:
            print("Could not evaluate problem {}!".format(os.path.basename(problem_path)))
            print(exception)

    if results_path is not None:
        with open(results_path, 'w') as item:
            json.dump(results_dict, item)
    else:
        printer = pprint.PrettyPrinter(indent=4)
        printer.pprint(results_dict)

    print("Done! :D")


    
if __name__ is not "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--problem_dir", nargs="?", required=True)
    parser.add_argument("-s", "--solution_dir", nargs="?", default=None)
    parser.add_argument("-t", "--temp_path", nargs="?", required=True)
    parser.add_argument("-m", "--model_path", nargs="?", required=True)
    parser.add_argument("-r", "--results_path", nargs="?", default=None)

    evaluate_tsp_problems(**vars(parser.parse_args()))


    
