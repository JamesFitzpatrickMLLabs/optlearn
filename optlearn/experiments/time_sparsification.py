import os
import json
import time
import argparse

import numpy as np

from pathlib import Path

from optlearn.experiments import experiment_utils


def evaluate_solve_vanilla(original_graph, solver_params=None):
    """ Evaluate a vanilla custom solve of a TSP problem """

    start = time.time()
    problem = experiment_utils.solve_problem_custom(original_graph, solver_params)
    finish = time.time()

    return {"optimal_value": problem.get_objective_value(), "timing": finish - start}


def evaluate_solve_pruned(wrapper, original_graph, solver_params=None):
    """ Evaluate a pruned custom solve of a TSP problem """

    start = time.time()
    pruned_graph = wrapper.prune_graph_with_logic(original_graph)
    problem = experiment_utils.solve_problem_custom(pruned_graph, solver_params)
    finish = time.time()

    return {"optimal_value": problem.get_objective_value(),
            "pruning_rate": 1 - len(pruned_graph.edges) / len(original_graph.edges),
            "timing": finish - start
    }


def evaluate_tsp_problem(wrapper, problem_path, solver_params=None):
    """ 
    Given the tsp problem and solution path, evalate the optimality ratio
    using the sparsification classifier stored in the wrapper. Return the 
    optmality ratio and problem order
    """

    original_graph = experiment_utils.load_problem(problem_path)

    vanilla_dict = evaluate_solve_vanilla(original_graph, solver_params)
    pruned_dict = evaluate_solve_pruned(wrapper, original_graph, solver_params)
    
    return {"vanilla_time": vanilla_dict["timing"],
            "pruned_time": pruned_dict["timing"],
            "pruning_rate": pruned_dict["pruning_rate"],
            "optimality_ratio": pruned_dict["optimal_value"] / vanilla_dict["optimal_value"], 
            "problem_order": len(original_graph.nodes)
    }


def evaluate_tsp_problems(model_path, problem_dir, results_path, solver_params_path=None):
    """ 
    Given the tsp problem and solution path, evalate the optimality ratio
    using the sparsification classifier stored in the wrapper. Return the 
    optmality ratio and problem order
    """

    if solver_params_path is not None:
        try:
            with open(model_param_json_path, "r") as json_file:
                solver_params = json.load(json_file)
        except FileNotFoundError as exception:
            print("Cannot find the file {}!".format(model_param_json_path))
            raise exception
        except Exception as exception:
            print("Unknown error when trying to load parameters from file!")
    else:
        solver_params = None
    
    try:
        experiment_utils.check_problems(problem_dir)
        filenames = [os.path.join(problem_dir, item) for item in os.listdir(problem_dir)]
    except Exception as exception:
        print("Something went wrong, checking if all files in problem dir are valid!")
        raise exception    
    try:
        print("Loading and wrapping trained model at {}!".format(model_path))
        wrapper = experiment_utils.load_and_wrap_model(model_path)
    except Exception as exception:
        print("Could not load the model and wrap it!")
        raise exception
    
    results_dict = {}
    
    for num, filename in enumerate(filenames):
        print("Loading problem {} of {}".format(num+1, len(filenames)))
        try:
            res = evaluate_tsp_problem(wrapper, filename, solver_params)
            results_dict[os.path.basename(filename)] = res
        except Exception as exception:
            print("Could not evaluate problem {}!".format(os.path.basename(filename)))
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
    parser.add_argument("-t", "--solver_params_path", nargs="?", required=False)
    parser.add_argument("-m", "--model_path", nargs="?", required=True)
    parser.add_argument("-r", "--results_path", nargs="?", default=None)

    evaluate_tsp_problems(**vars(parser.parse_args()))
