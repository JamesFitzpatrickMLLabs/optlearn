import copy
import time
import hashlib

import matplotlib.pyplot as plt

from optlearn.io import read
from optlearn.io.graph import graph_io

from optlearn.plotting import plot_utils
from optlearn.graph import process_utils
from optlearn.graph import generate_evrpnl
from optlearn.mip.routing import evrpnl_problem_builder


def check_if_solved(problem_builder):
    """ Check if the problem is solved to optimality """

    if problem_builder.problem.getProbStatus() == 6:
        is_solved = True
    else:
        is_solved = False

    return is_solved


def add_nodes_to_graph(graph, number_of_copies):
    """ Add station copies to the graph to restore feasibility """
    
    clone_graph = copy.deepcopy(graph)
    clone_graph = process_utils.duplicate_stations_uniform(clone_graph, number_of_copies)

    return clone_graph


def solve_until_optimal(graph, num_iterations=5):
    """ Solve the problem until we find the optimal solution """

    was_feasible = []
    objectives = []
    is_complete = False
    iterations = 0
    
    while is_complete is False and iterations < num_iterations:
        working_graph = copy.deepcopy(graph)
        working_graph = process_utils.duplicate_stations_uniform(working_graph, iterations)
        problem_builder = evrpnl_problem_builder.evrpnlProblemBuilder(
            solver_package="xpress",
            problem_type="evrpnl",
            is_directed=True,
            pwl_model="delta",
        )
        print("NODES: ", working_graph.nodes)
        problem_builder.build_arc_problem(working_graph)
        problem_builder.problem.solve()
        if check_if_solved(problem_builder):
            was_feasible.append(True)
            objectives.append(problem_builder.get_objective_value())
        else:
            was_feasible.append(False)
            objectives.append(999999999999999)
        if len(objectives) >= 2:
            if check_if_solved(problem_builder):
                if round(objectives[-1], 5) == round(objectives[-2], 5):
                    is_complete = True
        print("OBJECTIVES", objectives)
        iterations += 1
                
    return problem_builder, working_graph, iterations - 1


problem_hash = hashlib.sha1()
problem_hash.update(str(time.time()).encode("utf-8"))
problem_hash = problem_hash.hexdigest()[:10]

generator = generate_evrpnl.generateProblem()
graph = generator.generate_random_problem(number_of_stations=3)
graph = graph.to_directed()

try:
    start = time.time()
    problem_builder, latest_graph, iterations = solve_until_optimal(graph, num_iterations=7) 
    finish = time.time()
    solve_time = finish - start
    latest_graph.graph["solve_time"] = solve_time
    travel_solution = problem_builder.get_travel_solution()
    latest_graph.add_edges_from([(*edge, {"solval": val})
                                 for (edge, val) in zip(list(latest_graph.edges), travel_solution)])
    write_directory = "//"
    graph_io.write_pickle(graph, write_directory + problem_hash + ".pkl")
    graph_io.write_pickle(latest_graph, write_directory + problem_hash + "_solution.pkl")
    problem_builder.plot_problem(latest_graph, show=False, strict=True)
    problem_builder.plot_solution(latest_graph, show=False)
    plt.savefig(f"{write_directory}/{problem_hash}.png")
    plt.close("all")
except:
    pass


