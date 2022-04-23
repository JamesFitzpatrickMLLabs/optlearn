import numpy as np
import matplotlib.pyplot as plt

from optlearn import plotting

from optlearn.io import utils
from optlearn.mip import mip_model


def solve_problem(problem):
    """ Get an optimal solution to the given problem """

    problem.optimise()
    return problem.get_solution()


def add_solutions_as_constraints(problem, solutions):
    """ Given a problem, add known solutions as constraints to avoid them """

    for sol in solutions:
        lhs = problem._funcs["sum"]([var for (val, var) in
                                     zip(sol, problem.variable_dict.values()) if val > 0])
        problem.set_constraint(lhs, len(problem.vertices) - 1, "<=")
    return problem


def add_bound(problem, upper_bound):
    """ Using a known solution, add an lower bound """

    lhs = problem._funcs["sum"](list(problem.variable_dict.values()))
    problem.set_constraint(lhs, upper_bound, "<=")
    return problem
    

def solve_problem_without_previous_solutions(problem, solutions):
    """ Solve the given problem, but adding constraints to prevent previous solutions """

    problem = add_solutions_as_constraints(problem, solutions)
    return solve_problem(problem)


def get_all_optimal_tsp_solutions(graph, solver="scip"):
    """ Get all optimal solutions for the given problem """
    
    problem = mip_model.tspProblem(graph, solver=solver, var_type="binary")
    solutions = [solve_problem(problem=problem)]
    optimal_value, global_optimum = (problem.get_objective_value(), ) * 2 
    
    while optimal_value == global_optimum:
        problem = mip_model.tspProblem(graph, solver=solver, var_type="binary")
        # problem._funcs["add_solution"](problem.problem, solutions[0])
        # problem = add_bound(problem, global_optimum)
        solution = solve_problem_without_previous_solutions(problem=problem,
                                                            solutions=solutions)
        optimal_value = problem.get_objective_value()
        
        if optimal_value == global_optimum:
            solutions.append(solution)

    return (np.sum(solutions, axis=0) > 0.1).astype(int)
