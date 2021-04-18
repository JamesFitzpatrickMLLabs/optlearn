import xpress as xp
import numpy as np

from optlearn import io_utils
from optlearn.mip import mip_utils


_var_types = {
    "binary": {
        "vartype": xp.binary,
        },
    "continuous": {
        "vartype": xp.continuous,
        "ub": 1,
        "lb": 0
        },
}


def _check_npvar():
    """ Check if xpress has the npvar dtype, if it doesn't, set it to object """

    if hasattr(xp, "npvar"):
        return None
    else:
        xp.npvar = object
        print("Could not find xp.npvar, xpress license must be <= v8.8")
        print("Setting xp.npvar dtype to object")

        
def get_var_args(var_args=None):
    """ Set the variable type, default to continuous """

    self._var_args = _var_types.get(var_type) or _var_types["continuous"]  


def build_problem():
    """ Build an xpress problem instance """

    return xp.problem()


def create_variable(problem, edge, prefix="x", var_args=None):
    """ Create a single xpress variable and return it """

    name = mip_utils.name_variable(edge, prefix=prefix)
    return xp.var(name=name, **var_args)


def perform_relaxation(problem):
    """ Perform a linear relaxation of the problem """

    return problem.lpoptimize()


def add_variables(problem, variables):
    """ Add all of the variables to the given problem """

    problem.addVariable(*variables)


def xpress_sum(terms):
    """ Sum the given terms """

    return xp.Sum(terms)


def set_edge_objective(problem, variable_dict, graph, perturb=False):
    """ Set the edge objective """

    objective = xpress_sum(mip_utils.define_edge_objective(variable_dict, graph, perturb))
    problem.setObjective(objective)


def set_constraint(problem, lhs, rhs, operator):
    """ Set a constraint for the given problem """

    problem.addConstraint(mip_utils._operators[operator](lhs, rhs))


def solve_problem(problem, kwargs=None):
    """ Solve the problem using the default solver """

    kwargs = kwargs or {}
    return problem.solve(**kwargs)


def get_varnames(problem, variable_dict):
    """ Get the variable names for a given problem """

    return [problem.getVariable(item).name for item in range(len(variable_dict))]


def get_varval(problem, variable, variable_dict):
    """ Get a specific variable value """

    variable_index = list(variable_dict.keys()).index(variable.name)
    return problem.getlpsolval(variable_index, 0)[0]


def get_varvals(problem, variable_dict):
    """ Get the current variable values for a given problem """

    varvals = []
    problem.getlpsol(varvals)
    return varvals


def get_redcost(problem, variable, variable_dict):
    """ Get a specific reduced cost value """

    variable_index = list(variable_dict.keys()).index(variable.name)
    return problem.getlpsolval(variable_index, 0)[-1]


def get_redcosts(problem, variable_dict):
    """ Get the current reduced cost values for a given problem """

    redcosts = []
    problem.getlpsol(dj=redcosts)
    return redcosts


def get_solution(problem, variable_dict):
    """ Get the current variable values for a given problem """

    return problem.getSolution()


def add_mincut(problem, variables, variable_dict):
    """ Add a mincut constraint to the constraint matrix via callback """

    edges = mip_utils.get_edges_from_variables(variables)
    vertices = np.unique(edges)
    columns = len(variables)
    arcs = len(variable_dict) - len(edges)
    m = len(vertices)

    varlist = list(variable_dict.values())
    varinds = [varlist.index(var) for var in variables]

    print(columns, m)
    problem.addcuts([0], "L", [m - 1], [0, columns], variables, [1] * columns)


def get_varvals_by_name(problem, variable_dict, variable_keys):
    """ Get all variable values in a specific order """

    return [get_varval(problem, variable_dict[key], variable_dict) for
            key in variable_keys] 
    

def get_objective_value(problem):
    """ Get the objective value of the current solution to the problem """

    return problem.getObjVal()


def get_current_solution(problem):
    """ Gets the current solution during the B&B """

    solution = []
    problem.getlpsol(solution, None, None, None)
    return solution


def add_solution(problem, solution):
    """ Add the given solution to the problem """

    problem.addmipsol(solution)


_funcs = {
    "create_variable": create_variable,
    "build_problem": build_problem,
    "perform_relaxation": perform_relaxation,
    "add_variables": add_variables,
    "edge_objective": set_edge_objective,
    "set_constraint": set_constraint,
    "solve_problem": solve_problem,
    "get_varnames": get_varnames,
    "get_varvals": get_varvals,
    "get_varvals_by_name": get_varvals_by_name,
    "get_redcost": get_redcost,
    "get_redcosts": get_redcosts,
    "get_solution": get_solution,
    "get_current_solution": get_current_solution,
    "get_varval": get_varval,
    "sum": xpress_sum,
    "add_mincut": add_mincut,
    "get_objective_value": get_objective_value,
    "add_solution": add_solution,
}

    
