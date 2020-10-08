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


def set_edge_objective(problem, variable_dict, graph):
    """ Set the edge objective """

    objective = xpress_sum(mip_utils.define_edge_objective(variable_dict, graph))
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

    variables = list(variable_dict.values())
    return [get_varval(problem, variable, variable_dict) for variable in variables]


def get_varvals(problem, variable_dict):
    """ Get the current variable values for a given problem """

    varvals = []
    try:
        problem.getlpsol(varvals)
    except:
        varvals = problem.getSolution()
    return varvals


def add_mincut(problem, variables, variable_dict):
    """ Add a mincut constraint to the constraint matrix via callback """

    edges = mip_utils.get_edges_from_variables(variables)
    vertices = np.unique(edges)
    columns = len(variables)
    arcs = len(variable_dict) - len(edges)
    m = len(vertices)
    problem.addcuts([1], "L", [m - 1], [0, columns], variables, [1] * columns)


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
    "get_varval": get_varval,
    "sum": xpress_sum,
    "add_mincut": add_mincut,
}

    
