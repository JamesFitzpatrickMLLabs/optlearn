import xpress as xp
import numpy as np

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


def set_constraint(problem, lhs, rhs, operator, name=None):
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


def fix_binary_variable(problem, variable):
    """ Fix the given binary variable to zero """

    variable_index = problem.getIndex(variable)
    problem.chgbounds([variable_index], ["U"], [0])

    return None


def fix_binary_variables(problem, variables):
    """ Fix the given binary variables to zero """

    for variable in variables:
        _ = fix_binary_variable(problem, variable)
    
    return None


def fix_continuous_variable(problem, variable):
    """ Fix the given continuous variable to zero """

    variable_index = problem.getIndex(variable)
    problem.chgbounds([variable_index], ["U"], [0])

    return None


def fix_continuous_variables(problem, variables):
    """ Fix the given continuous variables to zero """

    for variable in variables:
        _ = fix_continuous_variable(problem, variable)
    
    return None


def release_binary_variable(problem, variable):
    """ Release the given binary variable, so that it can be nonzero """

    variable_index = problem.getIndex(variable)
    problem.chgbounds([variable_index], ["U"], [1])

    return None


def release_binary_variables(problem, variables):
    """ Release the given binary variables, so that they can be nonzero """

    for variable in variables:
        _ = release_binary_variable(problem, variable)
    
    return None


def release_continuous_variable(problem, variable, upper_bound):
    """ Release the given continuous variable, so take values uo to the bound  """

    variable_index = problem.getIndex(variable)
    problem.chgbounds([variable_index], ["U"], [upper_bound])

    return None


def release_continuous_variables(problem, variables, upper_bound):
    """ Release the given binary variables, so that they can be nonzero """

    if type(upper_bound) == int:
        upper_bound = [upper_bound] * len(variables)

    if len(upper_bound) != len(variables):
        _warning = "Can't release continuous variables, must have as many bounds as variables!"
        raise ValueError(_warning)
    
    for (variable, upper_bound) in variables:
        _ = release_binary_variable(problem, variable)
    
    return None


def relax_binary_variable(problem, variable):
    """ Release the given binary variable, so that it can be non-integral """

    variable_index = problem.getIndex(variable)
    problem.chgcoltype([variable_index], ["C"])

    return None


def relax_binary_variables(problem, variables):
    """ Relax the given binary variables, so that they can be non-integral """

    for variable in variables:
        _ = relax_binary_variable(problem, variable)
    
    return None


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
    "fix_binary_variable": fix_binary_variable,
    "fix_binary_variables": fix_binary_variables,
    "release_binary_variable": release_binary_variable,
    "release_binary_variables": release_binary_variables,
    "fix_continuous_variable": fix_continuous_variable,
    "fix_continuous_variables": fix_continuous_variables,
    "release_continuous_variable": release_continuous_variable,
    "release_continuous_variables": release_continuous_variables,
    "relax_binary_variable": relax_binary_variable,
    "relax_binary_variables": relax_binary_variables,

}

    
