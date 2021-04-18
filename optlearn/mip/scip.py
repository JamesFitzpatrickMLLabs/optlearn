import pyscipopt as ps

from optlearn import io_utils
from optlearn.mip import mip_utils


_var_types = {
    "binary": {
        "vtype": "B",
        },
    "continuous": {
        "vtype": "C",
        "ub": 1,
        "lb": 0
        },
}


def get_var_args(var_args=None):
    """ Set the variable type, default to continuous """

    self._var_args = _var_types.get(var_type) or _var_types["continuous"]  


def build_problem():
    """ Build an xpress problem instance """

    return ps.Model()


def create_variable(problem, edge, prefix="x", vtype=None):
    """ Create a single variable and return it """

    vtype = vtype or {}
    
    name = mip_utils.name_variable(edge, prefix=prefix)
    return problem.addVar(name, **vtype)


def perform_relaxation(problem):
    """ Perform a linear relaxation on the current problem  """
    
    return [item.getLPSol() for item in problem.getVars()]


def add_variables(problem, variables):
    """ Blank function to maintain consistent API """

    return None


def scip_sum(terms):
    """ Sum the given terms """

    return ps.quicksum(terms)


def set_edge_objective(problem, variable_dict, graph, perturb=False):
    """ Set the edge objective """

    objective = scip_sum(mip_utils.define_edge_objective(variable_dict, graph, perturb))
    problem.setObjective(objective, "minimize")


def set_constraint(problem, lhs, rhs, operator):
    """ Set a constraint for the given problem """

    problem.addCons(mip_utils._operators[operator](lhs, rhs))


def solve_problem(problem, kwargs=None):
    """ Solve the problem using the default solver """

    return problem.optimize()


def get_varnames(problem, variable_dict):
    """ Get the variable names for a given problem """

    return [item.name for item in problem.getVars()]


def get_varval(problem, variable, variable_dict):
    """ Get a specific variable value """

    return problem.getVal(variable)


def get_redcost(problem, variable, variable_dict):
    """ Get a specific reduced cost """

    return problem.getVarRedcost(variable)


def get_varvals(problem, variable_dict):
    """ Get all variable values """

    return [get_varval(problem, variable, variable_dict) for
            variable in variable_dict.values()] 


def get_redcosts(problem, variable_dict):
    """ Get all reduced costs """

    return [get_redcost(problem, variable, variable_dict) for
            variable in variable_dict.values()] 


def get_solvals(problem, solution, variable_dict):
    """ Get the solvals of the current solution """

    return [problem.getSolVal(solution, item) for item in variable_dict.values()]


def get_varvals_by_name(problem, variable_dict, variable_keys):
    """ Get all variable values in a specific order """

    return [get_varval(problem, variable_dict[key], variable_dict) for
            key in variable_keys] 


def get_redcosts_by_name(problem, variable_dict, variable_keys):
    """ Get all reduced costs in a specific order """

    return [get_redcost(problem, variable_dict[key], variable_dict) for
            key in variable_keys] 


def get_objective_value(problem):
    """ Get the current objective value of the problem """

    return problem.getObjVal()


def add_solution(problem, solution):
    """ Add the given solution to the problem """

    raise NotImplementedError
    

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
    "get_solution": get_varvals,
    "get_varvals_by_name": get_varvals_by_name,
    "get_solvals": get_solvals,
    "get_varval": get_varval,
    "get_redcosts": get_redcosts,
    "get_redcosts_by_name": get_redcosts_by_name,
    "get_redcost": get_redcost,
    "get_objective_value": get_objective_value,
    "sum": scip_sum,
    "add_solution": add_solution,
    }
