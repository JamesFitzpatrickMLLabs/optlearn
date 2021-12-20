import mip as mp

from optlearn.mip import mip_utils


_var_types = {
    "binary": {
        "var_type": mp.BINARY,
        },
    "continuous": {
        "var_type": mp.CONTINUOUS,
        # "ub": 1,
        # "lb": 0
        },
}


        
def get_var_args(var_args=None):
    """ Set the variable type, default to continuous """

    self._var_args = _var_types.get(var_type) or _var_types["continuous"]  


def build_problem():
    """ Build an xpress problem instance """

    return mp.Model()


def create_variable(problem, edge, prefix="x", var_type=None):
    """ Create a single variable and return it """

    name = mip_utils.name_variable(edge, prefix=prefix)
    return problem.add_var(name, **var_type)


def perform_relaxation(problem):
    """ Perform a linear relaxation on the current problem  """
    
    return problem.optimize(relax=True)


def add_variables(problem, variables):
    """ Blank function to maintain consistent API """

    return None


def coinor_sum(terms):
    """ Sum the given terms """

    return mp.xsum(terms)


def set_edge_objective(problem, variable_dict, graph):
    """ Set the edge objective """

    objective = coinor_sum(mip_utils.define_edge_objective(variable_dict, graph))
    problem.objective = objective


def set_constraint(problem, lhs, rhs, operator):
    """ Set a constraint for the given problem """

    problem += mip_utils._operators[operator](lhs, rhs)


def get_varnames(problem, variable_dict):
    """ Get the variable names for a given problem """

    return [item.name for item in problem.vars]


def get_varval(problem, variable, variable_dict):
    """ Get a specific variable value """

    return variable.x
    

def get_varvals(problem, variable_dict):
    """ Get the variable values for a given problem """

    return [get_varval(problem, variable, variable_dict) for variable in problem.vars]


def solve_problem(problem, kwargs=None):
    """ Solve the problem using the default solver """

    kwargs = kwargs or {}
    return problem.optimize(**kwargs)


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
    "sum": coinor_sum,
    }

    
