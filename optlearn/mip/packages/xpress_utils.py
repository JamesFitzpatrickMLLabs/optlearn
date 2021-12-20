import xpress as xp

from optlearn.mip import mip_utils


def build_problem(problem_name=None):
    """ Build an xpress problem instance """
    
    if problem_name is None:
        problem = xp.problem()
    else:
        problem = xp.problem(name=problem_name)

    return problem


def build_continuous_variable(problem, name="", lower_bound=None, upper_bound=None):
    """ Build a continuous variable with the given name and bounds """

    variable = xp.var(name=name, lb=lower_bound, ub=upper_bound, vartype=xp.continuous)

    return variable


def build_integer_variable(problem, name="", lower_bound=None, upper_bound=None):
    """ Build an integer variable with the given name and bounds """

    variable = xp.var(name=name, lb=lower_bound, ub=upper_bound, vartype=xp.integer)

    return variable


def build_binary_variable(problem, name=""):
    """ Build a binary variable with the given name """

    variable = xp.var(name=name, vartype=xp.binary)

    return variable


def build_continuous_variables(problem, names=[], lower_bounds=[], upper_bounds=[]):
    """ Build continuous variables with the given names and bounds """

    variables = []
    for (name, lower_bound, upper_bound) in zip(names, lower_bounds, upper_bounds):
        variable = build_continuous_variable(problem, name, lower_bound, upper_bound)
        variables.append(variables)

    return variables


def build_integer_variables(problem, names=[], lower_bounds=[], upper_bounds=[]):
    """ Build integer variables with the given names and bounds """

    variables = []
    for (name, lower_bound, upper_bound) in zip(names, lower_bounds, upper_bounds):
        variable = build_integer_variable(problem, name, lower_bound, upper_bound)
        variables.append(variables)

    return variables


def build_binary_variables(problem, names=[]):
    """ Build binary variables with the given names """

    variables = []
    for name in names:
        variable = build_binary_variable(problem, name)
        variables.append(variables)

    return variables


def add_variable_to_problem(problem, variable):
    """ Add the given variable to the given problem """

    problem.addVariable(variable)
    
    return variable


def add_variables_to_problem(problem, variables):
    """ Add the given variables to the given problem """

    _ = [add_variable_to_problem(problem, variable) for variable in variables]
    
    return variables


def set_continuous_variable(problem, name="", lower_bound=None, upper_bound=None):
    """ Build and add a continuous variable with the given name and bounds """

    variable = build_continuous_variable(problem, name, lower_bound, upper_bound)
    add_variable_to_problem(problem, variable)

    return variable


def set_integer_variable(problem, name="", lower_bound=None, upper_bound=None):
    """ Build and add an integer variable with the given name and bounds """

    variable = build_integer_variable(problem, name, lower_bound, upper_bound)
    add_variable_to_problem(problem, variable)

    return variable


def set_binary_variable(problem, name=""):
    """ Build and add a binary variable with the given name """

    variable = build_binary_variable(problem, name)
    add_variable_to_problem(problem, variable)

    return variable


def set_continuous_variables(problem, names=[], lower_bounds=[], upper_bounds=[]):
    """ Build and add continuous variables with the given names and bounds """

    variables = build_continuous_variables(
        problem,
        names,
        lower_bounds,
        upper_bounds
    )
    _ = add_variables_to_problem(variables)

    return variables


def set_integer_variables(problem, names=[], lower_bounds=[], upper_bounds=[]):
    """ Build and add integer variables with the given names and bounds """

    variables = build_integer_variables(
        problem,
        names,
        lower_bounds,
        upper_bounds
    )
    add_variables_to_problem(variables)

    return variables


def set_binary_variables(problem, names=[]):
    """ Build and add binary variables with the given names """

    variables = []
    for name in names:
        variable = build_binary_variable(problem, name)
        variables.append(variables)

    return variables


def sum_items(items):
    """ Sum the given items """

    return xp.Sum(items)


def build_linear_constraint(problem, lhs, rhs, operator, name=None):
    """ Build a constraint for the given problem """

    constraint = problem.addConstraint(mip_utils._operators[operator](lhs, rhs))

    return constraint


def add_constraint_to_problem(problem, constraint):
    """ Add the given constraint to the problem """

    return constraint


def add_constraints_to_problem(problem, constraints):
    """ Add the given constraints to the problem """

    return constraints


def set_linear_constraint(problem, lhs, rhs, operator, name=""):
    """ Build and add the given constraint to the problem """

    constraint = build_linear_constraint(problem, lhs, rhs, operator, name=name)
    add_constraint_to_problem(problem, constraint)

    return constraint


def set_sos1_constraint(problem, variables, name=""):
    """ Build and add an SOS(1) constraint for the given variables """

    indices = list(range(len(variables)))
    constraint = xp.sos(variables, indices, type=1, name=name)
    problem.addSOS(constraint)
    
    return constraint


def set_sos2_constraint(problem, variables, name=""):
    """ Build and add an SOS(2) constraint for the given variables """

    indices = list(range(len(variables)))
    constraint = xp.sos(variables, indices, type=2, name=name)
    problem.addSOS(constraint)    

    return constraint


def set_objective(problem, objective, name=None):
    """ Set the given objective to the problem """

    problem.setObjective(objective, sense=xp.minimize)

    return objective


def solve_problem(problem, kwargs={}):
    """ Solve the problem using the default solver """

    _ = problem.solve(**kwargs)

    return None


def get_number_of_variables(problem):
    """ Get the number of variables in the problem """

    fudge = []
    problem.getbasis(fudge, fudge)
    number_of_variables = len(fudge)

    return fudge


def get_variables(problem):
    """ Get the variables from the problem """

    number_of_variables = get_number_of_variables(problem)
    variables = [problem.getVariable(num) for num in range(number_of_variables)]

    return variables


def get_variable_name(problem, variable):
    """ Get the name of the given variables """

    name = variable.name

    return name


def get_variable_names(problem, variables):
    """ Get the names of the given variables """

    names = [get_variable_name(problem, variable) for variable in variables]

    return names


def get_variable_dict(problem):
    """ Get a dictionary of variables for the given problem """

    variables = get_variables(problem)
    names = get_variable_names(problem, variables)
    variable_dict = {
        name: variable for (name, variable) in zip(names, variables)
    }

    return variable_dict



def get_variable_index(problem, variable):
    """ Get the index of the variable in problem storage """

    index = problem.getIndex(variable)

    return index


def get_variable_indices(problem, variables):
    """ Get the index of the variable in problem storage """

    indices = [get_variable_index(problem, variable) for variable in variables]

    return indices


def get_variable_value(problem, variable):
    """ Get the current value of the given variable """

    index = get_variable_index(problem, variable)
    values = problem.getSolution()
    value = values[index]

    return value


def get_variable_values(problem, variables):
    """ Get the current value of the given variables """

    values = [get_variable_value(problem, variable) for variable in variables]

    return values


def get_reduced_cost(problem, variable):
    """ Get a the reduced cost of the given variable """

    index = get_variable_index(problem, variable)
    values = problem.getlpsolval(index, 0)[-1]
    value = values[index]

    return value


def get_reduced_costs(problem, variables):
    """ Get a the reduced cost of the given variables """

    costs = [get_reduced_cost(problem, variable) for variable in variables]

    return costs



def get_solution_value(problem, solution, variable):
    """ Get the value of the given variable in the given current solution """

    value = problem.getSolVal(solution, variable)

    return value


def get_solution(problem):
    """ Get the current solution for the problem """

    solution = problem.getSolution()

    return solution


def get_solutions(problem):
    """ Get the all identified solutions for the problem """

    solutions = problem.getSols()

    return solutions


def get_solution_values(problem, solution, variables):
    """ Get the values of the given variables in the given current solution """

    values = [get_solution_value(problem, solution, variable) for variable in variables]

    return values


def get_objective_value(problem):
    """ Get the current objective value of the problem """

    value = problem.getObjVal()

    return value


def get_objective_values(problem):
    """ Get the current objective value of the problem """

    solutions = problem.getSols()
    objective_values = [problem.getSolObjVal(solution) for solution in solutions]

    return objective_values


def add_solution_to_problem(problem, solution):
    """ Add the given solution to the problem """

    raise NotImplementedError

    
_functions = {
    "sum_items": sum_items,
    "build_problem": build_problem,
    "build_continuous_variables": build_continuous_variables,
    "build_continuous_variable": build_continuous_variable,
    "set_continuous_variables": set_continuous_variables,
    "set_continuous_variable": set_continuous_variable,
    "build_integer_variables": build_integer_variables,
    "build_integer_variable": build_integer_variable,
    "set_integer_variables": set_integer_variables,
    "set_integer_variable": set_integer_variable,
    "build_binary_variables": build_binary_variables,
    "build_binary_variable": build_binary_variable,
    "set_binary_variables": set_binary_variables,
    "set_binary_variable": set_binary_variable,
    "set_linear_constraint": set_linear_constraint,
    "set_sos1_constraint": set_sos1_constraint,
    "set_sos2_constraint": set_sos2_constraint,
    "build_linear_constraint": build_linear_constraint,
    "add_constraint_to_problem": add_constraint_to_problem,
    "add_constraints_to_problem": add_constraints_to_problem,
    "add_variables_to_problem": add_variables_to_problem,
    "add_variable_to_problem": add_variable_to_problem,
    "set_objective": set_objective,
    "solve_problem": solve_problem,
    "get_variables": get_variables,
    "get_variable_names": get_variable_names,
    "get_variable_name": get_variable_name,
    "get_variable_dict": get_variable_dict,
    "get_variable_values": get_variable_values,
    "get_variable_value": get_variable_value,
    "get_solution": get_solution,
    "get_solutions": get_solutions,
    "get_solution_values": get_solution_values,
    "get_solution_value": get_solution_value,
    "get_objective_value": get_objective_value,
    "get_objective_values": get_objective_values,
    "get_reduced_costs": get_reduced_costs,
    "get_reduced_cost": get_reduced_cost,
    "add_solution_to_problem": add_solution_to_problem,
}
