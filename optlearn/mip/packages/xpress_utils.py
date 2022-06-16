import time
import hashlib

import xpress as xp

from optlearn.mip import mip_utils


def generate_constraint_hash():
    """ Generate a hash to append to a constraint name """

    constraint_hash = hashlib.sha1()
    constraint_hash.update(str(time.time()).encode("utf-8"))
    constraint_hash = constraint_hash.hexdigest()[:10]

    return constraint_hash


def append_constraint_hash(constraint_name):
    """ Append a constraint hash to the constraint name """

    constraint_hash = generate_constraint_hash()
    constraint_name = constraint_name + f" --- {constraint_hash}"

    return constraint_name


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

    name = name or ""
    name = append_constraint_hash(name)
    constraint = xp.constraint(mip_utils._operators[operator](lhs, rhs), name=name)

    return constraint


def add_constraint_to_problem(problem, constraint):
    """ Add the given constraint to the problem """

    problem.addConstraint(constraint)
    
    return constraint


def add_constraints_to_problem(problem, constraints):
    """ Add the given constraints to the problem """

    for constraint in constraints:
        problem.addConstraint(constraint)
        
    return constraints


def set_linear_constraint(problem, lhs, rhs, operator, name=""):
    """ Build and add the given constraint to the problem """

    constraint = build_linear_constraint(problem, lhs, rhs, operator, name=name)
    add_constraint_to_problem(problem, constraint)

    return constraint


def set_sos1_constraint(problem, variables, name=""):
    """ Build and add an SOS(1) constraint for the given variables """

    name = append_constraint_hash(name)
    indices = list(range(len(variables)))
    constraint = xp.sos(variables, indices, type=1, name=name)
    problem.addSOS(constraint)
    
    return constraint


def set_sos2_constraint(problem, variables, name=""):
    """ Build and add an SOS(2) constraint for the given variables """

    name = append_constraint_hash(name)
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
    value = problem.getlpsolval(index, 0)[-1]

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


def count_problem_constraints(problem):
    """ Count the number of constraints """

    constraint_count = 0
    potential_constraints, encountered_index_error = range(100000), False
    while not encountered_index_error:
        for number in potential_constraints:
            try:
                _ = problem.getConstraint(number)
                constraint_count += 1
            except IndexError as exception:
                return constraint_count

    return constraint_count


def count_problem_variables(problem):
    """ Count the number of constraints """

    variable_count = 0
    potential_variables, encountered_index_error = range(100000), False
    while not encountered_index_error:
        for number in potential_variables:
            try:
                _ = problem.getVariable(number)
                variable_count += 1
            except IndexError as exception:
                return variable_count

    return variable_count


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
    "count_problem_constraints": count_problem_constraints,
    "count_problem_variables": count_problem_variables,
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
