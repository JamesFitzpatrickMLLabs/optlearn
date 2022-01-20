from optlearn.mip.packages import scip_utils
from optlearn.mip.packages import xpress_utils


_solver_packages = {
    "scip": scip_utils,
    "xpress": xpress_utils,
}


class mipWrapper():
    def __init__(self, solver_package, verbose=False):
        """ Setup problem """

        self.verbose = verbose
        self.solver_package = solver_package
        
        self._initialise_solver_functions()
        
    def _initialise_solver_functions(self, solver_package=None):
        """ Set up the solver function dictionary """

        if self.solver_package not in _solver_packages.keys():
            raise ValueError(f"Solver package {self.solver_package} not recognised!")
        self.solver_package = _solver_packages.get(self.solver_package)        
        self._functions = self.solver_package._functions

    def set_solver(self, solver_package):
        """ Set the solver for the wrapper """

        self.solver_package = solver_package
        self._initialise_solver_functions()
                
    def initialise_problem(self, problem_name=None):
        """ Setup the problem and add as an attribute """

        if self.verbose:
            print("Building problem!")
        self.problem = self._functions["build_problem"]()

    def set_continuous_variable(self, name, lower_bound, upper_bound):
        """ Build and add a continuous variable to the problem """

        variable = self._functions["set_continuous_variable"](
            name=name,
            problem=self.problem,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        return variable

    def set_integer_variable(self, name, lower_bound, upper_bound):
        """ Build and add an integer variable to the problem """

        variable = self._functions["set_integer_variable"](
            name=name,
            problem=self.problem,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        return variable

    def set_binary_variable(self, name):
        """ Build and add a binary variable to the problem """

        variable = self._functions["set_binary_variable"](
            name=name,
            problem=self.problem
        )

        return variable

    def set_continuous_variables(self, names, lower_bounds, upper_bounds):
        """ Build and add continuous variables to the problem """

        variables = self._functions["set_continuous_variables"](
            names=names,
            problem=self.problem,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        return variables

    def set_integer_variables(self, names, lower_bounds, upper_bounds):
        """ Build and add integer variables to the problem """

        variables = self._functions["set_integer_variables"](
            names=names,
            problem=self.problem,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        return variables

    def set_binary_variables(self, names):
        """ Build and add binary variables to the problem """

        variables = self._functions["set_binary_variables"](
            names=names,
            problem=self.problem
        )

        return variables

    def sum_items(self, items):
        """ Sum the given items """

        sum_expression = self._functions["sum_items"](items)

        return sum_expression

    def set_constraint(self, lhs, rhs, operator, name=""):
        """ Set the constraint with the given elements to the problem """

        constraint = self._functions["set_linear_constraint"](
            self.problem,
            operator=operator,
            name=name,
            lhs=lhs,
            rhs=rhs,
        )

        return constraint

    def set_sos1_constraint(self, variables, name=""):
        """ Set the SOS(1) constraint with the given variables to the problem """

        constraint = self._functions["set_sos1_constraint"](
            self.problem,
            variables=variables,
            name=name,
        )

        return constraint

    def set_sos2_constraint(self, variables, name=""):
        """ Set the SOS(2) constraint with the given variables to the problem """

        constraint = self._functions["set_sos2_constraint"](
            self.problem,
            variables=variables,
            name=name,
        )

        return constraint
    
    def set_objective(self, objective):
        """ Set the given objective function to the problem  """

        self._functions["set_objective"](
            self.problem,
            objective=objective,
        )
            
        return objective
    
    def get_variables(self):
        """ Get the variables from the problem """

        variables = self._functions["get_variables"](self.problem)

        return variables

    def get_variable_value(self, variable):
        """ Get the value of the given variable """

        value = self._functions["get_variable_value"](
            self.problem,
            variable,
        )

        return value

    def get_variable_values(self, variables):
        """ Get the values of the given variables """

        values = self._functions["get_variable_values"](
            self.problem,
            variables,
        )

        return values

    def get_variable_dict(self):
        """ Get a dictionary of the variables, keyed by their names """

        variable_dict = self._functions["get_variable_dict"](self.problem)

        return variable_dict

    def get_solution(self):
        """ Get the current solution """

        return self._functions["get_solution"](self.problem)

    def get_solutions(self):
        """ Get the all identified solution """

        return self._functions["get_solutions"](self.problem)
    
    def get_solution_value(self, solution, variable):
        """ Get the value of the given variable in the current solution """

        value = self._functions["get_solution_value"](
            self.problem,
            solution=solution,
            variable=variable,
        )

        return value

    def get_solution_values(self, solution, variables):
        """ Get the value of the given variables in the current solution """

        values = self._functions["get_solution_values"](
            self.problem,
            solution=solution,
            variables=variables,
        )

        return values

    def get_reduced_cost(self, variable):
        """ Get the reduced cost of the given variable in the current solution """

        value = self._functions["get_reduced_cost"](
            self.problem,
            variable=variable,
        )

        return value

    def get_reduced_costs(self, variables):
        """ Get the reduced costs of the given variables in the current solution """

        values = self._functions["get_reduced_costs"](
            self.problem,
            variables=variables,
        )

        return costs
    
    def get_objective_value(self):
        """ Get the objective value of the current solution """

        value = self._functions["get_objective_value"](self.problem)

        return value

    def get_objective_values(self):
        """ Get the objective values of all known solutions """

        objective_values = self._functions["get_objective_values"](self.problem)

        return objective_values

    def count_problem_constraints(self):
        """ Count the number of user-defined constraints in the problem """

        number_of_constraints = self._functions["count_problem_constraints"](self.problem)

        return number_of_constraints

    def fix_binary_variable(self, variable):
        """ Fix a binary variable to its lower bound """

        _ = self._functions["fix_binary_variable"](self.problem, variable)

        return None

    def fix_binary_variable(self, variables):
        """ Fix an iterable of binary variables to their lower bound """

        _ = self._functions["fix_binary_variables"](self.problem, variables)

        return None

    def release_binary_variable(self, variable):
        """ Release a binary variable from its lower bound """

        _ = self._functions["release_binary_variable"](self.problem, variable)

        return None

    def release_binary_variable(self, variables):
        """ Release an iterable of binary variables from their lower bound """

        _ = self._functions["release_binary_variables"](self.problem, variables)

        return None
