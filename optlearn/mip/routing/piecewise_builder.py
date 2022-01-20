import numpy as np

import networkx as nx

from optlearn.mip import mip_wrapper


class pwlBuilder(mip_wrapper.mipWrapper):

    def __init__(self, solver_package="scip", verbose=False):

        self.solver_package = solver_package
        self.verbose = verbose

        self._set_default_storage()

    def _set_lambda_storage(self):
        """ Set the storage function for the lambda variables """

        self.lambda_storage = nx.Graph()

        return None

    def _set_binary_storage(self):
        """ Set the storage function for the binary variables """

        self.binary_storage = nx.Graph()

        return None

    def _set_delta_storage(self):
        """ Set the storage function for the delta variables """

        self.delta_storage = nx.Graph()

        return None
    
    def _set_default_storage(self):
        """ Set the default variable storage """

        self._set_lambda_storage()
        self._set_delta_storage()
        self._set_binary_storage()

        return None

    def _warn_breakpoints(self):
        """ Print a warning about the breakpoints """

        return None
        
    def set_vertical_breakpoints(self, vertical_breakpoints):
        """ Set the vertical breakpoints for the PWL approximation """

        self._warn_breakpoints()
        self.vertical_breakpoints = vertical_breakpoints

        return None

    def set_horizontal_breakpoints(self, horizontal_breakpoints):
        """ Set the horizontal breakpoints for the PWL approximation """

        self._warn_breakpoints()
        self.horizontal_breakpoints = horizontal_breakpoints

        return None

    def get_vertical_breakpoints(self):
        """ Get the vertical breakpoints for the PWL approximation """

        vertical_breakpoints = self.vertical_breakpoints

        return vertical_breakpoints

    def get_horizontal_breakpoints(self):
        """ Get the horizontal breakpoints for the PWL approximation """

        horizontal_breakpoints = self.horizontal_breakpoints

        return horizontal_breakpoints

    def name_lambda_variable(self, name_stem="h_", breakpoint_index=0, base_index=None):
        """ Name the lambda variable, giving it indices to distinguish it """

        name = name_stem
        if base_index is not None:
            name = name + f"{base_index},"
        name = name + f"{breakpoint_index}"

        return name

    def name_delta_variable(self, name_stem="d_", breakpoint_index=0, base_index=None):
        """ Name the delta variable, giving it indices to distinguish it """

        name = name_stem
        if base_index is not None:
            name = name + f"{base_index},"
        name = name + f"{breakpoint_index}"

        return name
    
    def name_binary_variable(self, name_stem="w_", breakpoint_index=0, base_index=None):
        """ Name the binary variable, giving it indices to distinguish it """

        name = name_stem
        if base_index is not None:
            name = name + f"{base_index},"
        name = name + f"{breakpoint_index}"

        return name

    def store_variables(self, variables, storage_graph, base_index=None):
        """ Store the given variables in the given storage graph """

        if base_index is None:
            if len(storage_graph.nodes) == 0:
                base_index = 1
            else:
                base_index = max(storage_graph.nodes) + 1
        
        attribute_dict = {breakpoint: variable for (breakpoint, variable) in enumerate(variables)}
        storage_graph.add_nodes_from([(base_index, attribute_dict)])
            
        return None

    def build_lambda_variables(self, name_stem="h_", base_index=None, storage_graph=None):
        """ Build all of the necessary lambda variables, naming them as stated """

        variables = []
        for num in range(0, len(self.get_horizontal_breakpoints())):
            variable_name = self.name_lambda_variable(
                name_stem=name_stem,
                breakpoint_index=num,
                base_index=base_index,
            )
            variable = self.set_continuous_variable(variable_name, 0, 1)
            variables.append(variable)

        if storage_graph is None:
            storage_graph = self.lambda_storage
        self.store_variables(variables, storage_graph, base_index)

        return variables

    def build_delta_variables(self, name_stem="d_", base_index=None, storage_graph=None):
        """ Build all of the necessary lambda variables, naming them as stated """

        variables = []
        breakpoints = self.get_horizontal_breakpoints()
        for num in range(1, len(breakpoints)):
            difference = breakpoints[num] - breakpoints[num - 1]
            variable_name = self.name_delta_variable(
                name_stem=name_stem,
                breakpoint_index=num,
                base_index=base_index,
            )
            variable = self.set_continuous_variable(variable_name, 0, difference)
            variables.append(variable)

        if storage_graph is None:
            storage_graph = self.delta_storage
        self.store_variables(variables, storage_graph, base_index)

        return variables
    
    def build_binary_variables(self, name_stem="w_", base_index=None, storage_graph=None):
        """ Build all of the necessary binary variables, naming them as stated """

        variables = []
        for num in range(1, len(self.get_horizontal_breakpoints())):
            variable_name = self.name_binary_variable(
                name_stem=name_stem,
                breakpoint_index=num,
                base_index=base_index,
            )
            variable = self.set_binary_variable(variable_name)
            variables.append(variable)

        if storage_graph is None:
            storage_graph = self.binary_storage
        self.store_variables(variables, storage_graph, base_index)

        return variables

    def build_variables_lambda(self, lambda_stem="h_", binary_stem="w_",
                        binary_storage=None, lambda_storage=None, base_index=None):
        """ Build all of the lambda and binary variables needed for the model """

        _ = self.build_lambda_variables(
            name_stem=lambda_stem,
            base_index=base_index,
            storage_graph=lambda_storage,
        )
        _ = self.build_binary_variables(
            name_stem=binary_stem,
            base_index=base_index,
            storage_graph=binary_storage,
        )

        return None

    def build_variables_delta(self, delta_stem="d_", binary_stem="w_",
                              binary_storage=None, delta_storage=None, base_index=None):
        """ Build all of the lambda and binary variables needed for the model """

        _ = self.build_delta_variables(
            name_stem=delta_stem,
            base_index=base_index,
            storage_graph=delta_storage,
        )
        _ = self.build_binary_variables(
            name_stem=binary_stem,
            base_index=base_index,
            storage_graph=binary_storage,
        )

        return None

    def get_variables_from_storage(self, storage_graph, node=None):
        """ Get all of the variables from the given storage graph """

        if node is None:
            node = max(storage_graph.nodes)
        keys = sorted(storage_graph.nodes[node].keys()) 
        variables = [storage_graph.nodes[node][key] for key in keys]
        
        return variables

    def get_lambda_variables_from_storage(self, storage_graph=None, node=None):
        """ Get the lambda variables from the given (or default) storage """

        if storage_graph is None:
            storage_graph = self.lambda_storage

        variables = self.get_variables_from_storage(storage_graph, node)

        return variables

    def get_delta_variables_from_storage(self, storage_graph=None, node=None):
        """ Get the delta variables from the given (or default) storage """

        if storage_graph is None:
            storage_graph = self.delta_storage

        variables = self.get_variables_from_storage(storage_graph, node)

        return variables
    
    def get_binary_variables_from_storage(self, storage_graph=None, node=None):
        """ Get the binary variables from the given (or default) storage """

        if storage_graph is None:
            storage_graph = self.binary_storage
            
        variables = self.get_variables_from_storage(storage_graph, node)
        
        return variables
    
    def build_lambda_convexity_constraint(self, lambda_storage=None,
                                   binary_storage=None, base_index=None, rhs=None):
        """ Make sure the lambdas form a convex combination """

        lambda_variables = self.get_lambda_variables_from_storage(lambda_storage, base_index)
        lhs = self.sum_items(lambda_variables)
        constraint_name = "PWL convexity constraint"
        if base_index is not None:
            constraint_name = constraint_name + f" on {base_index}"
        if rhs is None:
            rhs = self.sum_items(self.get_binary_variables_from_storage(binary_storage, base_index))
        _ = self.set_constraint(lhs, rhs, "==", constraint_name)
        
        return None

    def build_lambda_equality_constraint(self, storage_graph=None, base_index=None, rhs=None):
        """ Make sure the lambdas form a convex combination """

        binary_variables = self.get_binary_variables_from_storage(storage_graph, base_index)
        lhs = self.sum_items(binary_variables)
        constraint_name = "PWL equality constraint"
        if base_index is not None:
            constraint_name = constraint_name + f" on {base_index}"
        if rhs is None:
            rhs = 1
        _ = self.set_constraint(lhs, rhs, "==", constraint_name)
        
        return None

    def build_leftmost_lambda_constraint(self, lambda_storage, binary_storage, base_index=None):
        """ Build the leftmost breakpoint ordering constraint """

        lambda_variables = self.get_lambda_variables_from_storage(lambda_storage, base_index)
        binary_variables = self.get_binary_variables_from_storage(binary_storage, base_index)
        lhs, rhs = lambda_variables[0], binary_variables[0]
        constraint_name = "Leftmost PWL constraint"
        if base_index is not None:
            constraint_name += f" on {base_index}"
        _ = self.set_constraint(lhs, rhs, "<=", constraint_name)
        
        return None

    def build_rightmost_lambda_constraint(self, lambda_storage, binary_storage, base_index=None):
        """ Build the leftmost breakpoint ordering constraint """

        lambda_variables = self.get_lambda_variables_from_storage(lambda_storage, base_index)
        binary_variables = self.get_binary_variables_from_storage(binary_storage, base_index)
        lhs, rhs = lambda_variables[-1], binary_variables[-1]
        constraint_name = "Rightmost PWL constraint"
        if base_index is not None:
            constraint_name += f" on {base_index}"
        _ = self.set_constraint(lhs, rhs, "<=", constraint_name)
        
        return None

    def build_middle_lambda_constraints(self, lambda_storage, binary_storage, base_index=None):
        """ Build the leftmost breakpoint ordering constraint """

        lambda_variables = self.get_lambda_variables_from_storage(lambda_storage, base_index)
        binary_variables = self.get_binary_variables_from_storage(binary_storage, base_index)
        for num in range(1, len(lambda_variables) - 1):
            lambda_variable = lambda_variables[num]
            left_binary_variable = binary_variables[num - 1]
            right_binary_variable = binary_variables[num]
            lhs, rhs = lambda_variable, left_binary_variable + right_binary_variable 
            constraint_name = f"Centre ({num}) PWL constraint"
            if base_index is not None:
                constraint_name += f" on {base_index}"
            _ = self.set_constraint(lhs, rhs, "<=", constraint_name)

        return None

    def build_sos_lambda_constraint(self, lambda_storage=None, base_index=None):
        """ Build an SOS(2) constraint for the lambda variables in the PWL model """

        variables = self.get_lambda_variables_from_storage(lambda_storage, base_index)
        constraint_name = "SOS(2) constraint for PWL model lambdas"
        if base_index is not None:
            constraint_name += f" for {base_index}"
        _ = self.set_sos2_constraint(variables, name=constraint_name)
        
        return None

    def build_delta_equality_constraint(self, storage_graph=None, base_index=None, rhs=None):
        """ Make sure the deltas form a linear combination """

        delta_variables = self.get_delta_variables_from_storage(storage_graph, base_index)
        breakpoints = self.get_horizontal_breakpoints()
        breakpoint = breakpoints[0]
        lhs = self.sum_items(delta_variables) + breakpoint
        constraint_name = "PWL equality constraint"
        if base_index is not None:
            constraint_name = constraint_name + f" on {base_index}"
        if rhs is None:
            rhs = 1
        _ = self.set_constraint(lhs, rhs, "==", constraint_name)
        
        return None
    
    def build_leftmost_delta_constraint(self, delta_storage, base_index=None):
        """ Build the leftmost breakpoint ordering constraint """

        delta_variables = self.get_delta_variables_from_storage(delta_storage, base_index)
        breakpoints = self.get_horizontal_breakpoints()
        lhs, rhs = delta_variables[0], breakpoints[1] - breakpoints[0] 
        constraint_name = "Leftmost PWL constraint"
        if base_index is not None:
            constraint_name += f" on {base_index}"
        _ = self.set_constraint(lhs, rhs, "<=", constraint_name)
        
        return None

    def build_middle_delta_constraints(self, delta_storage, binary_storage, base_index=None):
        """ Build the leftmost breakpoint ordering constraint """

        delta_variables = self.get_delta_variables_from_storage(delta_storage, base_index)
        binary_variables = self.get_binary_variables_from_storage(binary_storage, base_index)
        breakpoints = self.get_horizontal_breakpoints()
        for num in range(len(delta_variables) - 1):
            difference = breakpoints[num + 1] - breakpoints[num]
            delta_variable = delta_variables[num]
            binary_variable = binary_variables[num]
            lhs, rhs = delta_variable, binary_variable * (difference)
            constraint_name = f"Centre greater than ({num}) PWL constraint"
            if base_index is not None:
                constraint_name += f" on {base_index}"
            _ = self.set_constraint(lhs, rhs, ">=", constraint_name)
            
            difference = breakpoints[num + 2] - breakpoints[num + 1]
            delta_variable = delta_variables[num + 1]
            binary_variable = binary_variables[num]
            lhs, rhs = delta_variable, binary_variable * (difference)
            constraint_name = f"Centre less than ({num}) PWL constraint"
            if base_index is not None:
                constraint_name += f" on {base_index}"
            _ = self.set_constraint(lhs, rhs, "<=", constraint_name)
        
        return None
    
    def build_constraints_lambda(self, lambda_storage=None, binary_storage=None,
                                 base_index=None, equality_rhs=None, is_convex=False):
        """ Build all of the necessary constraints for the model """

        _ = self.build_lambda_convexity_constraint(lambda_storage, binary_storage, base_index)
        _ = self.build_lambda_equality_constraint(binary_storage, base_index, equality_rhs)
        if not is_convex:
            _ = self.build_leftmost_lambda_constraint(lambda_storage, binary_storage, base_index)
            _ = self.build_rightmost_lambda_constraint(lambda_storage, binary_storage, base_index)
            _ = self.build_middle_lambda_constraints(lambda_storage, binary_storage, base_index)

        return None

    def build_constraints_delta(self, delta_storage=None, binary_storage=None,
                                base_index=None, equality_rhs=None, is_convex=False):
        """ Build all of the necessary constraints for the model """

        _ = self.build_delta_equality_constraint(delta_storage, base_index, equality_rhs)
        if not is_convex:
            _ = self.build_leftmost_delta_constraint(delta_storage, base_index)
            _ = self.build_middle_delta_constraints(delta_storage, binary_storage, base_index)

        return None
    
    def build_vertical_lambda_combination_constraint(self, rhs_variable,
                                              lambda_storage=None, base_index=None):
        """ Build a constraint forcing the combination to equal the given variable """
        
        breakpoints = self.get_vertical_breakpoints()
        variables = self.get_lambda_variables_from_storage(lambda_storage, base_index)
        lhs = self.sum_items([breakpoint * variable
                              for (breakpoint, variable) in zip(breakpoints, variables)])
        constraint_name = "Vertical breakpoint equality constraint"
        if base_index is not None:
            constraint_name += f" on {base_index}"
        _ = self.set_constraint(lhs, rhs_variable, "==", constraint_name)
        
        return None

    def build_horizontal_lambda_combination_constraint(self, rhs_variable,
                                              lambda_storage=None, base_index=None):
        """ Build a constraint forcing the combination to equal the given variable """
        
        breakpoints = self.get_horizontal_breakpoints()
        variables = self.get_lambda_variables_from_storage(lambda_storage, base_index)
        lhs = self.sum_items([breakpoint * variable
                              for (breakpoint, variable) in zip(breakpoints, variables)])
        constraint_name = "Vertical breakpoint equality constraint"
        if base_index is not None:
            constraint_name += f" on {base_index}"
        _ = self.set_constraint(lhs, rhs_variable, "==", constraint_name)
        
        return None

    def build_horizontal_lambda_objective(self, lambda_storage=None, base_index=None):
        """ Build an objective that will model the PWL approximation """
        
        breakpoints = self.get_horizontal_breakpoints()
        variables = self.get_lambda_variables_from_storage(lambda_storage, base_index)
        summed_terms = self.sum_items([breakpoint * variable
                                       for (breakpoint, variable) in zip(breakpoints, variables)])
        self.set_objective(summed_terms)
        
        return None

    def build_vertical_lambda_objective(self, lambda_storage=None, base_index=None):
        """ Build an objective that will model the PWL approximation """
        
        breakpoints = self.get_vertical_breakpoints()
        variables = self.get_lambda_variables_from_storage(lambda_storage, base_index)
        summed_terms = self.sum_items([breakpoint * variable
                                       for (breakpoint, variable) in zip(breakpoints, variables)])
        self.set_objective(summed_terms)

        return None

    def build_vertical_delta_combination_constraint(self, rhs_variable,
                                                    delta_storage=None, base_index=None):
        """ Build a constraint forcing the combination to equal the given variable """

        verticals = np.diff(self.get_vertical_breakpoints())
        horizontals = np.diff(self.get_horizontal_breakpoints())
        breakpoints = [vertical / horizontal for (vertical, horizontal) in zip(verticals, horizontals)]
        offset = self.get_vertical_breakpoints()[0]
        variables = self.get_delta_variables_from_storage(delta_storage, base_index)
        summed_terms = self.sum_items([breakpoint * variable
                                       for (breakpoint, variable) in zip(breakpoints, variables)])
        lhs = summed_terms + offset
        constraint_name = "Vertical breakpoint equality constraint"
        if base_index is not None:
            constraint_name += f" on {base_index}"
        _ = self.set_constraint(lhs, rhs_variable, "==", constraint_name)
        
        return None

    def build_horizontal_delta_combination_constraint(self, rhs_variable,
                                                      delta_storage=None, base_index=None):
        """ Build a constraint forcing the combination to equal the given variable """
        
        breakpoints = self.get_horizontal_breakpoints()
        breakpoint = breakpoints[0]
        variables = self.get_delta_variables_from_storage(delta_storage, base_index)
        lhs = self.sum_items(variables) + breakpoint
        
        constraint_name = "Vertical breakpoint equality constraint"
        if base_index is not None:
            constraint_name += f" on {base_index}"
        _ = self.set_constraint(lhs, rhs_variable, "==", constraint_name)
        
        return None
    
    def build_horizontal_delta_objective(self, delta_storage=None, base_index=None):
        """ Build an objective that will model the PWL approximation """

        breakpoints = self.get_horizontal_breakpoints()
        breakpoint = breakpoints[0]
        variables = self.get_delta_variables_from_storage(delta_storage, base_index)
        summed_terms = self.sum_items(variables) + breakpoint
        self.set_objective(summed_terms)

        return None

    def build_vertical_delta_objective(self, delta_storage=None, base_index=None):
        """ Build an objective that will model the PWL approximation """
        
        verticals = np.diff(self.get_vertical_breakpoints())
        horizontals = np.diff(self.get_horizontal_breakpoints())
        breakpoints = [vertical / horizontal for (vertical, horizontal) in zip(verticals, horizontals)]
        offset = self.get_vertical_breakpoints()[0]
        variables = self.get_delta_variables_from_storage(delta_storage, base_index)
        summed_terms = self.sum_items([breakpoint * variable
                                       for (breakpoint, variable) in zip(breakpoints, variables)])
        self.set_objective(summed_terms + offset)

        return None


    def build_lambda_problem(self, lambda_storage=None, binary_storage=None, lambda_stem="h_",
                             binary_stem="w_", base_index=None, equality_rhs=None, rhs_variable=None):
        """ Build the variables and constraints for the given lambda PWL problem """

        self.build_variables_lambda(lambda_stem, binary_stem, binary_storage,
                                    lambda_storage, base_index)
        self.build_constraints_lambda(lambda_storage, binary_storage, base_index, equality_rhs)
        self.build_horizontal_lambda_combination_constraint(rhs_variable=rhs_variable)
        
        return None
    
    def build_delta_problem(self, delta_storage=None, binary_storage=None, delta_stem="d_",
                            binary_stem="w_", base_index=None, equality_rhs=None, rhs_variable=None):
        """ Build the variables and constraints for the given delta PWL problem """

        self.build_variables_delta(delta_stem, binary_stem, binary_storage,
                                     delta_storage, base_index)
        self.build_constraints_delta(delta_storage, binary_storage, base_index, equality_rhs)
        
        return None
