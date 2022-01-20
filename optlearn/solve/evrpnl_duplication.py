import os
import sys
import time

import numpy as np

from optlearn.graph import graph_utils
from optlearn.graph import process_utils

from optlearn.feature.vrp import feature_utils

from optlearn.mip.routing import evrpnl_problem_builder


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

        
class duplicationSolver():
    def __init__(self, problem_graph, iteration_limit=5, time_limit=3600, silent_solve=True):

        self.problem_graph = problem_graph
        self.iteration_limit = iteration_limit
        self.time_limit = time_limit
        self.silent_solve = silent_solve
        self.pruners = {}
        self.releasing_strategies = {}

        self.working_graph = self.set_working_graph()

    def wake_up(self):
        """ Wake up the prints! """

        self.silent_solve = False

        return None

    def set_customer_customer_pruner(self, pruner, strategy="auto"):
        """ Set pruner for the solver """

        self.pruners["customer_customer"] = pruner
        self.releasing_strategies["customer_customer"] = strategy

        return None

    def set_unreachability_pruner(self, pruner, strategy="auto"):
        """ Set pruner for the solver """

        self.pruners["unreachability"] = pruner
        self.releasing_strategies["unreachability"] = strategy

        return None
    
    def set_iteration_limit(self, iteration_limit):
        """ Set the iteration limit for the solver """

        self.iteration_limit = iteration_limit

        return None

    def set_time_limit(self, time_limit):
        """ Set the time limit for the solver """

        self.time_limit = time_limit

        return None

    def set_working_graph(self):
        """ Set the working graph """

        self._set_prime_stations()
        self.working_graph = graph_utils.clone_graph(self.problem_graph)

        return None

    def get_working_graph(self):
        """ Get the working graph """

        if not hasattr(self, "working_graph"):
            raise ValueError("Must have a working graph!")
        elif self.working_graph is None:
            raise ValueError("Working graph must be of graph type!")
        else:
            working_graph = self.working_graph

        return working_graph

    def build_evrpnl_problem(self):
        """ Build an EVRPNL problem to solve using the stored graph """

        self.problem_builder = evrpnl_problem_builder.evrpnlProblemBuilder(
            solver_package="xpress",
            is_directed=True,
            pwl_model="delta",
        )
        self.problem_builder.build_arc_problem(self.get_working_graph())
        self._set_solve_time_limit()

        return None

    def _get_prime_stations(self):
        """ Get the prime stations, so we know which ones to copy each time """

        prime_stations = self.problem_graph.graph["node_types"]["station"]

        return prime_stations

    def _set_prime_stations(self):
        """ Set the prime stations for the stored problem graph """

        self._prime_stations = self._get_prime_stations()

        return None

    def _build_duplication_dict(self):
        """ Build the duplication dict for the current interation """

        duplication_dict = {prime_station: self._iteration
                            for prime_station in self._prime_stations}
        
        return duplication_dict

    def _duplicate_stations(self):
        """ Duplicate all the prime stations uniformly in the working graph """

        duplication_dict = self._build_duplication_dict()
        self.set_working_graph()
        self.working_graph = process_utils.duplicate_stations_nonuniform(
            self.working_graph,
            duplication_dict,
        )

        return None

    def solve_current_problem(self):
        """ Solve the current problem in the problem builder """

        self.problem_builder.problem.solve()

        return None

    def solve_current_relaxed_problem(self):
        """ Solve the current problem in the problem builder """

        self.problem_builder.problem.lpoptimize()

        return None

    def _check_if_feasible(self):
        """ Check if the solved problem is feasible """ 

        if self.problem_builder.problem.getProbStatus() == 6:
            is_feasible = True
        elif self.problem_builder.problem.getProbStatus() == 2:
            is_feasible = True
        else:
            is_feasible = False
        
        return is_feasible

    def _stop_due_to_optimality(self):
        """ Check if we need to stop from reaching the time limit """

        if len(self._was_feasible) > 1:
            if self._was_feasible[-1] and self._was_feasible[-2]:
                if round(self._objective_values[-1], 5) == round(self._objective_values[-2], 5): 
                    self._stop = True

        return None

    def _stop_due_to_iterations(self):
        """ Check if we need to stop from reaching the iteration limit """

        if self._iteration == self.iteration_limit:
            self._stop = True

        return None

    def _start_timer(self):
        """ Start the wallclock timer """

        self._start_time = time.time()

        return None

    def _check_time(self):
        """ Check how long the solving has been going on for """

        solve_time = time.time() - self._start_time

        return solve_time

    def _calculate_remaining_time(self):
        """ Check the remaining time we have for solving """

        remaining_time = self.time_limit - self._check_time()

        return remaining_time
        
    def _stop_due_to_time(self):
        """ Check if we need to stop from reaching the time limit """

        if self._check_time() >= self.time_limit:
            self._stop = True

        return None

    def _set_solve_time_limit(self):
        """ Set the time limit for the solver """

        int_remaining_time = int(self._calculate_remaining_time()) - 1
        self.problem_builder.problem.controls.maxtime = int_remaining_time

        return None

    def _check_stop_conditions(self):
        """ Check if any of the stop conditions are met """

        stop_conditions = [
            self._stop_due_to_optimality(),
            self._stop_due_to_iterations(),
            self._stop_due_to_time(),
        ]

        return self._stop

    def _calculate_solve_time(self):
        """ Calculate the total solve time """

        solve_time = self._check_time()
        self._solve_time = solve_time

        return solve_time

    def _reset_stop_conditions(self):
        """ Reset the stop conditions for the solver """

        self._was_feasible = []
        self._stop = False

    def _reset_solution_parameters(self):
        """ Reset the solution parameters """

        self._start_timer()
        self._objective_values = []
        self._was_feasible = []
        self._solve_time = None
        self._solution = None
        self._iteration = 0

        return None

    def _increment_iteration(self):
        """ Increment the current interation """

        self._iteration += 1

        return None

    def _print_summary(self):
        """ Print current iteration summary """

        print(f"Iteration {self._iteration} of {self.iteration_limit}")
        if self._was_feasible[-1]:
            if len(self._was_feasible) > 1 and self._was_feasible[-2]:
                print(f"Optimal solution found")
            elif len(self._was_feasible) > 1 and not self._was_feasible[-2]:
                print(f"Feasible solution found")
        else:
            print(f"No feasible solution found!")
        if self._stop:
            print(f"Stopping condition reached, terminating!")

        return None

    def _get_solution_variables_names(self):
        """ Get the variable names for the variables in the stored solution """

        variables = [self.problem_builder.problem.getVariable(num)
                     for num in range(len(self._solution))]
        variable_names = [variable.name for variable in variables]

        return variable_names

    def _set_previous_solution_variables_names(self):
        """ Get the variable names for the variables in the stored solution """

        self._variable_names = self._get_solution_variables_names()

        return None

    def _get_previous_solution_variables_names(self):
        """ Get the stored variable names for the stored solution """

        variable_names = self._variable_names

        return variable_names

    def _get_current_problem_variables(self):
        """ Get the current variables with the given variable names """

        variable_names = self._get_previous_solution_variables_names()
        new_indices = [self.problem_builder.problem.getIndexFromName(2, variable_name)
                       for variable_name in variable_names]
        new_variables = [self.problem_builder.problem.getVariable(index)
                         for index in new_indices]

        return new_variables

    def _get_previous_solution_values(self):
        """ Get the values for the stored solution """

        solution_values = self._solution

        return solution_values

    def _set_warm_start_solution(self):
        """ Set the warm start solution for the current problem builder """

        variable_objects = self._get_current_problem_variables()
        solution_values = self._get_previous_solution_values()

        self.problem_builder.problem.addmipsol(solution_values, variable_objects, "warm-start")

        return None
        
    def _store_feasibility(self):
        """ Store the feasibility of the current problem """

        is_feasible = self._check_if_feasible()
        self._was_feasible.append(is_feasible)

        return None

    def _store_objective_value(self):
        """ Store the objective, if the problem was feasible """

        is_feasible = self._check_if_feasible()
        if is_feasible:
            self._objective_values.append(self.get_objective_value())
        else:
            self._objective_values.append(9999999999999999999999)

        return None

    def _store_current_solution(self):
        """ Store the solution (if there is one) to the current problem """

        self._store_feasibility()
        self._store_objective_value()
        self._solution = self.problem_builder.problem.getSolution()
        self._set_previous_solution_variables_names()
        
        return None

    def _warm_start_from_previous_solution(self):
        """ Using the previous solution, warm start the current problem """

        if self._solution is not None:
            self._set_warm_start_solution()
    
        return None

    def _regress_customer_customer_edges(self, reachability_radius):
        """ Regress the customer to customer edges with the pruner """

        customer_edges = self.get_customer_customer_arcs()
        predictions = self.pruners["customer-customer"](
            self.problem_graph,
            customer_edges,
            reachability_radius,
        )

        return predictions

    def _unreachability_pruner(self, working_graph, edges, reachability_radius):
        """ The pruner that fixes all unreachable edges in the graph """

        weights = graph_utils.get_edges_weights(working_graph, edges)
        prediction_dict = {edge: 1 if weight > reachability_radius else 0
                           for (edge, weight) in zip(edges, weights)}
        
        return prediction_dict

    def _set_unreachability_pruner(self):
        """ Prune any edges that are too long to be travelled """

        self._set_unreachability_pruner(self._unreachability_pruner)

        return None

    def _get_pruning_predictions(self, pruner, reachability_radius):
        """ Get the pruning predictions on the original graph """

        prediction_dict = pruner.classify_edges(self.problem_graph, reachability_radius)

        return prediction_dict

    def _get_travel_variable_from_arc(self, arc):
        """ Get the current travel variable from the given arc """

        variable_name = f"x_{arc[0]},{arc[1]}"
        variable_index = self.problem_builder.problem.getIndexFromName(2, variable_name)
        variable = self.problem_builder.problem.getVariable(variable_index)

        return variable

    def _get_energy_variable_from_arc(self, arc):
        """ Get the current travel variable from the given arc """

        variable_name = f"y_{arc[0]},{arc[1]}"
        variable_index = self.problem_builder.problem.getIndexFromName(2, variable_name)
        variable = self.problem_builder.problem.getVariable(variable_index)

        return variable

    def _get_time_variable_from_arc(self, arc):
        """ Get the current travel variable from the given arc """

        variable_name = f"t_{arc[0]},{arc[1]}"
        variable_index = self.problem_builder.problem.getIndexFromName(2, variable_name)
        variable = self.problem_builder.problem.getVariable(variable_index)

        return variable
        
    def _translate_to_arc_keys_to_variable_keys(self, prediction_dict):
        """ Translate the arc keys to the variable keys """

        translated_prediction_dict = {}
        for key in prediction_dict.keys():
            travel_variable = self._get_travel_variable_from_arc(key)
            energy_variable = self._get_energy_variable_from_arc(key)
            time_variable = self._get_time_variable_from_arc(key)
            translated_prediction_dict[travel_variable] = prediction_dict[key]
            translated_prediction_dict[energy_variable] = prediction_dict[key]
            translated_prediction_dict[time_variable] = prediction_dict[key]

        return translated_prediction_dict

    def _fix_pruning_predictions(self, translated_prediction_dict, threshold=0.5):
        """ Fix the arcs of the problem according to given predictions and threshold """

        pruning_count = 0
        for (variable, prediction) in translated_prediction_dict.items():
            print(prediction)
            if prediction <= threshold:
                self.problem_builder.fix_binary_variable(variable)
                
        return None

    def _release_pruning_predictions(self, translated_prediction_dict, threshold=0.5):
        """ Release the arcs of the problem according to given predictions and threshold """

        for (variable, prediction) in translated_prediction_dict.items():
            if prediction >= threshold:
                self.problem_builder.release_binary_variable(variable)

        return None

    def _exponential_releasing_strategy(self):

        pass
    
    def _customer_customer_prune(self, threshold, reachability_radius):
        """ Perform a customer-customer arc pruning, with the given threshold """

        customer_customer_pruner = self.pruners.get("customer_customer")
        if customer_customer_pruner is not None:
            self._customer_customer_predictions = self._get_pruning_predictions(
                customer_customer_pruner,
                reachability_radius
            )
            translated_prediction_dict = self._translate_to_arc_keys_to_variable_keys(
                self._customer_customer_predictions
            )
            self._fix_pruning_predictions(translated_prediction_dict, threshold)

        return None

    def _unreachability_prune(self):
        """ Perform unreachability arc pruning """

        unreachability_pruner = self.pruners.get("unreachability")
        if unreachability_pruner is not None:
            self._unreachability_predictions = self._get_pruning_predictions(
                unreachability_pruner
            )
            translated_prediction_dict = self._translate_to_arc_keys_to_variable_keys(
                self._unreachability_predictions
            )
            self._fix_pruning_predictions(translated_prediction_dict, threshold)

        return None
    
    def solve_problem(self, reachability_radius=128, with_prune=False, threshold=0.5):
        """ Solve the problem from the beginning """

        self._reset_stop_conditions()
        self._reset_solution_parameters()
        
        while self._stop is False:
            self._duplicate_stations()
            
            self.build_evrpnl_problem()
            if with_prune:
                self._unreachability_prune()
                self._customer_customer_prune(threshold, reachability_radius)
            self._warm_start_from_previous_solution()
            if self.silent_solve:
                with HiddenPrints():
                    self.solve_current_problem()
            else:
                self.solve_current_problem()

            self._store_current_solution()
            self._increment_iteration()
            self._check_stop_conditions()
            self._print_summary()
        self._calculate_solve_time()

    def lpsolve_problem(self, reachability_radius=128, with_prune=False, threshold=0.5):
        """ Solve the problem from the beginning """

        self._reset_stop_conditions()
        self._reset_solution_parameters()
        
        while self._stop is False:
            self._duplicate_stations()
            
            self.build_evrpnl_problem()
            if with_prune:
                self._unreachability_prune()
                self._customer_customer_prune(threshold, reachability_radius)
            self._warm_start_from_previous_solution()
            if self.silent_solve:
                with HiddenPrints():
                    self.solve_current_relaxed_problem()
            else:
                self.solve_current_relaxed_problem()

            self._store_current_solution()
            self._increment_iteration()
            self._check_stop_conditions()
            self._print_summary()
        self._calculate_solve_time()

    def get_travel_solution_dict(self):
        """ Get a solution dictionary for the travel edges """

        solution = self.problem_builder.problem.getSolution()
        travel_edges = self.problem_builder.get_all_travel_edges()
        solution_dict = {edge: value for (edge, value) in zip(travel_edges, solution)}

        return solution_dict

    def get_primed_travel_solution_dict(self, aggregator=np.mean):
        """ Get the travel solution dict, but with the edges in their prime form """

        stations = feature_utils.get_stations(self.working_graph)
        prime_solution_dict = {}
        solution_dict = self.get_travel_solution_dict()
        for edge in solution_dict.keys():
            value = solution_dict[edge]
            if edge[0] in stations:
                if self.working_graph.nodes[edge[0]].get("prime_node") is not None:
                    first_node = self.working_graph.nodes[edge[0]].get("prime_node")
                else:
                    first_node = edge[0]
            else:
                first_node = edge[0]
            if edge[1] in stations:
                if self.working_graph.nodes[edge[1]].get("prime_node") is not None:
                    second_node = self.working_graph.nodes[edge[1]].get("prime_node")
                else: second_node = edge[1]
            else:
                second_node = edge[1]
            if (first_node, second_node) not in prime_solution_dict.keys():
                prime_solution_dict[(first_node, second_node)] = [value]
            else:
                prime_solution_dict[(first_node, second_node)].append(value)
        for edge in prime_solution_dict.keys():
            prime_solution_dict[edge] = aggregator(prime_solution_dict[edge])

        return prime_solution_dict

    def get_objective_value(self):
        """ Get the incumbent objective value of the current problem """

        objective_value = self.problem_builder.get_objective_value()

        return objective_value

    def get_iterations(self):
        """ Get the number of iterations for the solve """

        iterations = self._iteration

        return iterations

    def get_solve_time(self):
        """ Get the stored latest solve time """

        solve_time = self._solve_time

        return solve_time
