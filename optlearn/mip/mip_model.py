import time
import random

import mip as mp
import numpy as np
import xpress as xp
import networkx as nx

from itertools import product
from collections import OrderedDict
from pyscipopt import SCIP_EVENTTYPE

from optlearn import io_utils
from optlearn import graph_utils
from optlearn.mip import mip_utils
from optlearn.mip import constraints
from optlearn.mip import xpress
from optlearn.mip import coinor
from optlearn.mip import scip


_solver_modules = {
    "xpress": xpress,
    "coinor": coinor,
    "scip": scip,
    }


_solver_funcs = {
    "xpress": xpress._funcs,
    "coinor": coinor._funcs,
    "scip": scip._funcs,
}

_formulations = [
    "dantzig",
]


def approx_int(number, epsilon=0.01):
    """ Check if number is approximately integer """

    return abs(number - round(number)) < abs(epsilon)


def approx_ints(numbers, epsilon=0.01):
    """ Check if list of number is approximately integer """

    return all(approx_int(number, epsilon) for number in numbers)


class tspProblem():
    """ This is designed to find solutions of IP TSPs """

    def __init__(self, graph,
                 var_type="binary",
                 formulation="dantzig",
                 solver="scip",
                 verbose=False,
                 shuffle_columns=False,
                 perturb=False,
                 get_quick=False,
                 cut_strategy="small_if_possible"
    ):
        """ Setup problem """

        self.verbose = verbose
        self.perturb = perturb
        self.get_quick = get_quick
        self.shuffle_columns = shuffle_columns
        self.times_optimised = 0
        self.cut_strategy = cut_strategy
        self.initialise_variable_dict()
        self.initialise_vertices(graph)
        self.initialise_min_vertex(graph)
        self.initialise_solver(solver)
        self.set_var_args(var_type)
        self.check_symmetry(graph)
        self.set_formulation(formulation)
        self.initialise_problem()
        self.set_variables(graph)
        self.add_variables()
        self.set_objective(graph)
        self.set_constraints()

    def initialise_variable_dict(self):
        """ Setup a dictionary to put the variables in """

        self.variable_dict = OrderedDict()

    def initialise_vertices(self, graph):
        """ Grab a list of the vertices from the graph """
        
        self.vertices = graph_utils.get_vertices(graph)

    def initialise_min_vertex(self, graph):
        """ Store the minimum vertex value """
        
        self.min_vertex = np.min(graph_utils.get_vertices(graph))

        if self.verbose:
            print("Minimum vertex: {}\n".format(self.min_vertex))
        
    def initialise_solver(self, solver):
        """ Get the setup function dictionary for the given solver """

        self._solver = solver
        self._funcs = _solver_funcs[solver]

    def set_formulation(self, formulation):
        """ Set the formulation type, needed to set the problem up """

        self.formulation = formulation

        if self.verbose:
            print("Formulation: {}\n".format(self.formulation))
        
    def set_var_args(self, var_type):
        """ Set the variable type, default to continuous """

        var_types = _solver_modules[self._solver]._var_types 
        self._var_args = var_types.get(var_type) or var_types["continuous"]

    def check_symmetry(self, graph):
        """ Check the symmetry of the given graph """

        self._is_symmetric = mip_utils.is_graph(graph, verbose=self.verbose)
        self._is_asymmetric = mip_utils.is_digraph(graph, verbose=self.verbose)
        
    def initialise_problem(self):
        """ Setup the problem and add as an attribute """

        print("Building problem!")
        self.problem = self._funcs["build_problem"]()

    def create_variable(self, edge, prefix="x", var_args=None):
        """ Grab the variable create function, create a single variable and return it """
        
        var_args = var_args or self.var_args
        return self._funcs["create_variable"](self.problem, edge, prefix, var_args)
    
    def set_variable(self, edge, prefix="x", var_args=None):
        """ Create an edge variable and set it """

        var_args = var_args or self.var_args
        variable = self.create_variable(edge, prefix, var_args)
        self.variable_dict[variable.name] = variable

    def set_edge_variables(self, graph):
        """ Create and set all TSP edge variables """

        print("Setting variables!")
        if self.shuffle_columns:
            for edge in random.sample(graph.edges, len(graph.edges)):
                self.set_variable(edge, prefix="x", var_args=self._var_args)
        else:
            for edge in graph.edges:
                self.set_variable(edge, prefix="x", var_args=self._var_args)

    def set_variables(self, graph):
        """ Create and set all variables for the formulation """

        if self.formulation == "dantzig":
            self.set_edge_variables(graph)
            return None

    def add_variables(self):
        """ Add the created variables to the problem """

        self._funcs["add_variables"](self.problem, self.variable_dict.values())

    def set_objective(self, graph):
        """ Set the objective function """

        print("Setting objective!")
        if self.formulation == "dantzig":
            self._funcs["edge_objective"](self.problem,
                                          self.variable_dict,
                                          graph, self.perturb)
            return None
        print("No objective function defined!")

    def get_variable_tuple(self, string):
        """ Get the (vertex_a, vertex_b) integer tuple from a variable name """

        return mip_utils.get_variable_tuple(string)
    
    def get_variable(self, edge, prefix="x"):
        """ Get a single variable by specifying its edge and prefix """

        return mip_utils.get_variable(self.variable_dict, edge, prefix)
        
    def get_outward_variables_sum(self, vertex, prefix="x"):
        """ Get the outward variables for a vertex and sum them """

        vars = mip_utils.get_outward_variables(self.variable_dict, vertex, prefix)
        return self._funcs["sum"](vars)

    def get_variables_sum(self, vertex, prefix="x"):
        """ Get all variables that have a given vertex and sum them """

        if self.get_quick:
            vars = mip_utils.get_variables_quick(self.variable_dict, self.vertices, vertex, prefix)
        else:
            vars = mip_utils.get_variables(self.variable_dict, vertex, prefix)
        return self._funcs["sum"](vars)
        
    def get_inward_variables_sum(self, vertex, prefix="x"):
        """ Get the inward variables for a vertex and sum them """

        vars = mip_utils.get_inward_variables(self.variable_dict, vertex, prefix)
        return self._funcs["sum"](vars)

    def set_constraint(self, lhs, rhs, operator):
        """ Set a constraint for the problem """

        self._funcs["set_constraint"](self.problem, lhs, rhs, operator)

    def set_degree_constraint(self, vertex):
        """ Define the degree constraint for a given vertex """

        if self._is_symmetric:
            lhs, rhs = self.get_variables_sum(vertex), 2
            if self.verbose:
                print("Degree constraint variables:\n {}\n".format(lhs))
            self.set_constraint(lhs, rhs, "==")
        if self._is_asymmetric:
            lhs, rhs = self.get_inward_variables_sum(vertex), 1
            if self.verbose:
                print("Inward degree constraint variables:\n {}\n".format(lhs))
            self.set_constraint(lhs, rhs, "==")
            lhs, rhs = self.get_outward_variables_sum(vertex), 1
            if self.verbose:
                print("Outward degree constraint variables:\n {}\n".format(lhs))
            self.set_constraint(lhs, rhs, "==")
                
    def set_degree_constraints(self):
        """ Define the degree constraints for all vertices """

        for vertex in self.vertices:
            self.set_degree_constraint(vertex)

    def set_dantzig_constraints(self):
        """ Define the constraints neccessary for a DFJ formulation """

        if self.verbose:
            print("Setting degree constraints... \n")
        self.set_degree_constraints()
        if self.verbose:
            print("Degree constraints set... \n")

    def set_constraints(self):
        """ Define the outward and inward constraints for all vertices """

        print("Setting constraints!")
        if self.formulation == "dantzig":
            self.set_dantzig_constraints()
            print("Constraints set!")
            return None
        print("No constraints set!")
        
    def perform_relaxation(self):
        """ Perform a linear relaxation on the current problem """

        self._funcs["perform_relaxation"](self.problem)

    def get_varnames(self):
        """ Get all variable names associated with the problem """

        return self._funcs["get_varnames"](self.problem, self.variable_dict)

    def get_variables(self, prefix="x"):
        """ Get all of the variables associated with the problem """

        keys = [item for item in self.variable_dict.keys() if prefix in item]
        return [self.variable_dict[key] for key in keys]
        
    def get_varvals(self):
        """ Get all variable values associated with the problem """

        return self._funcs["get_varvals"](self.problem, self.variable_dict)

    def get_redcosts(self):
        """ Get all reduced costs associated with the problem """

        return self._funcs["get_redcosts"](self.problem, self.variable_dict)
    
    def get_varvals_by_name(self, variable_names):
        """ Get all variable values, querying them in the order given """

        return self._funcs["get_varvals_by_name"](self.problem,
                                                  self.variable_dict,
                                                  variable_names,
        )
            
    def get_nonzero_varnames(self, prefix="x"):
        """ Get all nonzero variable names from the LP solution """

        values, names = self.get_varvals(), self.get_varnames()
        return [name for (value, name) in zip(values, names) if (value > 0 and prefix in name)]

    def get_solution_varnames(self, prefix="x"):
        """ Get all nonzero variable names from the LP solution """

        values, names = self.get_solution(), self.get_varnames()
        return [name for (value, name) in zip(values, names) if (value == 1 and prefix in name)]

    def get_unit_varnames(self, prefix="x"):
        """ Get all nonzero variable names from the LP solution """

        values, names = self.get_varvals(), self.get_varnames()
        return [name for (value, name) in zip(values, names) if (value == 1 and prefix in name)]

    def get_nonzero_varvals(self, prefix="x"):
        """ Get all nonzero variable values from the LP solution """

        values, names = self.get_varvals(), self.get_varnames()
        return [value for (value, name) in zip(values, names) if (value > 0 and prefix in name)]

    def get_unit_varvals(self, prefix="x"):
        """ Get all unit variable values from the LP solution """

        values, names = self.get_varvals(), self.get_varnames()
        return [value for (value, name) in zip(values, names) if (value >= 0.99 and prefix in name)]

    def get_solution(self):
        """ Get the current solution """

        return self._funcs["get_solution"](self.problem, self.variable_dict)

    def check_solution(self):
        """ Check if a current solution exists """

        try:
            self.get_solution()
            return True
        except:
            return False
            
    def get_edges(self, prefix="x"):
        """ Get the edges of the problem graph """

        varnames = [name for name in self.get_varnames() if prefix in name]
        return [mip_utils.get_edge_from_varname(varname) for varname in varnames]
    
    def get_nonzero_edges(self, prefix="x"):
        """ Get the nonzero edge tuples from the LP solution """

        varnames = self.get_nonzero_varnames(prefix=prefix)
        return [mip_utils.get_edge_from_varname(varname) for varname in varnames]

    def get_unit_edges(self, prefix="x"):
        """ Get the nonzero edge tuples from the LP solution """

        varnames = self.get_unit_varnames(prefix=prefix)
        return [mip_utils.get_edge_from_varname(varname) for varname in varnames]

    def get_solution_edges(self, prefix="x"):
        """ Get the nonzero edge tuples from the LP solution """

        varnames = self.get_solution_varnames(prefix=prefix)
        return [mip_utils.get_edge_from_varname(varname) for varname in varnames]
    
    def check_integral_solution(self):
        """ Check if the solution is integral """

        return mip_utils.all_values_integer(self.get_nonzero_varvals())
    
    def initialise_connection_graph(self):
        """ Set the mincut graph """
                    
        return  constraints.initialise_connection_graph(self._is_symmetric)
                                            
    def get_objective_value(self):
        """ Get the objective value of the current solution """

        return self._funcs["get_objective_value"](self.problem)

    def optimise_scip(self, max_nodes=1e10, max_rounds=1e10):
        """ Solve the problem using SCIP """

        if self.times_optimised < 1:
            contraint_handler = constraints.scip_constraint_handler(self,
                                                                    max_rounds=max_rounds)
            self.problem.includeConshdlr(contraint_handler,
                                         "TSP", "Subtour Elimination",
                                         sepapriority = -1, enfopriority = -1,
                                         chckpriority = -1, sepafreq = -1,
                                         propfreq = -1, eagerfreq = -1,
                                         maxprerounds = 0, delaysepa = False,
                                         delayprop = False, needscons = False,
            )
            event_handler = constraints.scip_event_handler(max_nodes=max_nodes)
            self.problem.includeEventhdlr(event_handler, "Event", "Event Handler")
        self.problem.optimize()

    def optimise_xpress(self, max_nodes=1e10, max_rounds=1e10):
        """ Solve the problem using Xpress """

        if self.times_optimised < 1:
            constrainer = constraints.xpress_constraint_callback(self,
                                                                 max_rounds=max_rounds)
            constrainer.add_cut_callback()
            constrainer.add_tour_callback()
        self.problem.mipoptimize()

    def optimise(self, max_nodes=1e10, max_rounds=1e10):
        """ Solve the problem """

        print("Solving problem!")
        
        self.counter = 0
        self.solutions = []
        
        if self._solver == "xpress":
            self.optimise_xpress(max_nodes=max_nodes, max_rounds=max_rounds)
        if self._solver == "scip":
            self.optimise_scip(max_nodes=max_nodes, max_rounds=max_rounds)
        if self._solver == "coin":
            raise NotImplementedError("Not implemented for COIN yet!")

        self.times_optimised += 1
