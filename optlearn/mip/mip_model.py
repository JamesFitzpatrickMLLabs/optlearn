import time

import mip as mp
import numpy as np
import xpress as xp
import networkx as nx

from itertools import product
from collections import OrderedDict

from optlearn import io_utils
from optlearn import graph_utils
from optlearn.mip import mip_utils
from optlearn.mip import constraints
from optlearn.mip import xpress
from optlearn.mip import coinor


_solver_modules = {
    "xpress": xpress,
    "coinor": coinor,
    }

_solver_funcs = {
    "xpress": xpress._funcs,
    "coinor": coinor._funcs,
}

_formulations = [
    "dantzig",
    "miller",
    "commodity",
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
                 var_type="continuous",
                 formulation="dantzig",
                 solver="xpress",
                 verbose=False,
    ):
        """ Setup problem """

        self.verbose = verbose
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

        if self.formulation == "dantzig":
            self._funcs["edge_objective"](self.problem, self.variable_dict, graph)
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

        if self.formulation == "dantzig":
            self.set_dantzig_constraints()
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
        
    def initialise_mincut_graph(self):
        """ Set the mincut graph """
                    
        self._mincut_graph = constraints.initialise_mincut_graph(self._is_symmetric)
            
    def set_mincut_graph(self):
        """ Build a graph structure for mincut """
        
        values = self.get_varvals()
        names = self.get_varnames()
        edges = [mip_utils.get_edge_from_varname(name) for name in names]

        self.initialise_mincut_graph()
        self._mincut_graph = constraints.set_mincut_graph(self._mincut_graph, edges, values)
        
    def set_mincut_constraint(self, vertex_a, vertex_b):
        """ Build a mincut constraint for a given pair of vertices """

        variables = constraints.get_mincut_variables(self._mincut_graph, vertex_a, vertex_b,
                                                     self.variable_dict, self._is_symmetric)
        if variables is None:
            return None

        varnames = [variable.name for variable in variables]
        edges = [mip_utils.get_edge_from_varname(varname) for varname in varnames]
        uniques = np.unique(edges)
        
        if variables is not None:
            self._funcs["add_mincut"](self.problem, variables, self.variable_dict)
        
    def set_mincut_constraints(self):
        """ Build mincut constraints """

        self.set_mincut_graph()
        
        for edge in self.get_edges(prefix="x"):
            self.set_mincut_constraint(*edge)

    def check_tour_other(self):
        """ Checks if the current solution gives a valid tour """

        varvals = self.get_varvals()
        edges = self.get_edges()
        vals = [1 if varval > 0 else 0 for varval in varvals]
        edges = [edge for (edge, varval) in zip(edges, varvals) if varval == 1]

        graph = constraints.initialise_mincut_graph(self._is_symmetric)
        graph.add_edges_from(edges)
        
        cycle = graph_utils.check_cycle(graph)
        return len(cycle) == len(self.vertices)

    def default_solver_args(self, kwargs):
        """ Set default arguments if not given """

        self.solution_dict = {}
        self.solution_counter = 0
        self.max_nodes = kwargs.get("max_nodes") or 99999999
        
    def solve(self, kwargs={}):
        """ Solve to optimality if possible """

        self.default_solver_args(kwargs)
        
        if self.formulation == "dantzig":
            if self._solver == "coinor":
                print("Not implemented...")
            if self._solver == "xpress":

                def check_nodes(problem, graph, isheuristic, cutoff):
                    self.solution_dict[self.solution_counter] = {
                        "solution": self.get_varvals(),
                        }
                    
                    self.solution_counter += 1
                    if self.solution_counter >= self.max_nodes:
                        self.problem.interrupt(1)
                    return (0, None)
                
                def check_tour(problem, graph, isheuristic, cutoff):
                    ser = np.logical_not(self.check_tour_other())
                    return (bool(ser), None)

                def add_cuts(problem, graph):
                    self.set_mincut_constraints()
                    return 0

                self.problem.addcbpreintsol(check_nodes, None, 1)
                self.problem.addcbpreintsol(check_tour, None, 1)
                self.problem.addcboptnode(add_cuts, None, 1)

                self.problem.solve()
                        
    def get_objective_value(self):
        """ Get the objective value of the current solution """

        return self._funcs["get_objective_value"](self.problem)
