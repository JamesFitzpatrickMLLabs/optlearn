import numpy as np

from itertools import product
from networkx import minimum_cut, DiGraph
from mip import Model, xsum, BINARY, CONTINUOUS, OptimizationStatus, CutType

from optlearn import io_utils
from optlearn import graph_utils


_var_types = {
    "binary": BINARY,
    "continuous": CONTINUOUS,
}

_formulations = ["dfj", "mil", "scf"]


class smallLinearTSP():
    """ This is designed to find relaxed solutions of very small TSPs """

    def __init__(self, digraph, var_type="continuous", formulation="dfj"):
        """ Setup problem """
        self.set_var_type(var_type)
        self.set_formulation(formulation)
        self.initialise_model()
        self.initialise_variable_dict()
        self.initialise_vertices(digraph)
        self.initialise_min_vertex(digraph)
        self.set_variables(digraph)
        self.set_objective(digraph)
        self.set_constraints()

    def set_formulation(self, formulation):
        """ Set the formulation type, needed to set the problem up """

        self.formulation = formulation

    def set_var_type(self, var_type):
        """ Set the variable type """

        self._var_type = _var_types.get(var_type) or _var_types["continuous"]  

    def initialise_model(self):
        """ Setup the model and add as an attribute """

        self.model = Model()

    def initialise_variable_dict(self):
        """ Setup a dictionary to put the variables in """

        self.variable_dict = {}

    def initialise_vertices(self, graph):
        """ Grab a list of the vertices from the graph """

        self.vertices = graph_utils.get_vertices(graph)

    def initialise_min_vertex(self, graph):
        """ Store the minimum vertex value """

        self.min_vertex = np.min(graph_utils.get_vertices(graph))

    def name_variable(self, edge, prefix="x"):
        """ Create a variable name """

        return "{}_{},{}".format(prefix, *edge)

    def create_variable(self, graph, edge, prefix="x", var_type=None):
        """ Create a single variable and return it """

        var_type = var_type or self._var_type
        name = self.name_variable(edge, prefix=prefix)
        return self.model.add_var(name, var_type=var_type)

    def set_variable(self, graph, edge, prefix="x", var_type=None):
        """ Create an edge variable and set it """

        variable = self.create_variable(graph, edge, prefix=prefix, var_type=var_type)
        name = self.name_variable(edge, prefix=prefix)
        self.variable_dict[name] = variable

    def set_edge_variables(self, graph):
        """ Create and set all DFJ  edge variables """

        for edge in graph.edges:
            self.set_variable(graph, edge, prefix="x", var_type=self._var_type)

    def set_flow_variables(self, graph):
        """ Create and set flow variables for Miller's formulation """

        for vertex in graph.nodes:
            self.set_variable(graph, (vertex, "_"), prefix="u", var_type=_var_types["continuous"])

    def set_commodity_variables(self, graph):
        """ Create and set all edge variables """

        for edge in graph.edges:
            self.set_variable(graph, edge, prefix="y", var_type=_var_types["continuous"])

    def set_variables(self, graph):
        """ Create and set all variables """

        if self.formulation == "dfj":
            self.set_edge_variables(graph)
            return None
        if self.formulation == "mil":
            self.set_edge_variables(graph)
            self.set_flow_variables(graph)
            return None
        if self.formulation == "scf":
            self.set_edge_variables(graph)
            self.set_commodity_variables(graph)
            return None

    def define_objective_term(self, graph, edge, prefix="x"):
        """ Define a single term in the objective """

        weight = graph_utils.get_edge_weight(graph, *edge)
        name = self.name_variable(edge, prefix=prefix)
        return self.variable_dict[name] * weight

    def define_dfj_objective_terms(self, graph):
        """ Define all objective terms  """

        return [self.define_objective_term(graph, edge, prefix="x") for edge in graph.edges]

    def define_scf_objective_terms(self, graph):
        """ Define all SCF objective terms  """

        tour_terms = [self.define_objective_term(graph, edge, prefix="x") for
                edge in graph.edges if self.min_vertex not in edge]
        commodity_terms_a = [self.define_objective_term(graph, edge, prefix="y") for
                edge in graph.edges if (edge[0] == self.min_vertex and edge[1] != self.min_vertex)]
        commodity_terms_b = [self.define_objective_term(graph, edge, prefix="y") for
                edge in graph.edges if (edge[1] == self.min_vertex and edge[0] != self.min_vertex)]
        commodity_terms_b = [item / len(self.vertices) for item in commodity_terms_b]
        return  tour_terms + commodity_terms_a + commodity_terms_b

    def set_dfj_objective(self, graph):
        """ Set the DFJ objective function """

        self.model.objective = xsum(self.define_dfj_objective_terms(graph))

    def set_scf_objective(self, graph):
        """ Set the DFJ objective function """

        self.model.objective = xsum(self.define_scf_objective_terms(graph))
        # self.model.objective = xsum(self.define_dfj_objective_terms(graph))
        print("OBJECTIVE IMPLEMENTATION NOT COMPLETED!!")

    def set_objective(self, graph):
        """ Set the objective function """

        if self.formulation == "dfj":
            self.set_dfj_objective(graph)
            return None
        if self.formulation == "mil":
            self.set_dfj_objective(graph)
            return None
        if self.formulation == "scf":
            self.set_scf_objective(graph)
            return None
        print("No objective function defined!")

    def get_variable_tuple(self, string):
        """ Get the (vertex_a, vertex_b) integer tuple from a variable name """

        string = string.split("_")[1]
        string = string.split(",")
        return (int(string[0]), int(string[1]))

    def get_variable(self, edge, prefix="x"):
        """ Get a single variable by specifying its edge and prefix """

        key = "{}_{},{}".format(prefix, *edge)
        return self.variable_dict[key]

    def get_outward_variables(self, vertex, prefix="x"):
        """ Get the outward variables for a vertex """

        keys = [key for key in self.variable_dict.keys() if prefix in key]
        return xsum(self.variable_dict[key]
                    for key in keys if self.get_variable_tuple(key)[0] == vertex)

    def get_inward_variables(self, vertex, prefix="x"):
        """ Get the inward variables for a vertex """

        keys = [key for key in self.variable_dict.keys() if prefix in key]
        return xsum(self.variable_dict[key]
                    for key in keys if self.get_variable_tuple(key)[1] == vertex)

    def set_outward_constraint(self, vertex):
        """ Define the outward constraints for a given vertex """

        self.model += self.get_outward_variables(vertex) == 1, "out_{}".format(vertex)

    def set_inward_constraint(self, vertex):
        """ Define the inward constraints for a given vertex """

        self.model += self.get_inward_variables(vertex) == 1, "in_{}".format(vertex)

    def set_inward_constraints(self):
        """ Define the inward constraints for all vertices """

        for vertex in self.vertices:
            self.model += self.get_inward_variables(vertex) == 1, "in_{}".format(vertex)

    def set_constraints(self):
        """ Define the outward and inward constraints for all vertices """

        if self.formulation == "dfj":
            self.set_dfj_constraints()
            return None
        if self.formulation == "mil":
            self.set_miller_constraints()
            return None
        if self.formulation == "scf":
            self.set_scf_constraints()
            return None
        print("No constraints set!")

    def set_in_out_constraints(self):
        """ Define the outward and inward constraints for all vertices """

        for vertex in self.vertices:
            self.set_outward_constraint(vertex)
            self.set_inward_constraint(vertex)

    def set_dfj_constraints(self):
        """ Define the constraints neccessary for a DFJ formulation """

        self.set_in_out_constraints()

    def set_miller_subtour_constraints(self):
        """ Set the subtour constraints for Miller's formulation  """

        for vertex_a in self.vertices:
            if vertex_a != self.min_vertex:
                for vertex_b in self.vertices:
                    if (vertex_b != vertex_a and vertex_b != self.min_vertex):
                        first_var = self.get_variable((vertex_a, "_"), "u")
                        second_var = self.get_variable((vertex_b, "_"), "u")
                        third_var = self.get_variable((vertex_a, vertex_b), "x")
                        en = len(self.vertices)
                        self.model += first_var - second_var + en * third_var <= en - 1

    def set_miller_constraints(self):
        """ Define the contsraints necessary for Miller's formulation """

        self.set_in_out_constraints()
        self.set_miller_subtour_constraints()

    def set_first_scf_constraint(self):
        """ Set the first P2 SCF commodity constraint (Gavish and Graves 1982) """

        variable_sum = xsum(self.get_variable((self.min_vertex, vertex), "y") for
                            vertex in self.vertices if vertex != self.min_vertex)
        self.model += variable_sum == 1, "SCF 1"

    def set_second_scf_constraints(self):
        """ Set the second P2 SCF commodity constraint set (Gavish and Graves 1982) """

        for vertex_a in self.vertices:
            if vertex_a != self.min_vertex:
                first_sum = xsum(self.get_variable((vertex_a, vertex_b), "y") for
                                 vertex_b in self.vertices if vertex_b != vertex_a)
                second_sum = xsum(self.get_variable((vertex_b, vertex_a), "y") for
                                  vertex_b in self.vertices if vertex_b != vertex_a)
                self.model += first_sum - second_sum == 1

    def set_third_scf_constraints(self):
        """ Set the third P2 SCF commodity constraint set (Gavish and Graves 1982) """

        for vertex_a in self.vertices:
            for vertex_b in self.vertices:
                if (vertex_b != vertex_a and vertex_b != self.min_vertex):
                    first_variable = self.get_variable((vertex_a, vertex_b), "y")
                    second_variable = self.get_variable((vertex_a, vertex_b), "x")
                    self.model += first_variable <= (len(self.vertices) - 1) * second_variable

    def set_fourth_scf_constraints(self):
        """ Set the fourth P2 SCF commodity constraint set (Gavish and Graves 1982) """

        for vertex_a in self.vertices:
            if vertex_a != self.min_vertex:
                first_variable = self.get_variable((vertex_a, self.min_vertex), "y")
                second_variable = self.get_variable((vertex_a, self.min_vertex), "x")
                self.model += first_variable == (len(self.vertices)) *  second_variable

    def set_scf_constraints(self):
        """ Define the constraints neccessary for a SCF formulation """

        self.set_in_out_constraints()
        self.set_first_scf_constraint()
        self.set_second_scf_constraints()
        self.set_third_scf_constraints()
        self.set_fourth_scf_constraints()

    def perform_relaxation(self):
        """ Perform a linear relaxation on the current problem  """

        self.model.optimize(relax=True)

    def set_mincut_graph(self):
        """ Build a graph structure for mincut """

        self._mincut_graph = DiGraph()
        edge_bunch = [(*self.get_variable_tuple(key), {"capacity": value.x})
                      for key, value in self.variable_dict.items()]
        self._mincut_graph.add_edges_from(edge_bunch)

    def set_mincut_constraint(self, vertex_a, vertex_b):
        """ Build a mincut constraint for a given pair of vertices """
        cut_value, (set_a, set_b) = minimum_cut(self._mincut_graph, vertex_a, vertex_b)

        if cut_value <= 0.99:
            self.model += (xsum(self.variable_dict[key] for key in self.variable_dict.keys()
                                 if (self.get_variable_tuple(key)[0] in set_a and
                                     self.get_variable_tuple(key)[1] in set_a)) <= len(set_a) - 1)
            self._add_constraints = True

    def set_mincut_constraints(self):
        """ Build mincut constraints """

        for (i, j) in product(self.vertices, self.vertices):
            if i != j:
                self.set_mincut_constraint(i, j)
            
    def set_valid_constraints(self):
        """ Generate more cutting planes """
        
        if not self._add_constraints and self.model.solver_name.lower() == "cbc":
            cutting_planes = self.model.generate_cuts([CutType.GOMORY,
                                                       CutType.MIR,
                                                       CutType.ZERO_HALF,
                                                       CutType.KNAPSACK_COVER])
            if cutting_planes.cuts:
                self.model += cutting_planes
                self._add_constraints = True
                
    def solve(self, kwargs):
        """ Solve to optimality if possible """

        if self.formulation == "dfj":
            self._add_constraints = True
            while self._add_constraints:
                self.perform_relaxation()
                self.set_mincut_graph()
                self._add_constraints = True
                self.set_mincut_constraints()
                self.set_valid_constraints()
        if self.formulation == "mil":
            self.model.optimize(**kwargs)
        if self.formulation == "scf":
            self.model.optimize(**kwargs)

    def get_edge_from_varname(self, varname):
        """ Get the edge tuple for a given variable name """

        (a, b) = varname.split(",")
        (a, b) = (a.split("_")[1], b)

        return tuple([int(a), int(b)])

    def get_nonzero_varnames(self, prefix="x"):
        """ Get all nonzero variable names from the LP solution """

        return [item.name for item in self.model.vars if (item.x > 0 and prefix in item.name)]

    def get_nonzero_edges(self, prefix="x"):
        """ Get the nonzero edge tuples from the LP solution """

        varnames = self.get_nonzero_varnames(prefix=prefix)
        return [self.get_edge_from_varname(varname) for varname in varnames]

    def get_nonzero_varvals(self, prefix="x"):
        """ Get all nonzero variable values from the LP solution """

        return [item.x for item in self.model.vars if (item.x > 0 and prefix in item.name)]
