from itertools import product
from networkx import minimum_cut, DiGraph
from mip import Model, xsum, BINARY, CONTINUOUS, OptimizationStatus, CutType

from optlearn import io_utils
from optlearn import graph_utils


_var_types = {
    "binary": BINARY,
    "continuous": CONTINUOUS,
}


class smallLinearTSP():
    """ this is designed to find relaxed solutions of very small TSPs """

    def __init__(self, graph, var_type="continuous"):
        """ setup problem """ 
        self.set_var_type(var_type)
        self.initialise_model()
        self.initialise_variable_dict()
        self.initialise_vertices(graph)
        self.set_variables(graph)
        self.set_objective(graph)
        self.set_constraints()

    def set_var_type(self, var_type):
        """ set the variable type """

        self._var_type = _var_types.get(var_type) or _var_types["continuous"]  

    def initialise_model(self):
        """ setup the model and add as an attribute """

        self.model = Model()

    def initialise_variable_dict(self):
        """ setup a dictionary to put the variables in """

        self.variable_dict = {}

    def initialise_vertices(self, graph):
        """ grab a list of the vertices from the graph """

        self.vertices = graph_utils.get_vertices(graph)

    def name_variable(self, edge):
        """ create a variable name """

        return "x_{},{}".format(*edge)

    def create_variable(self, graph, edge):
        """ create a single variable and return it """

        name = self.name_variable(edge)
        return self.model.add_var(name, var_type=self._var_type)

    def set_variable(self, graph, edge):
        """ create an edge variable and set it """

        variable = self.create_variable(graph, edge)
        name = self.name_variable(edge)
        self.variable_dict[name] = variable

    def set_variables(self, graph):
        """ create and set all edge variables """
        
        for edge in graph.edges:
            self.set_variable(graph, edge)

    def define_objective_term(self, graph, edge):
        """ define a single term in the objective """

        weight = graph_utils.get_edge_weight(graph, *edge)
        name = self.name_variable(edge)
        return self.variable_dict[name] * weight 

    def define_objective_terms(self, graph):
        """ define all objective terms  """

        return [self.define_objective_term(graph, edge) for edge in graph.edges]

    def set_objective(self, graph):
        """ set the objective function """

        self.model.objective = xsum(self.define_objective_terms(graph))

    def get_variable_tuple(self, string):
        """ get the (vertex_a, vertex_b) integer tuple from a variable name """

        string = string.split("_")[1]
        string = string.split(",")
        return (int(string[0]), int(string[1]))

    def get_outward_variables(self, vertex):
        """ get the outward variables for a vertex """

        keys = self.variable_dict.keys()
        return xsum(self.variable_dict[key]
                    for key in keys if self.get_variable_tuple(key)[0] == vertex)

    def get_inward_variables(self, vertex):
        """ get the inward variables for a vertex """

        keys = self.variable_dict.keys()
        return xsum(self.variable_dict[key]
                    for key in keys if self.get_variable_tuple(key)[1] == vertex)

    def set_outward_constraint(self, vertex):
        """ define the outward constraints for a given vertex """

        self.model += self.get_outward_variables(vertex) == 1, "out_{}".format(vertex)

    def set_inward_constraint(self, vertex):
        """ define the inward constraints for a given vertex """

        self.model += self.get_inward_variables(vertex) == 1, "in_{}".format(vertex)

    def set_constraints(self):
        """ define the outward and inward constraints for all vertices """

        for vertex in self.vertices:
            self.set_outward_constraint(vertex)
            self.set_inward_constraint(vertex)

    def perform_relaxation(self):
        """ perform a relaxation on the root of the branch and bound tree """

        self.model.optimize(relax=True)

    def set_mincut_graph(self):
        """ build a graph structure for mincut """

        self._mincut_graph = DiGraph()
        edge_bunch = [(*self.get_variable_tuple(key), {"capacity": value.x})
                      for key, value in self.variable_dict.items()]
        self._mincut_graph.add_edges_from(edge_bunch)

    def set_mincut_constraint(self, vertex_a, vertex_b):
        """ build a mincut constraint for a given pair of vertices """

        cut_value, (set_a, set_b) = minimum_cut(self._mincut_graph, vertex_a, vertex_b)
        if cut_value <= 0.99:
            self.model += (xsum(self.variable_dict[key] for key in self.variable_dict.keys()
                                 if (self.get_variable_tuple(key)[0] in set_a and
                                     self.get_variable_tuple(key)[1] in set_a)) <= len(set_a) - 1)
            self._add_constraints = True

    def set_mincut_constraints(self):
        """ build mincut constraints """

        for (i, j) in product(self.vertices, self.vertices):
            if i != j:
                self.set_mincut_constraint(i, j)
            
    def set_valid_constraints(self):
        """ generate more cutting planes """
        
        if not self._add_constraints and self.model.solver_name.lower() == "cbc":
            cutting_planes = self.model.generate_cuts([CutType.GOMORY,
                                                       CutType.MIR,
                                                       CutType.ZERO_HALF,
                                                       CutType.KNAPSACK_COVER])
            if cutting_planes.cuts:
                self.model += cutting_planes
                self._add_constraints = True
                
    def solve(self):
        """ solve to optimality if possible """
        
        self._add_constraints = True
        while self._add_constraints:
            self.perform_relaxation()
            self.set_mincut_graph()
            self._add_constraints = False
            self.set_mincut_constraints()
            self.set_valid_constraints()
