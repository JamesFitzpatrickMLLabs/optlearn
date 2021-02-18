import numpy as np
import networkx as nx

from pyscipopt import Conshdlr, Eventhdlr, SCIP_RESULT, quicksum, SCIP_EVENTTYPE

from optlearn.mip import mip_utils
from optlearn.mip import xpress
from optlearn.mip import scip


from optlearn import io_utils
from optlearn import plotting
import matplotlib.pyplot as plt


def initialise_connection_graph(is_symmetric=True):
    """ Set the mincut graph """
        
    if is_symmetric:
        graph_type = nx.Graph
    else:
        graph_type = nx.DiGraph
    return graph_type()


def build_connection_graph(solver):
    """ build the connection graph for the solver """

    graph = solver.initialise_connection_graph()
    nonzero_edges = solver.get_nonzero_edges()
    graph.add_edges_from(nonzero_edges)
    return graph


class xpress_constraint_callback():

    def __init__(self, solver, max_rounds=1e10):
        self.solver = solver
        self.max_rounds = max_rounds
        self.round_counter = 0
        self.is_optimal = False

    def is_tour_callback(self, problem=None):
        """ check if the current solution gives a valid tour """

        print("Called checker!")
        
        graph = nx.Graph()
        variables = list(self.solver.variable_dict.values())
        keys = list(self.solver.variable_dict.keys())
                
        
        solution = []
        try:
            problem.getlpsol(solution, None, None, None)
        except:
            return (True, None)
                
        indices = [num for num, item in enumerate(solution) if item >= 0.001]
        unit_edges = [mip_utils.get_variable_tuple(variables[index].name)
                         for index in indices]
        
        graph.add_edges_from(unit_edges)
        components = list(nx.connected_components(graph))

        try:
            cycle = nx.cycles.find_cycle(graph)
        except:
            return (True, None)
        
        return (len(components) > 1, None)
            
    def add_tour_callback(self):
        """ add the tour callback to the solver """
        
        def callback(problem, graph, isheuristic, cutoff):
            """ callback to check the existence of subtours """

            ret =  self.is_tour_callback(problem=problem)
            return ret

        
        self.solver.problem.addcbpreintsol(callback, None, 1)

    def cut_callback(self, problem):
        """ build the cuts for the cutting callback """

        print("Called cutter!")
        
        graph = nx.Graph()
        variables = list(self.solver.variable_dict.values())
        keys = list(self.solver.variable_dict.keys())

        solution = []
        
        try:
            self.solver.problem.getlpsol(solution, None, None, None)
        except:
            return False
        
        indices = [num for num, item in enumerate(solution) if item >= 0.001]
        unit_edges = [mip_utils.get_variable_tuple(variables[index].name)
                         for index in indices]
        graph.add_edges_from(unit_edges)

        self.solver.solutions.append(solution)
        
        try:
            cycle = nx.cycles.find_cycle(graph)
            print("Cycle found!")
        except:
            return False

        components = list(nx.connected_components(graph))

        for S in components:
            varnames = ["x_{},{}".format(i, j) for i in S
                        for j in S if j > i and "x_{},{}".format(i, j) in keys]
            variables = [self.solver.variable_dict[item] for item in varnames]
            xpress.add_mincut(self.solver.problem, variables, self.solver.variable_dict)

    def add_cut_callback(self):
        """ add the cutting callback to the optimiser """
        
        def callback(problem, graph):
            """ perform the subtour cutting """

            return self.cut_callback(problem)
            
        self.solver.problem.addcboptnode(callback, None, 1)
        


class crap_constraint_callback():

    def __init__(self, solver, max_rounds=1e10):
        self.solver = solver
        self.max_rounds = max_rounds
        self.round_counter = 0
        self.is_optimal = False

    def reverse(self, i, j):

        if i < j:
            return (i, j)
        else:
            return (j, i)

    def is_tour_callback(self, problem=None):
        """ check if the current solution gives a valid tour """

        variables = list(self.solver.variable_dict.values())
        keys = list(self.solver.variable_dict.keys())
        
        s = []

        print("Calling!")
        
        try:
            problem.getlpsol(s, None, None, None)
        except:
            return (True, None) 

        orignode = 1
        nextnode = 1
        card = 0

        while nextnode != orignode or card == 0:

            FS = [j for j in self.solver.vertices if j != nextnode and
                  s[keys.index("x_{},{}".format(*self.reverse(j, nextnode)))] == 1]
            card += 1
            
            if len(FS) < 1:
                return (True, None)

            nextnode = FS[0]

        return (card < len(self.solver.vertices), None)
                    
    def add_tour_callback(self):
        """ add the tour callback to the solver """
        
        def callback(problem, graph, isheuristic, cutoff):
            """ callback to check the existence of subtours """

            return self.is_tour_callback(problem=problem)
        
        self.solver.problem.addcbpreintsol(callback, None, 1)

    def cut_callback(self, problem):
        """ build the cuts for the cutting callback """

        print("Cutting!")
        
        graph = nx.Graph()
        variables = list(self.solver.variable_dict.values())
        keys = list(self.solver.variable_dict.keys())

        s = []

        try:
            problem.getlpsol(s, None, None, None)
        except:
            return 0

        orignode = 1
        nextnode = 1
        
        connset = []

        while nextnode != orignode or len(connset) == 0:
            
            connset.append(nextnode)

            FS = [j for j in self.solver.vertices if j != nextnode and
                  s[keys.index("x_{},{}".format(*self.reverse(j, nextnode)))] == 1]

            if len(FS) < 1:
                return 0

            nextnode = FS[0]

        if len(connset) < len(self.solver.vertices):

            if len(connset) <= len(self.solver.vertices)/2:
                columns = [variables[keys.index("x_{},{}".format(i, j))] for i in connset
                           for j in connset if i < j]
                nArcs = len(connset)
            else:
                columns = [variables[keys.index("x_{},{}".format(i, j))]
                           for i in self.solver.vertices for j in self.solver.vertices
                           if i not in connset and
                           j not in connset and i < j]
                nArcs = len(self.solver.vertices) - len(connset)

            nTerms = len(columns)
                
            problem.addcuts([1], ['L'], [nArcs - 1],
                     [0, nTerms], columns, [1] * nTerms)

        return 0

    def add_cut_callback(self):
        """ add the cutting callback to the optimiser """
        
        def callback(problem, graph):
            """ perform the subtour cutting """

            return self.cut_callback(problem)
            
        self.solver.problem.addcboptnode(callback, None, 1)


        
class scip_constraint_handler(Conshdlr):

    def __init__(self, solver, max_rounds=1e10):
        self.solver = solver
        self.max_rounds = max_rounds
        self.round_counter = 0
    
    def find_subtours(self, checkonly, solution, variable_dict):
        """ find subtours in the current solution """

        graph = nx.Graph()
        
        variables = list(self.solver.variable_dict.values())
        keys = list(self.solver.variable_dict.keys())
        solution = [self.model.getSolVal(solution, variable) for variable in variables]
        indices = [num for num, item in enumerate(solution) if item > 0]
        nonzero_edges = [mip_utils.get_variable_tuple(variables[index].name) for index in indices]
        graph.add_edges_from(nonzero_edges)
        components = list(nx.connected_components(graph))
        
        if len(components) == 1 or self.round_counter >= self.max_rounds:
            return False
        elif checkonly:
            return True
        for S in components:
            varnames = ["x_{},{}".format(i, j) for i in S
                        for j in S if j > i and "x_{},{}".format(i, j) in keys]
            self.model.addCons(quicksum(self.solver.variable_dict[name] for
                                        name in varnames) <= len(S) - 1)
        self.round_counter += 1
        self.solver.counter += 1
        return True

    def conscheck(self,
                  constraints,
                  solution,
                  checkintegrality,
                  checklprows,
                  printreason,
                  completelty):
        if self.find_subtours(checkonly=True,
                              solution=solution,
                              variable_dict=self.solver.variable_dict):
            return {"result": SCIP_RESULT.INFEASIBLE}
        else:
            return {"result": SCIP_RESULT.FEASIBLE}

    def consenfolp(self, constraints, nusefulconss, solinfeasible):
        if self.find_subtours(checkonly=False,
                              solution=None,
                              variable_dict=self.solver.variable_dict):
            return {"result": SCIP_RESULT.CONSADDED}
        else:
            return {"result": SCIP_RESULT.FEASIBLE}

    def conslock(self, constraint, locktype, nlockspos, nlocksneg):
        pass


class scip_event_handler(Eventhdlr):

    def __init__(self, max_nodes=50):
        self.max_nodes = max_nodes
        self.first_lp = False
        self.lp_counter = 0

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED, self)
        
    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.NODESOLVED, self)

    def eventexec(self, event):
        """ Interrupt search if maximum number of LP solves are reached """
        
        self.lp_counter += 1
        
        if self.lp_counter >= self.max_nodes:
            print("Maximum nodes ({}) reached! Interrupting!".format(self.max_nodes))
            self.model.interruptSolve()
            self.model.printStatistics()
        
