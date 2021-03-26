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


class subtourStrategy():

    def __init__(self, cut_strategy=None):

        self.cut_strategy = cut_strategy
        self.check_strategy()

    def check_strategy(self):

        if self.cut_strategy is None:
            self.cut_strategy = "small_if_possible"
        if self.cut_strategy not in ["small_if_possible", "all"]:
            self.cut_strategy = "small_if_possible"

    def get_subtour_lens(self, subtours):

        return np.array([len(subtour) for subtour in subtours])

    def get_subtours_smaller_than(self, subtours, threshold):

        subtour_lens = self.get_subtour_lens(subtours)
        indices = np.argwhere(subtour_lens < threshold).flatten()

        if len(indices) < 1:
            return []
        else:
            return np.array(subtours)[indices]

    def try_small_subtours(self, subtours):

        vertices_number = len(np.unique(subtours))
        threshold = int(vertices_number / int(np.ceil(np.log2(vertices_number))))
        subset_tours = self.get_subtours_smaller_than(subtours, threshold)
        if len(subset_tours) > 0:
            return subset_tours

        while len(subset_tours) < 1:
            threshold = int(threshold * 2)
            subset_tours = self.get_subtours_smaller_than(subtours, threshold)

        return subset_tours
    
    def get_all_subtours(self, subtours):

        return subtours


    def get_subtours(self, subtours):

        if self.cut_strategy == "small_if_possible":
            return self.try_small_subtours(subtours)
        else:
            return self.get_all_subtours(subtours)


class xpress_constraint_callback():

    def __init__(self, solver, max_rounds=1e10):
        self.solver = solver
        self.max_rounds = max_rounds
        self.round_counter = 0
        self.is_optimal = False

    def is_tour_callback(self, problem=None):
        """ check if the current solution gives a valid tour """

        graph = nx.Graph()
        variables = list(self.solver.variable_dict.values())
        keys = list(self.solver.variable_dict.keys())

        print("Check callback!")
        solution = []
        try:
            problem.getlpsol(solution, None, None, None)
        except:
            return (True, None)
        
        indices = [num for num, item in enumerate(solution) if item > 0.0001]
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

        if self.solver._is_symmetric:
            graph = nx.Graph()
        if self.solver._is_asymmetric:
            graph = nx.DiGraph()

        variables = list(self.solver.variable_dict.values())
        keys = list(self.solver.variable_dict.keys())

        solution = []

        print("Cut callback!")
        try:
            self.solver.problem.getlpsol(solution, None, None, None)
        except:
            return 0
        self.solver.solutions.append(solution)
        
        edges = [mip_utils.get_variable_tuple(key) for key in keys]
        tuples = [(*edge, value) for (edge, value) in zip(edges, np.abs(solution))]
        
        graph.add_weighted_edges_from(tuples)

        cut_value, (left_set, right_set) = nx.stoer_wagner(graph)

        if len(left_set) < len(right_set):
            cut_set = left_set
        else:
            cut_set = right_set

        if cut_value < 1.99:
            varnames = ["x_{},{}".format(i, j) for i in cut_set
                        for j in cut_set if j > i and "x_{},{}".format(i, j) in keys]
            variables = [self.solver.variable_dict[item] for item in varnames]
            xpress.add_mincut(self.solver.problem, variables, self.solver.variable_dict)

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

        subtour_selector = subtourStrategy(cut_strategy=self.solver.cut_strategy)
        
        for S in subtour_selector.get_subtours(components):
            if self.solver._is_symmetric:
                varnames = ["x_{},{}".format(i, j) for i in S
                            for j in S if j > i and "x_{},{}".format(i, j) in keys]
            if self.solver._is_asymmetric:
                varnames = ["x_{},{}".format(i, j) for i in S
                            for j in S if "x_{},{}".format(i, j) in keys]                
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
        
