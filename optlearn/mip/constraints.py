import numpy as np
import xpress as xp
import networkx as nx
import binpacking as bp

from pyscipopt import Conshdlr, Eventhdlr, SCIP_RESULT, quicksum, SCIP_EVENTTYPE

from optlearn import io_utils
from optlearn import plotting

from optlearn.mip import mip_utils
from optlearn.mip import xpress
from optlearn.mip import scip


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


class xpress_tsp_constraint_callback():

    def __init__(self, solver, max_rounds=1e10):
        self.solver = solver
        self.max_rounds = max_rounds
        self.round_counter = 0
        self.is_optimal = False

        self.solution = []
        self.variables = None
        self.keys = None

    def _check_variables(self, problem, solution):
        """ Check if the variables are known. If not, store them """

        if self.variables is None:
            self.variables = [problem.getVariable(item) for item in range(len(solution))]

    def _check_keys(self, solution):
        """ Check if the variable names are known. If not, store them """

        if self.keys is None:
            self.keys = [item.name for item in self.variables]


    def _check_helpers(self, problem, solution):
        """ If the keys and variables are not stored, store them """

        self._check_variables(problem, solution)
        self._check_keys(solution)

    def _set_graph(self):
        """ Set the graph to store the solution edges """

        if self.solver._is_symmetric:
            return nx.Graph()
        elif self.solver._is_asymmetric:
            return nx.DiGraph()

    def _fetch_solution(self, problem, checker=True):
        """ Fetch the solution to the LP at the current node, if possible """
        
        self.solution = []
        try:
            problem.getlpsol(self.solution, None, None, None)            
        except Exception as exception:
            if checker:
                return (True, None)
            else:
                return 0

    def _get_nonzero_edges(self):
        """ Get the edge tuples for the nonzero values of the solution """

        return [mip_utils.get_variable_tuple(key) for (key, value)
                in zip(self.keys, self.solution) if value > 0.0001]

    def _get_nonzero_triplets(self):
        """ Get the edge tuples plus edge weight for each nonzero value of the solution """

        return [mip_utils.get_variable_tuple(key) + (value, ) for (key, value)
                in zip(self.keys, self.solution) if value > 0.0001]

    def _build_nonzero_solution_graph(self):
        """ Build a solution graph with edge values equal to the nonzero solution values  """

        graph = self._set_graph()
        triplets = self._get_nonzero_triplets()
        graph.add_weighted_edges_from(triplets)
        return graph
    
    def _build_integer_nonzero_solution_graph(self):
        """ Build a solution graph with unit edges  """

        graph = self._set_graph()
        unit_edges = self._get_nonzero_edges()
        graph.add_edges_from(unit_edges)
        return graph

    def _check_hamiltonian_tour(self, graph):
        """ Check if the graph admits a hamiltonian tour """

        try:
            cycle = nx.cycles.find_cycle(graph)
        except Exception as exception:
            return (True, None)
        return  (len(cycle) != len(self.solver.vertices), None)
    
    def _check_feasible(self):
        """ Given an integer solution, check if it is feasible """

        if self.solver._is_symmetric:
            graph = self._build_integer_nonzero_solution_graph()
            return self._check_hamiltonian_tour(graph)
        if self.solver._is_asymmetric:
            raise NotImplementedError("Not implemented yet!")

    def _check_connected(self, graph):
        """ Given a solution graph, check if it is connected """

        if self.solver._is_symmetric:
            return nx.is_connected(graph)
        if self.solver._is_asymmetric:
            return nx.is_weakly_connected(graph)

    def get_subtour_variables(self, subtour):
        """ Given the vertices of the subtour, get the edge variables in the cut """

        if self.solver._is_symmetric:
            varnames = ["x_{},{}".format(i, j) for i in subtour
                        for j in subtour if i < j and "x_{},{}".format(i, j) in self.keys]
        if self.solver._is_asymmetric:
            varnames = ["x_{},{}".format(i, j) for i in subtour
                        for j in subtour if i != j and "x_{},{}".format(i, j) in self.keys]
        return [self.variables[self.keys.index(item)] for item in varnames]

    def _presolve_row(self, problem, variables, subtour):
        """ Presolve the cut for the given subtour and associated variables """

        return problem.presolverow("L", variables, [1] * len(variables),
                                   len(subtour) - 1, problem.attributes.cols,
                                   self.presolve_columns, self.presolve_coefficients)

    def _presolve_subtour(self, problem ,subtour):
        """ Given a subtour, presolve the cut needed to remove it """

        self._set_presolve_variables()
        variables = self.get_subtour_variables(subtour)
        presolve_rhs, _ = self._presolve_row(problem, variables, subtour)
        self._update_cut_variables(presolve_rhs)
        
    def _cut_disconnected(self, problem, graph, subtour_selector):
        """ Given a disconnected graph, perform subtour cuts """

        if self.solver._is_symmetric:
            components = nx.connected_components(graph)
        if self.solver._is_asymmetric:
            components = nx.weakly_connected_components(graph)
        components = list(components)
        for subtour in subtour_selector.get_subtours(components):
            self._presolve_subtour(problem, subtour)

    def _cut_connected(self, problem, graph):
        """ Given a connected graph, perform subtour cut, using Stoer-Wagner to separate """

        cut_value, (left_set, right_set) = nx.stoer_wagner(graph)

        if len(left_set) < len(right_set):
            cut_set = left_set
        else:
            cut_set = right_set

        if cut_value < 1.99:
            self._presolve_subtour(problem, cut_set)

    def _add_cuts(self, problem):
        """ Add the cuts to the problem """

        if self.num_cuts > 0:
            problem.addcuts([0] * self.num_cuts, ['L'] * self.num_cuts,
                            self.cuts_rhs, self.cuts_offsets,
                            self.cuts_indices, self.cuts_coefficients)
            self.round_counter += 1

    def _cut_subtours(self, problem, subtour_selector):
        """ Cut subtours from the LP solution where they can be found """

        graph = self._build_nonzero_solution_graph()
        
        if not self._check_connected(graph):
            self._cut_disconnected(problem, graph, subtour_selector)
        else:
            self._cut_connected(problem, graph)

        self._add_cuts(problem)

    def _set_cut_variables(self):
        """ Set the cut variables that are needed to properly index the constraint matrix """

        self.cuts_offsets = [0]
        self.cuts_indices = []
        self.cuts_coefficients = []
        self.cuts_rhs = []
        
        self.num_cuts = 0
        self.offset_start = 0

    def _update_cut_variables(self, presolve_rhs):
        """ Update the cut variables following the presolve of a new cut """

        self.offset_start += len(self.presolve_columns)
        self.num_cuts += 1
                
        self.cuts_indices.extend(self.presolve_columns)
        self.cuts_coefficients.extend(self.presolve_coefficients)
        self.cuts_rhs.append(presolve_rhs)
        self.cuts_offsets.append(self.offset_start)
        
    def _set_presolve_variables(self):
        """ Set the presolve variables that store presolved row information """

        self.presolve_columns = []
        self.presolve_coefficients = []
    
    def is_tour_callback(self, problem=None):
        """ check if the current solution gives a valid tour """

        fail_return = self._fetch_solution(problem, checker=True)
        if len(self.solution) < 1:
            return fail_return

        self._check_helpers(problem, self.solution)

        if self.round_counter >= self.max_rounds:
            print("Maximum number of cutting rounds ({}) reached!".format(self.max_rounds))
            return (False, None)
        
        return self._check_feasible()
        
    def add_tour_callback(self):
        """ add the tour callback to the solver """

        def callback(problem, graph, isheuristic, cutoff):
            """ callback to check the existence of subtours """

            return self.is_tour_callback(problem=problem)

        self.solver.problem.addcbpreintsol(callback, None)

    def cut_callback(self, problem):
        """ build the cuts for the cutting callback """
            
        fail_return = self._fetch_solution(problem, checker=True)
        if len(self.solution) < 1:
            return fail_return

        self._check_helpers(problem, self.solution)
        
        self._set_cut_variables()
        subtour_selector = subtourStrategy(cut_strategy="small_if_possible")
        self._cut_subtours(problem, subtour_selector)
        
        return 0

    def add_cut_callback(self):
        """ add the cutting callback to the optimiser """
        
        def callback(problem, graph):
            """ perform the subtour cutting """

            return self.cut_callback(problem)
            
        self.solver.problem.addcboptnode(callback, None)
        
        
class tsp_constraint_handler(Conshdlr):

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
        self.solver.solutions.append(solution)
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


class vrp_constraint_handler(Conshdlr):

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
        self.solver.solutions.append(solution)
        indices = [num for num, item in enumerate(solution) if item > 0]
        nonzero_edges = [mip_utils.get_variable_tuple(variables[index].name) for index in indices]
        graph.add_edges_from(nonzero_edges)

        if np.sum(solution) > 0 and np.sum(solution) < self.solver.vehicle_num + len(self.solver.vertices):
            cycles = []
            while(len(graph.edges)) > 0:
                edges = nx.find_cycle(graph)
                cycles.append(edges)
                graph.remove_edges_from(edges)

                components = [set(np.unique(item)) for item in cycles]
        else:
            components = nx.connected_components(graph)
        
        components = [item for item in components
                      if len(item.intersection(set(self.solver.depots))) == 0 or
                      np.sum([self.solver.demands[k] for k in item]) > self.solver.get_capacity()]
        
        if len(components) == 1 or self.round_counter >= self.max_rounds:
            return False
        elif checkonly:
            return True

        subtour_selector = subtourStrategy(cut_strategy=self.solver.cut_strategy)
        
        for S in subtour_selector.get_subtours(components):
            min_vehicles = len(bp.to_constant_volume({item: self.solver.demands[item]
                                                  for item in S}, self.solver.get_capacity()))
            if self.solver._is_symmetric:
                varnames = ["x_{},{}".format(i, j) for i in S
                            for j in S if j > i and "x_{},{}".format(i, j) in keys]
            if self.solver._is_asymmetric:
                varnames = ["x_{},{}".format(i, j) for i in S
                            for j in S if "x_{},{}".format(i, j) in keys]                
            self.model.addCons(quicksum(self.solver.variable_dict[name] for
                                        name in varnames) <= len(S) - min_vehicles)
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
        
