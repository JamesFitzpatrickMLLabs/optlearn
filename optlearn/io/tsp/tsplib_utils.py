import tsplib95

from optlearn import graph_utils

from optlearn.io import utils


def read_tsplib(filename):
    """ Read in tsplib file format """

    tsplib_object = tsplib95.load(filename)

    return tsplib_object


def read_solution(filename):
    """ Read and parse a solution file """

    lines = utils.read_file_into_list(filename)
    tour_index = utils.find_substring_in_stringlist(lines, "TOUR_SECTION")
    minus_index = utils.find_substring_in_stringlist(lines, "-1")
    lines = lines[tour_index+1:minus_index]
    solution = [int(item[:-1]) for item in lines]

    return solution


def read_tsplib_problem_as_graph(filename):
    """ Read in a TSPLib problem as a graph """

    reader = optObject()
    reader.read_problem_from_file(filename)
    graph = reader.get_graph()

    return graph


def read_tsplib_solution_as_array(filename):
    """ Read in a TSPLib solution as an array """

    reader = optObject()
    reader.read_solution_from_file(filename)
    array = reader.get_solution()

    return array 
    

class optObject():

    def read_problem_from_file(self, fname):
        self._problem = read_tsplib(fname)
        return self

    def read_solution_from_file(self, fname, symmetric=True):
        tour = read_solution(fname)
        edges = graph_utils.get_tour_edges(tour, symmetric=self.symmetric)
        self._solution = edges
        
        return tour

    def _check_graph(self):
        """ Check if the graph is already set """

        return hasattr(self, "_graph")

    def _set_graph(self):
        """ Set the graph """

        self._graph = self._problem.get_graph()
        self._graph = graph_utils.delete_self_weights(self._graph)
        if hasattr(self._problem, "node_coords"):
            self._graph.graph["coord_dict"] = self._problem.node_coords

    def get_graph(self):
        """ Get the graph """

        if not self._check_graph():
            self._set_graph()
        return self._graph

    def get_solution(self):
        return self._solution

    def get_dict(self):
        return self._problem.as_dict()

    def get_keyword_dict(self):
        return self._problem.as_keyword_dict()
