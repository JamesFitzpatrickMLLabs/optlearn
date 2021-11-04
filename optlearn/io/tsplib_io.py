import tsplib95

import networkx as nx

from optlearn import graph_utils

from optlearn.io import io_utils


def read_file(fname):
    """ Read in tsplib file format """

    return tsplib95.load(fname)


def read_solution_from_file(fname):
    """ Read and parse a solution file """

    lines = io_utils.read_file_into_list(fname)
    tour_index = io_utils.find_substring_in_stringlist(lines, "TOUR_SECTION")
    minus_index = io_utils.find_substring_in_stringlist(lines, "-1")
    lines = lines[tour_index+1:minus_index]
    return [int(item[:-1]) for item in lines]


class optObject():

    def read_problem_from_file(self, fname):
        self._problem = read_file(fname)
        return self

    def read_solution_from_file(self, fname, symmetric=True):
        tour = read_solution_from_file(fname)
        edges = graph_utils.get_tour_edges(tour, symmetric=symmetric)
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
