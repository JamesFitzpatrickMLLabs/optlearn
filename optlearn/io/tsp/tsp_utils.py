from optlearn.io.tsp import tsplib_utils


def read_tsp_problem_from_tsplib_file(filename):
    """ Read a TSP problem from a TSPLib file, return as a graph """

    graph = tsplib_utils.read_tsplib_problem_as_graph(filename)

    return graph


def read_tsp_solution_from_tsplib_file(filename):
    """ Read a TSP solution from a TSPLib solution, return as array """

    array = tsplib_utils.read_tsplib_solution_as_array(filename)

    return array
