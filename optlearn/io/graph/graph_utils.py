from optlearn.io.graph import graph_io


def read_graph_from_adjacency_list(filename):
    """ Read a graph from the given adjacency list """

    graph = graph_io.read_adjacency(filename)

    return graph


def read_graph_from_pickle(filename):
    """ Read a graph from the given pickle """

    graph = graph_io.read_pickle(filename)

    return graph


def write_graph_to_adjacency_list(graph, filename):
    """ Write a graph ro the given adjacency list filename """

    _ = graph_io.write_adjacency(graph, filename)

    return _


def write_graph_to_pickle(graph, filename):
    """ Write a graph to the given pickle filename """

    _ = graph_io.write_pickle(graph, filename)

    return _
