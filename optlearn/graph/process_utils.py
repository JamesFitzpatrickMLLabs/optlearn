import copy
import numpy as np


def get_node_type(graph, node):
    """ Get the type attribute of the given node """

    node_type = graph.nodes[node].get("type")

    return node_type


def get_node_types(graph, nodes):
    """ Get the node type attributes for the given nodes """

    node_types = [get_node_type(graph, node) for node in nodes]

    return node_types


def get_node_types_dict(graph, nodes):
    """ Get a dictionary of nodes, keyed by their type attribute """

    nodes = np.array(nodes)
    node_types = np.array(get_node_types(graph, nodes))
    unique_types = np.unique(node_types)
    node_dict = {
        node_type: list(nodes[node_types==node_type]) for node_type in unique_types
    }

    return node_dict


def get_node_attributes(graph, node):
    """ Get the attributes of the given node """

    node_attributes = graph.nodes[node]

    return node_attributes


def duplicate_node_attributes(graph, old_node, new_node):
    """ Duplicate the attributes of the old node for the new node """

    node_attributes = copy.deepcopy(get_node_attributes(graph, old_node))
    node_attributes["prime_node"] = old_node

    return node_attributes


def get_connected_nodes(graph, node):
    """ Find the nodes that this node is directly connected to """

    connected_nodes = list(graph[node])

    return connected_nodes
    


def duplicate_edges_outward(graph, old_node, new_node):
    """ Duplicate the outward edges for the old node """
    
    connected_nodes = np.sort(get_connected_nodes(graph, old_node) + [old_node])
    edges = [(new_node, node) for node in connected_nodes] + [(old_node, new_node)]
    
    return edges


def duplicate_edges_inward(graph, old_node, new_node):
    """ Duplicate the inward edges for the old node """

    edges = []
    for node in graph.nodes:
        connected_nodes = get_connected_nodes(graph, node)
        if old_node in connected_nodes and node != new_node:
            edges.append((node, new_node))
    
    return edges


def duplicate_edge_attributes_outward(graph, old_node, new_node):
    """ Duplicate the attributes for the old node's edges """

    connected_nodes = np.sort(list(graph[old_node]) + [old_node])
    insert_index = connected_nodes.tolist().index(old_node)
    edge_attributes = list(graph[old_node].values())
    edge_attributes.insert(insert_index, {"weight": 0.0})
    edge_attributes += [{"weight": 0.0}]

    return edge_attributes


def duplicate_edge_attributes_inward(graph, old_node, new_node):
    """ Duplicate the attributes for the old node's edges """

    node_edges = duplicate_edges_inward(graph, old_node, new_node)
    attribute_edges = duplicate_edges_inward(graph, old_node, old_node)
    edge_attributes = [graph[edge[0]][edge[1]] for edge in attribute_edges]

    return edge_attributes


def update_coordinate_dict(graph, old_node, new_node):
    """ Update the coordinate dict to include the new node """

    graph.graph["coord_dict"][new_node] = graph.graph["coord_dict"][old_node]

    return graph


def duplicate_node(graph, node):
    """ Duplicate the given node, giving the same node attributes """

    new_node = max(graph.nodes) + 1
    edges = duplicate_edges_outward(graph, node, new_node)
    node_attributes = duplicate_node_attributes(graph, node, new_node)
    edge_attributes = duplicate_edge_attributes_outward(graph, node, new_node)
    graph.add_nodes_from([(new_node, node_attributes)])
    graph.add_edges_from([
        (*edge, attribute) for (edge, attribute) in zip(edges, edge_attributes)
    ])
    edges = duplicate_edges_inward(graph, node, new_node)
    edge_attributes = duplicate_edge_attributes_inward(graph, node, new_node)
    graph.add_edges_from([
        (*edge, attribute) for (edge, attribute) in zip(edges, edge_attributes)
    ])
    update_coordinate_dict(graph, node, new_node)

    return graph


def get_station_nodes_from_metadata(graph):
    """ Get the station nodes from the metadata """

    node_types = graph.graph.get("node_types")
    if node_types is None:
        raise ValueError("Node types dict not found in the metadata!")
    stations = node_types.get("station")
    if stations is None:
        raise ValueError("No station nodes found for this graph!")

    return stations


def get_customer_nodes_from_metadata(graph):
    """ Get the customer nodes from the metadata """

    node_types = graph.graph.get("node_types")
    if node_types is None:
        raise ValueError("Node types dict not found in the metadata!")
    customers = node_types.get("customer")
    if customers is None:
        raise ValueError("No customer nodes found for this graph!")

    return customers


def get_depot_nodes_from_metadata(graph):
    """ Get the depot# nodes from the metadata """

    node_types = graph.graph.get("node_types")
    if node_types is None:
        raise ValueError("Node types dict not found in the metadata!")
    depots = node_types.get("depot")
    if depots is None:
        raise ValueError("No depot# nodes found for this graph!")

    return depots


def duplicate_stations_uniform(graph, num_duplicates):
    """ Duplicate the stations a given number of times """


    assert(type(num_duplicates) is int, "Parameter num_duplicates must be an int!")

    new_stations = []
    station_nodes = get_station_nodes_from_metadata(graph)
    for station_node in station_nodes:
        for num in range(num_duplicates):
            graph = duplicate_node(graph, station_node)
            new_stations.append(list(graph.nodes)[-1])
    graph.graph["node_types"]["station"] = graph.graph["node_types"]["station"] + new_stations

    return graph


def duplicate_depots_uniform(graph, num_duplicates):
    """ Duplicate the depots a given number of times, as stations """

    assert(type(num_duplicates) is int, "Parameter num_duplicates must be an int!")

    new_stations = []
    depot_nodes = get_depot_nodes_from_metadata(graph)
    for depot_node in depot_nodes:
        for num in range(num_duplicates):
            graph = duplicate_node(graph, depot_node)
            graph.nodes[list(graph.nodes)[-1]]["type"] = 2
            graph.nodes[list(graph.nodes)[-1]]["cs_type"] = "fast"
            new_stations.append(list(graph.nodes)[-1])
    graph.graph["node_types"]["station"] = graph.graph["node_types"]["station"] + new_stations
    graph = add_fast_charging_function(graph)
    return graph


def duplicate_stations_nonuniform(graph, num_duplicates):
    """ Duplicate the stations a given number of times """

    assert(type(num_duplicates) is dict, "Parameter num_duplicates must be dict!")

    new_stations = []
    station_nodes = num_duplicates.keys()
    for station_node in station_nodes:
        for num in range(num_duplicates[station_node]):
            graph = duplicate_node(graph, station_node)
            new_stations.append(list(graph.nodes)[-1])
    graph.graph["node_types"]["station"] = graph.graph["node_types"]["station"] + new_stations
            
    return graph


def duplicate_stations(graph, num_duplicates):
    """ Duplicate the stations a given number of times """

    if type(num_duplicates) is int:
        graph = duplicate_stations_uniform(graph, num_duplicates)
    elif type(num_duplicates) is dict:
        graph = duplicate_stations_nonuniform(graph, num_duplicates)
    else:
        raise ValueError("Parameter num_duplicates must be int or dict!")

    return graph


def add_fast_charging_function(graph):
    """ Add the charging function to the metadata """
    
    if graph.graph["fleet"]["functions_0"]["charging_functions"].get("fast") is None:
        graph.graph["fleet"]["functions_0"]["charging_functions"]["fast"] = {
            'cs_type': 'fast', 'breakpoints':[
            {'battery_level': 0.0, 'charging_time': 0.0},
            {'battery_level': 13600.0, 'charging_time': 0.31},
            {'battery_level': 15200.0, 'charging_time': 0.39},
            {'battery_level': 16000.0, 'charging_time': 0.51}
        ]}
            
    return graph


def process_tsp_graph(graph):
    """ Process the TSP graph """

    return graph


def process_cvrp_graph(graph):
    """ Process the CVRP graph """

    return graph


def process_grvp_graph(graph, num_duplicates):
    """ Process the GVRP graph """

    graph = duplicate_stations(graph, num_duplicates)

    return graph


def process_evrpnl_graph(graph, num_duplicates):
    """ Process the EVRP-NL graph """

    graph = duplicate_stations(graph, num_duplicates)

    return graph
