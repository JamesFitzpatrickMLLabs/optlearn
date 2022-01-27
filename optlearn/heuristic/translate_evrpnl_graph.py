import itertools

import numpy as np
import networkx as nx

from optlearn.graph import graph_utils

from optlearn.feature.vrp import feature_utils


def get_battery_energy_capacity(graph):
    """ Get the battery energy capacity from the graph """

    charging_function_dict = graph.graph.get("fleet").get("functions_0")
    battery_enegy_capacity = charging_function_dict.get("battery_capacity")
    if battery_enegy_capacity is None:
        print("COULD NOT FIND BATTERY ENERGY CAPACITY! ASSUMING 16kWh!")
        battery_enegy_capacity = 16
        
    return battery_enegy_capacity


def get_maximum_travel_time(graph):
    """ Get the maximum travel time allowed for the graph """

    maximum_travel_time = graph.graph.get("fleet").get("vehicle_0").get("max_travel_time")
    if maximum_travel_time is None:
        print("COULD NOT FIND MAXIMUM TRAVEL TIME! ASSUMING 10hrs!")
        maximum_travel_time = 10
    
    return maximum_travel_time


def get_average_vehicle_velocity(graph):
    """ Get the average vehicle velocity for the  graph """

    vehicle_velocity = graph.graph.get("fleet").get("vehicle_0").get("speed_factor")
    if vehicle_velocity is None:
        print("COULD NOT FIND VEHICLE VELOCITY! ASSUMING 40km/h!")
        vehicle_velocity = 40

    return vehicle_velocity


def get_energy_consumption_rate(graph):
    """ get the energy consumption rate for the graph """
    
    consumption_rate = graph.graph.get("fleet").get("functions_0").get("consumption_rate")    
    if consumption_rate is None:
        print("COULD NOT FIND ENERGY CONSUMPTION RATE! ASSUMING 0.125kW/km!")
        consumption_rate = 0.125
    
    return consumption_rate



def get_node_service_time(graph, node):
    """ Get the service time of the given node, otherwise return zero """

    service_time = graph.nodes[node].get("service_time") or 0

    return service_time


def get_node_service_times(graph):
    """ Get the service times for each of the nodes """

    service_times = [get_node_service_time(graph, node) for node in graph.nodes]

    return service_times


def get_nodes_service_times(graph, nodes):
    """ Get the service times for each of the nodes """

    service_times = [get_node_service_time(graph, node) for node in nodes]

    return service_times


def get_cs_type(graph, node):
    """ Get the cs_types for the given node """

    cs_type = graph.nodes[node].get("cs_type")

    return cs_type


def get_cs_types(graph):
    """ Get the cs_types for the station nodes """

    stations = feature_utils.get_stations(graph)
    cs_types = [get_cs_type(graph, station) for station in stations]

    return cs_types


def get_time_breakpoints(graph, cs_type):
    """ Get the time breakpoints for the given charging function """

    breakpoints = graph.graph.get("fleet").get("functions_0").get("charging_functions").get(cs_type)
    time_breakpoints = [item.get("charging_time") for item in breakpoints.get("breakpoints")]

    return time_breakpoints


def get_energy_breakpoints(graph, cs_type):
    """ Get the time breakpoints for the given charging function """

    breakpoints = graph.graph.get("fleet").get("functions_0").get("charging_functions").get(cs_type)
    energy_breakpoints = [item.get("battery_level") for item in breakpoints.get("breakpoints")]

    return energy_breakpoints


def get_energy_consumption_matrix(graph):
    """ Get the energy consumption matrix for the given graph """

    clone_graph = graph_utils.clone_graph(graph)
    clone_graph = clone_graph.to_directed()
    clone_graph.add_weighted_edges_from([(node, node, 0) for node in graph.nodes])

    edges = list(itertools.product(list(clone_graph.nodes), list(clone_graph.nodes)))
    weights = np.array(graph_utils.get_edges_weights(clone_graph, edges))
    weights = weights.reshape(len(clone_graph.nodes), len(clone_graph.nodes))
    consumption_rate = get_energy_consumption_rate(clone_graph) 
    consumption_matrix = (weights * consumption_rate * 1000).tolist()

    return consumption_matrix


def get_travel_time_matrix(graph):
    """ Get the travel time matrix for the given graph """

    clone_graph = graph_utils.clone_graph(graph)
    clone_graph = clone_graph.to_directed()
    clone_graph.add_weighted_edges_from([(node, node, 0) for node in graph.nodes])

    edges = list(itertools.product(list(clone_graph.nodes), list(clone_graph.nodes)))
    weights = np.array(graph_utils.get_edges_weights(clone_graph, edges))
    weights = weights.reshape(len(clone_graph.nodes), len(clone_graph.nodes))
    average_velocity = get_average_vehicle_velocity(clone_graph) 
    travel_time_matrix = (weights / average_velocity).tolist()

    return travel_time_matrix


def build_relabel_dict(graph):
    """ Build a relabelling dict for the stations """

    stations = feature_utils.get_stations(graph)
    last_nodes = list(graph.nodes)[::-1][:len(stations)]
    keys = stations + last_nodes
    values = last_nodes + stations
    relabel_dict = {key: value for (key, value) in zip(keys, values)}

    return relabel_dict


def recoordinate_stations(graph, relabel_dict):
    """ Update the coordinate dictioary to rflect the relabeling changes """

    relabel_keys = relabel_dict.keys()
    relabel_values = [relabel_dict[key] for key in relabel_keys]
    relabel_coordinates = [graph.graph["coord_dict"][key] for key in relabel_keys]
    for num, value in enumerate(relabel_values):
        graph.graph["coord_dict"][value] = relabel_coordinates[num]

    return graph


def retype_stations(graph, relabel_dict):
    """ Update the coordinate dictioary to rflect the relabeling changes """

    for (key, value) in relabel_dict.items():
        if key in graph.graph["node_types"]["station"]:
            index = graph.graph["node_types"]["station"].index(key)
            _ = graph.graph["node_types"]["station"].pop(index)
            graph.graph["node_types"]["station"].append(value)
        if key in graph.graph["node_types"]["customer"]:
            index = graph.graph["node_types"]["customer"].index(key)
            _ = graph.graph["node_types"]["customer"].pop(index)
            graph.graph["node_types"]["station"].append(value)

    return graph


def relabel_stations(graph):
    """ Relabel the stations so that they are the last nodes """

    relabel_dict = build_relabel_dict(graph)
    print(graph.graph["node_types"]["station"])
    graph = retype_stations(graph, relabel_dict)
    print(graph.graph["node_types"]["station"])
    graph = recoordinate_stations(graph, relabel_dict)
    graph = nx.relabel.relabel_nodes(graph, relabel_dict)

    return graph


def cstype_to_index(cs_type):
    """ Convert the CS type to an index """

    if cs_type == "fast":
        return 0
    elif cs_type == "normal":
        return 1
    elif cs_type == "slow":
        return 2
    else:
        raise ValueError("CS Type not recongised by translator!")


def graph_to_translated_instance(graph, relabel=False):
    """ Convert the graph to an FRVCPY instance """

    graph = graph_utils.clone_graph(graph)
    if relabel:
        graph = relabel_stations(graph)
    translated_instance = {}

    stations = feature_utils.get_stations(graph)
    cs_types = get_cs_types(graph)
    print(cs_types)
    
    translated_instance["max_q"] = get_battery_energy_capacity(graph) * 1000
    translated_instance["t_max"] = get_maximum_travel_time(graph)
    translated_instance["process_times"] = get_node_service_times(graph)
    translated_instance["css"] = [
        # {"node_id": station, "cs_type": cstype_to_index(cs_type)}
        {"node_id": station, "cs_type": cs_type}
        for (station, cs_type) in zip(stations, cs_types)
    ]
    translated_instance["breakpoints_by_type"] = [
        {
            # "cs_type": cstype_to_index(cs_type),
            "cs_type": cs_type,
            "time": get_time_breakpoints(graph, cs_type),
            "charge": get_energy_breakpoints(graph, cs_type),
        } for cs_type in np.unique(cs_types)
    ]
    translated_instance["energy_matrix"] = get_energy_consumption_matrix(graph)
    translated_instance["time_matrix"] = get_travel_time_matrix(graph)

    return translated_instance


