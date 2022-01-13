import numpy as np

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


def get_service_times(graph):
    """ Get the service times for each of the nodes """

    service_times = [graph.nodes[node].get("service_time") or 0 for node in graph.nodes]

    return service_times


def get_cs_types(graph):
    """ Get the cs_types for the station nodes """

    stations = feature_utils.get_stations(graph)
    cs_types = [graph.nodes[station].get("cs_type") for station in stations]

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
    
    weights = np.array(graph_utils.get_all_weights(clone_graph))
    weights = weights.reshape(len(clone_graph.nodes), len(clone_graph.nodes))
    consumption_rate = get_energy_consumption_rate(clone_graph) 
    consumption_matrix = (weights * consumption_rate * 1000).tolist()

    return consumption_matrix


def get_travel_time_matrix(graph):
    """ Get the travel time matrix for the given graph """

    clone_graph = graph_utils.clone_graph(graph)
    clone_graph = clone_graph.to_directed()
    clone_graph.add_weighted_edges_from([(node, node, 0) for node in graph.nodes])
    
    weights = np.array(graph_utils.get_all_weights(clone_graph))
    weights = weights.reshape(len(clone_graph.nodes), len(clone_graph.nodes))
    average_velocity = get_average_vehicle_velocity(clone_graph) 
    travel_time_matrix = (weights / average_velocity).tolist()

    return travel_time_matrix


def graph_to_translated_instance(graph):
    """ Convert the graph to an FRVCPY instance """

    translated_instance = {}

    stations = feature_utils.get_stations(graph)
    cs_types = get_cs_types(graph)
    
    translated_instance["max_q"] = get_battery_energy_capacity(graph) * 1000
    translated_instance["t_max"] = get_maximum_travel_time(graph)
    translated_instance["process_times"] = get_service_times(graph)
    translated_instance["css"] = [
        {"node_id": station, "cs_type": cs_type} for (station, cs_type) in zip(stations, cs_types)
    ]
    translated_instance["breakpoints_by_type"] = [
        {
            "cs_type": cs_type,
            "time": get_time_breakpoints(graph, cs_type),
            "charge": get_energy_breakpoints(graph, cs_type),
        } for cs_type in np.unique(cs_types)
    ]
    translated_instance["energy_matrix"] = get_energy_consumption_matrix(graph)
    translated_instance["time_matrix"] = get_travel_time_matrix(graph)

    return translated_instance


