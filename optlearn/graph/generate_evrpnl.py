import itertools

import numpy as np
import networkx as nx

from optlearn.io.vrp import vrp_utils
from optlearn.graph import graph_utils


_slow_energy_breakpoints = [0.0, 13600.0, 15200.0, 16000.0]
_normal_energy_breakpoints = [0.0, 13600.0, 15200.0, 16000.0]
_fast_energy_breakpoints = [0.0, 13600.0, 15200.0, 16000.0]
_slow_time_breakpoints = [0.0, 1.26, 1.54, 2.04]
_normal_time_breakpoints = [0.0, 0.62, 0.77, 1.01]
_fast_time_breakpoints = [0.0, 0.31, 0.39, 0.51]


def initialise_coordinate_dict(graph):
    """ Initialise a coordinate dictionary for the graph """

    graph.graph["coord_dict"] = {}

    return graph

def initialise_decimals(graph):
    """ Initialise a decimals attribute for the graph """

    graph.graph["decimals"] = 14

    return graph


def initialise_fleet(graph):
    """ Initialise a fleet for the graph """

    graph.graph["fleet"] = {}

    return graph


def initialise_node_types(graph):
    """ Initialise node types in the given graph """

    graph.graph["node_types"] = {
        "customer": [],
        "station": [],
        "depot": [],
    }

    return graph


def generate_breakpoints(energy_breakpoints, time_breakpoints):
    """ Generate breakpoints attributes for the graph """

    breakpoints = [{"battery_level": battery_level, "charging_time": charging_time}
                   for (battery_level, charging_time) in zip(energy_breakpoints, time_breakpoints)]

    return breakpoints


def generate_charging_function(cs_type, energy_breakpoints, time_breakpoints):
    """  Generate a charging function with the given breakpoints and type name """

    charging_function_attributes = {
        "cs_type": cs_type,
        "breakpoints": generate_breakpoints(energy_breakpoints, time_breakpoints)
    }

    return charging_function_attributes


def generate_slow_charging_function():
    """ Generate a slow charging function """

    slow_function_attributes = generate_charging_function(
        cs_type="slow",
        energy_breakpoints=_slow_energy_breakpoints,
        time_breakpoints=_slow_time_breakpoints,
    )

    return slow_function_attributes


def generate_normal_charging_function():
    """ Generate a normal charging function """

    normal_function_attributes = generate_charging_function(
        cs_type="normal",
        energy_breakpoints=_normal_energy_breakpoints,
        time_breakpoints=_normal_time_breakpoints,
    )

    return normal_function_attributes


def generate_fast_charging_function():
    """ Generate a fast charging function """

    fast_function_attributes = generate_charging_function(
        cs_type="fast",
        energy_breakpoints=_fast_energy_breakpoints,
        time_breakpoints=_fast_time_breakpoints,
    )

    return fast_function_attributes


def generate_standard_charging_functions():
    """ Generate the standard charging functions """

    charging_functions = {
        "slow": generate_slow_charging_function(),
        "normal": generate_normal_charging_function(),
        "fast": generate_fast_charging_function(),
    }

    return charging_functions
    

def generate_vehicle(max_travel_time, speed_factor, capacity, consumption_rate, depot_node):
    """ Generate vehicle characteristics information attributes for the graph """

    vehicle_attributes = {
        "max_travel_time": max_travel_time,
        "speed_factor": speed_factor,
        "departure_node": depot_node,
        "arrival_node": depot_node,
        "capacity": None,
        "type": 0,
    }

    return vehicle_attributes
    

def initialise_num_vehicles(graph):
    """ Initialise the number of vehicles """

    graph.graph["num_vehicles"] = None

    return graph


def build_evrpnl_graph():
    """ Build a blank EVRP-NL graph """

    graph = graph_utils.create_graph()
    graph = initialise_coordinate_dict(graph)
    graph = initialise_decimals(graph)
    graph = initialise_fleet(graph)
    graph = initialise_num_vehicles(graph)
    graph = initialise_node_types(graph)
    
    return graph


def generate_depot_coordinate_attributes(x_limits=(0,100), y_limits=(0,100)):
    """ Generate a coordinate for the depot """

    x_ordinate = round(np.random.uniform(*x_limits), 2)
    y_ordinate = round(np.random.uniform(*y_limits), 2)
    coordinate_attributes = {"cx": x_ordinate, "cy": y_ordinate}
    
    return coordinate_attributes


def generate_depot_type_attribute():
    """ Generate a type attribute for the depot """

    type_attribute = {"type": 0}

    return type_attribute


def generate_depot_cs_type_attribute(cs_type):
    """ Generate a CS type attribute for the depot """

    cs_type_attribute = {"cs_type": cs_type}

    return cs_type_attribute


def generate_depot_id_attribute(maximum_id):
    """ Generate an ID attribute for the depot """

    id_attribute = {"id": int(maximum_id + 1)}

    return id_attribute


def generate_depot_attributes(x_limits, y_limits, cs_type, maximum_id):
    """ Generate attributes for the given depot """

    type_attribute = generate_depot_type_attribute()
    cs_type_attribute = generate_depot_cs_type_attribute(cs_type)
    id_attribute = generate_depot_id_attribute(maximum_id)
    coordinate_attributes = generate_depot_coordinate_attributes(x_limits, y_limits)

    depot_attributes = {
        **type_attribute,
        **cs_type_attribute,
        **id_attribute,
        **coordinate_attributes,
    }

    return depot_attributes


def generate_new_key(graph):
    """ Generate a new key for the next node """

    new_key = len(graph.nodes)

    return new_key


def update_depot_node_types(graph, depot_node):
    """ Update the node types with the given depot """

    graph.graph["node_types"]["depot"].append(depot_node)

    return graph


def update_coordinate_dict(graph, depot_node, coordinate):
    """ Update the graph coordinate dict with the given coordinate """

    graph.graph["coord_dict"][depot_node] = coordinate

    return graph


def generate_depot(graph, x_limits, y_limits, cs_type):
    """ Generate a depot for the given graph """

    depot_node = generate_new_key(graph)
    depot_attributes = generate_depot_attributes(
        x_limits,
        y_limits,
        cs_type,
        depot_node,
    )
    graph.add_nodes_from([(depot_node, depot_attributes)])
    graph = update_depot_node_types(graph, depot_node)
    coordinate = [depot_attributes["cx"], depot_attributes["cy"]]
    graph = update_coordinate_dict(graph, depot_node, coordinate)
    
    return graph


def generate_reachable_coordinate(coordinate, reachability_radius):
    """ Generate a new coordinate that is reachable from the given coordinate """

    x_limits = (coordinate[0] - reachability_radius, coordinate[0] + reachability_radius)
    x_limits = tuple([float(np.clip(item, 0, 120)) for item in x_limits])
    x_ordinate = np.random.uniform(*x_limits)
    y_limit = np.sqrt(reachability_radius ** 2 - (coordinate[0] - x_ordinate) ** 2)
    y_limits = (coordinate[1] - y_limit, coordinate[1] + y_limit)
    y_limits = tuple([float(np.clip(item, 0, 120)) for item in y_limits])
    y_ordinate = np.random.uniform(*y_limits)
    coordinate = [x_ordinate, y_ordinate]
    
    return coordinate


def generate_strictly_reachable_coordinate(coordinate, reachability_radius):
    """ Generate a new coordinate that is strictly reachable from the given coordinate """

    x_limits = (coordinate[0] - (reachability_radius/3), coordinate[0] + (reachability_radius/3))
    x_limits = tuple([float(np.clip(item, 0, 120)) for item in x_limits])
    x_ordinate = np.random.uniform(*x_limits)
    y_limit = np.sqrt((reachability_radius/3) ** 2 - (coordinate[0] - x_ordinate) ** 2)
    y_limits = (coordinate[1] - y_limit, coordinate[1] + y_limit)
    y_limits = tuple([float(np.clip(item, 0, 120)) for item in y_limits])
    y_ordinate = np.random.uniform(*y_limits)
    coordinate = [x_ordinate, y_ordinate]

    return coordinate


def generate_station_coordinate_attributes(parent_coordinate, reachability_radius):
    """ Generate a coordinate for the station """

    coordinate = generate_reachable_coordinate(parent_coordinate, reachability_radius)
    coordinate_attributes = {"cx": round(coordinate[0], 2), "cy": round(coordinate[1], 2)}
    
    return coordinate_attributes


def generate_station_type_attribute():
    """ Generate a type attribute for the station """

    type_attribute = {"type": 2}

    return type_attribute


def generate_station_cs_type_attribute(cs_type):
    """ Generate a CS type attribute for the station """

    cs_type_attribute = {"cs_type": cs_type}

    return cs_type_attribute


def generate_station_id_attribute(maximum_id):
    """ Generate an ID attribute for the station """

    id_attribute = {"id": int(maximum_id + 1)}

    return id_attribute


def generate_station_attributes(parent_coordinate, reachability_radius, cs_type, maximum_id):
    """ Generate attributes for the given station """

    type_attribute = generate_station_type_attribute()
    cs_type_attribute = generate_station_cs_type_attribute(cs_type)
    id_attribute = generate_station_id_attribute(maximum_id)
    coordinate_attributes = generate_station_coordinate_attributes(
        parent_coordinate,
        reachability_radius
    )
    station_attributes = {
        **type_attribute,
        **cs_type_attribute,
        **id_attribute,
        **coordinate_attributes,
    }

    return station_attributes


def update_station_node_types(graph, station_node):
    """ Update the node types with the given station """

    graph.graph["node_types"]["station"].append(station_node)

    return graph


def generate_station(graph, parent_coordinate, reachability_radius, cs_type):
    """ Generate a station within the reachability_radius of the parent coordinate """

    station_node = generate_new_key(graph)
    station_attributes = generate_station_attributes(
        parent_coordinate,
        reachability_radius,
        cs_type,
        station_node,
    )
    graph.add_nodes_from([(station_node, station_attributes)])
    graph = update_station_node_types(graph, station_node)
    coordinate = [station_attributes["cx"], station_attributes["cy"]]
    graph = update_coordinate_dict(graph, station_node, coordinate)
    
    return graph


def generate_customer_coordinate_attributes(parent_coordinate, reachability_radius):
    """ Generate a coordinate for the customer """

    coordinate = generate_strictly_reachable_coordinate(parent_coordinate, reachability_radius)
    coordinate_attributes = {"cx": round(coordinate[0], 2), "cy": round(coordinate[1], 2)}
    
    return coordinate_attributes


def generate_customer_type_attribute():
    """ Generate a type attribute for the customer """

    type_attribute = {"type": 1}

    return type_attribute


def generate_customer_cs_type_attribute():
    """ Generate a CS type attribute for the customer """

    cs_type_attribute = {"cs_type": None}

    return cs_type_attribute


def generate_customer_quantity_attribute(limits=None):
    """ Generate a quantity attribute for the customer """

    if limits is not None:
        quantity = round(np.random.uniform(*limits), 2)
    else:
        quantity = None
    quantity = {"quantity": None}

    return quantity


def generate_customer_service_time_attribute(limits=None):
    """ Generate a service time attribute for the customer """

    if limits is not None:
        service_time = round(np.random.uniform(*limits), 2)
    else:
        service_time = 0.5
    service_time = {"service_time": service_time}

    return service_time


def generate_customer_id_attribute(maximum_id):
    """ Generate an ID attribute for the customer """

    id_attribute = {"id": int(maximum_id + 1)}

    return id_attribute


def generate_customer_attributes(parent_coordinate, reachability_radius,
                                 service_time_limits, quantity_limits, maximum_id):
    """ Generate attributes for the given customer """

    type_attribute = generate_customer_type_attribute()
    cs_type_attribute = generate_customer_cs_type_attribute()
    quantity_attribute = generate_customer_quantity_attribute(quantity_limits)
    service_time_attribute = generate_customer_service_time_attribute(service_time_limits)    
    id_attribute = generate_customer_id_attribute(maximum_id)
    coordinate_attributes = generate_customer_coordinate_attributes(
        parent_coordinate,
        reachability_radius
    )
    customer_attributes = {
        **type_attribute,
        **cs_type_attribute,
        **quantity_attribute,
        **service_time_attribute,
        **id_attribute,
        **coordinate_attributes,
    }

    return customer_attributes


def update_customer_node_types(graph, customer_node):
    """ Update the node types with the given customer """

    graph.graph["node_types"]["customer"].append(customer_node)

    return graph


def generate_customer(graph, parent_coordinate, reachability_radius,
                      service_time_limits, quantity_limits):
    """ Generate a customer within the strict reachability radius of the parent coordinate """

    customer_node = generate_new_key(graph)
    customer_attributes = generate_customer_attributes(
        parent_coordinate,
        reachability_radius,
        service_time_limits,
        quantity_limits, 
        customer_node,
    )
    graph.add_nodes_from([(customer_node, customer_attributes)])
    graph = update_customer_node_types(graph, customer_node)
    coordinate = [customer_attributes["cx"], customer_attributes["cy"]]
    graph = update_coordinate_dict(graph, customer_node, coordinate)
    
    return graph


class generateProblem():

    def __init__(self, service_time_limits=(0.4999, 0.5001),
                 quantity_limits=None, max_travel_time=10.0,
                 energy_consumption_rate=0.125,
                 battery_energy_capacity=16.0,
                 use_standard_functions=True,
                 x_limits=(0, 100), y_limits=(0,100),
                 speed_factor=40,
    ):

        self.service_time_limits = service_time_limits
        self.quantity_limits = quantity_limits
        self.battery_energy_capacity = battery_energy_capacity
        self.energy_consumption_rate = energy_consumption_rate
        self.max_travel_time = max_travel_time
        self.use_standard_functions = use_standard_functions
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.speed_factor = speed_factor

        self._initialise_function_storage()
    

    def _initialise_function_storage(self):
        """ Initialise storage for the charging functions """

        if self.use_standard_functions:
            self.functions = generate_standard_charging_functions()
        else:
            self.functions = {}
            
        return None
    
    def set_function(self, function_name, energy_breakpoints, time_breakpoints):
        """ Set a charging function with the given name and breakpoints """

        self.functions[function_name] = {
            "energy_breakpoints": energy_breakpoints,
            "time_breakpoints": time_breakpoints,
        }

        return None

    def set_service_time_limits(self, service_time_limits):
        """ Set service time limits for the problem generator """

        self.service_time_limits = service_time_limits

        return None

    def set_service_time_limits(self, quantity_limits):
        """ Set quantity limits for the problem generator """
        
        self.quantity_limits = quantity_limits
    
        return None

    def set_max_travel_time(self, max_travel_time):
        """ Set the maximum travel time for the problem generator """
        
        self.max_travel_time = max_travel_time
    
        return None

    def set_energy_consumption_rate(self, energy_consumption_rate):
        """ Set the energy consumption rate for the problem generator """
        
        self.energy_consumption_rate = energy_consumption_rate
    
        return None

    def set_x_limits(self, x_limits):
        """ Set horizontal positional limits for depots for the problem generator """
        
        self.x_limits = x_limits
    
        return None

    def set_y_limits(self, y_limits):
        """ Set vertical positional limits for depots for the problem generator """
        
        self.y_limits = y_limits
    
        return None

    def build_graph(self):
        """ Build a fresh graph """

        graph = build_evrpnl_graph()

        return graph

    def add_depot_to_graph(self, graph):
        """ Add a depot to the graph """

        cs_type = self.select_random_cs_type()
        graph = generate_depot(graph, self.x_limits, self.y_limits, cs_type)

        return graph

    def add_depots_to_graph(self, graph, number_of_depots):
        """ Add the given number of depots to the problem """

        for num in range(number_of_depots):
            graph = self.add_depot_to_graph(graph)

        return graph
    
    def select_random_cs_type(self):
        """ Select a random CS type from storage """

        cs_types = list(self.functions.keys())
        random_cs_type = np.random.choice(cs_types)

        return random_cs_type

    def compute_reachability_radius(self):
        """ Compute the reachability_radius of the given graph """

        battery_capacity = self.battery_energy_capacity
        energy_consumption_rate = self.energy_consumption_rate
        reachability_radius = battery_capacity / energy_consumption_rate

        return reachability_radius

    def select_random_station(self, graph, depot_priority=2):
        """ Select a random station from the graph """

        depot_nodes = graph.graph["node_types"]["depot"] * depot_priority
        station_nodes = graph.graph["node_types"]["station"]
        stations = depot_nodes + station_nodes
        random_station = np.random.choice(stations)

        return random_station

    def add_station_to_graph(self, graph, reachable_node, cs_type):
        """ Add a station to the graph that is reachable for the given node """

        coordinate = graph.graph["coord_dict"][reachable_node]
        reachability_radius = self.compute_reachability_radius()
        graph = generate_station(graph, coordinate, reachability_radius, cs_type)

        return graph

    def add_random_stations_to_graph(self, graph, number_of_stations):
        """ Add random stations to the graph """

        for num in range(number_of_stations):
            random_station = self.select_random_station(graph)
            cs_type = self.select_random_cs_type()
            graph = self.add_station_to_graph(graph, random_station, cs_type)

        return graph

    def add_customer_to_graph(self, graph, reachable_node):
        """ Add a customer into the strictly reachable region of the given node """

        coordinate = graph.graph["coord_dict"][reachable_node]
        reachability_radius = self.compute_reachability_radius()
        service_time_limits = self.service_time_limits
        quantity_limits = self.quantity_limits
        graph = generate_customer(graph, coordinate, reachability_radius,
                                  service_time_limits, quantity_limits)

        return graph

    def add_random_customers_to_graph(self, graph, number_of_customers):
        """ Add the given number of customers to the graph """

        for num in range(number_of_customers):
            random_station = self.select_random_station(graph)
            graph = self.add_customer_to_graph(graph, random_station)
            
        return graph

    def add_vehicle_to_graph(self, graph, depot_node):
        """ Add a vehicle to the graph """

        graph.graph["fleet"]["vehicle_0"] = generate_vehicle(
            self.max_travel_time,
            self.speed_factor,
            self.battery_energy_capacity,
            self.energy_consumption_rate,
            depot_node
        )

        return graph

    def add_vehicles_to_graph(self, graph):
        """ Add vehicles to the depots """

        depot_nodes = graph.graph["node_types"]["depot"]
        for depot_node in depot_nodes:
            graph = self.add_vehicle_to_graph(graph, depot_node)

        return graph

    def add_functions_to_graph(self, graph):
        """ Add the charging functions to the graph """

        graph.graph["fleet"]["functions_0"] = {
            "consumption_rate": None,
            "battery_capacity": None,
            "charging_functions": self.functions
        }

        return graph

    def add_complete_edges_to_graph(self, graph):
        """ Add the complete weighted edge set to the graph """

        nodes = np.sort(list(graph.nodes))
        coordinates = [graph.graph["coord_dict"][node] for node in nodes]
        edges = np.array(list(itertools.combinations(nodes, 2)))
        weights = vrp_utils.compute_pairwise_distances(coordinates, vrp_utils.euclidean, 16)
        graph.add_weighted_edges_from([(*edge, weight)
                                       for ((edge), weight) in zip(edges, weights)])

        return graph
        
    def generate_random_problem(self, number_of_depots=1, number_of_stations=2,
                                number_of_customers=10):
        """ Generate a random problem with the given number of depots, stations and customers """

        graph = self.build_graph()
        graph = self.add_depots_to_graph(graph, number_of_depots)
        graph = self.add_random_stations_to_graph(graph, number_of_stations)
        graph = self.add_random_customers_to_graph(graph, number_of_customers)
        graph = self.add_vehicles_to_graph(graph)
        graph = self.add_functions_to_graph(graph)
        graph = self.add_complete_edges_to_graph(graph)
        
        return graph
