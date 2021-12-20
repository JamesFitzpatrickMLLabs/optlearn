import numpy as np
import networkx as nx

from optlearn.mip.routing import problem_builder


class gvrpProblemBuilder(problem_builder.basicProblemBuilder):

    def __init__(self, solver_package, problem_type, is_directed):

        problem_builder.basicProblemBuilder.__init__(self, solver_package, problem_type, is_directed)

        self.initialise_problem()
        self._initialise_arc_travel_variable_graph()
        self._initialise_node_time_variable_graph()
        self._initialise_node_fuel_variable_graph()
        self.assume_hyperparameters()
        self.warn_hyperparmaters()

    def assume_hyperparameters(self):
        """ Assume some of the hyperparmaeters """

        self.customer_service_time = 0.5
        self.station_service_time = 0.25

    def check_hyperparameters(self):
        """ Make sure everything is set up correctly """

        pass

    def warn_hyperparmaters(self):
        """ Warn the user about the hyper-parameters """

        print(f"WARNING! Attribute customer_service_time set to {self.customer_service_time}")
        print(f"WARNING! Attribute station_service_time set to {self.station_service_time}")
        
    def get_fuel_tank_capacity(self, graph):
        """ Get the fuel capacity for the given graph. Assume fleet homogeneity. """

        first_vehicle_info = graph.graph["fleet"]["vehicle_0"]
        fuel_tank_capacity = first_vehicle_info["fuel_tank_capacity"]

        return fuel_tank_capacity

    def get_average_velocity(self, graph):
        """ Get the average velocity of a vehicle in the given graph. Assume homogeneity. """

        first_vehicle_info = graph.graph["fleet"]["vehicle_0"]
        average_velocity = first_vehicle_info["average_velocity"]

        return average_velocity

    def get_fuel_consumption_rate(self, graph):
        """ Get the fuel consumption rate of a vehicle in the given graph. Assume homogeneity. """
        
        first_vehicle_info = graph.graph["fleet"]["vehicle_0"]
        fuel_consumption_rate = first_vehicle_info["fuel_consumption_rate"]

        return fuel_consumption_rate

    def get_maximum_travel_time(self, graph):
        """ Get the maximum travel time allowed for any one vehicle. Assume homogeneity. """

        first_vehicle_info = graph.graph["fleet"]["vehicle_0"]
        maximum_travel_time = first_vehicle_info["tour_length"]

        return maximum_travel_time
        
    def distance_to_duration(self, distance, speed):
        """ Convert the given travel distance to a duration. Assume metric units. """
        
        duration = distance / speed

        return duration

    def distances_to_durations(self, distances, speed):
        """ Convert the given travel distance to a duration. Assume metric units. """
        
        durations =  [distance / speed for distance in distances]

        return distances

    def build_node_service_time_graph(self, graph):
        """ Build a graph containing the node service times """

        self.service_time_graph = nx.Graph()
        self.service_time_graph.add_nodes_from(
            self.get_customers(graph),
            service_time=self.customer_service_time,
        )
        self.service_time_graph.add_nodes_from(
            self.get_stations(graph),
            service_time=self.station_service_time,
        )
        self.service_time_graph.add_nodes_from(
            self.get_depots(graph),
            service_time=0,
        )

    def get_node_service_time(self, graph, node):
        """ Get the service time for a given node """

        if not hasattr(self, "service_time_graph"):
            self.build_node_service_time_graph(graph)
        service_time = self.service_time_graph.nodes[node]["service_time"]

        return service_time

    def get_node_service_times(self, graph, nodes):
        """ Get the service times for the given nodes """

        service_times = [self.get_node_service_time(graph, node) for node in nodes]
        
        return service_times
    
    def get_travel_time(self, graph, first_node, second_node):
        """ Get the travel time between the two given nodes """

        velocity = self.get_average_velocity(graph)
        weight = graph[first_node][second_node]["weight"]
        time = self.distance_to_duration(weight, velocity)

        return time

    def get_fuel_consumption(self, graph, first_node, second_node):
        """ Get the fuel consumption between the two given nodes """

        fuel_consumption_rate = self.get_fuel_consumption_rate(graph)
        weight = graph[first_node][second_node]["weight"]
        fuel_consumption = weight * fuel_consumption_rate

        return fuel_consumption

    def build_primary_time_tracking_constraint(self, graph, first_node, second_node):
        """ Build the main time tracking constraint for the GVRP """

        first_time_variable = self.get_node_time_variable_from_storage(first_node)
        second_time_variable = self.get_node_time_variable_from_storage(second_node)
        travel_variable = self.get_travel_variable_from_storage((first_node, second_node))
        travel_time = self.get_travel_time(graph, first_node, second_node)
        service_time = self.get_node_service_time(graph, first_node)
        # service_time = self.get_node_service_time(graph, second_node)
        maximum_travel_time = self.get_maximum_travel_time(graph)

        lhs = second_time_variable
        rhs = first_time_variable + (travel_time + service_time) * travel_variable
        # rhs = first_time_variable + (travel_time - service_time) * travel_variable
        rhs = rhs - maximum_travel_time * (1 - travel_variable)
        name = f"Time tracking constraint for travel between nodes {first_node} and {second_node}"
        _ = self.set_constraint(lhs, rhs, ">=", name)

        return None

    def build_primary_time_tracking_constraints(self, graph):
        """ Build the time tracking constraint for all pairs of nodes """

        all_nodes = self.get_nodes(graph)
        all_nodes_without_depots = self.get_nodes_without_depots(graph)

        for first_node in all_nodes:
            for second_node in all_nodes_without_depots:
                if first_node != second_node:
                    self.build_primary_time_tracking_constraint(graph, first_node, second_node)

        return None

    def build_depot_time_tracking_constraint(self, graph, depot_node):
        """ Build the time tracking constraint for the depot nodes """

        depot_time_variable = self.get_node_time_variable_from_storage(depot_node)
        maximum_travel_time = self.get_maximum_travel_time(graph)
        
        # rhs = maximum_travel_time
        rhs = 0
        name = f"Time tracking constraint for depot node {depot_node}"
        _ = self.set_constraint(depot_time_variable, maximum_travel_time, "<=", name)

        return None

    def build_depot_time_tracking_constraints(self, graph):
        """ Build the time tracking constraints for all of the depots """

        depot_nodes = self.get_depots(graph)
        for depot_node in depot_nodes:
            self.build_depot_time_tracking_constraint(graph, depot_node)

        return None

    def build_secondary_time_tracking_constraint(self, graph, depot_node, other_node):
        """ Build the second major constraint type for time tracking between the given nodes """

        node_time_variable = self.get_node_time_variable_from_storage(other_node)
        travel_time_from_depot = self.get_travel_time(graph, depot_node, other_node)
        travel_time_to_depot = self.get_travel_time(graph, other_node, depot_node)
        service_time = self.get_node_service_time(graph, other_node)
        maximum_travel_time = self.get_maximum_travel_time(graph)

        first_name = f"Lower bound constraint on travel between nodes {depot_node} and {other_node}"
        second_name = f"Upper bound constraint on travel between nodes {depot_node} and {other_node}"
        _ = self.set_constraint(node_time_variable, travel_time_from_depot, ">=", first_name)
        rhs = maximum_travel_time - (travel_time_to_depot + service_time)
        _ = self.set_constraint(node_time_variable, rhs, "<=", second_name)

        return None

    def build_secondary_time_tracking_constraints(self, graph):
        """ Build the second major constraint type for time tracking between depot and other nodes """

        depot_nodes = self.get_depots(graph)
        all_nodes_without_depots = self.get_nodes_without_depots(graph)

        for depot_node in depot_nodes:
            for other_node in all_nodes_without_depots:
                self.build_secondary_time_tracking_constraint(graph, depot_node, other_node)
        
        return None

    def build_valid_constraint(self, graph, station_node):
        """ Additional constraints on the time for the given station """

        incident_arcs = self.get_inward_incident_arcs(graph, station_node)
        inward_variables = self.get_travel_variables_from_storage(incident_arcs)
        node_time_variable = self.get_node_time_variable_from_storage(station_node)
        
        maximum_travel_time = self.get_maximum_travel_time(graph)

        lhs = node_time_variable
        rhs = self.sum_items(inward_variables) * maximum_travel_time
        constraint = self.set_constraint(lhs, rhs, "<=", f"Extra constraint on {station_node}")

        return None

    def build_valid_constraints(self, graph):
        """ Additional constraints on the times for the stations """

        station_nodes = self.get_stations(graph)
        for station_node in station_nodes:
            self.build_valid_constraint(graph, station_node)

    def build_primary_fuel_tracking_constraint(self, graph, other_node, customer_node):
        """ Build the main fuel tracking constraint for the GVRP """

        other_fuel_variable = self.get_node_fuel_variable_from_storage(other_node)
        customer_fuel_variable = self.get_node_fuel_variable_from_storage(customer_node)
        travel_variable = self.get_travel_variable_from_storage((other_node, customer_node))

        fuel_consumption = self.get_fuel_consumption(graph, other_node, customer_node)
        fuel_tank_capacity = self.get_fuel_tank_capacity(graph)
        
        lhs = customer_fuel_variable
        rhs = other_fuel_variable - (fuel_consumption) * travel_variable
        rhs = rhs + fuel_tank_capacity * (1 - travel_variable)
        name = f"Fuel tracking constraint for travel between nodes {other_node} and {customer_node}"
        constraint = self.set_constraint(lhs, rhs, "<=", name)

        return None

    def build_primary_fuel_tracking_constraints(self, graph):
        """ Build the fuel tracking constraint for all pairs of nodes """

        all_nodes = self.get_nodes(graph)
        customer_nodes = self.get_customers(graph)

        for first_node in all_nodes:
            for second_node in customer_nodes:
                if first_node != second_node:
                    self.build_primary_fuel_tracking_constraint(graph, first_node, second_node)

        return None

    def build_fuel_reset_constraint(self, graph, station_node):
        """ Build the fuel reset constraint for the GVRP """

        station_fuel_variable = self.get_node_fuel_variable_from_storage(station_node)

        fuel_tank_capacity = self.get_fuel_tank_capacity(graph)
        
        name = f"Fuel reset constraint for station {station_node}"
        constraint = self.set_constraint(station_fuel_variable, fuel_tank_capacity, "<=", name)

        return None
    
    def build_fuel_reset_constraints(self, graph):
        """ Build the fuel reset constraints for all station nodes """

        station_nodes = self.get_stations(graph)
        depot_nodes = self.get_depots(graph)
        
        for station_node in station_nodes:
            self.build_fuel_reset_constraint(graph, station_node)
        for depot_node in depot_nodes:
            self.build_fuel_reset_constraint(graph, station_node)

        return None

    def build_secondary_fuel_tracking_constraint(self, graph, depot_node, customer_node, station_node):
        """ Build the secondary fuel tracking constraints for the given nodes """

        customer_fuel_variable = self.get_node_fuel_variable_from_storage(customer_node)
        customer_to_depot_consumption = self.get_fuel_consumption(graph, customer_node, depot_node)
        station_to_depot_consumption = self.get_fuel_consumption(graph, station_node, depot_node)
        customer_to_station_consumption = self.get_fuel_consumption(graph, customer_node, station_node)
        combined_consumption = station_to_depot_consumption + customer_to_station_consumption

        lhs = customer_fuel_variable
        rhs = min([customer_to_depot_consumption, combined_consumption])
        name= f"Return fuel constraint for nodes {customer_node}, {station_node}, and {depot_node}"
        _ = self.set_constraint(lhs, rhs, ">=", name)

        return None

    def build_secondary_fuel_tracking_constraints(self, graph):
        """ Build the secondary fuel tracking constraints for all needed npodes """

        customer_nodes = self.get_customers(graph)
        station_nodes = self.get_stations(graph)
        depot_nodes = self.get_depots(graph)

        for depot_node in depot_nodes:
            for customer_node in customer_nodes:
                for station_node in station_nodes:
                    self.build_secondary_fuel_tracking_constraint(graph, depot_node,
                                                                  customer_node, station_node)

        return None
