import torch

from optlearn.nets import masking_utils


class evrpMasker():
    def __init__(self, problem_parameters, graph_parameters, vehicle_parameters={}):

        self._set_problem_parameters(
            problem_parameters=problem_parameters
        )
        self._set_graph_parameters(
            graph_parameters=graph_parameters
        )
        self._set_vehicle_parameters(
            vehicle_parameters=vehicle_parameters
        )
        
        return None

    def _get_num_problems(self, problem_parameters):
        """ Get the current number of problems """

        num_problems = problem_parameters.get("num_problems")
        if num_problems is None:
            raise ValueError("Must specify to masker the number of problems!")

        return num_problems

    def _set_num_problems(self, problem_parameters):
        """ Set the number of problems """

        num_problems = self._get_num_problems(problem_parameters)
        self._num_problems = num_problems
        
        return None

    def _set_problem_parameters(self, problem_parameters):
        """ Set the problem parameters """

        self._set_num_problems(problem_parameters)

        return None

    def _get_num_customers(self, graph_parameters):
        """ Get the number of customers """

        num_customers = graph_parameters.get("num_customers")
        if num_customers is None:
            raise ValueError("Number of customers must be nonzero!")

        return num_customers

    def _set_num_customers(self, graph_parameters):
        """ Set the number of customers """

        num_customers = self._get_num_customers(graph_parameters)
        self._num_customers = num_customers

        return None

    def _get_num_stations(self, graph_parameters):
        """ Get the number of stations """

        num_stations = graph_parameters.get("num_stations")
        if num_stations is None:
            num_stations = 0

        return num_stations
        
    def _set_num_stations(self, graph_parameters):
        """ Set the number of stations under consideration """

        num_stations = self._get_num_stations(graph_parameters)
        self._num_stations = num_stations

        return None

    def _set_num_nodes(self):
        """ Set the number of nodes under consideration """

        self._num_nodes = 1 + self._num_customers + self._num_stations

        return None

    def _set_graph_parameters(self, graph_parameters):
        """ Set the graph parameters """

        self._set_num_customers(graph_parameters)
        self._set_num_stations(graph_parameters)
        self._set_num_nodes()

        return None

    def _get_vehicle_capacity_parameter(self, vehicle_parameters):
        """ Get the vehicle carrying capacity parameter and default it if not found """

        vehicle_capacity = vehicle_parameters.get("vehicle_capacity")
        if vehicle_capacity is None:
            vehicle_capacity = 0

        return vehicle_capacity

    def _set_vehicle_capacity(self, vehicle_parameters):
        """ Set the carrying capacity of the vehicle """

        vehicle_capacity = self._get_vehicle_capacity_parameter(vehicle_parameters)
        self._vehicle_capacity = vehicle_capacity

        return None

    def _get_battery_capacity_parameter(self, vehicle_parameters):
        """ Get the battery capacity parameter and default it if not found """

        battery_capacity = vehicle_parameters.get("battery_capacity")
        if battery_capacity is None:
            battery_capacity = None

        return battery_capacity

    def _set_battery_capacity(self, vehicle_parameters):
        """ Set the battery capacity of the vehicle """

        battery_capacity = self._get_battery_capacity_parameter(vehicle_parameters)
        self._battery_capacity = battery_capacity

        return None

    def _get_time_limit_parameter(self, vehicle_parameters):
        """ Get the time limit parameter and default it if not found """

        time_limit = vehicle_parameters.get("time_limit")
        if time_limit is None:
            time_limit = 0

        return time_limit

    def _set_time_limit(self, vehicle_parameters):
        """ Set the time limit for a route """

        time_limit = self._get_time_limit_parameter(vehicle_parameters)
        self._time_limit = time_limit

        return None

    def _set_vehicle_parameters(self, vehicle_parameters):
        """ Set the parameters for the vehicles """

        self._set_vehicle_capacity(vehicle_parameters)
        self._set_battery_capacity(vehicle_parameters)
        self._set_time_limit(vehicle_parameters)

        return None
    
    def _build_customer_set_mask(self):
        """ Build a mask the disincludes all of the customer nodes """
        
        customer_set_mask = masking_utils.are_nodes_customer_nodes(
            self._num_problems,
            self._num_nodes,
            self._num_customers,
        )
        
        return customer_set_mask
    
    def _build_station_set_mask(self):
        """ Build a mask that disincludes all of the station nodes """
        
        station_set_mask = masking_utils.are_nodes_station_nodes(
            self._num_problems,
            self._num_nodes,
            self._num_customers,
        )
        
        return station_set_mask
    
    def _build_time_reachability_mask(self):
        """ Build a mask that disincludes all nodes that are not directly time-reachable """
        
        time_reachability_mask = masking_utils.are_nodes_time_reachable(
            self._time_tensor,
            self._time_remaining,
            self._batch_indices,
            self._last_nodes,
        )
        
        return time_reachability_mask
    
    def _build_energy_reachability_mask(self):
        """ Build a mask that disincludes all nodes that are not directly time-reachable """
        
        energy_reachability_mask = masking_utils.are_nodes_energy_reachable(
            self._energy_tensor,
            self._energy_remaining,
            self._batch_indices,
            self._last_nodes,
        )
        
        return energy_reachability_mask
    
    def _build_time_returnability_mask(self):
        """ Build a mask that disincludes nodes we don't have time to travel to the depot through """
        
        time_returnability_mask = masking_utils.are_nodes_time_returnable(
            self._taba_service_time_tensor, 
            self._time_remaining, 
            self._batch_indices, 
            self._last_nodes,
        )
        
        return time_returnability_mask
    
    def _build_energy_returnability_mask(self):
        """ Build a mask that disincludes nodes we don't have energy to travel to the depot through """
        
        energy_returnability_mask = masking_utils.are_nodes_energy_returnable(
            self._taba_energy_tensor,
            self._energy_remaining,
            self._batch_indices,
            self._last_nodes,
        )
        
        return energy_returnability_mask
    
    def _build_direct_station_detourability_mask(self):
        """ Build a mask that disincludes nodes we don't have energy to travel to a station through """
        
        direct_station_detourability_mask = masking_utils.are_any_detour_stations_energy_reachable(
            self._tac_energy_tensor, 
            self._energy_remaining, 
            self._batch_indices, 
            self._last_nodes,
        )
        
        return direct_station_detourability_mask
        
    def _build_direct_station_returnability_mask(self):
        """ Build a  mask masking stations we cannot directly reach from depot with a full battery """
        
        station_returnability_mask = masking_utils.are_any_detour_stations_directly_energy_returnable(
            self._energy_tensor, 
            self._num_customers,
            self._battery_capacity,
        )
        
        return station_returnability_mask
        
    def _build_detour_time_rechargeability_mask(self):
        
        detour_time_rechargeability_mask = masking_utils.are_any_detour_stations_time_rechargeable(
            self._energy_tensor, 
            self._time_tensor, 
            self._tac_service_time_tensor, 
            self._tac_energy_tensor, 
            self._station_features, 
            self._num_customers, 
            self._energy_remaining, 
            self._time_remaining, 
            self._batch_indices, 
            self._last_nodes
        )
        
        return detour_time_rechargeability_mask
    
    def _build_time_rechargeability_mask(self):
        
        time_rechargeability_mask = masking_utils.are_station_nodes_time_rechargeable(
            self._taba_service_time_tensor, 
            self._energy_tensor, 
            self._station_features, 
            self._num_customers, 
            self._time_remaining, 
            self._energy_remaining, 
            self._batch_indices, 
            self._last_nodes
        )
        
        return time_rechargeability_mask
    
    def _build_visited_customer_mask(self):
        
        visited_customer_mask = masking_utils.was_customer_visited(
            self._energy_tensor, 
            self._visited_customers
        )
        
        return visited_customer_mask
    
    def _build_just_visited_mask(self):
        
        just_visited_mask = masking_utils.was_node_visited_last(
            self._energy_tensor, 
            self._batch_indices, 
            self._last_nodes
        )
        
        return just_visited_mask

    def _build_blank_mask(self):
        """ Build a blank mask """

        output_mask = torch.zeros(
            (self._num_problems, self._num_customers + self._num_stations + 1)
        ).bool()

        return output_mask
    
    def _build_output_mask(self):
        
        time_reachability_mask = self._build_time_reachability_mask()
        time_returnability_mask = self._build_time_returnability_mask()
        energy_reachability_mask = self._build_energy_reachability_mask()
        energy_returnability_mask = self._build_energy_returnability_mask()
        direct_station_detourability_mask = self._build_direct_station_detourability_mask()
        direct_station_returnability_mask = self._build_direct_station_returnability_mask()
        detour_time_rechargeability_mask = self._build_detour_time_rechargeability_mask()
        time_rechargeability_mask = self._build_time_rechargeability_mask()
        customer_set_mask = self._build_customer_set_mask()
        station_set_mask = self._build_station_set_mask()
        visited_customer_mask = self._build_visited_customer_mask()
        just_visited_mask = self._build_just_visited_mask()
        
        output_mask = self._build_blank_mask()
        
        output_mask[time_reachability_mask == False] = True
        output_mask[energy_reachability_mask == False] = True
        output_mask[time_returnability_mask == False] = True
        output_mask[
            (output_mask == False) *
            (energy_returnability_mask == False) *
            (customer_set_mask == True) *
            (direct_station_detourability_mask == False)
        ] = True
        output_mask[
            (output_mask == False) *
            (energy_returnability_mask == False) *
            (customer_set_mask == True) *
            (direct_station_detourability_mask == True) *
            (direct_station_returnability_mask == False)
        ] = True
        output_mask[
            (output_mask == False) *
            (energy_returnability_mask == False) *
            (customer_set_mask == True) *
            (direct_station_detourability_mask == True) *
            (direct_station_returnability_mask == True) *
            (detour_time_rechargeability_mask == False)
        ] = True
        output_mask[
            (station_set_mask == True) *
            (output_mask == False) *
            (time_rechargeability_mask == False)
        ] = True
        output_mask[visited_customer_mask] = True
        output_mask[just_visited_mask] = True
        
        return output_mask

    def _are_all_customers_visited(self):
        
        are_all_customers_visited = self._build_visited_customer_mask()
        are_all_customers_visited = are_all_customers_visited[:, 1:self._num_customers+1].prod(1).bool()
        
        return are_all_customers_visited
