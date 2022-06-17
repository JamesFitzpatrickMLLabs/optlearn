import torch

from optlearn.nets import charging
from optlearn.nets import routing_utils


_sample_parameter_dict = {
    "time_limit": 10.0,
    "battery_capacity": 16.0,
    "vehicle_capacity": 50.0,
    "average_velocity": 40.0,
    "energy_consumption_rate": 0.15,
}

_sample_state_dict = {
    "num_problems": 1,
    "num_customers": 10,
    "num_stations": 3,
    "problem_indices": [[0]],
    "last_node_indices": [[0]],
    "time_remaining": torch.tensor([[10.0]]),
    "energy_remaining": torch.tensor([[16.0]]),
    "capacity_remaining": torch.tensor([[50.0]]),
}



def get_time_limit(parameter_dict):
    """ Get the time limit from the parameter dictionary """

    time_limit = parameter_dict.get("time_limit")
    if time_limit is None:
        raise ValueError("No route time limit specified!")

    return time_limit


def get_battery_capacity(parameter_dict):
    """ Get the battery capacity from the parameter dictionary """

    battery_capacity = parameter_dict.get("battery_capacity")
    if battery_capacity is None:
        raise ValueError("No battery capacity specified!")

    return battery_capacity


def get_average_velocity(parameter_dict):
    """ Get the average velocity from the parameter dictionary """

    average_velocity = parameter_dict.get("average_velocity")
    if average_velocity is None:
        raise ValueError("No average velocity specified!")

    return average_velocity


def get_energy_consumption_rate(parameter_dict):
    """ Get the energy consumption rate from the parameter dictionary """

    energy_consumption_rate = parameter_dict.get("energy_consumption_rate")
    if energy_consumption_rate is None:
        raise ValueError("No energy consumption rate specified!")

    return None


def get_num_problems(state_dictionary):
    """ Get the number of problems from the state dictionary """

    num_problems = state_dictionary.get("num_problems")
    if num_problems is None:
        raise ValueError("No number of problems specified!")

    return num_problems


def get_num_customers(state_dictionary):
    """ Get the number of customers from the state dictionary """

    num_customers = state_dictionary.get("num_customers")
    if num_customers is None:
        raise ValueError("No number of customers specified!")

    return num_customers


def get_num_stations(state_dictionary):
    """ Get the number of stations from the state dictionary """

    num_stations = state_dictionary.get("num_stations")
    if num_stations is None:
        raise ValueError("No number of stations specified!")

    return num_stations


def get_problem_indices(state_dictionary):
    """ Get the problem indices from the state dictionary """

    problem_indices = state_dictionary.get("problem_indices")
    if problem_indices is None:
        raise ValueError("The problem indices were not specified!")

    return problem_indices


def get_last_node_indices(state_dictionary):
    """ Get the last node indices from the state dictionary """

    last_node_indices = state_dictionary.get("last_node_indices")
    if last_node_indices is None:
        raise ValueError("The indices of the last nodes were not specified!")

    return last_node_indices


def get_time_remaining(state_dictionary):
    """ Get the time remaining in the current route from the state dictionary """

    time_remaining = state_dictionary.get("time_remaining")
    if time_remaining is None:
        raise ValueError("The time remaining has not been specified!")

    return time_remaining


def get_energy_remaining(state_dictionary):
    """ Get the energy remaining in the current route from the state dictionary """

    energy_remaining = state_dictionary.get("energy_remaining")
    if energy_remaining is None:
        raise ValueError("The energy remaining has not been specified!")

    return energy_remaining


def get_capacity_remaining(state_dictionary):
    """ Get the capacity remaining in the current route from the state dictionary """

    capacity_remaining = state_dictionary.get("capacity_remaining")
    if capacity_remaining is None:
        raise ValueError("The capacity remaining has not been specified!")

    return capacity_remaining


def get_weight_tensor(state_dictionary):
    """ Get the weight tensor from the state dictionary """

    weight_tensor = state_dictionary.get("weight_tensor")
    if weight_tensor is None:
        raise ValueError("Weight tensor has not been specified!")

    return weight_tensor


def compute_time_tensor(state_dictionary, parameter_dictionary):
    """ Using the state and parameter dictionaries, compute the time tensor """

    weight_tensor = get_weight_tensor(state_dictionary)
    average_velocity = get_average_velocity(parameter_dictionary)
    time_tensor = routing_utils.compute_time_tensor(weight_tensor, average_velocity)

    return time_tensor


def compute_energy_tensor(state_dictionary, parameter_dictionary):
    """ Using the state and parameter dictionaries, compute the energy tensor """

    weight_tensor = get_weight_tensor(state_dictionary)
    energy_consumption_rate = get_energy_consumption_rate(parameter_dictionary)
    energy_tensor = routing_utils.compute_energy_tensor(weight_tensor, energy_consumption_rate)

    return energy_tensor


def compute_service_time_tensor(state_dictionary, parameter_dictionary):
    """ Using the state and parameter dictionaries, compute the service time tensor """

    weight_tensor = get_weight_tensor(state_dictionary)
    average_velocity = get_average_velocity(parameter_dictionary)
    num_customers = get_num_customers(parameter_dictionary)
    service_time_tensor = routing_utils.compute_service_time_tensor(
        weight_tensor, num_customers, average_velocity
    )

    return service_time_tensor


def compute_taba_tensor(state_dictionary):
    """ Using the state dictionary, compute the taba tensor """

    weight_tensor = get_weight_tensor(state_dictionary)
    taba_tensor = routing_utils.compute_taba_tensor(weight_tensor)

    return taba_tensor


def compute_taba_time_tensor(state_dictionary, parameter_dictionary):
    """ Using the state and parameter dictionaries, compute the taba time tensor """

    taba_tensor = compute_taba_tensor(state_dictionary)
    average_velocity = get_average_velocity(parameter_dictionary)
    taba_time_tensor = routing_utils.compute_taba_time_tensor(taba_tensor, average_velocity)

    return taba_time_tensor


def compute_taba_service_time_tensor(state_dictionary, parameter_dictionary):
    """ Using the state and parameter dictionaries, compute the taba service time tensor """

    taba_tensor = compute_taba_tensor(state_dictionary)
    num_customers = get_num_customers(state_dictionary)
    average_velocity = get_average_velocity(parameter_dictionary)
    taba_service_time_tensor = routing_utils.compute_taba_service_time_tensor(
        taba_tensor, average_velocity, num_customers,
    )

    return taba_service_time_tensor


def compute_taba_energy_tensor(state_dictionary, parameter_dictionary):
    """ Using the state and parameter dictionaries, compute the taba energy tensor """

    taba_tensor = compute_taba_tensor(state_dictionary)
    energy_consumption_rate = get_energy_consumption_rate(parameter_dictionary)
    taba_energy_tensor = routing_utils.compute_taba_energy_tensor(taba_tensor, energy_consumption_rate)

    return taba_energy_tensor


def compute_tac_tensor(state_dictionary):
    """ Using the state and parameter dictionaries, compute the tac tensor """

    weight_tensor = get_weight_tensor(state_dictionary)
    num_customers = get_num_customers(state_dictionary)
    tac_tensor = routing_utils.compute_tac_tensor(weight_tensor, num_customers)

    return tac_tensor


def compute_tac_time_tensor(state_dictionary, parameter_dictionary):
    """ Using the state and parameter dictionaries, compute the tac time tensor """

    average_velocity = get_average_velocity(parameter_dictionary)
    tac_tensor = compute_tac_tensor(state_dictionary)
    tac_time_tensor = routing_utils.compute_tac_time_tensor(tac_tensor, average_velocity)

    return tac_time_tensor


def compute_tac_service_time_tensor(state_dictionary, parameter_dictionary):
    """ Using the state and parameter dictionaries, compute the tac service time tensor """

    average_velocity = get_average_velocity(parameter_dictionary)
    num_customers = get_num_customers(state_dictionary)
    tac_tensor = compute_tac_tensor(state_dictionary)
    tac_service_time_tensor = routing_utils.compute_tac_service_time_tensor(
        tac_tensor, num_customers, average_velocity,
    )

    return tac_service_time_tensor


def compute_tac_energy_tensor(state_dictionary, parameter_dictionary):
    """ Using the state and parameter dictionaries, compute the tac energy tensor """

    energy_consumption_rate = get_energy_consumption_rate(parameter_dictionary)
    tac_tensor = compute_tac_tensor(state_dictionary)
    tac_energy_tensor = routing_utils.compute_tac_energy_tensor(tac_tensor, energy_consumption_rate)

    return tac_energy_tensor


def indexify_tensor(tensor):
    """ Flatten and turn the tensor into a list """

    indices = tensor.flatten().tolist()

    return indices


# def get_transition_costs(tensor, state_dictionary):
#     """ Get the cost to transition from the current (last) nodes to the next for each problem """

#     problem_indices = get_problem_indices(state_dictionary)
#     last_node_indices = get_last_node_indices(state_dictionary)
#     problem_indices = indexify_tensor(problem_indices)
#     last_node_indices = indexify_tensor(last_node_indices)
#     transition_costs = tensor[problem_indices, last_node_indices]

#     return transition_costs


def get_transition_costs(tensor, problem_indices, last_node_indices):
    """ Get the cost to transition from the current (last) nodes to the next for each problem """

    problem_indices = indexify_tensor(problem_indices)
    last_node_indices = indexify_tensor(last_node_indices)
    transition_costs = tensor[problem_indices, last_node_indices]

    return transition_costs


def are_nodes_customer_nodes(state_dictionary):
    """ For each node, specify if it is a customer node """

    num_problems = get_num_problems(state_dictionary)
    num_customers = get_num_customers(state_dictionary)
    are_nodes_customer_nodes = torch.zeros((num_problems, num_nodes)).bool()
    are_nodes_customer_nodes[:, 1:num_customers+1] = True

    return are_nodes_customer_nodes


def are_nodes_customer_nodes(num_problems, num_nodes, num_customers):
    """ For each node, specify if it is a customer node """

    are_nodes_customer_nodes = torch.zeros((num_problems, num_nodes)).bool()
    are_nodes_customer_nodes[:, 1:num_customers+1] = True

    return are_nodes_customer_nodes


# def are_nodes_station_nodes(state_dictionary):
#     """ For each node, specify if it is a station node """

#     num_problems = get_num_problems(state_dictionary)
#     num_customers = get_num_customers(state_dictionary)
#     are_nodes_station_nodes = torch.zeros((num_problems, num_nodes)).bool()
#     are_nodes_station_nodes[:, num_customers+1:] = True
    
#     return are_nodes_station_nodes


def are_nodes_station_nodes(num_problems, num_nodes, num_customers):
    """ For each node, specify if it is a station node """

    are_nodes_station_nodes = torch.zeros((num_problems, num_nodes)).bool()
    are_nodes_station_nodes[:, num_customers+1:] = True
    
    return are_nodes_station_nodes


def are_nodes_time_reachable(time_tensor, time_remaining, batch_indices, last_nodes):
    """ Figure out if, given the current time remaining, are the other nodes time-reachable """

    travel_times = get_transition_costs(time_tensor, batch_indices, last_nodes)
    next_node_travel_times = time_remaining - travel_times
    are_nodes_time_reachable = next_node_travel_times >= 0
    
    return are_nodes_time_reachable


def are_nodes_energy_reachable(energy_tensor, energy_remaining, batch_indices, last_nodes):
    """ Figure out if, given the current state of charge, are the other nodes energy-reachable """

    travel_costs = get_transition_costs(energy_tensor, batch_indices, last_nodes)
    next_node_travel_costs = energy_remaining - travel_costs
    are_nodes_energy_reachable = next_node_travel_costs >= 0
    
    return are_nodes_energy_reachable


def compute_taba_time_remaining(taba_service_time_tensor, time_remaining, batch_indices, last_nodes):
    """ Compute the tensor to determine if there is enough time to travel to a node and home """

    taba_times = get_transition_costs(taba_service_time_tensor, batch_indices, last_nodes)
    next_node_taba_times = time_remaining - taba_times

    return next_node_taba_times


def are_nodes_time_returnable(taba_service_time_tensor, time_remaining, batch_indices, last_nodes):
    """ Figure out if, given the current time remaining, are the other nodes time-returnable """

    next_node_taba_times = compute_taba_time_remaining(
        taba_service_time_tensor, time_remaining, batch_indices, last_nodes
    )
    are_nodes_taba_time_returnable = next_node_taba_times >= 0

    return are_nodes_taba_time_returnable


def compute_taba_energy_remaining(taba_energy_tensor, energy_remaining, batch_indices, last_nodes):
    """ Compute the tensor to determine if there is enough energy to travel to a node and home """

    taba_costs = get_transition_costs(taba_energy_tensor, batch_indices, last_nodes)
    next_node_taba_costs = energy_remaining - taba_costs

    return next_node_taba_costs

    
def are_nodes_energy_returnable(taba_energy_tensor, energy_remaining, batch_indices, last_nodes):
    """ Figure out if, given the current state of charge, are the other nodes energy-reaturnable """

    next_node_taba_costs = compute_taba_energy_remaining(
        taba_energy_tensor, energy_remaining, batch_indices, last_nodes
    )
    are_nodes_taba_energy_returnable = next_node_taba_costs >= 0
    
    return are_nodes_taba_energy_returnable


def compute_tac_energy_remaining(tac_energy_tensor, energy_remaining, batch_indices, last_nodes):
    """ Compute the tensor to determine if there is enough energy to travel to a station after j """

    tac_energy_differences = (energy_remaining.unsqueeze(-1).unsqueeze(-1) - tac_energy_tensor)
    relevant_energy_values = get_transition_costs(tac_energy_differences, batch_indices, last_nodes)

    return relevant_energy_values


def are_any_detour_stations_energy_reachable(tac_tensor, energy_remaining, batch_indices, last_nodes):
    """ Figure out if, given we travel (i,j) for a non-station j, is there a k we can reach  """

    relevant_energy_values = compute_tac_energy_remaining(
        tac_tensor, energy_remaining, batch_indices, last_nodes
    )
    are_any_detour_stations_energy_reachable = (relevant_energy_values >= 0).any(-1)

    return are_any_detour_stations_energy_reachable


def are_any_detour_stations_directly_energy_returnable(energy_tensor, num_customers, battery_capacity):
    """ For each arc (i,j), check if any station k is within range of the depot with  full battery """
    
    shape = energy_tensor.shape[1]
    station_return_energies = energy_tensor[:, 0:1, num_customers + 1:].repeat(1, shape, 1)
    are_stations_directly_energy_returnable = (station_return_energies < battery_capacity).sum(-1) >= 1
    
    return are_stations_directly_energy_returnable


def compute_tac_time_remaining(tac_time_tensor, energy_remaining, batch_indices, last_nodes):
    """ Compute the tensor to determine if there is enough energy to travel to a station after j """

    tac_energy_differences = (energy_remaining.unsqueeze(-1).unsqueeze(-1) - tac_tensor)
    relevant_energy_values = get_transition_costs(tac_energy_differences, batch_indices, last_nodes)

    return relevant_energy_values


def are_any_detour_stations_time_rechargeable(energy_tensor, time_tensor, tac_service_time_tensor, tac_energy_tensor, station_features, num_customers, energy_remaining, time_remaining, batch_indices, last_nodes):
    """ For each arc (i,j,k,0), check if we have enough time to travel and charge and return """

    tac_service_time_remaining = time_remaining.unsqueeze(-1) - tac_service_time_tensor[batch_indices.flatten().tolist(), last_nodes.flatten().tolist()]
    travel_time_remaining = tac_service_time_remaining - time_tensor[:, 0:1, num_customers + 1:].repeat(1, time_tensor.shape[1], 1)

    initial_energy = (energy_remaining.unsqueeze(-1).unsqueeze(-1) - tac_energy_tensor)[batch_indices.flatten().tolist(), last_nodes.flatten().tolist()]
    final_energy = (energy_tensor[:, 0:1, num_customers + 1:].repeat(1, energy_tensor.shape[1], 1))
    initial_energy = initial_energy.data.numpy()
    final_energy = final_energy.data.numpy()

    slow_charge_time = charging.charge_time_slow(initial_energy, final_energy)
    normal_charge_time = charging.charge_time_normal(initial_energy, final_energy)
    fast_charge_time = charging.charge_time_fast(initial_energy, final_energy)

    slow_charge_time = torch.tensor(slow_charge_time).float()
    normal_charge_time = torch.tensor(normal_charge_time).float()
    fast_charge_time = torch.tensor(fast_charge_time).float()

    slow_tacaba_time = travel_time_remaining - slow_charge_time
    normal_tacaba_time = travel_time_remaining - normal_charge_time
    fast_tacaba_time = travel_time_remaining - fast_charge_time

    are_any_detour_stations_time_rechargeable = torch.zeros(travel_time_remaining.shape).float()
    
    slow_indices = (station_features[..., 2] == 0.0).unsqueeze(1).repeat(1, energy_tensor.shape[1], 1).bool()
    normal_indices = (station_features[..., 2] == 0.5).unsqueeze(1).repeat(1, energy_tensor.shape[1], 1).bool()
    fast_indices = (station_features[..., 2] == 1.0).unsqueeze(1).repeat(1, energy_tensor.shape[1], 1).bool()

    are_any_detour_stations_time_rechargeable[slow_indices] = slow_tacaba_time[slow_indices].float()
    are_any_detour_stations_time_rechargeable[normal_indices] = normal_tacaba_time[normal_indices].float()
    are_any_detour_stations_time_rechargeable[fast_indices] = fast_tacaba_time[fast_indices].float()

    are_any_detour_stations_time_rechargeable[are_any_detour_stations_time_rechargeable > 10.0] = -1
    are_any_detour_stations_time_rechargeable[are_any_detour_stations_time_rechargeable < 0] = -1
    
    are_any_detour_stations_time_rechargeable = (are_any_detour_stations_time_rechargeable >= 0).sum(-1).bool()

    return are_any_detour_stations_time_rechargeable


def are_station_nodes_time_rechargeable(taba_service_time_tensor, energy_tensor, station_features, num_customers, time_remaining, energy_remaining, batch_indices, last_nodes):
    """ For any tarversal (i,j) for j a station, see if we have enough time to charge to return home """

    taba_time_remaining = time_remaining - taba_service_time_tensor[
        indexify_tensor(batch_indices), indexify_tensor(last_nodes)
    ][:, num_customers + 1:]
    initial_energy = energy_remaining - energy_tensor[batch_indices.flatten().tolist(), last_nodes.flatten().tolist()][:, num_customers + 1:]
    final_energy = energy_tensor[:, 0, num_customers + 1:]
    initial_energy = initial_energy.data.numpy()
    final_energy = final_energy.data.numpy()
                                    
    slow_charge_time = charging.charge_time_slow(initial_energy, final_energy)
    normal_charge_time = charging.charge_time_normal(initial_energy, final_energy)
    fast_charge_time = charging.charge_time_fast(initial_energy, final_energy)

    slow_taba_time = taba_time_remaining - torch.tensor(slow_charge_time).float()
    normal_taba_time = taba_time_remaining - torch.tensor(normal_charge_time).float()
    fast_taba_time = taba_time_remaining - torch.tensor(fast_charge_time).float()
    
    slow_indices = (station_features[..., 2] == 0.0).bool()
    normal_indices = (station_features[..., 2] == 0.5).bool()
    fast_indices = (station_features[..., 2] == 1.0).bool()
    
    are_station_nodes_time_rechargeable = torch.zeros(taba_time_remaining.shape).float()
    
    are_station_nodes_time_rechargeable[slow_indices] = slow_taba_time[slow_indices].float()
    are_station_nodes_time_rechargeable[normal_indices] = normal_taba_time[normal_indices].float()
    are_station_nodes_time_rechargeable[fast_indices] = fast_taba_time[fast_indices].float()
    
    are_station_nodes_time_rechargeable[are_station_nodes_time_rechargeable > 10.0] = -1
    are_station_nodes_time_rechargeable[are_station_nodes_time_rechargeable < 0.0] = -1
    
    are_station_nodes_time_rechargeable = (are_station_nodes_time_rechargeable >= 0.0).bool()
    stacker = torch.ones(energy_tensor.shape[:2]).bool()[:, :num_customers+1]
    are_station_nodes_time_rechargeable = torch.cat([stacker, are_station_nodes_time_rechargeable], -1)

    return are_station_nodes_time_rechargeable

                                                                                                        
def was_node_visited_last(energy_tensor, batch_indices, last_nodes):

    was_node_visited_last = torch.zeros(energy_tensor.shape[:2]).bool()
    was_node_visited_last[batch_indices.flatten().tolist(), last_nodes.flatten().tolist()] = True

    return was_node_visited_last


def was_customer_visited(energy_tensor, visited_customers):
    
    was_customer_visited = torch.zeros(energy_tensor.shape[:2]).bool()
    batch_indices = [[num for i in range(len(item))] for (num, item) in enumerate(visited_customers)]
    
    flat_batch_indices = [item for sublist in batch_indices for item in sublist]
    flat_visited_customers = [item for sublist in visited_customers for item in sublist]
    
    was_customer_visited[flat_batch_indices, flat_visited_customers] = True
    
    return was_customer_visited
