import torch

from scipy.spatial import distance


def compute_weight_tensor_nonbatch(node_coordinates):

    numpy_coordinates = node_coordinates.data.numpy()
    weight_matrix = distance.cdist(numpy_coordinates, numpy_coordinates)
    weight_tensor = torch.tensor(weight_matrix)

    return weight_tensor


def compute_weight_tensor_batch(node_coordinates):

    weight_tensors = []
    for node_subcoordinates in node_coordinates:
        weight_subtensor = compute_weight_tensor_nonbatch(node_subcoordinates)
        weight_tensors.append(weight_subtensor)
    weight_tensor = torch.stack(weight_tensors, 0)

    return weight_tensor


def compute_energy_tensor(weight_tensor, energy_consumption_rate):
    
    energy_tensor = weight_tensor * energy_consumption_rate

    return energy_tensor


def compute_time_tensor(weight_tensor, average_velocity):
    
    time_tensor = weight_tensor / average_velocity

    return time_tensor


def compute_service_time_tensor(weight_tensor, num_customers, average_velocity, service_time=0.5):

    service_time_tensor = compute_time_tensor(weight_tensor, average_velocity)
    service_time_tensor[..., :, 1:num_customers + 1] += service_time
    
    return service_time_tensor


def compute_taba_tensor_nonbatch(weight_tensor):
    
    there_and_back_again_tensor = weight_tensor + weight_tensor[:, 0:1].T
    
    return there_and_back_again_tensor


def compute_taba_tensor_batch(weight_tensor):

    taba_tensor = []
    for weight_subtensor in weight_tensor:
        taba_subtensor = compute_taba_tensor_nonbatch(weight_subtensor)
        taba_tensor.append(taba_subtensor)
    taba_tensor = torch.stack(taba_tensor, 0)

    return taba_tensor


def compute_taba_tensor(weight_tensor):

    there_and_back_again_tensor = compute_taba_tensor_batch(weight_tensor)
    
    return there_and_back_again_tensor


def compute_taba_energy_tensor(taba_tensor, energy_consumption_rate):
    """ For node-node arcs (i,j), compute the energy cost for travel along (i,j,0) """

    taba_energy_tensor = taba_tensor * energy_consumption_rate

    return taba_energy_tensor


def compute_taba_time_tensor(taba_tensor, average_velocity):
    """ For node-node arcs (i,j), compute the time for travel along (i,j,0) """
    
    taba_time_tensor = taba_tensor / average_velocity

    return taba_time_tensor


def compute_taba_service_time_tensor(taba_tensor, num_customers, average_velocity, service_time=0.5):
    """ For node-customer arcs (i,j), compute travel and service time for (i,j,0) """

    taba_service_time_tensor = compute_taba_time_tensor(taba_tensor, average_velocity)
    taba_service_time_tensor[..., :, 1:num_customers + 1] += service_time
    
    return taba_service_time_tensor


def compute_tac_tensor(weight_tensor, num_customers):
    """ Compute the weight required to get from i to j and then to station k for each station k """
    
    there_and_charge_tensor = weight_tensor.unsqueeze(-1) + weight_tensor[..., num_customers + 1:].unsqueeze(1)
    
    return there_and_charge_tensor


def compute_tac_time_tensor(tac_tensor, average_velocity):
    """ For node-customer arcs (i,j), compute travel time for (i,j,k) for each station k """
    
    tac_time_tensor = tac_tensor / average_velocity
    
    return tac_time_tensor


def compute_tac_energy_tensor(tac_tensor, energy_consumption_rate):
    """ For node-node arcs (i,j), compute the energy cost for travel along (i,j,k) for each k """
    
    tac_energy_tensor = tac_tensor * energy_consumption_rate
    
    return tac_energy_tensor


def compute_tac_service_time_tensor(tac_tensor, num_customers, average_velocity, service_time=0.5):
    """ For node-customer arcs (i,j), compute travel and service time for (i,j,k) for each station k """
    
    tac_service_time_tensor = compute_tac_time_tensor(tac_tensor, average_velocity)
    tac_service_time_tensor[:, :, 1:num_customers + 1] += service_time
    
    return tac_service_time_tensor 
