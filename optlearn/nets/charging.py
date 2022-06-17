import torch

import numpy as np


slow_functions = [
        lambda x: 0.0 + (1.26 - 0.0) * ((x - 0.0) / (13.6 - 0.0)),
        lambda x: 1.26 + (1.54 - 1.26) * ((x - 13.6) / (15.2 - 13.6)),
        lambda x: 1.54 + (2.06 - 1.54) * ((x - 15.2) / (16.0 - 15.2)),
        lambda x: np.inf,
    ]
normal_functions = [
        lambda x: 0.0 + (0.62 - 0.0) * ((x - 0.0) / (13.6 - 0.0)),
        lambda x: 0.62 + (0.77 - 0.62) * ((x - 13.6) / (15.2 - 13.6)),
        lambda x: 0.77 + (1.01 - 0.77) * ((x - 15.2) / (16.0 - 15.2)),
        lambda x: np.inf,
    ]
fast_functions = [
        lambda x: 0.0 + (0.31 - 0.0) * ((x - 0.0) / (13.6 - 0.0)),
        lambda x: 0.31 + (0.39 - 0.31) * ((x - 13.6) / (15.2 - 13.6)),
        lambda x: 0.39 + (0.51 - 0.39) * ((x - 15.2) / (16.0 - 15.2)),
        lambda x: np.inf,
    ]
slow_energy_functions = [
        lambda x: 0.0 + (13.6 - 0.0) * ((x - 0.0) / (1.26 - 0.0)),
        lambda x: 13.6 + (15.2 - 13.6) * ((x - 1.26) / (1.54 - 1.26)),
        lambda x: 15.2 + (16.0 - 15.2) * ((x - 1.54) / (2.06 - 1.54)),
        lambda x: np.inf,
    ]
normal_energy_functions = [
        lambda x: 0.0 + (13.6 - 0.0) * ((x - 0.0) / (0.62 - 0.0)),
        lambda x: 13.6 + (15.2 - 13.6) * ((x - 0.62) / (0.77 - 0.62)),
        lambda x: 15.2 + (16.0 - 15.2) * ((x - 0.77) / (1.01 - 0.77)),
        lambda x: np.inf,
    ]
fast_energy_functions = [
        lambda x: 0.0 + (13.6 - 0.0) * ((x - 0.0) / (0.31 - 0.0)),
        lambda x: 13.6 + (15.2 - 13.6) * ((x - 0.31) / (0.39 - 0.31)),
        lambda x: 15.2 + (16.0 - 15.2) * ((x - 0.39) / (0.51 - 0.39)),
        lambda x: np.inf,
    ]


def piecewise_slow(x):

    conditions = [
        (x >= 0.0) & (x < 13.6),
        (x >= 13.6) & (x < 15.2),
        (x >= 15.2) & (x <= 16.0),
        (x > 16.0),
    ]
    functions = slow_functions

    return np.piecewise(x, conditions, functions)

def piecewise_slow_energy(x):

    conditions = [
        (x >= 0.0) & (x < 1.26),
        (x >= 1.26) & (x < 1.54),
        (x >= 1.54) & (x <= 2.06),
        (x > 2.06),
    ]
    functions = slow_energy_functions
    
    return np.piecewise(x, conditions, functions)


def piecewise_normal(x):

    conditions = [
        (x >= 0.0) & (x < 13.6),
        (x >= 13.6) & (x < 15.2),
        (x >= 15.2) & (x <= 16.0),
        (x > 16.0),
    ]
    functions = normal_functions
    
    return np.piecewise(x, conditions, functions)


def piecewise_normal_energy(x):
    
    conditions = [
        (x >= 0.0) & (x < 0.62),
        (x >= 0.62) & (x < 0.77),
        (x >= 0.77) & (x <= 1.01),
        (x > 1.01),
    ]
    functions = normal_energy_functions
    
    return np.piecewise(x, conditions, functions)

def piecewise_fast(x):

    conditions = [
        (x >= 0.0) & (x < 13.6),
        (x >= 13.6) & (x < 15.2),
        (x >= 15.2) & (x <= 16.0),
        (x > 16.0),
    ]
    functions = fast_functions

    return np.piecewise(x, conditions, functions)

def piecewise_fast_energy(x):
                                                                
    conditions = [
        (x >= 0.0) & (x < 0.31),
        (x >= 0.31) & (x < 0.39),
        (x >= 0.39) & (x <= 0.51),
        (x > 0.51),
    ]
    functions = fast_energy_functions
    
    return np.piecewise(x, conditions, functions)


def charge_time_slow(x_before, x_after):

    charge_time_before = piecewise_slow(x_before)
    charge_time_after = piecewise_slow(x_after)
    
    charge_time = charge_time_after - charge_time_before
    
    charge_time[x_before < 0] = - np.inf
    charge_time[x_after < x_before] = 0
    charge_time[x_before > 16.0] = - np.inf
    charge_time[x_after < 0] = - np.inf
    charge_time[x_after > 16.0] = - np.inf
    
    return charge_time


def charge_time_normal(x_before, x_after):
    
    charge_time_before = piecewise_normal(x_before)
    charge_time_after = piecewise_normal(x_after)

    charge_time = charge_time_after - charge_time_before

    charge_time[x_before < 0] = - np.inf
    charge_time[x_after < x_before] = 0
    charge_time[x_before > 16.0] = - np.inf
    charge_time[x_after < 0] = - np.inf
    charge_time[x_after > 16.0] = - np.inf
    
    return charge_time


def charge_time_fast(x_before, x_after):
    
    charge_time_before = piecewise_fast(x_before)
    charge_time_after = piecewise_fast(x_after)

    charge_time = charge_time_after - charge_time_before
    
    charge_time[x_before < 0] = - np.inf
    charge_time[x_after < x_before] = 0
    charge_time[x_before > 16.0] = - np.inf
    charge_time[x_after < 0] = - np.inf
    charge_time[x_after > 16.0] = - np.inf
    
    return charge_time


def compute_minimum_station_charge_time(energy_tensor, num_customers, energy_remaining, station_features, batch_indices, last_nodes):

    initial_energy = (energy_remaining.unsqueeze(-1) - energy_tensor)[batch_indices.flatten().tolist(), last_nodes.flatten().tolist(), num_customers + 1:]
    final_energy = energy_tensor[:, 0, num_customers + 1:]

    initial_energy = initial_energy.data.numpy()
    final_energy = final_energy.data.numpy()

    slow_time = charge_time_slow(initial_energy, final_energy)
    normal_time = charge_time_normal(initial_energy, final_energy)
    fast_time = charge_time_fast(initial_energy, final_energy)
    
    slow_time = torch.tensor(slow_time).float()
    normal_time = torch.tensor(normal_time).float()
    fast_time = torch.tensor(fast_time).float()
    
    min_charge_time = torch.zeros_like(slow_time)
    
    min_charge_time[station_features[..., 2] == 0.0] = slow_time[station_features[..., 2] == 0.0]
    min_charge_time[station_features[..., 2] == 0.5] = normal_time[station_features[..., 2] == 0.5]
    min_charge_time[station_features[..., 2] == 1.0] = fast_time[station_features[..., 2] == 1.0]
    
    min_charge_time[min_charge_time < 0] = 0
    
    return min_charge_time


def compute_maximum_station_charge_time(energy_tensor, taba_time_tensor, num_customers, energy_remaining, time_remaining, station_features, batch_indices, last_nodes):

    initial_energy = (energy_remaining.unsqueeze(-1) - energy_tensor)[batch_indices.flatten().tolist(), last_nodes.flatten().tolist(), num_customers + 1:]
    final_energy = torch.ones_like(energy_tensor[:, 0, num_customers + 1:]) * 16.0
    
    initial_energy = initial_energy.data.numpy()
    final_energy = final_energy.data.numpy()
    
    slow_time = charge_time_slow(initial_energy, final_energy)
    normal_time = charge_time_normal(initial_energy, final_energy)
    fast_time = charge_time_fast(initial_energy, final_energy)
    
    slow_time = torch.tensor(slow_time).float()
    normal_time = torch.tensor(normal_time).float()
    fast_time = torch.tensor(fast_time).float()
    
    slow_time[slow_time == - torch.inf] = torch.inf
    normal_time[normal_time == - torch.inf] = torch.inf
    fast_time[fast_time == - torch.inf] = torch.inf
    
    leftover_time = (time_remaining.unsqueeze(-1) - taba_time_tensor)[batch_indices.flatten().tolist(), last_nodes.flatten().tolist(), num_customers + 1:]
    
    time_check_slow = ((leftover_time - slow_time) >= 0).bool()
    time_check_normal = ((leftover_time - normal_time) >= 0).bool()
    time_check_fast = ((leftover_time - fast_time) >= 0).bool()
    
    type_check_slow = (station_features[..., 2] == 0.0).bool()
    type_check_normal = (station_features[..., 2] == 0.5).bool()
    type_check_fast = (station_features[..., 2] == 1.0).bool()
    
    charge_time = torch.zeros_like(slow_time)
    
    charge_time[type_check_slow * time_check_slow] = slow_time[type_check_slow * time_check_slow].float()
    charge_time[type_check_slow * torch.logical_not(time_check_slow)] = leftover_time[type_check_slow * torch.logical_not(time_check_slow )].float()
    charge_time[type_check_normal * time_check_normal] = normal_time[type_check_normal * time_check_normal].float()
    charge_time[type_check_normal * torch.logical_not(time_check_normal)] = leftover_time[type_check_normal * torch.logical_not(time_check_normal)].float()
    charge_time[type_check_fast * time_check_fast] = fast_time[type_check_fast * time_check_fast].float()
    charge_time[type_check_fast * torch.logical_not(time_check_fast)] = leftover_time[type_check_fast * torch.logical_not(time_check_fast)].float()
    
    charge_time[(torch.tensor(initial_energy) < 0).bool()] = 0

    return charge_time
