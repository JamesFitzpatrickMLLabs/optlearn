import random


def check_sample_viability(coord_dict, reasonable_order=40):
    """ Make sure that the TSP is big enough to get a reasonable sub-TSP """

    if len(coord_dict) <= reasonable_order:
        return False
    else:
        return True


def sample_from_coord_dict(coord_dict, indices):
    """ Subsample a smaller TSP from one specified by the coord_dict """

    return {num: coord_dict[index] for num, index in enumerate(indices)}


def random_sample_from_coord_dict(coord_dict, reasonable_order=40, min_order=20):
    """ Subsample the TSP to a smaller of TSP of random order """

    if not check_sample_viability(coord_dict, reasonable_order):
        print("TSP too small to sample from... aborting!")
        return False
    order = random.randint(20, len(coord_dict))
    indices = random.sample(list(range(len(coord_dict))), order)

    return sample_from_coord_dict(coord_dict, indices)
    
        
