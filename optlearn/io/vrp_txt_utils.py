import os

from optlearn.io import txt_utils
from optlearn.io import vrp_txt_info_utils
from optlearn.io import vrp_txt_network_utils
from optlearn.io import vrp_txt_fleet_utils
from optlearn.io import vrp_txt_requests_utils


def get_instance_name(filename):
    """ Get the instance name from the filename """

    instance_name = os.path.basename(filename).split(".")[0]

    return instance_name


def read_base_lines_dict(filename):
    """ Read in the given VRP TXT file and do some processing """

    base_lines = txt_utils.read_vrp_file_from_txt(filename)
    instance_name = get_instance_name(filename)
    base_lines_dict = {
        instance_name: base_lines
    }

    return base_lines_dict


def read_vrp_txt(filename):
    """ Read in the given VRP TXT file and do some processing """

    base_lines_dict = read_base_lines_dict(filename)
    info_dict = {
        "info": vrp_txt_info_utils.get_info_dict(base_lines_dict),
        "fleet": vrp_txt_fleet_utils.get_fleet_dict(base_lines_dict),
        "network": vrp_txt_network_utils.get_network_dict(base_lines_dict),
        "requests": vrp_txt_requests_utils.get_requests_dict(base_lines_dict),
    }

    return info_dict


def get_vehicle_num(filename):
    """ Read in the given VRP TXT file and do some processing """

    info_dict = read_vrp_txt(filename)
    vehicle_num = info_dict["fleet"]["vehicle_0"]["num_vehicles"]

    return vehicle_num
