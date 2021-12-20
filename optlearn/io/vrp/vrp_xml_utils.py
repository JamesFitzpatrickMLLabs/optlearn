from optlearn.io import xml_utils
from optlearn.io.vrp import vrp_xml_info_utils
from optlearn.io.vrp import vrp_xml_network_utils
from optlearn.io.vrp import vrp_xml_fleet_utils
from optlearn.io.vrp import vrp_xml_requests_utils


def get_root_element_dict(root_element):
    """ Get a dictionary of the children elements for the root element """

    root_element_dict = xml_utils.build_child_element_dict_from_element(root_element)

    return root_element_dict


def parse_vrp_root_element(root_element):
    """ Parse the root element into a dictionary """

    root_element_dict = get_root_element_dict(root_element)
    info_dict = {
        "info": vrp_xml_info_utils.get_info_dict(root_element_dict["info"]),
        "fleet": vrp_xml_fleet_utils.get_fleet_dict(root_element_dict["fleet"]),
        "network": vrp_xml_network_utils.get_network_dict(root_element_dict["network"]),
        "requests": vrp_xml_requests_utils.get_requests_dict(root_element_dict["requests"]),
    }

    return info_dict


def read_vrp_xml(filename):
    """ Read and parse the XML VRP file into a dictionary """

    element_tree = xml_utils.read_xml_as_element_tree(filename)
    root_element = xml_utils.get_element_tree_root(element_tree)
    info_dict = parse_vrp_root_element(root_element)
    
    return info_dict
