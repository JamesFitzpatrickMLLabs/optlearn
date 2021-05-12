import numpy as np
import networkx as nx

import xml.etree.ElementTree as et


def check_xml(fname):
    """ Check if the file is an xml file """

    return fname[-4:] == ".xml"


def read_xml(fname):
    """ Read an xml file """

    return et.parse(fname)


def get_children_xml(xml_element):
    """ Get the the children of the xml element """

    return [item for item in xml_element.getchildren()]


def get_tags_xml(xml_elements):
    """ Get the tags of the given xml elements """

    return [element.tag for element in xml_elements]


def get_child_tags_xml(xml_element):
    """ Get the tags of the children of the xml element """

    children = get_children_xml(xml_element)
    return get_tags_xml(children)


def get_child_tag_dict_xml(xml_element):
    """ Get a dictionary of child elements and their tags """

    children  = get_children_xml(xml_element)
    tags = get_tags_xml(children)
    return {tag: child for (tag, child) in zip(tags, children)}


def check_element_xml(network_element, child_tag):
    """ Check if the network element has a child with the given tag """

    child_tags = get_child_tags_xml(network_element)
    return child_tag in child_tags


def parse_ordinate_xml(ordinate_element):
    """ Parse the given ordinate """

    return float(ordinate_element.text)
        

def parse_node_coordinate_xml(node_element):
    """ Get the x and y coordinates of the node """

    coord_dict = get_child_tag_dict_xml(node_element)
    x, y = coord_dict["cx"], coord_dict["cy"]
    return parse_ordinate_xml(x), parse_ordinate_xml(y)


def parse_node_info_xml(node_element):
    """ Get the x and y coordinates of the node """

    attributes = node_element.attrib
    node_idx = int(attributes["id"])
    node_type = int(attributes["type"])
    return node_idx, node_type


def parse_node_xml(node_element):
    """ Parse a single node in xml format """

    x, y = parse_node_coordinate_xml(node_element)
    node_idx, node_type = parse_node_info_xml(node_element)
    node_dict = {"xy": [x, y], "node_type": node_type}
    return node_idx, node_dict
        

def parse_nodes_xml(nodes_element):
    """ Parse the nodes, their types, and their coordinates """

    node_elements = get_children_xml(nodes_element)
    parsed_elements = [parse_node_xml(element) for element in node_elements]
    return {node_idx: node_dict for (node_idx, node_dict) in parsed_elements}


def parse_metric_xml(network_element):
    """ Parse the metric from the network element """

    if "euclidean" in get_child_tags_xml(network_element):
        return "euclidean"
    else:
        raise ValueError("Metric either not specified or not recognised!")


def parse_decimals_xml(decimal_element):
    """ Parse the rounding places for the metric """

    return int(decimal_element.text)


def parse_network_xml(network_element):
    """ Parse the elements of the network section """

    tag_dict = get_child_tag_dict_xml(network_element)
    
    if check_element_xml(network_element, "nodes"):
        node_dict = parse_nodes_xml(tag_dict["nodes"])
        metric = parse_metric_xml(network_element)
        if check_element_xml(network_element, "decimals"):
            decimals = parse_decimals_xml(tag_dict["decimals"])
        else:
            print("Decimals not found, deafulting to nearest integer rounding!")
            decimals = 0
        return {
            "node_info": node_dict,
            "metric": metric,
            "decimals": decimals
        }
    else:
        raise ValueError("Unrecognised network format!")
        

def parse_capacity_xml(capacity_element):
    """ Parse the capacity of a vehicle """

    return float(capacity_element.text)


def parse_departure_xml(departure_element):
    """ Parse the departure node for a vehicle """

    return int(departure_element.text)


def parse_arrival_xml(arrival_element):
    """ Parse the arrival node for a vehicle """

    return int(arrival_element.text)


def parse_vehicle_type_xml(vehicle_element):
    """ Parse the vehicle type """

    return int(vehicle_element.attrib["type"])


def parse_vehicle_xml(vehicle_element):
    """ Parse the information for a specific vehicle """

    children = get_child_tag_dict_xml(vehicle_element)

    for (key, value) in children.items():
        if key == "capacity":
            capacity = parse_capacity_xml(value)
        elif key == "departure_node":
            departure_node = parse_departure_xml(value)
        elif key == "arrival_node":
            arrival_node = parse_arrival_xml(value)
        else:
            print("Unrecognised vehicle element!")
    vehicle_type = parse_vehicle_type_xml(vehicle_element)

    return {
        "capacity": capacity,
        "departure_node": departure_node,
        "arrival_node": arrival_node,
        "vehicle_type": vehicle_type,
    }


def parse_fleet_xml(fleet_element):
    """ Parse the fleet information """

    children = get_children_xml(fleet_element)
    
    return {
        num: parse_vehicle_xml(item) for (num, item) in enumerate(children)
    }


def parse_quantity_xml(quantity_element):
    """ Parse a quantity element """

    return float(quantity_element.text)


def parse_request_xml(request_element):
    """ Parse a request for a single node """

    tag_dict = get_child_tag_dict_xml(request_element)
    
    node_id = int(request_element.attrib["node"])
    if check_element_xml(request_element, "quantity"):
        quantity = parse_quantity_xml(tag_dict["quantity"])
    else:
        raise ValueError("No demand quantity found for node {}".format(node_id))

    return node_id, quantity


def parse_requests_xml(requests_element):
    """ Parse all of the requests """

    children = get_children_xml(requests_element)
    request_pairs = [parse_request_xml(child) for child in children]
    return {num: request for (num, request) in request_pairs}


def parse_vrp_file_xml(xml_root):
    """ Parse a VRP file in xml format """

    tag_dict = get_child_tag_dict_xml(xml_root)
    
    network = parse_network_xml(tag_dict["network"])
    fleet = parse_fleet_xml(tag_dict["fleet"])
    requests = parse_requests_xml(tag_dict["requests"])

    return {
        "network": network,
        "fleet": fleet,
        "requests": requests,
    }
