from optlearn.io import xml_utils


def get_network_element(root_element_dict):
    """ Get the network element from the given element dict """

    network_element = root_element_dict.get("network")

    return network_element


def get_nodes_element(network_element_dict):
    """ Get the nodes element from the given element dict """

    nodes_element = network_element_dict.get("nodes")

    return nodes_element


def get_node_elements(nodes_element):
    """ Get the child node elements from the given nodes element """

    node_elements = get_child_elements(nodes_element)

    return node_elements


def get_decimals_element(network_element_dict):
    """ Get the decimals element from the given element dict """

    decimals_element = network_element_dict.get("decimals")

    return decimals_element


def get_cx_element(node_element_dict):
    """ Get the cx element from the given element dict """

    cx_element = node_element_dict.get("cx")

    return cx_element


def get_cy_element(node_element_dict):
    """ Get the cy element from the given element dict """

    cy_element = node_element_dict.get("cy")

    return cy_element


def get_cs_type_element(custom_element_dict):
    """ Get the cs type element from the given element dict """

    cs_type_element = custom_element_dict.get("cs_type")

    return cs_type_element


def parse_cx_element(cx_element):
    """ Parse the cx element as a float """

    cx_content = xml_utils.parse_element_content_as_float(cx_element)

    return cx_content


def parse_cy_element(cy_element):
    """ Parse the cy element as a float """

    cy_content = xml_utils.parse_element_content_as_float(cy_element)

    return cy_content


def fetch_cx_element(node_element_dict):
    """ Fetch and parse the cx element from the given node element dict """

    cx_element = get_cx_element(node_element_dict)
    if cx_element is None:
        return None
    cx_content = parse_cx_element(cx_element)

    return cx_content


def fetch_cy_element(node_element_dict):
    """ Fetch and parse the cy element from the given node element dict """

    cy_element = get_cy_element(node_element_dict)
    if cy_element is None:
        return None
    cy_content = parse_cy_element(cy_element)

    return cy_content


def parse_cs_type_element(cs_type_element):
    """ Parse the cs type element as a string """

    cs_type_content = xml_utils.parse_element_content_as_str(cs_type_element)

    return cs_type_content


def fetch_cs_type_element(node_element_dict):
    """ Get and parse the cs type element from the given node_element_dict """

    custom_element = xml_utils.get_custom_element(node_element_dict)

    if custom_element is None:
        return None
    custom_element_dict = xml_utils.build_child_element_dict_from_element(custom_element)
    cs_type_element = get_cs_type_element(custom_element_dict)
    if cs_type_element is None:
        return None
    cs_type_content = parse_cs_type_element(cs_type_element)

    return cs_type_content


def parse_node_id_attribute(node_id_attribute):
    """ Parse the node ID attribute as an integer """

    node_id_attribute = int(node_id_attribute)

    return node_id_attribute


def parse_node_type_attribute(node_type_attribute):
    """ Parse the node type attribute as an integer """

    node_type_attribute = int(node_type_attribute)

    return node_type_attribute


def fetch_node_id_attribute(node_element):
    """ Get and parse the node ID element """

    node_element_attributes = xml_utils.get_element_attributes(node_element)
    node_id_attribute = node_element_attributes.get("id")
    if node_id_attribute is None:
        return None
    else:
        node_id_attribute = parse_node_id_attribute(node_id_attribute)

    return node_id_attribute


def fetch_node_type_attribute(node_element):
    """ Get and parse the node type element """
    
    node_element_attributes = xml_utils.get_element_attributes(node_element)
    node_type_attribute = node_element_attributes.get("type")
    if node_type_attribute is None:
        return None
    else:
        node_type_attribute = parse_node_type_attribute(node_type_attribute)

    return node_type_attribute
         

def get_node_dict(node_element):
    """ Get a dict of node information from the given node element """

    node_element_dict = xml_utils.build_child_element_dict_from_element(node_element)
    node_dict = {
        "cx": fetch_cx_element(node_element_dict),
        "cy": fetch_cy_element(node_element_dict),
        "id": fetch_node_id_attribute(node_element),
        "type": fetch_node_type_attribute(node_element),
        "cs_type": fetch_cs_type_element(node_element_dict),
    }

    return node_dict


def get_nodes_dict(nodes_element):
    """ Get a dict of all the nodes with their nodes as keys """

    node_elements = xml_utils.get_child_elements(nodes_element)
    node_dicts = [get_node_dict(node_element) for node_element in node_elements]
    nodes_dict = {node_dict["id"]: node_dict for node_dict in node_dicts}

    return nodes_dict


def parse_decimals_element(decimals_element):
    """ Parse the decimals element """

    decimals_content = xml_utils.parse_element_content_as_int(decimals_element)

    return decimals_content


def fetch_decimals_element(network_element_dict):
    """ Get and parse the decimals element from the given element dict """

    decimals_element = get_decimals_element(network_element_dict)
    decimals_content = parse_decimals_element(decimals_element)

    return decimals_content


def get_network_dict(network_element):
    """ Fetch all of the information about the network """

    network_element_dict = xml_utils.build_child_element_dict_from_element(network_element)
    nodes_element = get_nodes_element(network_element_dict)
    network_dict = {
        "nodes": get_nodes_dict(nodes_element),
        "decimals": fetch_decimals_element(network_element_dict),
    }

    return network_dict
