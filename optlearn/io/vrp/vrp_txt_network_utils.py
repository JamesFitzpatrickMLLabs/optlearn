from optlearn.io import txt_utils


def get_decimals_line(base_lines_dict):
    """ Get the decimals of precision from the base lines dict """

    decimals_line = None

    return decimals_line


def get_nodes_lines(base_lines_dict):
    """ Get the nodes lines from the base lines dict """

    base_lines = txt_utils.get_base_lines(base_lines_dict)
    nodes_lines = [item for item in base_lines if len(item) == 4]

    return nodes_lines


def get_cx_string(node_line):
    """ Get the cx string from the node line """

    cx_string = node_line[2]

    return cx_string


def get_cy_string(node_line):
    """ Get the cy string from the node line """

    cy_string = node_line[3]

    return cy_string


def get_node_type_string(node_line):
    """ Get the node type  string from the node line """

    node_type_string = node_line[1]

    return node_type_string


def get_node_name_string(node_line):
    """ Get the node name string from the node line """

    node_name_string = node_line[0]

    return node_name_string


def parse_cx_string(cx_string):
    """ Parse the cx coordinate from the given string """

    cx_content = float(cx_string)

    return cx_content


def parse_cy_string(cy_string):
    """ Parse the cy coordinate from the given string """

    cy_content = float(cy_string)

    return cy_content


def parse_node_type_string(node_type_string):
    """ Parse the node type from the given string"""

    return node_type_string


def parse_node_name_string(node_name_string):
    """ Parse the node name from the given string """

    return node_name_string


def fetch_cx_line(node_line):
    """ Get and parse the cx line from the given node line """

    cx_string = get_cx_string(node_line)
    cx_content = parse_cx_string(cx_string)

    return cx_content


def fetch_cy_line(node_line):
    """ Get and parse the cy line from the given node line """

    cy_string = get_cy_string(node_line)
    cy_content = parse_cy_string(cy_string)

    return cy_content


def fetch_node_type_line(node_line):
    """ Get and parse the node type line from the given node line """

    node_type_string = get_node_type_string(node_line)
    node_type_content = parse_node_type_string(node_type_string)

    return node_type_content


def fetch_node_name_line(node_line):
    """ Get and parse the node name line from the given node line """

    node_name_string = get_node_name_string(node_line)
    node_name_content = parse_node_name_string(node_name_string)

    return node_name_content


def get_node_dict(node_line):
    """ Parse the node line as a dictionary """

    node_dict = {
        "name": fetch_node_name_line(node_line),
        "type": fetch_node_type_line(node_line),
        "cx": fetch_cx_line(node_line),
        "cy": fetch_cy_line(node_line),
    }

    return node_dict


def get_nodes_dict(nodes_lines):
    """ Get the dict of nodes information from the nodes lines """

    nodes_dict = {
        num: get_node_dict(line) for (num, line) in enumerate(nodes_lines)
    }
    nodes_dict = {
        key: {**value, "id": key} for (key, value) in nodes_dict.items()
    }

    return nodes_dict


def get_network_dict(base_lines_dict):
    """ get the network information from the base lines dict """

    nodes_lines = get_nodes_lines(base_lines_dict)
    network_dict = {
        "nodes": get_nodes_dict(nodes_lines),
        "decimals": get_decimals_line(base_lines_dict),
    }

    return network_dict
