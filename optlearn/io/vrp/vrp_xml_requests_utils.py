from optlearn.io import xml_utils


def get_requests_element(root_element_dict):
    """ Get the requests element from the given element dict """

    requests_element = root_element_dict.get("requests")

    return requests_dict


def get_request_elements(requests_element_dict):
    """ Get the request elements from the given element dict """

    request_elements = xml_utils.get_child_elements(requests_element_dict)

    return request_elements


def get_service_time_element(request_element_dict):
    """ Get the service time element from the given element dict """

    service_time_element = request_element_dict.get("service_time")

    return service_time_element


def get_quantity_element(request_element_dict):
    """ Get the quantity element from the given element dict """

    quantity_element = request_element_dict.get("quantity")

    return quantity_element


def parse_service_time_element(service_time_element):
    """ Parse the service time element content as a float """

    service_time_content = xml_utils.parse_element_content_as_float(service_time_element)

    return service_time_content


def parse_quantity_element(quantity_element):
    """ Parse the quantity element as a float """

    quantity_content = xml_utils.parse_element_content_as_float(quantity_element)

    return quantity_content


def fetch_service_time_element(request_element_dict):
    """ Get and parse the service time element content from the given element dict """

    service_time_element = get_service_time_element(request_element_dict)
    if service_time_element is None:
        return None
    service_time_content = parse_service_time_element(service_time_element)

    return service_time_content


def fetch_quantity_element(request_element_dict):
    """ Get and parse the quantity element content from the given element dict """

    quantity_element = get_quantity_element(request_element_dict)
    if quantity_element is None:
        return None
    quantity_content = parse_quantity_element(quantity_element)

    return quantity_content


def parse_request_id_attribute(request_id_attribute):
    """ Parse the request ID attribute as an integer """

    request_id_attribute = int(request_id_attribute)

    return request_id_attribute


def parse_request_node_attribute(request_node_attribute):
    """ Parse the request node attribute as an integer """

    request_node_attribute = int(request_node_attribute)

    return request_node_attribute


def fetch_request_node_attribute(request_element):
    """ Get and parse the request node element """

    request_element_attributes = xml_utils.get_element_attributes(request_element)
    request_node_attribute = request_element_attributes.get("node")
    if request_node_attribute is None:
        return None
    else:
        request_node_attribute = parse_request_node_attribute(request_node_attribute)

    return request_node_attribute


def get_request_dict(request_element):
    """ Get a dict of request information from the given request element """

    request_element_dict = xml_utils.build_child_element_dict_from_element(request_element)
    request_dict = {
        "quantity": fetch_quantity_element(request_element_dict),
        "service_time": fetch_service_time_element(request_element_dict),
        "node": fetch_request_node_attribute(request_element)
    }

    return request_dict


def get_requests_dict(requests_element):
    """ Get a dict of all the requests with their nodes as keys0 """

    request_elements = xml_utils.get_child_elements(requests_element)
    request_dicts = [get_request_dict(request_element) for request_element in request_elements]
    requests_dict = {request_dict["node"]: request_dict for request_dict in request_dicts}

    return requests_dict
