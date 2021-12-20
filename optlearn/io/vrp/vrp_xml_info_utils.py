from optlearn.io import xml_utils


def get_info_element(root_element_dict):
    """ Get the info element from the given element dict """

    info_element = root_element_dict.get("info")

    return info_element


def get_dataset_element(info_element_dict):
    """ Get the dataset element from the given element dict """

    dataset_element = info_element_dict.get("dataset")

    return dataset_element


def get_name_element(info_element_dict):
    """ Get the name element from the given element dict """

    name_element = info_element_dict.get("name")

    return name_element


def parse_dataset_element(dataset_element):
    """ Parse the dataset info element as a string """


    dataset_content = xml_utils.parse_element_content_as_str(dataset_element) 
    
    return dataset_content


def parse_name_element(name_element):
    """ Parse the name info element as a string """

    name_content = xml_utils.parse_element_content_as_str(name_element)

    return name_content


def fetch_dataset_element(info_element_dict):
    """ Get and parse the dataset information from the info element dict """

    dataset_element = get_dataset_element(info_element_dict)
    if dataset_element is None:
        return None
    dataset_content = parse_dataset_element(dataset_element)

    return  dataset_content


def fetch_name_element(info_element_dict):
    """ Get and parse the name information from the info element dict """

    name_element = get_name_element(info_element_dict)
    if name_element is None:
        return None
    name_content = parse_name_element(name_element)

    return  name_content


def get_info_dict(info_element):
    """ Get the info element as a dictionary of details """

    info_element_dict = xml_utils.build_child_element_dict_from_element(info_element)
    info_dict = {
        "dataset": fetch_dataset_element(info_element_dict),
        "name": fetch_name_element(info_element_dict),
    }

    return info_dict


