import xml.etree.ElementTree as et


def read_xml_as_element_tree(filepath):
    """ Read an xml file as an element tree """

    element_tree = et.parse(filepath)
    
    return element_tree


def get_element_tree_root(element_tree):
    """ Get the root from the element tree """

    root_element = element_tree.getroot()

    return root_element


def get_child_elements(xml_element):
    """ Get the child elements from the given element (if they exist)  """

    child_elements = xml_element.getchildren()

    return child_elements


def get_element_tag(xml_element):
    """ Get the tag for the given xml element """

    xml_element_tag = xml_element.tag

    return xml_element_tag


def get_elements_tags(xml_elements):
    """ Get the tags for each of the given elements """

    elements_tags = [element.tag for element in xml_elements]

    return elements_tags


def get_child_tags(xml_element):
    """ Get the tags of the children of the given element """

    child_elements = get_child_elements(xml_element)
    child_tags = get_elements_tags(child_elements)

    return child_tags


def build_element_dict_from_elements(xml_elements):
    """ Build a dictionary of tags and their elements """

    xml_elements_tags = get_elements_tags(xml_elements)
    element_dict = {tag: element for (tag, element) in zip(xml_elements_tags, xml_elements)}

    return element_dict


def build_child_element_dict_from_element(xml_element):
    """ Build a tag-element dictionary from the given element's children """

    xml_elements = get_child_elements(xml_element)
    element_dict = build_element_dict_from_elements(xml_elements)

    return element_dict
    

def check_element_has_child_tag(xml_element, child_tag):
    """ Check if the network element has a child with the given tag """

    child_tags = get_child_tags(xml_element)
    boolean_value = child_tag in child_tags

    return boolean_value


def get_element_attributes(xml_element):
    """ Get the attributes of the given element """

    attributes = xml_element.attrib

    return attributes


def parse_element_content_as_str(xml_element):
    """ Parse the content of the given element as a string """

    string_content = xml_element.text

    return string_content


def parse_element_content_as_int(xml_element):
    """ Parse the content of the given element as an integer """

    integer_content = int(xml_element.text)

    return integer_content


def parse_element_content_as_float(xml_element):
    """ Parse the content of the given element as a float """

    float_content = float(xml_element.text)

    return float_content


def get_custom_element(element_dict):
    """ Get the custom element from the given element dict """

    custom_element = element_dict.get("custom")

    return custom_element
