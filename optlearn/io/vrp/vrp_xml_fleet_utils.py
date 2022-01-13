from optlearn.io import xml_utils


def get_fleet_element(root_element_dict):
    """ Get the fleet element from the element dict """

    fleet_element = root_element_dict.get("fleet")

    return fleet_element


def get_vehicle_profile_elements(fleet_element):
    """ Get the vehicle profile elements from the given fleet element """

    vehicle_profile_elements = xml_utils.get_child_elements(fleet_element)

    return vehicle_profile_elements


def get_departure_node_element(vehicle_profile_element_dict):
    """ Get the departure node element from the given element dict """

    departure_node_element = vehicle_profile_element_dict.get("departure_node")

    return departure_node_element


def get_arrival_node_element(vehicle_profile_element_dict):
    """ Get the arrival node element from the given element dict """

    arrival_node_element = vehicle_profile_element_dict.get("arrival_node")

    return arrival_node_element


def get_capacity_node_element(vehicle_profile_element_dict):
    """ Get the capacity element from the given element dict """

    capacity_element = vehicle_profile_element_dict.get("capacity")

    return capacity_element


def get_max_travel_time_element(vehicle_profile_element_dict):
    """ Get the max travel time element from the given element dict """

    max_travel_time_element = vehicle_profile_element_dict.get("max_travel_time")

    return max_travel_time_element


def get_speed_factor_element(vehicle_profile_element_dict):
    """ Get the speed factor element from the given element dict """

    speed_factor_element = vehicle_profile_element_dict.get("speed_factor")

    return speed_factor_element


def get_consumption_rate_element(custom_element_dict):
    """ Get the consumption rate element from the given element  dict """

    consumption_rate_element = custom_element_dict.get("consumption_rate")

    return consumption_rate_element
    

def get_battery_capacity_element(custom_element_dict):
    """ Get the battery capacity element from the given element dict """

    battery_capacity_element = custom_element_dict.get("battery_capacity")

    return battery_capacity_element


def get_charging_functions_element(custom_element_dict):
    """ Get the charging functions element from the given element dict """

    charging_functions_element = custom_element_dict.get("charging_functions")

    return charging_functions_element


def get_function_elements(charging_functions_element):
    """ Get the function element from the given element_dict """

    function_elements = xml_utils.get_child_elements(charging_functions_element)

    return function_elements


def get_breakpoint_elements(function_element):
    """ Get the breakpoint elements from the given function element """

    breakpoint_elements = xml_utils.get_child_elements(function_element)

    return breakpoint_elements

    
def get_battery_level_element(breakpoint_element_dict):
    """ Get the battery level element from the given element dict """

    battery_level_element = breakpoint_element_dict.get("battery_level")

    return battery_level_element


def get_charging_time_element(breakpoint_element_dict):
    """ Get the charging time element from the given element dict """

    charging_time_element = breakpoint_element_dict.get("charging_time")

    return charging_time_element


def parse_vehicle_type_attribute(vehicle_type_attribute):
    """ Parse the node type attribute as an integer """

    vehicle_type_attribute = int(vehicle_type_attribute)

    return vehicle_type_attribute


def fetch_vehicle_type_attribute(vehicle_profile_element):
    """ Get and parse the node ID element """

    vehicle_profile_element_attributes = xml_utils.get_element_attributes(vehicle_profile_element)
    vehicle_type_attribute = vehicle_profile_element_attributes.get("type")
    if vehicle_type_attribute is None:
        return None
    else:
        vehicle_type_attribute = parse_vehicle_type_attribute(vehicle_type_attribute)

    return vehicle_type_attribute


def parse_departure_node_element(departure_node_element):
    """ Parse the departure node element as an integer """

    departure_node_content = xml_utils.parse_element_content_as_int(departure_node_element)

    return departure_node_content


def parse_arrival_node_element(arrival_node_element):
    """ Parse the arrival node element as an integer """

    arrival_node_content = xml_utils.parse_element_content_as_int(arrival_node_element)

    return arrival_node_content


def parse_capacity_element(capacity_element):
    """ Parse the capacity element as a float """

    capacity_content = xml_utils.parse_element_content_as_float(capacity_element)

    return capacity_content


def parse_max_travel_time_element(max_travel_time_element):
    """ Parse the max travel time element as a float """

    max_travel_time_content = xml_utils.parse_element_content_as_float(max_travel_time_element)

    return max_travel_time_content


def parse_speed_factor_element(speed_factor_element):
    """ Parse the speed factor element as float """

    speed_factor_content = xml_utils.parse_element_content_as_float(speed_factor_element)

    return speed_factor_content


def fetch_departure_node_element(vehicle_profile_element_dict):
    """ Get and parse the departure node from the given element dict """

    departure_node_element = get_departure_node_element(vehicle_profile_element_dict)
    if departure_node_element is None:
        return None
    departure_node_content = parse_departure_node_element(departure_node_element)

    return departure_node_content


def fetch_arrival_node_element(vehicle_profile_element_dict):
    """ Get and parse the arrival node for the given element dict """

    arrival_node_element = get_arrival_node_element(vehicle_profile_element_dict)
    if arrival_node_element is None:
        return None
    arrival_node_content = parse_arrival_node_element(arrival_node_element)

    return arrival_node_content


def fetch_capacity_element(vehicle_profile_element_dict):
    """ get and parse the vheicle capacity for the given element dict """

    capacity_element = get_capacity_node_element(vehicle_profile_element_dict)
    if capacity_element is None:
        return None
    capacity_content = parse_capacity_element(capacity_element)

    return capacity_content


def fetch_max_travel_time_element(vehicle_profile_element_dict):
    """ Get and parse the max travel time element from the given element dict """

    max_travel_time_element = get_max_travel_time_element(vehicle_profile_element_dict)
    if max_travel_time_element is None:
        return None
    max_travel_time_content = parse_max_travel_time_element(max_travel_time_element)

    return max_travel_time_content


def fetch_speed_factor_element(vehicle_profile_element_dict):
    """ Get and parse the speed factor element from the given element dict """

    speed_factor_element = get_speed_factor_element(vehicle_profile_element_dict)
    if speed_factor_element is None:
        return None
    speed_factor_content = parse_speed_factor_element(speed_factor_element)

    return speed_factor_content


def parse_consumption_rate_element(consumption_rate_element):
    """ Parse the consumption rate element as a float """

    consumption_rate_content = xml_utils.parse_element_content_as_float(consumption_rate_element)
    consumption_rate_content = consumption_rate_content / 1000
    
    return consumption_rate_content


def parse_battery_capacity_element(battery_capacity_element):
    """ Parse the battery capacity element content as a float """

    battery_capacity_content = xml_utils.parse_element_content_as_float(battery_capacity_element)
    battery_capacity_content = battery_capacity_content / 1000
    
    return battery_capacity_content


def parse_battery_level_element(battery_level_element):
    """ Parse the battery level element content as a float """

    battery_level_content = xml_utils.parse_element_content_as_float(battery_level_element)

    return battery_level_content


def parse_charging_time_element(charging_time_element):
    """ Parse the charging time element as a float """

    charging_time_content = xml_utils.parse_element_content_as_float(charging_time_element)

    return charging_time_content


def fetch_battery_level_element(breakpoint_element_dict):
    """ Get and parse the battery level element from the given element dict """

    battery_level_element = get_battery_level_element(breakpoint_element_dict)
    if battery_level_element is None:
        return None
    battery_level_content = parse_battery_level_element(battery_level_element)

    return battery_level_content


def fetch_charging_time_element(breakpoint_element_dict):
    """ Get and parse the charging time element from the given element dict """

    charging_time_element = get_charging_time_element(breakpoint_element_dict)
    if charging_time_element is None:
        return None
    charging_time_content = parse_charging_time_element(charging_time_element)

    return charging_time_content


def fetch_battery_capacity_element(custom_element_dict):
    """ Get and parse the battery capacity element from the given element dict """

    battery_capacity_element = get_battery_capacity_element(custom_element_dict)
    if battery_capacity_element is None:
        return None
    battery_capacity_content = parse_battery_capacity_element(battery_capacity_element)

    return battery_capacity_content


def fetch_consumption_rate_element(custom_element_dict):
    """ Get and parse the consumption rate element from the given element dict """

    consumption_rate_element = get_consumption_rate_element(custom_element_dict)
    if consumption_rate_element is None:
        return None
    consumption_rate_content = parse_consumption_rate_element(consumption_rate_element)

    return consumption_rate_content
    

def get_breakpoint_dict(breakpoint_element):
    """ Get the information from the breakpoint element """

    breakpoint_element_dict = xml_utils.build_child_element_dict_from_element(breakpoint_element)
    breakpoint_dict = {
        "battery_level": fetch_battery_level_element(breakpoint_element_dict),
        "charging_time": fetch_charging_time_element(breakpoint_element_dict),
    }

    return breakpoint_dict


def get_breakpoints_dict(function_element):
    """ Get all of the breakpoint element information from the given function element """

    breakpoint_elements = xml_utils.get_child_elements(function_element)
    breakpoints_dict = [get_breakpoint_dict(breakpoint_element)
                        for breakpoint_element in breakpoint_elements]

    return breakpoints_dict


def parse_cs_type_attribute(cs_type_attribute):
    """ Parse the CS type attribute as a string """

    cs_type_attribute = str(cs_type_attribute)

    return cs_type_attribute


def fetch_cs_type_attribute(function_element):
    """ Get and parse the CS type element """

    function_element_attributes = xml_utils.get_element_attributes(function_element)
    cs_type_attribute = function_element_attributes.get("cs_type")
    if cs_type_attribute is None:
        return None
    else:
        cs_type_attribute = parse_cs_type_attribute(cs_type_attribute)

    return cs_type_attribute


def get_function_dict(function_element):
    """ Get the information realting to the given function element """
    
    function_dict = {
        "cs_type": fetch_cs_type_attribute(function_element),
        "breakpoints": get_breakpoints_dict(function_element)
    }

    return function_dict


def get_charging_functions_dict(charging_functions_element):
    """ Get the information relating to the charging functions element """

    function_elements = xml_utils.get_child_elements(charging_functions_element)
    function_dicts = [get_function_dict(function_element)
                      for function_element in function_elements]
    charging_functions_dict = {
        function_dict["cs_type"]: function_dict for function_dict in function_dicts
    }

    return charging_functions_dict


def get_custom_dict(vehicle_profile_element):
    """ Get custom info from the vehicle profile element """
    
    profile_element_dict = xml_utils.build_child_element_dict_from_element(vehicle_profile_element)
    custom_element = xml_utils.get_custom_element(profile_element_dict)
    if custom_element is None:
        return None
    custom_element_dict = xml_utils.build_child_element_dict_from_element(custom_element)
    charging_functions_element = get_charging_functions_element(custom_element_dict)
    if charging_functions_element is not None:
        charging_functions_content = get_charging_functions_dict(charging_functions_element)
    else:
        charging_functions_content = None
    custom_dict = {
        "consumption_rate": fetch_consumption_rate_element(custom_element_dict), 
        "battery_capacity": fetch_battery_capacity_element(custom_element_dict),
        "charging_functions": charging_functions_content,
    }

    return custom_dict


def get_vehicle_profile_dict(vehicle_profile_element):
    """ Get information relating to the given vehicle profile element """

    profile_element_dict = xml_utils.build_child_element_dict_from_element(vehicle_profile_element)
    vehicle_profile_dict = {
        "max_travel_time": fetch_max_travel_time_element(profile_element_dict),
        "departure_node": fetch_departure_node_element(profile_element_dict),
        "arrival_node": fetch_arrival_node_element(profile_element_dict),
        "speed_factor": fetch_speed_factor_element(profile_element_dict),
        "capacity": fetch_capacity_element(profile_element_dict),
        "type": fetch_vehicle_type_attribute(vehicle_profile_element),
    }

    return vehicle_profile_dict


def get_fleet_dict(fleet_element):
    """ Get the information relating to the given fleet element """

    vehicle_profile_elements = get_vehicle_profile_elements(fleet_element)
    vehicle_dict = {
        f"vehicle_{num}": get_vehicle_profile_dict(vehicle_profile_element)
        for (num, vehicle_profile_element) in enumerate(vehicle_profile_elements)
    }
    custom_dict = {
        f"functions_{num}": get_custom_dict(vehicle_profile_element)
        for (num, vehicle_profile_element) in enumerate(vehicle_profile_elements)
    }
    fleet_dict = {
        **vehicle_dict,
        **custom_dict,
    }

    return fleet_dict
