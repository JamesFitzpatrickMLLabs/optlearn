from optlearn.io import txt_utils


def get_lines_substring_index(fleet_lines, substring):
    """ Get the index of the line containing the substring in the fleet lines """

    content_index = [num for (num, string) in enumerate(fleet_lines) if substring in string]
    if len(content_index) != 1:
        return None
    else:
        content_index = content_index[0]
    
    return content_index


def get_fleet_lines(base_lines_dict):
    """ Get the fleet lines from the base lines dict """

    base_lines = txt_utils.get_base_lines(base_lines_dict)
    fleet_lines = [line[0] for line in base_lines if len(line) <= 1]

    return fleet_lines
    

def get_fuel_consumption_rate_line(fleet_lines):
    """ Get the line dealing with fuel consumption rate from the fleet lines """ 

    content_index = get_lines_substring_index(fleet_lines, "fuel consumption rate")
    if content_index is None:
        return None
    fuel_consumption_rate_line = fleet_lines[content_index]
    
    return fuel_consumption_rate_line


def get_fuel_tank_capacity_line(fleet_lines):
    """ Get the line dealing with fuel tank capacity from the fleet lines """

    content_index = get_lines_substring_index(fleet_lines, "fuel tank capacity")
    if content_index is None:
        return None
    fuel_tank_capacity_line = fleet_lines[content_index]
    
    return fuel_tank_capacity_line


def get_average_velocity_line(fleet_lines):
    """ Get the line dealing with average velocity from the fleet lines """

    content_index = get_lines_substring_index(fleet_lines, "average Velocity")
    if content_index is None:
        return None
    average_velocity_line = fleet_lines[content_index]
    
    return average_velocity_line


def get_tour_length_line(fleet_lines):
    """ Get the line dealing with tour length from the fleet lines """

    content_index = get_lines_substring_index(fleet_lines, "TourLength")
    if content_index is None:
        return None
    tour_length_line = fleet_lines[content_index]
    
    return tour_length_line


def get_number_of_vehicles_line(fleet_lines):
    """ Get the line dealing with number of vehicles from the fleet lines """

    content_index = get_lines_substring_index(fleet_lines, "numVeh")
    if content_index is None:
        return None
    number_of_vehicles_line = fleet_lines[content_index]
    
    return number_of_vehicles_line


def parse_fuel_consumption_rate_line(fuel_consumption_rate_line):
    """ Parse the fuel consumption rate line as a float """

    if fuel_consumption_rate_line is None:
        return None
    fuel_consumption_rate_content = float(fuel_consumption_rate_line.split("/")[1])

    return fuel_consumption_rate_content


def parse_fuel_tank_capacity_line(fuel_tank_capacity_line):
    """ Parse the fuel tank capacity line as a float """

    if fuel_tank_capacity_line is None:
        return None
    fuel_tank_capacity_content = float(fuel_tank_capacity_line.split("/")[1])

    return fuel_tank_capacity_content


def parse_tour_length_line(tour_length_line):
    """ Parse the tour length line as a float """

    if tour_length_line is None:
        return None
    tour_length_content = float(tour_length_line.split("/")[1])

    return tour_length_content


def parse_average_velocity_line(average_velocity_line):
    """ Parse the average velocity line as a float """

    if average_velocity_line is None:
        return None
    average_velocity_content = float(average_velocity_line.split("/")[1])

    return average_velocity_content


def parse_number_of_vehicles_line(number_of_vehicles_line):
    """ Parse the numbe of vehciles line as an integer """

    if number_of_vehicles_line is None:
        return None
    number_of_vehicles_content = int(number_of_vehicles_line.split("/")[1])

    return number_of_vehicles_content


def fetch_fuel_consumption_rate_line(fleet_lines):
    """ Get and parse the fuel consumption rate line from the given fleet lines """

    fuel_consumption_rate_line = get_fuel_consumption_rate_line(fleet_lines)
    fuel_consumption_rate_content = parse_fuel_consumption_rate_line(fuel_consumption_rate_line)

    return fuel_consumption_rate_content


def fetch_fuel_tank_capacity_line(fleet_lines):
    """ Get and parse the fuel tank capacity line from the given fleet lines """

    fuel_tank_capacity_line = get_fuel_tank_capacity_line(fleet_lines)
    fuel_tank_capacity_content = parse_fuel_tank_capacity_line(fuel_tank_capacity_line)

    return fuel_tank_capacity_content


def fetch_tour_length_line(fleet_lines):
    """ Get and parse the tour length line from the given fleet lines """

    tour_length_line = get_tour_length_line(fleet_lines)
    tour_length_content = parse_tour_length_line(tour_length_line)

    return tour_length_content


def fetch_average_velocity_line(fleet_lines):
    """ Get and parse the average velocity line from the given fleet lines """

    average_velocity_line = get_average_velocity_line(fleet_lines)
    average_velocity_content = parse_average_velocity_line(average_velocity_line)

    return average_velocity_content


def fetch_number_of_vehicles_line(fleet_lines):
    """ Get and parse the number of vehicles line from the given fleet lines """

    number_of_vehicles_line = get_number_of_vehicles_line(fleet_lines)
    number_of_vehicles_content = parse_number_of_vehicles_line(number_of_vehicles_line)

    return number_of_vehicles_content


def get_vehicle_dict(base_lines_dict):
    """ Get the vehicle information dict from the base lines dict """

    fleet_lines = get_fleet_lines(base_lines_dict)
    vehicle_dict = {
        "num_vehicles": fetch_number_of_vehicles_line(fleet_lines),
        "fuel_consumption_rate": fetch_fuel_consumption_rate_line(fleet_lines),
        "fuel_tank_capacity": fetch_fuel_tank_capacity_line(fleet_lines),
        "average_velocity": fetch_average_velocity_line(fleet_lines),
        "tour_length": fetch_tour_length_line(fleet_lines),
        
    }

    return vehicle_dict


def get_fleet_dict(base_lines_dict):
    """ Get the fleet information dict from the base lines dict """

    
    vehicle_dict = get_vehicle_dict(base_lines_dict)
    fleet_dict = {
        "vehicle_0": vehicle_dict
    }
    
    return fleet_dict
