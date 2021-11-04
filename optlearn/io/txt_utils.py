def read_txt_file(filename):
    """ Open the text file into a list """
    
    with open(filename, "r") as fid:
    	 lines = fid.readlines()

    return lines


def remove_newlines(string):
    """ Remove the newlines delimiters """

    string = string.replace("\n", "")

    return string


def split_tabs(string):
    """ Split a string along its tabs """

    split_strings = string.split("\t")

    return split_strings


def read_vrp_file_from_txt(filename):
    """ Read the VRP file from a txt file and process the input """

    lines = read_txt_file(filename)
    lines = [remove_newlines(string) for string in lines]
    lines = [split_tabs(string) for string in lines]

    return lines


def get_base_lines(base_lines_dict):
    """ Get the base lines from the base lines dict """

    base_lines = list(base_lines_dict.values())[0]

    return base_lines
