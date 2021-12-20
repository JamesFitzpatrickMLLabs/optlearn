from optlearn.io import txt_utils


def get_info_lines(base_lines_dict):
    """ Get the info lines from the base list of lines """

    info_lines = []
    
    return info_lines


def get_dataset_line(base_lines_dict):
    """ Get the dataset line from the information lines """

    dataset_line = None

    return dataset_line


def get_name_line(base_lines_dict):
    """ Get the name line from the information lines """

    name_line = list(base_lines_dict.keys())[0]

    return name_line


def get_info_dict(base_lines_dict):
    """ Get the info bits as a dictionary of details """

    info_dict = {
        "dataset": get_dataset_line(base_lines_dict),
        "name": get_dataset_line(base_lines_dict),
    }

    return info_dict
