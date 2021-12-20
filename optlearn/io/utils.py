def build_string(items):
    """ Build a string from a bunch of items, witn a newline at the end """

    empty = "{} " * len(items) 
    return empty[:-1].format(*items) + "\n"


def read_file_into_list(fname):
    """ Read a file into a list """

    with open(fname) as f:
        lines = f.readlines()
    return lines


def find_substring_in_stringlist(stringlist, substring):
    """ Find the first time a substring appears in a list of strings """

    for num, item in enumerate(stringlist):
        if substring in item:
            return num
    raise ValueError("Substring not found in the stringlist!")
