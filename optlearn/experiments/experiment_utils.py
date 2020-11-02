import os


def list_files_recursive(directory):
    """ Get all files in each subdirectory """

    directories = [os.path.join(directory, item) for item in os.listdir(directory)]

    filenames = []

    for item in directories:
        files = os.listdir(item)
        files = [os.path.join(item, file) for file in files]
        filenames += files

    return filenames
