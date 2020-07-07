import os
import pathlib
import pkg_resources

from setuptools import setup

version = "?.?.?"
local_dir = os.path.dirname(__file__)
if len(local_dir) == 0:
    local_dir = "."
try:
    with open(os.sep.join([local_dir, "VERSION"]), "r") as fid:
        version = fid.read().strip()
except:
    pass

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]


setup(
    name='optlearn',
    version=version,
    install_requires=install_requires,
)

os.system("pip3 install git+https://github.com/jvkersch/pyconcorde.git")
