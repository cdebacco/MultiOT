# -*- coding: utf-8 -*-
"""
MultiOT - Optimal Transport in Multilayer networks (https://github.com/cdebacco/MultiOT)

Licensed under the GNU General Public License v3.0

Note: docstrings have been generated semi-automatically
"""

import os

packages = [
    "jupyterlab",
    "notebook",
    "matplotlib",
    "pandas",
    "networkx",
    "scipy",
    "libpysal",
    "pysal",
]

for st_ in packages:
    os.system("pip install " + st_)

# os.system("python3 -m pip install -U --pre shapely")
# os.system("pip install geopandas")
