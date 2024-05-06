# -*- coding: utf-8 -*-
"""
MultiOT - Optimal Transport in Multilayer networks (https://github.com/cdebacco/MultiOT)

Licensed under the GNU General Public License v3.0

Note: docstrings have been generated semi-automatically
"""

import pickle as pkl
from typing import Union, Optional

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix


def get_array_from_string(s: str):
    """
    Convert string separated by space in np.ndarray

    Parameters:
     - s: String.
    """

    return np.array(s.split(" "))


def is_synthetic(topol: str):
    """
    Check if topology is synthetic.

    Parameters:
     - topol: Chosen network topology.
    """

    if topol == "synthetic":
        return True
    elif topol == "real":
        return False


def serialize(data, opath: str, fname: str):
    """
    Serialize output.

    Parameters:
     - data: Data to serialize.
     - fname: File name for output folder.
    """

    with open(
        opath + fname,
        "wb",
    ) as result_folder:
        pkl.dump(data, result_folder, protocol=pkl.HIGHEST_PROTOCOL)

    return None


def openfile(fname: str):
    """
    Open pickled file.

    Parameters:
     - fname: File name.
    """

    file = open(fname, "rb")
    data = pkl.load(file)

    return data


def assert_mass_matrix(S: np.ndarray):
    """
    Check if columns of mass matrix sum to zero.

    Parameters:
     - S: Mass matrix.
    """

    for s in S.T:
        if abs(np.sum(s)) < 1e-12:
            pass
        else:
            raise ValueError("[ERROR] S in not well-defined")

    return None


def assert_mass_conservation(B: csr_matrix, F: np.ndarray, S: np.ndarray):
    """
    Raise Exception if mass is not balanced in OT updates.

    Parameters:
     - B: Incidence matrix.
     - F: Fluxes.
     - S: Mass Matrix.
    """

    eps = 1.0e-4
    check = np.max(abs(B * F - S))

    if check < eps:
        pass
    else:
        raise ValueError(f"[ERROR] mass is not conserved: {check} > {eps}")

    return None


def print_verbose_params(
    verbose: bool, verbosetimestep: int, it: int, dmu_max: float, dJ: float
):
    """
    Print parameters to evaluate convergence.

    Parameters:
     - verbose: Print metadata algorithm.
     - verbosetimestep: Frequency to print metadata.
     - it: Iteration number.
     - dmu_max: Max difference of conductivities.
     - dJ: Cost difference.
    """

    if verbose and it % verbosetimestep == 0:
        print("IT:", it, "delta_mu:", round(dmu_max, 12), "dJ:", round(dJ, 12))

    return None


def assert_j_decrease(J: float, Jnew: float):
    """
     Raise Exception if transportation cost does not decrease.

    Parameters:
      - J: Old cost.
      - Jnew: Updated cost.
    """

    if Jnew < J:
        conv = False
        unstable_result = False
    else:
        conv = True
        unstable_result = True

    return conv, unstable_result
