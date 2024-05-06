# -*- coding: utf-8 -*-
"""
MultiOT - Optimal Transport in Multilayer networks (https://github.com/cdebacco/MultiOT)

Licensed under the GNU General Public License v3.0

Note: docstrings have been generated semi-automatically
"""

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import spsolve
from src.tools import assert_mass_conservation, is_synthetic

from .dijkstra import get_ot_spreal_cost
from .dynamics import (
    get_beta_index,
    get_beta_index_real,
    check_convergence,
    compute_diff_cost,
)


def fit_dyn_sp(
    G: nx.Graph,
    M: int,
    Ns: np.ndarray,
    seedmu: int,
    betas: np.ndarray,
    delta: float,
    w: np.ndarray,
    w_real: np.ndarray,
    mu: np.ndarray,
    S: np.ndarray,
    relax: float,
    T: int,
    epsJ: float,
    epsmu: float,
    topol: str,
    verbose: bool,
    verbosetimestep: int,
):
    """
    OT dynamics for each separate inflow.

    Parameters:
     - G: Network.
     - M: Number of commodities.
     - Ns: Number of nodes in each layer.
     - seedmu: Seed for random noise initialization of conductivities.
     - betas: Betas in each layer.
     - delta: Discrete time step.
     - w: Weights for synthetic networks.
     - w_real: Real lengths.
     - mu: Initial conductivities/capacities.
     - S: Mass matrix.
     - relax: Relaxation of Laplacian pseudoinverse.
     - T: Stopping time.
     - epsJ: Convergence threshold for the cost function J.
     - epsmu: Convergence threshold for conductivities/capacities.
     - topol: Type of network topology, can be 'synthetic' or 'real'.
     - verbose: Verbose flag for additional output.
     - verbosetimestep: Frequency when to print algorithm metadata.
    """

    N, E, B, num_layers = get_network_variables_sp(G, Ns)

    if is_synthetic(topol):
        beta_index = get_beta_index(G, num_layers)
    else:
        beta_index = get_beta_index_real(G)

    betas_with_intelayer = np.array(list(betas) + [1])

    # Initialization.
    it = 0
    conv = False
    Jstack = list()
    Jotstack = list()
    mustack = list()
    Fstack = list()
    wtile = np.tile(w, (M, 1)).transpose()

    p = get_potential(mu, w, S, B, N, M, relax, seedmu)
    J, F, Fnorm = get_sp_cost_and_flux(mu, w, p, B, M, E)
    J_ot, J_sp_real = get_ot_spreal_cost(F, w, w_real, betas_with_intelayer, beta_index)

    Jstack.append(J)
    mustack.append(mu)
    Fstack.append(F)

    # Update.
    while it < T and conv is False:
        mu_new = update_mu(mu, p, wtile, B, delta)
        p = get_potential(mu_new, w, S, B, N, M, relax, seedmu)
        J_new, F, Fnorm = get_sp_cost_and_flux(mu_new, w, p, B, M, E)
        dJ = compute_diff_cost(J_new, J, delta)
        J_ot, J_sp_real = get_ot_spreal_cost(
            F, w, w_real, betas_with_intelayer, beta_index
        )

        conv = check_convergence(
            it, verbose, verbosetimestep, conv, dJ, epsJ, mu_new, mu, epsmu, delta
        )

        mu = mu_new
        J = J_new

        Jstack.append(J)
        Jotstack.append(J_ot)
        mustack.append(mu)
        Fstack.append(F)

        assert_mass_conservation(B, F, S)

        it += 1

    return J, F, J_ot, J_sp_real


def get_potential(
    mu: np.ndarray,
    w: np.ndarray,
    S: np.ndarray,
    B: csr_matrix,
    N: int,
    M: int,
    relax: float,
    seedmu: int,
):
    """
    Compute potential.

    Parameters:
     - mu: Conductivities/capacities.
     - w: Weights.
     - S: Mass matrix.
     - B: Incidence matrix.
     - N: Number of nodes.
     - M: Number of commodities.
     - relax: Relaxation of Laplacian pseudoinverse.
     - seedmu: Seed for random noise initialization of conductivities.
    """

    np.random.seed(seed=seedmu)
    p = np.zeros((N, M))

    for i in range(M):
        Li = B * diags(mu[:, i] / w) * B.T
        relax_noise = np.random.uniform(low=0.0, high=1.0)
        L_relax = Li + relax * relax_noise * identity(N)

        p[:, i] = spsolve(L_relax, S[:, i])

    return p


def get_sp_cost_and_flux(
    mu: np.ndarray, w: np.ndarray, p: np.ndarray, B: csr_matrix, M: int, E: int
):
    """
    Compute J_sp and F.

    Parameters:
     - mu: Conductivities/capacities.
     - w: Weights.
     - p: Potential.
     - B: Incidence matrix.
     - M: Number of commodities.
     - E: Number of edges.
    """

    F = np.zeros((E, M))

    for i in range(M):
        F[:, i] = csr_matrix.dot(diags(mu[:, i] / w) * B.T, p[:, i])

    Fnorm = np.linalg.norm(F, axis=1, ord=1)
    J = np.dot(w, Fnorm)

    return J, F, Fnorm


def update_mu(
    mu: np.ndarray,
    p: np.ndarray,
    wtile,
    B: csr_matrix,
    delta: float,
):
    """
    Update conductivities.

    Parameters:
     - mu: Current conductivities/capacities.
     - p: Potential.
     - wtile: Tiled weights for each commodity.
     - B: Incidence matrix.
     - delta: Discrete time step.
    """

    dp = B.T * p
    rhs = (mu * (dp**2)) / (wtile**2) - mu
    mu = mu + delta * rhs

    return mu


def get_network_variables_sp(G: nx.Graph, Ns: np.ndarray):
    """
    Compute useful variables from network.

    Parameters:
     - G: Network.
     - Ns: Number of nodes in each layer.
    """

    N = G.number_of_nodes()
    E = G.number_of_edges()
    B = csr_matrix(
        nx.incidence_matrix(G, nodelist=list(range(G.number_of_nodes())), oriented=True)
    )
    num_layers = len(Ns)

    return N, E, B, num_layers
