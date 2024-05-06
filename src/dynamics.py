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

from src.tools import (
    is_synthetic,
    assert_mass_conservation,
    assert_j_decrease,
    print_verbose_params,
)


def fit(
    G: nx.Graph,
    Ns: np.ndarray,
    betas: np.ndarray,
    delta: float,
    w: np.ndarray,
    mu: np.ndarray,
    S: np.ndarray,
    relax: float,
    seedmu: int,
    T: int,
    epsJ: float,
    epsmu: float,
    topol: str,
    verbose: bool,
    verbosetimestep: int,
):
    """
    Update conductivities.

    Parameters:
     - G: Network.
     - Ns: Number of nodes in each layer.
     - betas: Betas in each layer.
     - delta: Discrete time step.
     - w: Weights.
     - mu: Conductivities/capacities.
     - S: Mass matrix.
     - relax: Relaxation of Laplacian pseudoinverse.
     - T: Stopping time.
     - epsJ: convergence threshold J.
     - epsmu: convergence threshold mu.
     - topol: topology of network.
     - seedmu: Seed for random noise initialization of conductivities.
     - verbose: Verbose flag for additional output.
     - verbosetimestep: Frequency when to print algorithm metadata.
    """

    N, B, num_layers = get_network_variables(G, Ns)
    if is_synthetic(topol):
        beta_index = get_beta_index(G, num_layers)
    else:
        beta_index = get_beta_index_real(G)

    betas_with_intelayer = np.array(list(betas) + [1])

    # Initialization.
    it = 0
    conv = False
    conv_unstable = False
    unstable_result = False
    Jstack = list()
    Jsp_dyn_stack = list()
    mustack = list()
    Fstack = list()
    results = dict()

    p = get_potential(N, mu, w, S, B, relax, seedmu)
    J, F, Fnorm = get_ot_cost_and_flux(betas_with_intelayer, beta_index, mu, w, p, B)
    Jsp_dyn = get_sp_cost(F, w)

    Jstack.append(J)
    Jsp_dyn_stack.append(Jsp_dyn)
    mustack.append(mu)
    Fstack.append(F)

    # Update.
    while it < T and conv is False and conv_unstable is False:
        mu_new = update_mu(mu, Fnorm, delta, betas_with_intelayer, beta_index)

        p = get_potential(N, mu_new, w, S, B, relax, seedmu)
        J_new, F, Fnorm = get_ot_cost_and_flux(
            betas_with_intelayer, beta_index, mu_new, w, p, B
        )
        Jsp_dyn = get_sp_cost(F, w)
        dJ = compute_diff_cost(J_new, J, delta)

        conv = check_convergence(
            it, verbose, verbosetimestep, conv, dJ, epsJ, mu_new, mu, epsmu, delta
        )

        assert_mass_conservation(B, F, S)
        # Add assertion for more stable results.
        conv_unstable, unstable_result = assert_j_decrease(J, J_new)

        mu = mu_new
        J = J_new

        Jstack.append(J)
        Jsp_dyn_stack.append(Jsp_dyn)
        mustack.append(mu)
        Fstack.append(F)

        it += 1

    results["mu"] = mu
    results["F_dyn"] = F
    results["J_dyn"] = J
    results["J_sp_dyn"] = Jsp_dyn
    results["unstable_result"] = unstable_result

    return results


def get_network_variables(G: nx.Graph, Ns: np.ndarray):
    """
    Compute useful variables from network.

    Parameters:
     - G: Network.
     - Ns: Number of nodes in each layer.
    """

    N = G.number_of_nodes()
    B = csr_matrix(
        nx.incidence_matrix(G, nodelist=list(range(G.number_of_nodes())), oriented=True)
    )
    num_layers = len(Ns)

    return N, B, num_layers


def get_beta_index(G: nx.Graph, num_layers: int):
    """
    Find index of beta for each edge.

    Parameters:
     - G: Network.
     - num_layers: Number of layers.
    """

    beta_index = list()
    for edge in G.edges():
        layeri = G.nodes[edge[0]]["original_label"][1]
        layerj = G.nodes[edge[1]]["original_label"][1]
        if layeri == layerj:
            beta_index.append(layeri)
        else:
            beta_index.append(num_layers)

    return beta_index


def get_beta_index_real(G: nx.Graph):
    """
    Find index of beta for each edge.

    - G: Network.
    """

    beta_index = list()
    for edge in G.edges():
        edge_label = G.edges[edge]["label"]
        if edge_label == "car":
            beta_index.append(0)
        if edge_label == "bike":
            beta_index.append(1)
        if edge_label == "inter":
            beta_index.append(2)

    return beta_index


def get_potential(
    N: int,
    mu: np.ndarray,
    w: np.ndarray,
    S: np.ndarray,
    B: csr_matrix,
    relax: float,
    seedmu: int,
):
    """
    Calculate the potential function.

    Parameters:
     - N: Number of nodes.
     - mu: Conductivities/capacities.
     - w: Weights.
     - S: Mass matrix.
     - B: Incidence matrix.
     - relax: Relaxation of Laplacian pseudoinverse.
     - seedmu: Seed for random noise initialization of conductivities.
    """

    np.random.seed(seed=seedmu)

    L = B * diags(mu / w) * B.T
    relax_noise = np.random.uniform(low=0.0, high=1.0)
    L_relax = L + relax * relax_noise * identity(N)

    p = spsolve(L_relax, S)

    return p


def get_ot_cost_and_flux(
    betas_with_intelayer: np.ndarray,
    beta_index: list,
    mu: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    B: csr_matrix,
):
    """
    Compute J and F.

    Parameters:
     - betas_with_intelayer: Beta values with inter-layer weights.
     - beta_index: Index of beta for each edge.
     - mu: Conductivities/capacities.
     - w: Weights.
     - p: Potential.
     - B: Incidence matrix.
    """

    F = csr_matrix.dot((diags(mu / w) * B.T), p)
    Fnorm = np.linalg.norm(F, axis=1, ord=2)

    gamma_exp = (
        2
        * (2 - betas_with_intelayer[beta_index])
        / (3 - betas_with_intelayer[beta_index])
    )

    J = np.dot(w, (Fnorm**gamma_exp))

    return J, F, Fnorm


def update_mu(
    mu: np.ndarray,
    Fnorm: np.ndarray,
    delta: float,
    betas_with_intelayer: np.ndarray,
    beta_index: list,
):
    """
    Update conductivities.

    Parameters:
     - mu: Initial conductivities/capacities.
     - Fnorm: Norm of the flux.
     - delta: Discrete time step.
     - betas_with_intelayer: Beta values with inter-layer weights.
     - beta_index: Index of beta for each edge.
    """

    beta_exp = betas_with_intelayer[beta_index] - 2
    rhs = mu**beta_exp * Fnorm**2 - mu
    mu = mu + delta * rhs

    return mu


def compute_diff_cost(
    cost_new: float,
    cost: float,
    delta: float,
):
    """
    Compute difference of J between two consecutive time steps.

    Parameters:
     - cost_new: New cost value.
     - cost: Current cost value.
     - delta: Discrete time step.
    """

    return abs(cost - cost_new) / delta


def check_convergence(
    it: int,
    verbose: bool,
    verbosetimestep: int,
    conv: bool,
    dJ: float,
    epsJ: float,
    mu_new: np.ndarray,
    mu: np.ndarray,
    epsmu: float,
    delta: float,
):
    """
    Check convergence of MultiOT.

    Parameters:
     - it: Current iteration.
     - verbose: Verbose flag for additional output.
     - verbosetimestep: Frequency when to print algorithm metadata.
     - conv: Flag indicating if convergence is achieved.
     - dJ: Difference in cost (J) between consecutive time steps.
     - epsJ: Convergence threshold for J.
     - mu_new: Updated conductivities/capacities.
     - mu: Current conductivities/capacities.
     - epsmu: Convergence threshold for mu.
     - delta: Discrete time step.
    """

    conv_J = False
    conv_mu = False

    if it >= 1:
        if dJ < epsJ:
            conv_J = True

        dmu_max = np.max(abs(mu_new - mu)) / delta

        if dmu_max < epsmu:
            conv_mu = True

        print_verbose_params(verbose, verbosetimestep, it, dmu_max, dJ)

    if conv_mu is True and conv_J is True:
        conv = True

    return conv


def get_sp_cost(F: np.ndarray, w: np.ndarray):
    """
    Compute 1-norm cost.

    Parameters:
     - F: Flux matrix.
     - w: Weights.
    """

    Fnorm1 = np.linalg.norm(F, axis=1, ord=1)

    return np.dot(w, Fnorm1)
