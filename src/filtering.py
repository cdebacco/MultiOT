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
from src.tools import assert_mass_conservation

from .dynamics import (
    is_synthetic,
    get_beta_index,
    get_beta_index_real,
    get_network_variables,
    check_convergence,
    compute_diff_cost,
)
from .initialization import init_conductivities


def filtering_sp(
    topol: str,
    G: nx.Graph,
    w: np.ndarray,
    w_real: np.ndarray,
    S: np.ndarray,
    M: int,
    Ns: np.ndarray,
    betas: np.ndarray,
    verbose: bool,
    F_dyn_sp: np.ndarray,
    tau_filtering: float,
    seedmu: int,
    delta: float,
    relax: float,
    T: int,
    epsJ: float,
    epsmu: float,
    verbosetimestep: int,
):
    """
    Filter shortest path fluxes and cost with Dijstra or OT dynamics.

    Parameters:
     - topol: Type of network topology, can be 'synthetic' or 'real'.
     - G: Network.
     - w: Weights.
     - w_real: Real lengths.
     - Mass matrix.
     - M: Number of commodities.
     - Ns: Number of nodes in each layer.
     - betas: Betas in each layer.
     - verbose: Verbose flag for additional output.
     - F_dyn_sp: Fluxes.
     - tau_filtering: Filtering threshold for fluxes.
     - seedmu: Seed random init of conductivities.
     - delta: Discrete time step.
     - relax: Relaxation of Laplacian pseudoinverse.
     - T: Stopping time.
     - epsJ: Convergence threshold for the cost function J.
     - epsmu: Convergence threshold for conductivities/capacities.
     - verbosetimestep: Frequency when to print algorithm metadata.
    """

    edges_list = np.array(G.edges(), dtype="int,int")

    J_sp_filtered_tot = 0
    J_sp_real_filtered_tot = 0

    N, E, B, num_layers = get_network_variables_sp(G, Ns)
    F_all = np.zeros((E, M))

    # Exponents to compute OT cost.
    if is_synthetic(topol):
        beta_index = get_beta_index(G, num_layers)
    else:
        beta_index = get_beta_index_real(G)

    betas_with_intelayer = np.array(list(betas) + [1])

    edge_labels = {}
    if topol != "synthetic":
        edge_labels = {edge: G.edges[edge]["label"] for edge in G.edges()}

    for source_index in range(M):
        F_each_i = abs(F_dyn_sp[:, source_index])
        F_each_i_filtered = np.where(F_each_i >= tau_filtering, F_each_i, 0)
        index_filtered = list(np.nonzero(F_each_i_filtered)[0])
        w_filtered = w[index_filtered]
        w_real_filtered = w_real[index_filtered]
        w_filtered_dict = {
            tuple(e): w_real[i] for i, e in enumerate(edges_list[index_filtered])
        }

        G_filtered = nx.Graph()
        G_filtered.add_nodes_from(G.nodes(data=True))
        G_filtered.add_edges_from(edges_list[index_filtered])
        nx.set_edge_attributes(G_filtered, w_filtered_dict, "length")
        if topol != "synthetic":
            edge_labels_filtered = {
                tuple(e): edge_labels[tuple(e)] for e in edges_list[index_filtered]
            }
            nx.set_edge_attributes(G_filtered, edge_labels_filtered, name="label")

        try:
            nx.find_cycle(G_filtered)

            if M <= 50:
                J_sp_filtered, F_filtered, J_sp_real_filtered = fit_dijkstra_filtering(
                    topol,
                    G_filtered,
                    w_filtered,
                    w_real_filtered,
                    S,
                    source_index,
                    Ns,
                    betas,
                    verbose,
                )

            else:
                mu = init_conductivities(G_filtered, seedmu)

                (
                    J_sp_filtered,
                    F_filtered,
                    J_sp_real_filtered,
                ) = fit_dyn_filtering(
                    source_index,
                    G_filtered,
                    Ns,
                    seedmu,
                    delta,
                    w_filtered,
                    w_real_filtered,
                    mu,
                    S,
                    relax,
                    T,
                    epsJ,
                    epsmu,
                    verbose,
                    verbosetimestep,
                )

            J_sp_filtered_tot += J_sp_filtered
            J_sp_real_filtered_tot += J_sp_real_filtered
            F_all[index_filtered, source_index] = F_filtered

        except nx.NetworkXNoCycle:
            J_sp_filtered_tot += np.dot(w, F_each_i)
            J_sp_real_filtered_tot += np.dot(w_real, F_each_i)
            F_all[:, source_index] = F_dyn_sp[:, source_index]

    # OT cost after filtering
    gamma_exp = (
        2
        * (2 - betas_with_intelayer[beta_index])
        / (3 - betas_with_intelayer[beta_index])
    )
    J_ot_filtered_tot = np.dot(w, np.linalg.norm(F_all, axis=1, ord=2) ** gamma_exp)

    return J_sp_filtered_tot, F_all, J_ot_filtered_tot, J_sp_real_filtered_tot


def fit_dyn_filtering(
    source_index: int,
    G: nx.Graph,
    Ns: np.ndarray,
    seedmu: int,
    delta: float,
    w: np.ndarray,
    w_real: np.ndarray,
    mu: np.ndarray,
    S: np.ndarray,
    relax: float,
    T: int,
    epsJ: float,
    epsmu: float,
    verbose: bool,
    verbosetimestep: int,
):
    """
    OT dynamics for each separate inflow.

    Parameters:
     - source_index: Index of inflow.
     - G: Network.
     - Ns: Number of nodes in each layer.
     - seedmu: Seed for random noise initialization of conductivities.
     - delta: Discrete time step.
     - w: Weights for synthetic networks.
     - w_real: Real lengths.
     - mu: Initial conductivities/capacities.
     - S: Mass matrix.
     - relax: Relaxation of Laplacian pseudoinverse.
     - T: Stopping time.
     - epsJ: Convergence threshold for the cost function J.
     - epsmu: Convergence threshold for conductivities/capacities.
     - verbose: Verbose flag for additional output.
     - verbosetimestep: Frequency when to print algorithm metadata.
    """

    N, E, B, num_layers = get_network_variables_sp(G, Ns)

    # Initialization.
    it = 0
    conv = False
    Jstack = list()
    mustack = list()
    Fstack = list()

    p = get_potential(source_index, mu, w, S, B, N, relax, seedmu)
    J, F, Fnorm = get_sp_cost_and_flux(mu, w, p, B)
    J_sp_real = get_ot_spreal_cost_filtering(F, w_real)

    Jstack.append(J)
    mustack.append(mu)
    Fstack.append(F)

    # Update.
    while it < T and conv is False:
        mu_new = update_mu(mu, p, w, B, delta)
        p = get_potential(source_index, mu_new, w, S, B, N, relax, seedmu)
        J_new, F, Fnorm = get_sp_cost_and_flux(mu_new, w, p, B)
        dJ = compute_diff_cost(J_new, J, delta)
        J_sp_real = get_ot_spreal_cost_filtering(
            F,
            w_real,
        )

        conv = check_convergence(
            it, verbose, verbosetimestep, conv, dJ, epsJ, mu_new, mu, epsmu, delta
        )

        assert_mass_conservation(B, F, S.T[source_index])
        mu = mu_new
        J = J_new

        Jstack.append(J)
        mustack.append(mu)
        Fstack.append(F)

        it += 1

    return J, F, J_sp_real


def get_potential(
    source_index: int,
    mu: np.ndarray,
    w: np.ndarray,
    S: np.ndarray,
    B: csr_matrix,
    N: int,
    relax: float,
    seedmu: int,
):
    """
    Compute potential.

    Parameters:
     - source_index: Index of inflow.
     - mu: Conductivities/capacities.
     - w: Weights.
     - S: Mass matrix.
     - B: Incidence matrix.
     - N: number of nodes
     - relax: Relaxation of Laplacian pseudoinverse.
     - seedmu: Seed for random noise initialization of conductivities.
    """

    np.random.seed(seed=seedmu)

    L = B * diags(mu / w) * B.T
    relax_noise = np.random.uniform(low=0.0, high=1.0)
    L_relax = L + relax * relax_noise * identity(N)

    p = spsolve(L_relax, S[:, source_index])

    return p


def get_sp_cost_and_flux(
    mu: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    B: csr_matrix,
):
    """
    Compute J_sp, F and its norm.

    Parameters:
     - mu: Conductivities/capacities.
     - w: Weights.
     - p: Potential.
     - B: Incidence matrix.
    """

    F = csr_matrix.dot(diags(mu / w) * B.T, p)
    Fnorm = abs(F)
    J = np.dot(w, Fnorm)

    return J, F, Fnorm


def update_mu(
    mu: np.ndarray,
    p: np.ndarray,
    w: np.ndarray,
    B: csr_matrix,
    delta: float,
):
    """
    Update conductivities.

    Parameters:
     - mu: Conductivities/capacities.
     - p: Potential.
     - w: Weights.
     - p: Potential.
     - B: Incidence matrix.
     - delta: Discrete time step.
    """

    dp = B.T * p
    rhs = (mu * (dp**2)) / (w**2) - mu
    mu = mu + delta * rhs

    return mu


def get_ot_spreal_cost_filtering(F_sp: np.ndarray, w_real: np.ndarray):
    """
    Compute J.

    Parameters:
     - F_sp: Fluxes.
     - w_real: Real lengths.
    """

    Fnorm1 = abs(F_sp)

    J_real = np.dot(w_real, Fnorm1)

    return J_real


def fit_dijkstra_filtering(
    topol: str,
    G: nx.Graph,
    w: np.ndarray,
    w_real: np.ndarray,
    S: np.ndarray,
    source_index: int,
    Ns: np.ndarray,
    betas: np.ndarray,
    verbose: bool,
):
    """
    Find shortest path fluxes and cost with Dijstra.

    Parameters:
     - topol: Type of network topology, can be 'synthetic' or 'real'.
     - G: Network.
     - w: Inverse velocity for all layers.
     - w_real: Inverse velocity for real flows.
     - S: Mass matrix.
     - source_index: Index of inflow.
     - Ns: Number of nodes in each layer.
     - betas: Betas in each layer.
     - verbose: Verbose flag for additional output.
    """

    # Setup.
    w_dict = {e: we for e, we in list(zip(G.edges(), w))}
    edge_index = {e: n for n, e in enumerate(G.edges())}

    N, B, num_layers = get_network_variables(G, Ns)

    nx.set_edge_attributes(G, w_dict, "length")
    edges_list = list(G.edges())
    num_layers = len(Ns)
    if is_synthetic(topol):
        beta_index = get_beta_index(G, num_layers)
    else:
        beta_index = get_beta_index_real(G)
    betas_with_intelayer = np.array(list(betas) + [1])

    # Extract sources and sinks.
    sources, sinks = get_sources_sinks_mass(S)

    # Run Dijkstra's algorithm.
    J_sp, F_sp = compute_fluxes_and_cost(
        G, S, sources, sinks, edge_index, w_dict, source_index, verbose, edges_list, B
    )
    J_dijkstra, J_dijkstra_real = get_ot_spreal_cost(
        F_sp, w, w_real, betas_with_intelayer, beta_index
    )

    return J_sp, F_sp, J_dijkstra_real


def get_sources_sinks_mass(S: np.ndarray):
    """
    Extract origin and destination nodes from mass matrix.

    Parameters:
     - S: Mass matrix.
    """

    sources = list()
    sinks = list()

    for s in S.T:
        index_source = list(np.where(s > 0)[0])
        index_sinks = list(np.where(s < 0)[0])

        sources.append(index_source)
        sinks.append(index_sinks)

    return sources, sinks


def compute_fluxes_and_cost(
    G: nx.Graph,
    S: np.ndarray,
    sources: list,
    sinks: list,
    edge_index: dict,
    w_dict: dict,
    source_index: int,
    verbose: bool,
    edges_list: list,
    B: csr_matrix,
):
    """
    Compute shortest path fluxes and transport cost.

    Parameters:
     - G: Network.
     - S: mass matrix
     - sources: list of sources
     - sinks: list of sinks
     - edge_index: Mapping of edges to indexes.
     - w_dict: Edge weights.
     - source_index: Index of inflow.
     - verbose: Verbose flag for additional output.
     - edges_list: List of edges in the network.
     - B: Incidence matrix of the graph.
    """

    J_sp = 0
    F_sp = np.zeros(G.number_of_edges())
    for s in [S.T[source_index]]:
        # Compute shortest path only if there is inflowing mass.
        index_source = np.where(s > 0)[0][0]
        if len([index_source]) >= 1:
            single_so = sources[source_index][0]
            si = sinks[source_index]

            if verbose is True:
                print("so / |sources|:", source_index / len(sources))

            for single_si in si:
                # Compute shortest path.
                sp = nx.shortest_path(
                    G, source=single_so, target=single_si, weight="length"
                )

                sp_edges = [(u, v) for u, v in list(zip(sp, sp[1:]))]

                # Compute shortest path length and transport cost.
                for e in sp_edges:
                    if e in edges_list:
                        J_sp += w_dict[e] * abs(s[single_si])
                        edge_orientation = B[e[0], edge_index[e]]
                        F_sp[edge_index[e]] += edge_orientation * abs(s[single_si])
                    else:
                        J_sp += w_dict[(e[1], e[0])] * abs(s[single_si])
                        edge_orientation = B[e[0], edge_index[(e[1], e[0])]]
                        F_sp[edge_index[(e[1], e[0])]] += edge_orientation * abs(
                            s[single_si]
                        )

    return J_sp, F_sp


def get_ot_spreal_cost(
    F_sp: np.ndarray,
    w: np.ndarray,
    w_real: np.ndarray,
    betas_with_intelayer: np.ndarray,
    beta_index: list,
):
    """
    Compute J.

    Parameters:
     - F_sp: Fluxes.
     - w: Weights.
     - w_real: Real lengths.
     - betas_with_intelayer: Beta values with inter-layer weights.
     - beta_index: Index of beta for each edge.
    """

    Fnorm = abs(F_sp)

    gamma_exp = (
        2
        * (2 - betas_with_intelayer[beta_index])
        / (3 - betas_with_intelayer[beta_index])
    )

    J = np.dot(w, (Fnorm**gamma_exp))
    J_real = np.dot(w_real, Fnorm)

    return J, J_real


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
