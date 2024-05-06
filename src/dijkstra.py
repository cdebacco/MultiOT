# -*- coding: utf-8 -*-
"""
MultiOT - Optimal Transport in Multilayer networks (https://github.com/cdebacco/MultiOT)

Licensed under the GNU General Public License v3.0

Note: docstrings have been generated semi-automatically
"""

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

from .dynamics import (
    get_beta_index,
    get_beta_index_real,
    is_synthetic,
    get_network_variables,
)


def fit_dijkstra(
    topol: str,
    G: nx.Graph,
    w: np.ndarray,
    w_real: np.ndarray,
    S: np.ndarray,
    M: int,
    Ns: np.ndarray,
    betas: np.ndarray,
    verbose: bool,
):
    """
    Find shortest path fluxes and cost with Dijkstra.

    Parameters:
     - topol: Type of network topology, can be 'synthetic' or 'real'.
     - G: Network.
     - w: Inverse velocity for all layers.
     - w_real: Inverse velocity for real flows.
     - S: Mass matrix.
     - M: Number of commodities.
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
        G, S, sources, sinks, edge_index, w_dict, M, verbose, edges_list, B
    )
    J_dijkstra, J_dijkstra_real = get_ot_spreal_cost(
        F_sp, w, w_real, betas_with_intelayer, beta_index
    )

    return J_sp, F_sp, J_dijkstra, J_dijkstra_real


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
    M: int,
    verbose: bool,
    edges_list: list,
    B: csr_matrix,
):
    """
    Compute shortest path fluxes and transport cost.

    Parameters:
     - G: Network.
     - S: Mass matrix.
     - sources: List of source nodes.
     - sinks: List of sink nodes.
     - edge_index: Dictionary mapping edges to their corresponding index.
     - w_dict: Dictionary of weights for different edge types.
     - M: Number of layers in the network.
     - verbose: Verbose flag for additional output.
     - edges_list: List of edges in the network.
     - B: Incidence matrix of the graph.
    """

    J_sp = 0
    F_sp = np.zeros((G.number_of_edges(), M))
    for i_index, s in enumerate(S.T):
        # Compute shortest path only if there is inflowing mass.
        index_source = list(np.where(s > 0)[0])
        if len(index_source) >= 1:
            single_so = sources[i_index][0]
            si = sinks[i_index]

            if verbose is True:
                print("so / |sources|:", i_index / len(sources))

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
                        F_sp[edge_index[e]][i_index] += edge_orientation * abs(
                            s[single_si]
                        )
                    else:
                        J_sp += w_dict[(e[1], e[0])] * abs(s[single_si])
                        edge_orientation = B[e[0], edge_index[(e[1], e[0])]]
                        F_sp[edge_index[(e[1], e[0])]][
                            i_index
                        ] += edge_orientation * abs(s[single_si])

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
     - F_sp: Flux vector for shortest path dynamics.
     - w: Weights.
     - w_real: Real lengths.
     - betas_with_intelayer: Beta for each layer.
     - beta_index: Index of Beta for each edge.
    """

    Fnorm = np.linalg.norm(F_sp, axis=1, ord=2)
    Fnorm1 = np.linalg.norm(F_sp, axis=1, ord=1)

    gamma_exp = (
        2
        * (2 - betas_with_intelayer[beta_index])
        / (3 - betas_with_intelayer[beta_index])
    )

    J = np.dot(w, (Fnorm**gamma_exp))
    J_real = np.dot(w_real, Fnorm1)

    return J, J_real
