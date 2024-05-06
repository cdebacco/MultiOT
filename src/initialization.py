# -*- coding: utf-8 -*-
"""
MultiOT - Optimal Transport in Multilayer networks (https://github.com/cdebacco/MultiOT)

Licensed under the GNU General Public License v3.0

Note: docstrings have been generated semi-automatically
"""

import networkx as nx
import numpy as np
from scipy.spatial import distance
from src.generate_planar import planar_graph
from src.tools import is_synthetic, assert_mass_matrix, openfile


def multiot_init(
    topol: str,
    seedG: int,
    Ns: np.ndarray,
    ws: np.ndarray,
    seedmu: int,
    p: float,
    seedS: int,
    ifolder: str,
):
    """
    Initialize the multi-commodity optimal transport simulation.

    Parameters:
     - topol: Type of network topology, can be 'synthetic' or 'real'.
     - seedG: Seed for random graph generation.
     - Ns: Number of nodes in each layer.
     - ws: Inverse velocity for all layers (also referred to as alpha for effective lengths).
     - seedmu: Seed for random noise initialization of conductivities.
     - p: Monocentric/random inflows of mass for synthetic networks.
     - seedS: Random seed for mass matrix.
     - ifolder: Input folder containing data.
    """

    G = nx.Graph()

    # Contstruct/load network topology.
    if is_synthetic(topol):
        G = planar_graph(seedG, Ns)
    elif not is_synthetic(topol):
        network_file_name = ifolder + topol + "_network.pkl"
        G = openfile(network_file_name)

    # Initialize edge weights and conductivities.
    if is_synthetic(topol):
        weff, w = init_weights(G, ws)
    else:
        w_file_name = ifolder + topol + "_w.pkl"
        w = openfile(w_file_name)
        weff = get_effective_lengths_real(G, w, ws)

    mu = init_conductivities(G, seedmu)

    # Initalize/load forcing function.
    if is_synthetic(topol):
        S = init_forcing(G, p, Ns, seedS)
    else:
        S_file_name = ifolder + topol + "_S.pkl"
        S = openfile(S_file_name)
        assert_mass_matrix(S)

    return G, weff, w, mu, S


def init_weights(G: nx.Graph, ws: np.ndarray):
    """
    Initialize edge weights.

    Parameters:
     - G: Network.
     - ws: Inverse velocity for all layers (also referred to as alpha for effective lengths).
    """

    # Euclidean lengths.
    euclidean_lengths = get_euclidean_lengths(G)

    # Effective lengths with inverse velocities.
    effective_lengths = get_effective_lengths(G, euclidean_lengths, ws)

    # Add non-zero term to inter-layer edges.
    weff = add_inter_layer_lengths(effective_lengths)
    w = add_inter_layer_lengths(euclidean_lengths)

    return weff, w


def init_conductivities(G: nx.Graph, seedmu: int):
    """
    Initialize edge conductivities.

    Parameters:
     - G: network.
     - seedmu: Seed for random noise initialization of conductivities.
    """

    np.random.seed(seed=seedmu)

    return np.ones(G.number_of_edges()) + 1e-1 * np.random.uniform(
        low=0.0, high=1.0, size=G.number_of_edges()
    )


def init_forcing(G: nx.Graph, p: float, Ns: np.ndarray, seedS: int):
    """
    Initialize forcing term S.
    Setting p = 0 returns a monocentric forcing at each layer, p = 1 corresponds to random inflows init.
    All nodes are taken as inflows of mass, every entry node corresponds to a different commodity index i.

    Parameters:
     - G: Network.
     - p: Monocentric/random inflows of mass parameter.
     - Ns: Number of nodes in each layer, with each entry corresponding to a different commodity index i.
     - seedS: Seed for random choice of sources/sinks.
    """

    closest_to_centroid, nodes_each_layer = find_central_nodes(G, Ns)
    S = build_mass_matrix(G, p, closest_to_centroid, nodes_each_layer, Ns, seedS)
    assert_mass_matrix(S)

    return S


def get_euclidean_lengths(G: nx.Graph):
    """
    Compute Euclidean lengths. Inter-layer edges have automatically length zero.

    Parameters:
     - G: Network.
    """

    euclidean_lengths = np.array(
        [
            distance.euclidean(G.nodes[edge[0]]["pos"], G.nodes[edge[1]]["pos"])
            for edge in G.edges()
        ]
    )

    return euclidean_lengths


def get_effective_lengths(G: nx.Graph, euclidean_lengths: np.ndarray, ws: np.ndarray):
    """
    Compute effective lengths using inverse velocities.

    Parameters:
     - G: Network.
     - euclidean_lengths: An array of Euclidean lengths.
     - ws: An array of inverse velocities for all layers.
    """

    inverse_velocity_edges = list()
    for edge in G.edges():
        layer_first_node = G.nodes[edge[0]]["original_label"][1]
        layer_second_node = G.nodes[edge[1]]["original_label"][1]
        if layer_first_node == layer_second_node:
            inverse_velocity_edges.append(ws[layer_first_node])
        else:
            inverse_velocity_edges.append(0)
    inverse_velocity_edges = np.array(inverse_velocity_edges)

    effective_lengths = euclidean_lengths * inverse_velocity_edges

    return effective_lengths


def get_effective_lengths_real(
    G: nx.Graph, euclidean_lengths: np.ndarray, ws: np.ndarray
):
    """
    Compute effective lengths using inverse velocities in real network.

    Parameters:
     - G: Network.
     - euclidean_lengths: An array of Euclidean lengths.
     - ws: An array of inverse velocities for all layers.
    """

    inverse_velocity_edges = list()
    for edge in G.edges():
        edge_label = G.edges[edge]["label"]
        if edge_label == "car":
            inverse_velocity_edges.append(ws[0])
        if edge_label == "bike":
            inverse_velocity_edges.append(ws[1])
        if edge_label == "inter":
            inverse_velocity_edges.append(1)
    inverse_velocity_edges = np.array(inverse_velocity_edges)

    effective_lengths = euclidean_lengths * inverse_velocity_edges

    return effective_lengths


def add_inter_layer_lengths(effective_lengths: np.ndarray):
    """
    Add non-zero small term to inter layer edges.

    Parameters:
     - effective_lengths: An array of effective lengths, including inter-layer edges.
    """

    samll_length = 1e-3 * min(effective_lengths[effective_lengths > 0])
    effective_lengths[effective_lengths == 0] = samll_length

    return effective_lengths


def find_central_nodes(G: nx.Graph, Ns: np.ndarray):
    """
    Add non-zero small term to inter layer edges.

    Parameters:
     - G: Network.
     - Ns: Number of nodes in each layer.
    """

    number_of_layers = len(Ns)
    pos_nodes_layers = {layer_index: list() for layer_index in range(number_of_layers)}
    nodes_each_layer = {layer_index: list() for layer_index in range(number_of_layers)}
    centroid_layers = {
        layer_index: np.zeros(2) for layer_index in range(number_of_layers)
    }
    closest_to_centroid = {layer_index: 0 for layer_index in range(number_of_layers)}

    # Compute centroid.
    for node in G.nodes():
        layer_of_node = G.nodes[node]["original_label"][1]
        pos_nodes_layers[layer_of_node].append(np.array(G.nodes[node]["pos"]))
        nodes_each_layer[layer_of_node].append(node)
    for layer_index in range(number_of_layers):
        centroid_layers[layer_index] = np.mean(
            np.array(pos_nodes_layers[layer_index]), axis=0
        )

    # Find closest point to centroid for each layer with global indexing.
    for layer_index in range(number_of_layers):
        argmin_distance = np.argmin(
            [
                distance.euclidean(centroid_layers[layer_index], pos)
                for pos in pos_nodes_layers[layer_index]
            ]
        )
        closest_to_centroid[layer_index] = nodes_each_layer[layer_index][
            argmin_distance
        ]

    return closest_to_centroid, nodes_each_layer


def build_mass_matrix(
    G: nx.Graph,
    p: float,
    closest_to_centroid: dict,
    nodes_each_layer: dict,
    Ns: np.ndarray,
    seedS: int,
):
    """
    Build right-hand side of Kirchhoff's law.

    Parameters:
     - G: Network.
     - p: Monocentric/random inflows of mass parameter.
     - closest_to_centroid: Dictionary containing closest nodes to the centroid.
     - nodes_each_layer: Dictionary containing the nodes in each layer.
     - Ns: Number of nodes in each layer.
     - seedS: Seed for random choice of sources/sinks.
    """

    S = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
    np.random.seed(seed=seedS)

    number_of_layers = len(Ns)
    nodes_inflows_layers = {
        layer_index: list() for layer_index in range(number_of_layers)
    }

    # All nodes are positive inflows but the central ones.
    for layer_index in range(number_of_layers):
        nodes_inflows_layers[layer_index] = list(
            set(nodes_each_layer[layer_index]) - {closest_to_centroid[layer_index]}
        )

    # All nodes are positive inflows but the central one.
    for layer_index in range(number_of_layers):
        for source in nodes_inflows_layers[layer_index]:
            r = np.random.uniform(low=0.0, high=1.0)
            if r < p:
                # Choose random number of sinks at random from all possible nodes.

                # Random number of sinks.
                all_sinks_to_choose = list(
                    (set(nodes_inflows_layers[layer_index]) - {source}).union(
                        {closest_to_centroid[layer_index]}
                    )
                )
                number_of_sinks = np.random.choice(
                    np.arange(1, len(all_sinks_to_choose) + 1, dtype=int)
                )

                # Random selection of sinks.
                sink = np.random.choice(
                    all_sinks_to_choose, size=number_of_sinks, replace=False
                )

                # Distribute outflows with zero-sum small noise.
                if len(sink) > 1:
                    xi = np.random.rand(len(sink))
                    xi -= np.mean(xi)
                    xi /= max(abs(xi))
                    xi *= 1e-2

                    S[source, source] = 1
                    S[sink, source] = -1 / len(sink) + xi

                else:
                    S[source, source] = 1
                    S[sink, source] = -1

            else:
                # Set sink at closest point to centroid.
                sink = closest_to_centroid[layer_index]

                S[source, source] = 1
                S[sink, source] = -1

    return S
