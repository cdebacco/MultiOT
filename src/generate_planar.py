# -*- coding: utf-8 -*-
"""
MultiOT - Optimal Transport in Multilayer networks (https://github.com/cdebacco/MultiOT)

Licensed under the GNU General Public License v3.0

Note: docstrings have been generated semi-automatically
"""

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay


def planar_graph(seedG: int, Ns: np.ndarray):
    """
    Generate a planar random graph at each layer and then snap them together into a multilater network.

    Parameters:
     - seedG: Seed for random graph generation.
     - Ns: Number of nodes in each layer.
    """

    np.random.seed(seed=seedG)

    sorted_layers, sorted_nnodes = np.argsort(Ns), np.sort(Ns)
    layer_max, nnode_max = sorted_layers[-1], sorted_nnodes[-1]
    domain = (0, 0, 1, 1)

    nodes_largest_layer = np.arange(nnode_max, dtype=int)

    # Generate largest layer of the network by randomly placing nodes in square domain.
    graph_largest_layer, pos_largest_layer = generate_largest_layer(
        nodes_largest_layer, domain
    )

    # Place nodes of all the other layers.
    nodes_all_layers, pos_all_layers = generate_all_layers(
        nodes_largest_layer, pos_largest_layer, sorted_layers, sorted_nnodes, layer_max
    )

    # Delaunay triangulation.
    edges_all_layers = delaunay_triangulation(
        sorted_layers, nodes_all_layers, pos_all_layers
    )

    # Obtain dictionary for node relabelling.
    reldict = get_relabelling_dict(nodes_all_layers, sorted_layers)

    # Rename all nodes and create layered network structure.
    total_nodes_list, total_pos_dict, total_edge_list = generate_layered_network(
        sorted_layers, reldict, pos_all_layers, edges_all_layers
    )

    # Add inter-layer edges.
    total_edge_list = add_inter_layer_edges(
        Ns, nodes_all_layers, total_edge_list, reldict
    )

    # Create multilayer network.
    G = generate_multilayer(total_nodes_list, total_pos_dict, total_edge_list, reldict)
    assert_sizes(G, edges_all_layers, nodes_all_layers)

    return G


def generate_largest_layer(nodes_largest_layer: np.ndarray, domain: tuple):
    """
    Each node of the largest layer gets randomly positioned in the given domain.

    Parameters:
     - nodes_largest_layer: Nodes to place.
     - domain: Square domain where to place nodes.
    """

    graph_largest_layer = nx.Graph()
    graph_largest_layer.add_nodes_from(nodes_largest_layer)
    xmin, ymin, xmax, ymax = domain[0], domain[1], domain[2], domain[3]
    mins = np.array([xmin, ymin])
    maxs = np.array([xmax, ymax])
    pos_largest_layer = {
        v: tuple(np.random.uniform(mins, maxs, size=2)) for v in graph_largest_layer
    }
    nx.set_node_attributes(graph_largest_layer, pos_largest_layer, "pos")

    return graph_largest_layer, pos_largest_layer


def generate_all_layers(
    nodes_largest_layer: np.ndarray,
    pos_largest_layer: dict,
    sorted_layers: np.ndarray,
    sorted_nnodes: np.ndarray,
    layer_max: int,
):
    """
    Place nodes of all the other layers.

    Parameters:
     - nodes_largest_layer: Nodes to place.
     - pos_largest_layer: Position of nodes in largest layer.
     - sorted_layers: Layers from largest to smallest.
     - sorted_nnodes: Number of nodes of layers sorted according to layer size.
     - layer_max: Index largest layer.
    """

    nodes_all_layers = dict()
    pos_all_layers = dict()
    nodes_all_layers[layer_max] = nodes_largest_layer
    pos_all_layers[layer_max] = pos_largest_layer

    for lay_nodes in list(zip(sorted_layers[:-1], sorted_nnodes[:-1])):
        nodes_layer = lay_nodes[1]
        index_layer = lay_nodes[0]
        nodes_all_layers[index_layer] = np.random.choice(
            nodes_largest_layer, size=nodes_layer, replace=False
        )
        pos_all_layers[index_layer] = {
            item[0]: item[1]
            for item in pos_largest_layer.items()
            if item[0] in nodes_all_layers[index_layer]
        }

    return nodes_all_layers, pos_all_layers


def delaunay_triangulation(
    sorted_layers: np.ndarray, nodes_all_layers: dict, pos_all_layers: dict
):
    """
    Generate edges of each layer with Delaunay triangulation of nodes.

    Parameters:
     - sorted_layers: Layers from largest to smallest.
     - nodes_all_layers: Nodes of all network layers.
     - pos_all_layers: Positions of all nodes.
    """

    edges_all_layers = dict()

    for index_layer in sorted_layers[::-1]:
        dummy_graph = nx.Graph()
        dummy_graph.add_nodes_from(nodes_all_layers[index_layer])

        postri = [
            [item[1][0], item[1][1]] for item in pos_all_layers[index_layer].items()
        ]

        # Note that degeneracies arise when four or more unique points lie on the same circle.
        tri = Delaunay(postri, furthest_site=False)
        for path in tri.simplices:
            nx.add_path(dummy_graph, path)

        layer_size = np.arange(len(nodes_all_layers[index_layer]), dtype=int)
        reldictdel = get_nodes_relabelling_delaunay(
            layer_size, list(nodes_all_layers[index_layer])
        )

        edges_all_layers[index_layer] = [
            (reldictdel[e[0]], reldictdel[e[1]]) for e in dummy_graph.edges()
        ]

    return edges_all_layers


def get_nodes_relabelling_delaunay(sorted_: np.ndarray, unsorted: list):
    """
    Get dictionary to relabel nodes Delaunay triangulation.

    Parameters:
     - sorted_: Sorted nodes for layer size.
     - unsorted: Unsorted nodes.
    """

    reldictdel = dict()
    for i, sorted_node in enumerate(sorted_):
        reldictdel[sorted_node] = unsorted[i]

    return reldictdel


def get_relabelling_dict(nodes_all_layers: dict, sorted_layers: np.ndarray):
    """
    Dictionary for node relabelling in multilayer network.

    Parameters:
     - nodes_all_layers: Nodes of all network layers.
     - sorted_layers: Layers from largest to smallest.
    """

    previous_layer_size = 0
    reldict = dict()
    for index_layer in sorted_layers[::-1]:
        nodes_sorted = (
            np.arange(len(nodes_all_layers[index_layer]), dtype=int)
            + previous_layer_size
        )
        previous_layer_size += len(nodes_sorted)

        reldict = get_nodes_relabelling_dict(
            reldict, nodes_all_layers, nodes_sorted, index_layer
        )

    return reldict


def get_nodes_relabelling_dict(
    reldict: dict, nodes_all_layers: dict, nodes_sorted: np.ndarray, index_layer: int
):
    """
    Get dictionary to relabel nods in different label sequentially.

    Parameters:
     - reldict: relabelling dictionary.
     - nodes_sorted: Nodes of all network layers sorted according to layer size..
     - index_layer: Index of network layer.
    """

    for idx_nodes in enumerate(nodes_sorted):
        reldict[(nodes_all_layers[index_layer][idx_nodes[0]], index_layer)] = idx_nodes[
            1
        ]

    return reldict


def generate_layered_network(
    sorted_layers: np.ndarray,
    reldict: dict,
    pos_all_layers: dict,
    edges_all_layers: dict,
):
    """
    Generate layers of network in unique structure.

    Parameters:
     - sorted_layers: Layers from largest to smallest.
     - reldict: Relabelling dictionary for nodes.
     - pos_all_layers: Positions of all nodes.
     - edges_all_layers: Edges of all network layers.
    """

    total_nodes_list = list(reldict.values())
    total_edge_list = list()
    total_pos_dict = dict()
    for index_layer in sorted_layers[::-1]:
        for e in edges_all_layers[index_layer]:
            mapvalue0 = reldict[(e[0], index_layer)]
            mapvalue1 = reldict[(e[1], index_layer)]
            total_edge_list.append(tuple(sorted((mapvalue0, mapvalue1))))
        for pos in pos_all_layers[index_layer].items():
            mapvalue = reldict[(pos[0], index_layer)]
            total_pos_dict[mapvalue] = pos[1]

    return total_nodes_list, total_pos_dict, total_edge_list


def add_inter_layer_edges(
    Ns: np.ndarray, nodes_all_layers: dict, total_edge_list: list, reldict: dict
):
    """
    Add inter-layer edges.

    Parameters:
     - Ns: Number of nodes in each layer.
     - nodes_all_layers: Nodes of all network layers.
     - total_edge_list: Edge list.
     - reldict: Relabelling dictionary for nodes.
    """

    number_of_layers = len(Ns)
    total_inter_edges = list()
    for index_layer in list(zip(range(number_of_layers), range(number_of_layers)[1:])):
        nodes_layer_0 = nodes_all_layers[index_layer[0]]
        nodes_layer_1 = nodes_all_layers[index_layer[1]]
        common_nodes = list(set(nodes_layer_0).intersection(nodes_layer_1))
        for com_node in common_nodes:
            e_0 = reldict[(com_node, index_layer[0])]
            e_1 = reldict[(com_node, index_layer[1])]
            total_inter_edges.append(tuple(sorted((e_0, e_1))))

    total_edge_list += total_inter_edges

    return total_edge_list


def assert_sizes(G: nx.Graph, edges_all_layers: dict, nodes_all_layers: dict):
    """
    Check that number of nodes and number of edges in multilayer network are consistent.

    Parameters:
     - G: Newtork.
     - edges_all_layers: Edges of all network layers.
     - nodes_all_layers: Nodes of all network layers.
    """

    nodes_check = list()
    for nodes in nodes_all_layers.values():
        nodes_check.append(list(nodes))
    nodes_check = len(sum(nodes_check, []))

    edges_check = list()
    for edges in edges_all_layers.values():
        edges_check.append(list(edges))
    edges_check = len(sum(edges_check, []))

    number_of_layers = len(nodes_all_layers)
    range_layers = range(number_of_layers)
    interstection_lengths = 0
    for ij in list(zip(range_layers, range_layers[1:])):
        i, j = ij[0], ij[1]
        set1 = set(nodes_all_layers[i])
        set2 = set(nodes_all_layers[j])
        interstection_lengths += len(set1.intersection(set2))
    edges_check += interstection_lengths

    assert G.number_of_nodes() == nodes_check
    assert G.number_of_edges() == edges_check

    return None


def generate_multilayer(
    total_nodes_list: list, total_pos_dict: dict, total_edge_list: list, reldict: dict
):
    """
    Generate final multilayer network.

    Parameters:
     - total_nodes_list: Final list of all nodes in multilayer network.
     - total_pos_dict: Final list of all nodes' position in multilayer network.
     - total_edge_list: Final list of all edges in multilayer network.
     - reldict: Relabelling dictionary for nodes.
    """

    G = nx.Graph()
    G.add_nodes_from(total_nodes_list)
    original_nodes = {v: k for k, v in list(zip(reldict.keys(), reldict.values()))}
    nx.set_node_attributes(G, original_nodes, "original_label")
    nx.set_node_attributes(G, total_pos_dict, "pos")
    G.add_edges_from(total_edge_list)

    return G
