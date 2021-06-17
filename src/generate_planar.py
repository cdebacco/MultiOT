from libpysal import weights, examples
from libpysal.cg import voronoi_frames
#import geopandas
import networkx as nx
import numpy as np
from itertools import combinations
#from math import  sqrt
import random

def planar_graph(nnode, G0 = None, subN = None, L_max=None, L_min = None, domain=(0, 0, 1, 1), metric=None, seed=10, connected_component=True):
    """Returns a planar random graph modified so that only egdes within a distance are possible

    """

    prng = np.random.RandomState(seed)
    if G0 is None:
        G = nx.Graph()
        G.add_nodes_from(np.arange(nnode))
        (xmin, ymin, xmax, ymax) = domain
        # Each node gets a uniformly random position in the given rectangle.
        pos = {v: (prng.uniform(xmin, xmax), prng.uniform(ymin, ymax)) for v in G}
        nx.set_node_attributes(G, pos, "pos")
    else:
        G = nx.Graph()
        G.add_nodes_from(G0.nodes(data=True))
        nnode = G.number_of_nodes()
        pos = nx.get_node_attributes(G, "pos")

    if subN is None:
        subN = nnode

    if subN < nnode:
        random.seed(seed)
        node2remove = random.sample(list(G.nodes()),k = nnode-subN)
        G.remove_nodes_from(node2remove)

    ### Extracts planar graph
    nodes = list(G.nodes())
    coordinates = np.array([(pos[n][0],pos[n][1]) for n in list(G.nodes())])
    cells, generators = voronoi_frames(coordinates, clip="convex hull")
    delaunay = weights.Rook.from_dataframe(cells)
    G = delaunay.to_networkx()
    G = nx.relabel_nodes(G, { i: nodes[i] for i in range(G.number_of_nodes())})
    positions = { n: coordinates[i] for i,n in enumerate(list(G.nodes())) }
    nx.set_node_attributes(G, positions, "pos")

    # If no distance metric is provided, use Euclidean distance.
    if metric is None:
        metric = euclidean

    if L_max is None:
        L_max = max(metric(x, y) for x, y in combinations(positions.values(), 2))
    if L_min is None:
        L_min = 0

    def dist(u, v):
        return metric(positions[u], positions[v])

    edges2remove = [e for e in list(G.edges()) if np.logical_or(dist(*e) > L_max,dist(*e) < L_min ) == True]
    G.remove_edges_from(edges2remove)

    if connected_component == True:
        Gc = max(nx.connected_components(G), key=len)
        nodes_to_remove = set(G.nodes()).difference(Gc)
        G.remove_nodes_from(list(nodes_to_remove))

    return G



def euclidean(x, y):
    """Returns the Euclidean distance between the vectors ``x`` and ``y``.

    Each of ``x`` and ``y`` can be any iterable of numbers. The
    iterables must be of the same length.

    """
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))