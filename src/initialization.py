"""

Optimal Transport in Multilayer network

Python implementation of the MultiOT algorithm described in:

- [1] Ibrahim, A.A.; Lonardi, A.; De Bacco, C. (2021). *Optimal transport in multilayer networks*.  

This is a an algorithm that uses optimal transport theory to find optimal path trajectories in multilayer networks. 

If you use this code please cite [1].   

Copyright (c) 2021 Abdullahi Adinoyi Ibrahim, Alessandro Lonardi and Caterina De Bacco
"""

#######################################
# PACKAGES
#########################
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import random
from scipy.spatial import distance
from scipy.sparse import coo_matrix
from scipy import sparse

import math

from itertools import combinations

from math import radians, cos, sin, asin, sqrt
#######################################

def topology_generation(topol_mode, input_path, nnode, nnode_inter, ncomm, ncomm_inter):
    """create or import topology file"""

    def continuefunc():
        print("topology file: imported")
        return 0

    switcher = {
        "0": lambda: generate_topol_file(input_path, nnode, nnode_inter, ncomm, ncomm_inter),
        "1": lambda: continuefunc()
    }

    return switcher.get(topol_mode, lambda: print("ERROR: invalid flag (topology)"))()


def generate_topol_file(input_path, nnode):
    """generate topology file"""

    print("topology file: generated")

    graph_file_path = input_path + "/graph_generated.dat"
    adjacency_file_path = input_path + "/adj_generated.dat"
    graph_coord_path = input_path + "/coord_generated.dat"
    graph_weight_path = input_path + "/weight_generated.dat"

    # generate Waxman graph
    g = nx.waxman_graph(nnode, alpha=0.25, beta=0.25, L=1.0, domain=(0, 0, 1, 1))
    coord = {i: (g.nodes[i]["pos"][0], g.nodes[i]["pos"][1]) for i in range(nnode)}
    nedge = len(g.edges)
    
    # print graph topology in file
    with open(graph_file_path, "w") as f_topol:
        f_topol.write(str(nnode) + "\n")
        f_topol.write(str(ncomm) + "\n")
        f_topol.write(str(nedge) + "\n")

    # print adjacency list in file
    with open(adjacency_file_path, "w") as f_adj:
        i = 0
        for e in g.edges():
            f_adj.write(str(e[0]) + " " + str(e[1]) + "\n")
            i += 1

    # print node coordinates (automatically generated if Waxman model)
    with open(graph_coord_path, "w") as f_coord:
        for i in range(nnode):
            f_coord.write(str(coord[i][0]) + " " + str(coord[i][1]) + "\n")
            i += 1
            
    return 0

def extract_node_layer(df):
    nodes_inter = {}
    for n0,g in df.groupby(by=['fromNode']):
        n = str(n0)
        layer_i = set(g['layer'].unique())
        if n in nodes_inter:
            nodes_inter[n] = layer_i.union(nodes_inter[n])
        else:
            nodes_inter[n] = set(layer_i)
    for n0,g in df.groupby(by=['toNode']):
        n = str(n0)
        layer_i = set(g['layer'].unique())
        if n in nodes_inter:
            nodes_inter[n] = layer_i.union(nodes_inter[n])
        else:
            nodes_inter[n] = set(layer_i)

    for n in nodes_inter:
        nodes_inter[n] = list(nodes_inter[n])

    return nodes_inter   

def file2graph(edges_file_name,sep = ' ',weight_inter = 1, w0 = 1 ):
    '''
    Generate graph from edgelist of rows:
    layerId sourceId targetId weight
    '''

    edges_file = pd.read_csv(edges_file_name, names=['layer', 'fromNode', 'toNode', 'weight'], sep=sep)

    nodes_inter = extract_node_layer(edges_file) #dict(nodeID)

    layers = list(np.unique(edges_file.layer))
    nlayer = len(layers)
    
    '''
    Build graph:
    - single-layer network
    - add dummy nodes for nodes that belong to more than one layer
    - add dummy supernode connecting them
    - inter-layer edges are between dummy nodes
    '''

    g = nx.Graph()
    for n,row in edges_file.iterrows():
        l = row[0]
        lid = layers.index(l)
        u0,v0 = str(row[1]), str(row[2])
        w = row[3]

        u = str(u0) + '_' + str(lid)
        v = str(v0) + '_' + str(lid)

        g.add_edge(u,v, weight = w, etype= lid )

    '''
    Add inter-layer edges
    '''
    for n in nodes_inter:
        if len(nodes_inter[n]) > 1:
            g.add_node(str(n), ntype='super')
            for l in nodes_inter[n]:
                u = str(n) + '_' + str(l)
                g.add_edge(str(n),u,weight=w0,etype='inter-super')
                g.nodes[u]['ntype'] = 'inter'
                # for m in nodes_inter[n]:
                #     if l != m :
                #         v = str(n) + '_' + str(m)
                #         g.add_edge(u,v,weight=weight_inter,etype='inter')
        else:
            u = str(n) + '_' + str(nodes_inter[n][0])
            g.nodes[u]['ntype'] = 'intra'


    nodes = list(g.nodes()) #list of stations
    nnode = len(nodes)

    '''
    Relabel nodes so that nodeId is the same as node name
    '''
    nodeName2Id = {}
    nodeId2Name = {}
    for i,n in enumerate(nodes): 
        nodeName2Id[n] = i
        nodeId2Name[i] = n
        if '_' in n: # inter-layer nodes
            n0 = n.split('_')[0]
            nodeName2Id[n0] = i

    g = nx.relabel_nodes(g, nodeName2Id)
    
    nodes = list(np.arange(nnode))
        
    nedge = g.number_of_edges() 

    return g, nnode, nedge, nodes, nodes_inter, nodeName2Id, nodeId2Name

############length
def eucledian_bias(length_mode, g, length_inter0 = None, haversine_on = False):
    """length assignation: bias or eucledian"""

    # length type
    switcher = {
        "bias": lambda: bias(g,length_inter0),
        "eucl": lambda: eucledian(g,length_inter0,haversine_on = haversine_on)
    }

    return switcher.get(length_mode, lambda: print("ERROR: invalid flag (length)"))()

def from_latlon_to_cartesian(pos,R=6371000):
    '''
    pos = (lat,lon)
    x = R cos(lat)cos(lon)
    y = R cos(lat)sin(lon)
    R is the Earth's radius in meters
    '''
    x = R * np.cos(pos[0]) * np.cos(pos[1])
    y = R * np.cos(pos[0]) * np.sin(pos[1])
    return (x,y)


def eucledian(g, length_inter0=None,R=6371000,haversine_on = True):
    '''
    eucledian lengths assigned to edges
    R is the Earth's radius in meters
    First convert lat/lon to cartesian coords:
    x = R cos(lat)cos(lon)
    y = R cos(lat)sin(lon)
    '''
    print("length: eucledian")
    
    length = np.zeros(len(g.edges())) 
    for i,edge in enumerate(g.edges()):
        if haversine_on == False:
            length[i] = distance.euclidean(g.nodes[edge[0]]["pos"], g.nodes[edge[1]]["pos"])
        else:
            length[i] = haversine(g.nodes[edge[0]]["pos"], g.nodes[edge[1]]["pos"])
        if np.allclose(length[i],0):
            length[i] +=     1e-4
        # length[l][i] = distance.euclidean(from_latlon_to_cartesian(g[l].nodes[edge[0]]["pos"]), from_latlon_to_cartesian(g[l].nodes[edge[1]]["pos"]))
    
    if length_inter0 is None:
        length_inter0 = min(length)
    
    for i,edge in enumerate(g.edges()):
        if g.edges[edge]['etype'] == 'inter': 
            length[i] = length_inter0
    # length_inter = [np.full((L,L), min_val) for i in range(E)]
    
    return length

def bias(g,length, nedge_inter, length_inter0=None):
    """fake constant lengths assigned to edges"""

    print("length: bias")

    length = np.zeros(len(g.edges()))
    for i in range(len(g.edges())):
        length[i] = 1 + 0.001 * random.uniform(0, 1)

    return length

#########################
########  weight ########
#########################
def weight_generation(weight_mode, input_path, weight_file_name, g, nnode):
    """create or import coordinate file"""

    switcher = {
        "0": lambda: weightgenerating(input_path, g, nnode),
        "1": lambda: weightimporting(weight_file_name, g, nnode)
    }

    return switcher.get(weight_mode, lambda: print("ERROR: invalid flag (weight)"))()

def weightgenerating(input_path, g, nnode):
    """weight generated in square [0,1]"""

    print("weights: generated")

    # generating weight
    weight = np.zeros([nnode, 1])
    for inode in range(nnode):
        weight[inode][:] = [random.uniform(0, 1)]
        inode += 1

    # assigning weight as attributes and printing in file
    for i in range(len(weight)):
        g.edges[i] = weight[i]

    weight_file = open(input_path + "/weight_generated.dat", "w")
    for i in range(int(nnode)):
        weight_file.write(str(weight[i][0]) + "\n")
    weight_file.close()

    return 0

def weightimporting(edge_file_name, g, nnode):
    """weight imported from file"""

    print("weight: imported")

    edge_file = open(edge_file_name, "r")
    input_lines = edge_file.readlines()

    weight = np.zeros([nnode, 1])
    inode = 0
    for line in input_lines:
        weight[inode][:] = [float(w) for w in line.split()[3:4]]
        inode += 1

    for i in range(nnode):
        g.edges[i] = (weight[i][0])

    return 0


#########################

## coord
def coord_generation(coord_mode, input_path, coord_file_name, g, nnode,sep=' '):
    """create or import coordinate file"""

    switcher = {
        "0": lambda: coordgenerating(input_path, g, nnode),
        "1": lambda: coordimporting(coord_file_name, g, nnode,sep=sep)
    }

    return switcher.get(coord_mode, lambda: print("ERROR: invalid flag (coordinates)"))()


def coordgenerating(input_path, g,  nnode):
    """coordinates generated in square [0,1]x[0,1]"""

    print("coordinates: generated")
    # generating coordinates
    coord = np.zeros([nnode, 2])
    for inode in range(nnode):
        coord[inode][:] = [random.uniform(0, 1),random.uniform(0, 1)]
        inode += 1

    # assigning coord as attributes and printing in file
    for i in range(len(coord)):
        g.nodes[i]["pos"] = coord[i]

    coord_file = open(input_path + "/coord_generated.dat", "w")
    for i in range(int(nnode)):
        coord_file.write(str(coord[i][0]) + " " + str(coord[i][1]) + "\n")
    coord_file.close()

    return 0


def coordimporting(coord_file_name, g, nnode,nid='nodeID',nlat='nodeLat',nlon='nodeLong',sep = ' '):
    """coordinates imported from file"""

    print("coordinates: imported")

    '''
    Build dictionary of (lat,lon)
    '''
    df_coord = pd.read_csv(coord_file_name, names=['nodeID', 'name', 'nodeLat', 'nodeLong'], sep = sep)

    pos_lat = df_coord.set_index(nid).to_dict()[nlat]
    pos_lon = df_coord.set_index(nid).to_dict()[nlon]
    pos = dict()
    for i in pos_lat: pos[i] = (pos_lat[i],pos_lon[i])   

    for i in list(g.nodes):
        if i in pos:
            g.nodes[i]["pos"] = pos[i]
        else:
            print('node ',i, 'does not have coordinates!')


def rhs_generation(rhs_mode, input_path, nodes, ncomm, tot_mass):
    """mass generation mode: 0 = import, 1 = generate"""

    def continuefunc():
        print("rhs: imported")
        return 0

    switcher = {
        "0": lambda: rhsgeneration(input_path, nodes, ncomm, tot_mass),
        "1": lambda: continuefunc()
    }

    return switcher.get(rhs_mode, lambda : print("ERROR: invalid flag (rhs)"))()


def rhsgeneration(input_path, nodes, ncomm, tot_mass):
    """generate an artificial forcing file"""

    print("rhs: generated")

    #generate sources/sinks list
    comm_list = random.sample(nodes, ncomm)

    f_mass = np.zeros(int(ncomm))
    g_mass = np.zeros(int(ncomm))

    # generating g and assigning h to keep mass balance (h are equal to 0, can be modified in future implementation)
    for i in range(0, len(comm_list)):
        assigned_mass = random.randint(1, tot_mass)
        indexes = list(range(0, int(ncomm)))
        indexes.pop(i)
        g_mass[i] = assigned_mass

    # write rhs file
    rhs_file = open(input_path + "/rhs_generated.dat", "w")
    for i in range(int(ncomm)):
        rhs_file.write(str(comm_list[i]) + " " + str(g_mass[i]) + " " + str(f_mass[i]) + "\n")

    rhs_file.close()

    return 0


def file2forcing(assignation_mode, file_name,  input_path, nodes,nnode, ncomm, g, nodeName2Id):
    """switcher mass generation mode: 0 = import, 1 = generate"""

    switcher = {
        "ia": lambda: file2forcing_impassign(file_name,  nodes, nnode, ncomm,g, nodeName2Id),
        "im": lambda: forcing_importing(input_path, nodes)
    }

    return switcher.get(assignation_mode, lambda : print("ERROR: invalid flag (forcing assignment)"))()


def file2forcing_impassign(file_name, nodes, nnode, ncomm, g, nodeName2Id):
    """generate forcing using Influence Assignment method"""

    print("forcing: Influence Assignment")

    # upload file and generate lists: icomm, g^i, h_i
    with open(file_name) as f:
        lines = f.readlines()
        temp_list = []
        for line in lines:
            temp_array = line.strip().split(" ")
            temp_list.append(np.array([int(float(element)) for element in temp_array]))
        temp_list = np.array(temp_list)

        comm_list = temp_list[:,0]  # list of sources and sinks
        temp_g = temp_list[:,1]     # list {g^i}i
    
    #print(temp_list)
    # creation of junction list
    transit_list = [node for node in nodes if nodeName2Id[node] not in comm_list]

    # generation of the rhs matrix S
    norm_g = []
    for i in range(ncomm):
        sum_g = np.sum(np.array(temp_g))
        sum_g -= temp_g[i]
        norm_g.append(sum_g)

    norm_mat = []
    for i in range(ncomm):
        norm_mat.append(np.array(temp_g)/norm_g[i])

    # synthetic initialization of S with Influence Assignment method
    mat_g = np.transpose(np.tile(temp_g, (ncomm, 1)))

    usable_nodes = []
    forcing = np.zeros((ncomm, nnode)) 
    
    forcing[:,comm_list] = - np.floor(np.multiply(mat_g, norm_mat))  # temp_g

    for idx,i in enumerate(comm_list):
        forcing[idx]= float(temp_g[i])
        usable_nodes.append(list(set(list(range(int(nnode)))) - set(transit_list) - set([nodeName2Id[i]])))

    residual = np.sum(np.array(forcing), axis=1)
    
    for j in range(len(comm_list)):
        chosen_index = random.choice(usable_nodes[j])
        forcing[j][chosen_index] -= residual[j]

    return forcing, comm_list, transit_list

def forcing_importing(input_path, nodes):
    """import forcing from last run"""

    print("forcing: imported")
    
    with open(input_path + "comm_list.pkl", "rb") as f_comm_list:
        comm_list = pickle.load(f_comm_list)
    with open(input_path + "forcing.pkl", "rb") as f_forcing:
        forcing = pickle.load(f_forcing)
    
    transit_list = [node for node in nodes if node not in comm_list]

    return forcing, comm_list, transit_list

def forcing_importing_from_file( input_path, nodes,nodes_inter,nodeName2Id = None,sep = ' ',header=True,source='source',sink='sink'):
    """import forcing from last run
    source sink weight
    0 1 10
    """
    print("forcing: imported from file")
    rhs_df = pd.read_csv(input_path,sep=sep,header=header)
    comm_list = list(rhs_df[source].unique().astype('str')) # keep only comm with g^i > 0
    sink_list = list(rhs_df[sink].unique().astype('str')) # keep only comm with h^i != 0
    
    transit_list = list(set(nodes).difference(set(comm_list).union(set(sink_list))))
    print('Ncom:',len(comm_list),' Ntransit:',len(transit_list))

    nnode = len(nodes)
    ncomm = len(comm_list)

    forcing = np.zeros((ncomm, nnode))
    for idx,row in rhs_df.iterrows():
        source_i = str(int(row[0]))
        sink_i = str(int(row[1]))
        g_i = row[2]
        idx_com = comm_list.index(source_i)

        if len(nodes_inter[source_i]) > 1: #super node
            if nodeName2Id is not None:
                i = nodeName2Id[source_i]
            else:
                i = source_i
            forcing[idx_com][i] += g_i
        else:
            if nodeName2Id is not None:
                i = nodeName2Id[source_i + '_' + str(nodes_inter[source_i][0])]
            else:
                i = source_i + '_' + str(nodes_inter[source_i][0])
            forcing[idx_com][i] += g_i
        if len(nodes_inter[sink_i]) > 1: # super node
            if nodeName2Id is not None:
                j = nodeName2Id[sink_i]
            else:
                j = sink_i
            forcing[idx_com][j] -= g_i
        else:
            if nodeName2Id is not None:
                j = nodeName2Id[sink_i + '_' + str(nodes_inter[sink_i][0])]
            else:
                j = sink_i + '_' + str(nodes_inter[sink_i][0])
            forcing[idx_com][j] -= g_i
      
    return forcing, comm_list, transit_list


def haversine(pos1, pos2, R = 6371):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lat1, lon1 = pos1
    lat2, lon2 = pos2
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = R * c
    return km 

def waxman_graph_modified(n, beta=0.4, alpha=0.1, L_max=None, L_min = None, domain=(0, 0, 1, 1), metric=None, seed=10, connected_component=True):
    r"""Returns a Waxman random graph modified so that only egdes within a distance are possible
    Puts a hard-threshold (cut-off) to standard waxman model

    The Waxman random graph model places `n` nodes uniformly at random
    in a rectangular domain. Each pair of nodes at distance `d` is
    joined by an edge with probability

    .. math::
            p = \beta \exp(-d / \alpha L).

    This function implements both Waxman models, using the `L` keyword
    argument.

    * Waxman-1: if `L` is not specified, it is set to be the maximum distance
      between any pair of nodes.
    * Waxman-2: if `L` is specified, the distance between a pair of nodes is
      chosen uniformly at random from the interval `[0, L]`.

    Parameters
    ----------
    n : int or iterable
        Number of nodes or iterable of nodes
    beta: float
        Model parameter
    alpha: float
        Model parameter
    L : float, optional
        Maximum distance between nodes.  If not specified, the actual distance
        is calculated.
    domain : four-tuple of numbers, optional
        Domain size, given as a tuple of the form `(x_min, y_min, x_max,
        y_max)`.
    metric : function
        A metric on vectors of numbers (represented as lists or
        tuples). This must be a function that accepts two lists (or
        tuples) as input and yields a number as output. The function
        must also satisfy the four requirements of a `metric`_.
        Specifically, if $d$ is the function and $x$, $y$,
        and $z$ are vectors in the graph, then $d$ must satisfy

        1. $d(x, y) \ge 0$,
        2. $d(x, y) = 0$ if and only if $x = y$,
        3. $d(x, y) = d(y, x)$,
        4. $d(x, z) \le d(x, y) + d(y, z)$.

        If this argument is not specified, the Euclidean distance metric is
        used.

        .. _metric: https://en.wikipedia.org/wiki/Metric_%28mathematics%29

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    Graph
        A random Waxman graph, undirected and without self-loops. Each
        node has a node attribute ``'pos'`` that stores the position of
        that node in Euclidean space as generated by this function.

    Examples
    --------
    Specify an alternate distance metric using the ``metric`` keyword
    argument. For example, to use the "`taxicab metric`_" instead of the
    default `Euclidean metric`_::

        >>> dist = lambda x, y: sum(abs(a - b) for a, b in zip(x, y))
        >>> G = nx.waxman_graph(10, 0.5, 0.1, metric=dist)

    .. _taxicab metric: https://en.wikipedia.org/wiki/Taxicab_geometry
    .. _Euclidean metric: https://en.wikipedia.org/wiki/Euclidean_distance

    Notes
    -----
    Starting in NetworkX 2.0 the parameters alpha and beta align with their
    usual roles in the probability distribution. In earlier versions their
    positions in the expression were reversed. Their position in the calling
    sequence reversed as well to minimize backward incompatibility.

    References
    ----------
    .. [1]  B. M. Waxman, *Routing of multipoint connections*.
       IEEE J. Select. Areas Commun. 6(9),(1988) 1617--1622.
    """
    prng = np.random.RandomState(seed)
    nodes = n
    G = nx.Graph()
    G.add_nodes_from(np.arange(nodes))
    (xmin, ymin, xmax, ymax) = domain
    # Each node gets a uniformly random position in the given rectangle.
    pos = {v: (prng.uniform(xmin, xmax), prng.uniform(ymin, ymax)) for v in G}
    nx.set_node_attributes(G, pos, "pos")
    # If no distance metric is provided, use Euclidean distance.
    if metric is None:
        metric = euclidean
    # If the maximum distance L is not specified (that is, we are in the
    # Waxman-1 model), then find the maximum distance between any pair
    # of nodes.
    #
    # In the Waxman-1 model, join nodes randomly based on distance. In
    # the Waxman-2 model, join randomly based on random l.
    # if L is None:
    #     L = max(metric(x, y) for x, y in combinations(pos.values(), 2))
    if L_max is None:
        L_max = max(metric(x, y) for x, y in combinations(pos.values(), 2))
    if L_min is None:
        L_min = 0

    def dist(u, v):
        return metric(pos[u], pos[v])

    # `pair` is the pair of nodes to decide whether to join.
    def should_join(pair):
        if np.logical_or(dist(*pair) > L_max,dist(*pair) < L_min ): 
            return False
        else:
            return prng.rand() < beta * math.exp(-dist(*pair) / (alpha * L_max))

    G.add_edges_from(filter(should_join, combinations(G, 2)))

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
    return sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))


def assign_weights(length0, edges, alphas):
    length = np.array(length0)
    for i, (u,v,d) in enumerate(edges):
        l = d['etype']
        length[i] *= alphas[l]
    return length
    
def from_nx_graphs2df(list_G,cols = ['layer', 'fromNode', 'toNode', 'weight'],outfile = 'adjacency_tmp.csv'):
    data = []
    for l in range(len(list_G)):
        for i,j in list(list_G[l].edges()):
            data.append([l,i,j,1])
            
    df = pd.DataFrame(data,columns = cols)
    if outfile is not None:
        df.to_csv(outfile,index=False,header=0)
    return df

def from_nx_graph2coord(list_G, nodes0,cols = ['nodeID', 'name', 'nodeLat', 'nodeLong'],outfile = 'coord_tmp.csv',mapping = None, nodes_inter = None):
    data = []
    nodes = []
    for l in range(len(list_G)):
        for n0_int in list(list_G[l].nodes()):
            n0 = str(n0_int)
            for m in nodes_inter[n0]:
                n_name = str(n0) + '_' + str(m)
                if mapping is not None:
                    n = mapping[str(n_name)]
                else:
                    n = n0
                if n_name not in nodes:
                    i = nodes0.index(n)
                    data.append([i,n_name,list_G[l].nodes[n0_int]['pos'][0],list_G[l].nodes[n0_int]['pos'][1]])
                    nodes.append(n_name)
            if len(nodes_inter[n0]) > 1:
                n_name = str(n0)
                if n_name not in nodes:
                    if mapping is not None:
                        n = mapping[str(n_name)]
                    else:
                        n = n0
                    i = nodes0.index(n)
                    data.append([i,n_name,list_G[l].nodes[n0_int]['pos'][0],list_G[l].nodes[n0_int]['pos'][1]])
                    nodes.append(n_name)

    df = pd.DataFrame(data,columns = cols)
    if outfile is not None:
        df.to_csv(outfile,index=False,header=0)
    return df

