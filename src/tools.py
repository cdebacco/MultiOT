import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import dynamics as dyn


def find_central_nodes(G, pos = None, pos_center_of_mass = None):
    
    if pos is None:
        pos = nx.get_node_attributes(G,'pos')
    if pos_center_of_mass is None:
        positions = np.array(list(pos.values()))
        pos_center_of_mass = np.mean(positions,axis = 0)
    
    nodes = list(G.nodes)
    distances = [euclidean(pos[n],pos_center_of_mass) for n in nodes]
    nid = np.argmin(distances)
    n = nodes[nid]
    return n

def forcing_generate(G, p = 0., weigth = 10, pos = None, pos_center_of_mass = (0.5,0.5), seed = 10):
    """G is the graph in the first layer"""
    
    prng = np.random.RandomState(seed)
    
    n_center = find_central_nodes(G, pos = pos, pos_center_of_mass = pos_center_of_mass)

    nodes = set(G.nodes()) - {n_center} # all nodes except central one
    forcing = []
    for source in nodes:
        if source != n_center:
            r = prng.rand()
            if r < p: # rewire: extract a random node
                sink = random.choice(list(nodes - {source}))
                forcing.append([source,sink,weigth])
            else:
                forcing.append([source,n_center,weigth])
    
    return forcing

def euclidean(x, y):
    """Returns the Euclidean distance between the vectors ``x`` and ``y``.

    Each of ``x`` and ``y`` can be any iterable of numbers. The
    iterables must be of the same length.

    """
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))

def plot_results(G,H,graph,opttdens, optpot, optflux,length, flag_trim = False,  ns = 20,
                 outfigure = None, tau = None,dpi=300,
                 wl = False ,w0 = 1.0, edge_width = 'linear',figsize = (15, 5),
                 colors_map = {0:'b',1:'r',2:'g',3:'magenta','inter':'black','inter-super':'black'}):
    
    if flag_trim == True:
        tau = tau
        graph_final,  opttdens_final,  flux_norm_final, length_final = dyn.abs_trimming_dyn(graph, opttdens, optpot, length, tau = tau)
    else:
        graph_final = nx.Graph(graph)
        opttdens_final = opttdens.copy()
        flux_norm_final =  optflux.copy()
        
    '''
    Assign colors
    '''
    etypes = nx.get_edge_attributes(graph_final,'etype')
    nlayer = len(set(etypes.values()).difference(set(['inter','inter-super'])))
    for e in list(graph_final.edges):
        graph_final[e[0]][e[1]]['color'] = colors_map[etypes[e]]
    colors = nx.get_edge_attributes(graph_final,'color')
    
    '''
    Assign edge widths
    '''
    if edge_width == 'linear':
        widths = [ 0 + w0 * flux_norm_final[idx] for idx, e in enumerate(list(graph_final.edges())) ]
    else:
        widths = [ 0 + w0 * np.log(1 + flux_norm_final[idx]) for idx, e in enumerate(list(graph_final.edges())) ]
    
    for idx, e in enumerate(list(graph_final.edges(data=True))):
        if e[2]['etype'] == 'inter':
            widths[idx] = 0.1
        if e[2]['etype'] == 'inter-super':
            widths[idx] = 0.1
            
    '''
    Build graph with the solution
    '''
    G_plot = nx.Graph(graph_final)
    # G_plot = nx.relabel_nodes(G_plot, nodeId2Name)
    # colors = nx.get_edge_attributes(G_plot,'color')
    G_plot.number_of_edges(),G_plot.number_of_nodes()
    weights = { e: flux_norm_final[idx] for idx, e in enumerate(list(graph_final.edges())) }
    inv_weights = { e: 1./(flux_norm_final[idx]+1e-12) for idx, e in enumerate(list(graph_final.edges())) }
    widths_dict = { e: widths[idx] for idx, e in enumerate(list(graph_final.edges())) }
    nx.set_edge_attributes(G_plot, weights, 'flux')
    nx.set_edge_attributes(G_plot, inv_weights, 'inv_flux')
    nx.set_edge_attributes(G_plot, widths_dict, 'widths')
    
    '''
    Plot
    '''

    figsize = figsize

    fig, axes = plt.subplots(nrows=1, ncols=3,figsize=figsize)
    ax = axes.flatten()

    edges = G.edges()
    colors = 'grey' #nx.get_edge_attributes(G,'color')
    pos = nx.get_node_attributes(G,'pos')
    nx.draw(G,pos, with_labels=wl,node_size = ns, node_color = 'b', font_weight="bold",ax=ax[0])
#     ax[0].set_axis_off()
    # fig, ax = plt.subplots(1, 1, figsize=figsize)

    edges = H.edges()
    colors = 'r' #[H[u][v]['color'] for u,v in edges]
    pos = nx.get_node_attributes(H,'pos')
    nx.draw(H, pos, with_labels=wl, node_size = ns, edge_color=colors, font_weight="bold", node_color='red',ax=ax[1])
    #     ax[1].set_axis_off()
    ax[1].set_xlim([1.*x for x in ax[0].get_xlim()])
    ax[1].set_ylim([1.*y for y in ax[0].get_ylim()])

    edges = G_plot.edges()
    pos_plot = nx.get_node_attributes(G_plot,'pos')
    colors = list(nx.get_edge_attributes(G_plot,'color').values())
#     for n in pos_plot.keys():
#         if G_plot.nodes[n]['ntype'] == 'inter': pos_plot[n] = G_plot.nodes[n]['pos'] * (1 + 0.2 * np.random.rand(2))
#         if G_plot.nodes[n]['ntype'] == 'super': pos_plot[n] = G_plot.nodes[n]['pos'] * (1 + 0.2 * np.random.rand(2))
    nx.draw(G_plot,pos_plot, node_size = ns, with_labels = wl, edge_color=colors, font_weight="bold",width = widths, ax=ax[2])
    
    ax[2].set_xlim([1.*x for x in ax[0].get_xlim()])
    ax[2].set_ylim([1.*y for y in ax[0].get_ylim()])
    
    if outfigure is not None:
        plt.savefig(outfigure + '.png', dpi=dpi, format='png', bbox_inches='tight',pad_inches=0.1)
        
    plt.show()
    
    return G_plot

def plot_individual_G_final(G_plot, ns = 20,outfigure = None, colors = None,
                 wl = False , figsize = (15, 5), widths = None,dpi=300,
                 colors_map = {0:'b',1:'r',2:'g',3:'magenta','inter':'black','inter-super':'black'}):

    # plt.figure(figsize=figsize)
    plt.figure()
    pos_plot = nx.get_node_attributes(G_plot,'pos')
    if colors is None:
        colors = list(nx.get_edge_attributes(G_plot,'color').values())
    if widths is None:
        widths = list(nx.get_edge_attributes(G_plot,'widths').values())
    nx.draw(G_plot,pos_plot, node_size = ns, with_labels = wl, edge_color=colors, font_weight="bold",width = widths)
    if outfigure is not None:
        plt.savefig(outfigure + '.png', dpi=dpi, format='png', bbox_inches='tight',pad_inches=0.1)
        
    plt.show()
    return plt.gcf()

def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    if isinstance(x,(dict)):
        x = np.array(list(x.values()))

    diffsum = np.sum(np.abs(np.subtract.outer(x, x)))
    gini = 0.5 * (diffsum /  (len(x)**2 * np.mean(x)))
    
    return gini

def flt(x,d=1):
    return round(x, d)