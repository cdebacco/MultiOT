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
#######################################

import pickle,time, warnings
import numpy as np
import networkx as nx
import random
import copy
import math
import scipy as sp

from scipy.sparse import csr_matrix,lil_matrix,issparse,csc_matrix
from scipy.sparse import diags
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
from scipy import sparse

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
#######################################

warnings.filterwarnings("ignore", message="Matrix is exactly singular")

def tdensinit(nedge, seed=10):
    """initialization of the conductivities: mu_e ~ U(0,1)"""
    prng = np.random.RandomState(seed = seed)

    tdens_0 = prng.uniform(0, 1,size = nedge)
  
    weight = np.ones(nedge) + 0.01 * prng.uniform(0, 1, size = nedge)
    
    return tdens_0, weight

def dyn(g,  nodes,pflux, nedge, length, forcing0, tol_var_tdens, comm_list, seed=10,verbose=False, N_real = 1, plot_cost = False):
    """dynamics method"""

    print("\ndynamics...")

    relax_linsys = 1.0e-5       # relaxation for stiffness matrix
    tot_time = 1000            # upper bound on number of time steps
    time_step = 0.5             # delta_t (reduce as beta gets close to 2)
    threshold_cost = 1.0e-6   # threshold for stopping criteria using cost
    prng = np.random.RandomState(seed=seed) # only needed if spsolve has problems (inside update)

    nnode = len(nodes)
    ncomm = forcing0.shape[0]
    forcing = forcing0.transpose()

    minCost = 1e14
    minCost_list = []

    inc_mat = csr_matrix(nx.incidence_matrix(g, nodelist=nodes, oriented=True)) # B    
    inc_transpose = csr_matrix(inc_mat.transpose())      # B^T
    inv_len_mat = diags( 1 / length, 0)     # diag[1/l_e]

    for r in range(N_real):

        tdens_0, weight = tdensinit(nedge,seed=seed+r)  # conductivities initialization
        # forcing = csc_matrix(forcing0.transpose())
        # --------------------------------------------------------------------------------


        tdens = tdens_0.copy()
        td_mat = diags(tdens.astype(float), 0)   # matrix M
        
        stiff = inc_mat * td_mat * inv_len_mat * inc_transpose  # B diag[mu] diag[1/l_e] B^T
        stiff_relax = stiff + relax_linsys * identity(nnode)     # avoid zero kernel
        pot = spsolve(stiff_relax, forcing).reshape((nnode,ncomm))      # pressure

        # --------------------------------------------------------------------------------
        # Run dynamics
        convergence_achieved = False
        cost_update = 0
        cost_update_inter = 0
        cost_list = []
        time_iteration = 0

        fmax = forcing0.max() 

        while not convergence_achieved and time_iteration < tot_time:

            time_iteration += 1

            # update tdens-pot system
            tdens_old = tdens.copy()
            pot_old = pot.copy()

            # equations update
            tdens, pot, grad, info = update(tdens,  pot, weight, inc_mat, inc_transpose, inv_len_mat, forcing, time_step, pflux, relax_linsys, nnode)
            
            # singular stiffness matrix
            if info != 0:
                tdens = tdens_old + prng.rand(*tdens.shape) * np.mean(tdens_old)/1000.
                pot = pot_old + prng.rand(*pot.shape) * np.mean(pot_old)/1000. 

            # 1) convergence with conductivities
            # var_tdens = max(np.abs(tdens - tdens_old))/time_step
            # print(time_iteration, var_tdens)
            if verbose > 1 :
                print('==========')
            
            # 2) an alternative convergence criteria: using total cost and maximum variation of conductivities
            var_tdens = max(np.abs(tdens - tdens_old))/time_step 
            
            #var_tdens_inter = ([ max(np.abs(tdens_inter[i] - tdens_old_inter[i]))/time_step for i in range(nnode) ] )
            convergence_achieved, cost_update, abs_diff_cost,  flux_norm = cost_convergence(threshold_cost, cost_update, tdens, pot, inc_mat, inv_len_mat, length, weight, pflux,convergence_achieved, var_tdens)#, var_tdens_inter)
            
            if verbose > 1: 
                # print(time_iteration, var_tdens/forcing.max(), abs_diff_cost)

                # print('\r','It=',it,'err=', abs_diff,'J-J_old=',abs_diff_cost,sep=' ', end='', flush=True)
                print('\r','it=%3d, err/max_f=%5.2f, J_diff=%8.2e' % (time_iteration,var_tdens/fmax,abs_diff_cost),sep=' ', end=' ', flush=True)
                time.sleep(0.05)

            cost_list.append(cost_update)

            if (var_tdens < tol_var_tdens):# or (var_tdens_inter < tol_var_tdens):
                convergence_achieved = True

            elif time_iteration >= tot_time:
                convergence_achieved = True
                tdens = tdens_old.copy()
                if verbose > 0 :
                    print("ERROR: convergence dynamics not achieved, iteration time > maxit")

        if convergence_achieved:
            if verbose > 0 :
                print('cost:',cost_update, ' - N_real:',r, '- Best cost', minCost)   
            if cost_update < minCost:
                minCost = cost_update
                minCost_list = cost_list
                tdens_best = tdens.copy()
                pot_best = pot.copy()
                flux_best = flux_norm.copy()

        else:
            print("ERROR: convergence dynamics not achieved")

    if plot_cost:
        plot_J(minCost_list, int_ticks=True)

    return tdens_best, pot_best, flux_best, minCost, minCost_list

##*********************************************

def update(tdens,  pot,  weight,  inc_mat,  inc_transpose, inv_len_mat, forcing,time_step, pflux, relax_linsys, nnode):
    """dynamics update"""

    t = 1
    ctr = 1 / (1 + math.exp((t - 250)/50) )

    nedge = tdens.shape[0]
    nnode, ncomm = pot.shape
    
    # weight = [ np.sum(weight[l]) for l in range(L)]
    weight = 1.
    grad = inv_len_mat * inc_transpose * pot   # discrete gradient 
    if isinstance(grad,(np.ndarray)):
        rhs_ode = ((tdens**pflux) * ((grad**2).sum(axis=1)) / (weight**2)) - tdens 
    else:
        rhs_ode = ((tdens**pflux) * ((grad.power(2)).sum(axis=1)) / (weight**2)) - tdens 
    rhs_ode = rhs_ode.reshape(tdens.shape)

    # update conductivity
    if rhs_ode.ndim > 1:
        tdens = tdens + time_step * np.ravel(rhs_ode[:,0])
        td_mat = diags(tdens.astype(float), 0) 
    else:
        tdens = tdens + time_step * rhs_ode
        td_mat = diags(tdens.astype(float), 0) 
    

    # update stiffness matrix
    stiff = inc_mat * td_mat * inv_len_mat * inc_transpose
    
    # spsolve
    stiff_relax = stiff + relax_linsys * identity(nnode) # avoid zero kernel
    # update potential
    pot = spsolve(stiff_relax, forcing).reshape((nnode,ncomm))  # pressure
    if np.any(np.isnan(pot)):# or np.any(np.isnan(pot_ctr)): # or np.any(pot != pot)
        info = -1
        pass
    else:
        info = 0

    return tdens, pot, grad, info

def cost_convergence(threshold_cost, cost,  tdens,  pot, inc_mat, inv_len_mat,length,weight, pflux, convergence_achieved, var_tdens):#, var_tdens_inter):
    """computing convergence using total cost: setting a high value for maximum conductivity variability"""

    L = len(pot)
    nnode = pot[0].shape[0]

    td_mat = np.diag(tdens.astype(float))
    
    flux_mat = np.matmul(td_mat * inv_len_mat * np.transpose(inc_mat), pot) 
    flux_norm = np.linalg.norm(flux_mat, axis=1) 

    cost_update = np.sum(weight * length * (flux_norm**(2*(2-pflux)/(3-pflux)))) 
  
    abs_diff_cost = abs(cost_update - cost)
    
    convergence_achieved = bool(convergence_achieved)
    if min(pflux) > 1.40:
        if abs_diff_cost < threshold_cost :
            convergence_achieved = True
    else: 
        if abs_diff_cost < threshold_cost and var_tdens < 1:
            convergence_achieved  = True

    return convergence_achieved, cost_update, abs_diff_cost, flux_norm

def abs_trimming_dyn(g, opttdens, optpot, length, output_path=None, tau=None):
    """obtaining the trimmed graph using an absolute threshold"""

    print("trimming dynamics graph...\n")

    nnode = len(g.nodes())

    #nnode_inter = len(g_inter.nodes())
    
    inc_mat = csr_matrix(nx.incidence_matrix(g, nodelist=list(range(nnode)), oriented=True))    # delta
    inv_len_mat = diags(1/length, 0)    # diag[1/l_e]

    mu_mat = np.diag(opttdens) 
    
    flux_mat = np.matmul(mu_mat* inv_len_mat * np.transpose(inc_mat), optpot) 
    
    flux_norm = np.linalg.norm(flux_mat, axis=1) 

    if tau is None:
        tau = np.percentile(flux_norm,20)
    
    edges_list = list(g.edges()) 

    g_trimmed = copy.deepcopy(g) 
    
    g_final = copy.deepcopy(g) 

    index_to_remove = [] 

    for i in range(len(flux_norm)):
        g_trimmed.remove_edge(edges_list[i][0], edges_list[i][1])
        if flux_norm[i] < tau:
            index_to_remove.append(i)
            # iteratively trim the edges
            node_1 = edges_list[i][0]
            node_2 = edges_list[i][1]
            g_final.remove_edge(node_1, node_2)
                
    opttdens_final = np.delete(opttdens, index_to_remove) 

    flux_norm_final = np.delete(flux_norm, index_to_remove) 
    
    length_final = np.delete(length, index_to_remove) 

    # dumping for graph visualization
    if output_path is not None:
        pickle.dump(np.array(index_to_remove), open(output_path + "index_to_remove_dyn.pkl", "wb"))
        pickle.dump(index_to_remove_inter, open(output_path + "index_to_remove_dyn_inter.pkl", "wb"))

    return g_final,  opttdens_final, flux_norm_final, length_final


def incidence_matrix_from_sparse_adjacency(A,oriented = True,weight = None):
    assert sparse.issparse(A)
    # A = A.tocoo() # too handle easily the inidices
    subs_nz = A.nonzero()
    N,E = A.shape[0],subs_nz[0].shape[0]
    B = sparse.lil_matrix((N, E))
    for e in range(E):
        u,v = subs_nz[0][e],subs_nz[1][e]
        if weight is None:
            w = 1
        else:
            w = A.data[e]
        if oriented == True:
            if u < v:
                B[u,e] = - w
                B[v,e] = w
            # else:
            #     B[u,e] = w
            #     B[v,e] = - w
        else:
            B[u,e] = w
            B[v,e] = w
    return B.asformat("csr")

def extract_inter_incidence_matrix(A_inter,oriented = True,weight = None):
    '''
    If you want to calculate for the whole list
    '''
    N = len(A_inter)

    B = []
    for i in range(N):
        B.append(incidence_matrix_from_sparse_adjacency(A[i],oriented = oriented,weight = weight))
    
    return B

def plot_J(values, indices = None, k_i = 5, figsize=(7, 3), int_ticks=False, xlab='Iterations'):

    fig, ax = plt.subplots(1,1, figsize=figsize)
    #print('\n\nL: \n\n',values[k_i:])

    if indices is None:
        ax.plot(values[k_i:])
    else:
        ax.plot(indices[k_i:], values[k_i:])
    ax.set_xlabel(xlab)
    ax.set_ylabel('J')
    if int_ticks:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid()

    plt.tight_layout()
    plt.show()

