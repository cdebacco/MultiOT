# -*- coding: utf-8 -*-
"""
MultiOT - Optimal Transport in Multilayer networks (https://github.com/cdebacco/MultiOT)

Licensed under the GNU General Public License v3.0

Note: docstrings have been generated semi-automatically
"""

from argparse import ArgumentParser
from time import time
import os

import numpy as np
from src.dijkstra import fit_dijkstra
from src.dynamics import fit
from src.dynamics_sp import fit_dyn_sp
from src.filtering import filtering_sp
from src.initialization import multiot_init
from src.tools import get_array_from_string, serialize


def main():
    """
    MultiOT.

    Parameters:
     - topol: Type of network topology, can be 'synthetic' or 'real'.
     - Ns: Number of nodes in each layer.
     - betas: Critical exponents in each layer.
     - ws: Inverse velocity for all layers (also referred to as alpha for effective lengths).
     - p: Monocentric/random inflows of mass for synthetic networks.
     - V: Verbose flag for additional output.
     - Vtimestep: Frequency when to print algorithm metadata.
     - relax: Relaxation of Laplacian pseudoinverse.
     - delta: Discrete time step.
     - delta_filtering: Discrete time step for filtering.
     - tot_iterations: Maximum iteration limit for the algorithm.
     - epsilonmu: Convergence threshold for conductivities/capacities.
     - epsilonJ: Convergence threshold for OT cost.
     - epsilonmu_filtering: Convergence threshold for conductivities/capacities in filtering.
     - epsilonJ_filtering: Convergence threshold for OT cost in filtering.
     - tau_filtering: Threshold to trim fluxes in dynamics filtering.
     - seedG: Seed for random graph generation.
     - seedmu: Seed for random noise initialization of conductivities.
     - seedS: Seed for random choice of sources/sinks.
     - dynamics_flag: Flag to run multi-commodity dynamics.
     - dynamics_sp_flag: Flag to run shortest path dynamics.
     - dijkstra_flag: Flag to run multi-source multi-sinks Dijkstra's algorithm.
     - filtering_flag: Flag to run filtering.
     - ifolder: Input folder containing data.
     - ofolder: Output folder for storing results.
    """

    pars = ArgumentParser()

    pars.add_argument("-topol", "--topol", type=str, default="synthetic")
    pars.add_argument("-Ns", "--Ns", type=str, default="10 10")
    pars.add_argument("-betas", "--betas", type=str, default="1.0 1.0")
    # Car, Bikes coeff for effective lengths. Smaller ws yield shorter effective length, hence more favorable edges.
    pars.add_argument("-ws", "--ws", type=str, default="1.0 1.0")
    pars.add_argument("-p", "-p", type=float, default=0)
    pars.add_argument("-V", "--V", type=lambda x: bool(int(x)), default=False)
    pars.add_argument("-Vtimestep", "--Vtimestep", type=int, default=20)
    pars.add_argument("-relax", "--relax", type=float, default=1e-10)
    pars.add_argument("-delta", "--delta", type=float, default=0.9)
    pars.add_argument("-delta_filtering", "--delta_filtering", type=float, default=0.9)
    pars.add_argument("-tot_iterations", "--tot_iterations", type=int, default=100)
    pars.add_argument(
        "-epsilonmu", "--epsilonmu", type=float, default=1e-1
    )  # 1e-1 copenhagen
    pars.add_argument(
        "-epsilonJ", "--epsilonJ", type=float, default=1e-1
    )  # 1e-1 copenhagen
    pars.add_argument(
        "-epsilonmu_filtering", "--epsilonmu_filtering", type=float, default=1e-3
    )
    pars.add_argument(
        "-epsilonJ_filtering", "--epsilonJ_filtering", type=float, default=1e-3
    )
    pars.add_argument("-tau_filtering", "--tau_filtering", type=float, default=1e-12)
    pars.add_argument("-seedG", "--seedG", type=int, default=0)
    pars.add_argument("-seedmu", "--seedmu", type=int, default=0)
    pars.add_argument("-seedS", "--seedS", type=int, default=0)
    pars.add_argument(
        "-dynamics_flag", "--dynamics_flag", type=lambda x: bool(int(x)), default=False
    )
    pars.add_argument(
        "-dynamics_sp_flag",
        "--dynamics_sp_flag",
        type=lambda x: bool(int(x)),
        default=False,
    )
    pars.add_argument(
        "-dijkstra_flag", "--dijkstra_flag", type=lambda x: bool(int(x)), default=False
    )
    pars.add_argument(
        "-filtering_flag",
        "--filtering_flag",
        type=lambda x: bool(int(x)),
        default=False,
    )
    pars.add_argument(
        "-ifolder", "--ifolder", type=str, default="./data/input/real-data/"
    )
    pars.add_argument(
        "-ofolder", "--ofolder", type=str, default="./data/output/synthetic/"
    )

    args = pars.parse_args()

    topol = args.topol
    seedG = args.seedG
    Ns = np.array(get_array_from_string(args.Ns), dtype=int)
    M = 0
    if topol == "synthetic":
        M = np.sum(Ns)
    p = args.p
    ws = np.array(get_array_from_string(args.ws), dtype=float)
    seedmu = args.seedmu
    betas = np.array(get_array_from_string(args.betas), dtype=float)
    seedS = args.seedS
    delta = args.delta
    delta_filtering = args.delta_filtering
    relax = args.relax
    tot_iterations = args.tot_iterations
    epsilonJ = args.epsilonJ
    epsilonmu = args.epsilonmu
    epsilonJ_filtering = args.epsilonJ_filtering
    epsilonmu_filtering = args.epsilonmu_filtering
    tau_filtering = args.tau_filtering
    V = args.V
    Vtimestep = args.Vtimestep
    ifolder = args.ifolder
    ofolder = args.ofolder
    dynamics_flag = args.dynamics_flag
    dynamics_sp_flag = args.dynamics_sp_flag
    filtering_flag = args.filtering_flag
    dijkstra_flag = args.dijkstra_flag

    results = dict()
    time_dyn_end, time_dyn_start = 0, 0
    time_dyn_sp_end, time_dyn_sp_start = 0, 0
    time_dijkstra_end, time_dijkstra_start = 0, 0
    time_filtering_end, time_filtering_start = 0, 0

    # Initialization.
    G, weff, w, mu, S = multiot_init(topol, seedG, Ns, ws, seedmu, p, seedS, ifolder)

    # Dynamics with effective lengths.
    if dynamics_flag is True:
        if V is True:
            print("** Dyn")

        time_dyn_start = time()

        results = fit(
            G,
            Ns,
            betas,
            delta,
            weff,
            mu,
            S,
            relax,
            seedmu,
            tot_iterations,
            epsilonJ,
            epsilonmu,
            topol,
            V,
            Vtimestep,
        )

        time_dyn_end = time()

    # Shortest path dynamics.
    J_dyn_sp = 0
    J_sp_dyn_sp = 0
    J_dyn_sp_real = 0
    F_dyn_sp = np.zeros((G.number_of_edges(), M))

    J_sp_dyn_sp_filtered = 0
    J_dyn_sp_filtered = 0
    J_dyn_sp_real_filtered = 0
    F_dyn_sp_filtered = np.zeros((G.number_of_edges(), M))

    if dynamics_sp_flag is True:
        if V is True:
            print("** Dyn sp")

        if topol != "synthetic":
            M = S.shape[1]

        mu_tiled = np.tile(mu, (M, 1)).transpose()

        time_dyn_sp_start = time()
        if filtering_flag is True:
            time_filtering_start = time()

        J_sp_dyn_sp, F_dyn_sp, J_dyn_sp, J_dyn_sp_real = fit_dyn_sp(
            G,
            M,
            Ns,
            seedmu,
            betas,
            delta,
            weff,
            w,
            mu_tiled,
            S,
            relax,
            tot_iterations,
            epsilonJ,
            epsilonmu,
            topol,
            V,
            Vtimestep,
        )

        time_dyn_sp_end = time()

        if filtering_flag is True:
            if V is True:
                print("** Filtering")

            (
                J_sp_dyn_sp_filtered,
                F_dyn_sp_filtered,
                J_dyn_sp_filtered,
                J_dyn_sp_real_filtered,
            ) = filtering_sp(
                topol,
                G,
                weff,
                w,
                S,
                M,
                Ns,
                betas,
                V,
                F_dyn_sp,
                tau_filtering,
                seedmu,
                delta_filtering,
                relax,
                tot_iterations,
                epsilonJ_filtering,
                epsilonmu_filtering,
                Vtimestep,
            )

            time_filtering_end = time()

    # Dijkstra.
    J_sp_dijkstra = 0
    J_dijkstra = 0
    J_dijkstra_real = 0
    F_dijkstra = np.zeros((G.number_of_edges(), M))

    if dijkstra_flag is True:
        if V is True:
            print("** Dijkstra")

        if topol != "synthetic":
            M = S.shape[1]

        time_dijkstra_start = time()

        J_sp_dijkstra, F_dijkstra, J_dijkstra, J_dijkstra_real = fit_dijkstra(
            topol, G, weff, w, S, M, Ns, betas, V
        )

        time_dijkstra_end = time()

    # Serialization.
    net = dict()
    net["G"] = G
    net["w"] = w
    net["weff"] = weff

    params = dict()
    params["topol"] = topol
    params["betas"] = betas
    params["ws"] = ws
    params["S"] = S
    params["delta"] = delta
    params["Ns"] = Ns
    params["M"] = M
    params["relax"] = relax
    params["tot_iterations"] = tot_iterations
    params["epsilonJ"] = epsilonJ
    params["epsilonmu"] = epsilonmu
    params["seedG"] = seedG
    params["seedmu"] = seedmu
    params["seedS"] = seedS

    if dynamics_flag:
        elapsed_time_dyn = float(time_dyn_end - time_dyn_start)
        results["time_dyn"] = elapsed_time_dyn

    if dijkstra_flag:
        results["J_sp_dijkstra"] = J_sp_dijkstra
        results["F_dijkstra"] = F_dijkstra
        results["J_dijkstra"] = J_dijkstra
        results["J_dijkstra_real"] = J_dijkstra_real
        elapsed_time_dijkstra = float(time_dijkstra_end - time_dijkstra_start)
        results["time_dijkstra"] = elapsed_time_dijkstra

    if dynamics_sp_flag:
        results["J_sp_dyn_sp"] = J_sp_dyn_sp
        results["F_dyn_sp"] = F_dyn_sp
        results["J_dyn_sp"] = J_dyn_sp
        results["J_dyn_sp_real"] = J_dyn_sp_real
        elapsed_time_dyn_sp = float(time_dyn_sp_end - time_dyn_sp_start)
        results["time_dyn_sp"] = elapsed_time_dyn_sp

        if filtering_flag:
            results["J_sp_filtering"] = J_sp_dyn_sp_filtered
            results["F_filtering"] = F_dyn_sp_filtered
            results["J_filtering"] = J_dyn_sp_filtered
            results["J_filtering_real"] = J_dyn_sp_real_filtered
            elapsed_time_filtering = float(time_filtering_end - time_filtering_start)
            results["time_filtering"] = elapsed_time_filtering

    serialize(net, ofolder, "network.pkl")
    serialize(params, ofolder, "params.pkl")
    serialize(results, ofolder, "results.pkl")


if __name__ == "__main__":
    main()
