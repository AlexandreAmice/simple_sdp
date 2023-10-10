import numpy as np
import cvxpy as cp
import networkx as nx
import sdp_problems


if __name__ == "__main__":
    n = 100
    # Generate any graph you would want. Networkx generators are here https://networkx.org/documentation/stable/reference/generators.html
    G = nx.fast_gnp_random_graph(n, 0.2)

    # Other solvers available at https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver
    first_order_solver = (
        cp.SCS
    )  # Splitting cone solver based on ADMM: https://www.cvxgrp.org/scs/
    commercial_second_order_solver = (
        cp.MOSEK
    )  # Mosek, commerical interior point solver: https://www.mosek.com/
    open_source_second_order_solver = (
        cp.CLARABEL
    )  # Clarabel, an open source interior point solve: https://oxfordcontrol.github.io/ClarabelDocs/stable/

    def solve_sdp_with_solver(sdp, name="sdp"):
        first_order = sdp.solve(solver=first_order_solver)
        print(f"{name} first order value {first_order}")
        second_order_commerical = sdp.solve(solver=commercial_second_order_solver)
        print(f"{name} second order commerical value {second_order_commerical}")
        second_order_open = sdp.solve(solver=open_source_second_order_solver)
        print(f"{name} second order open source value {second_order_open}")
        return first_order, second_order_commerical, second_order_open

    max_cut_sdp = sdp_problems.maxcut_sdp(G)
    solve_sdp_with_solver(max_cut_sdp, "MAXCUT")
    print()
    lovasz_sdp = sdp_problems.lovasz_theta_sdp(G)
    solve_sdp_with_solver(lovasz_sdp, "LOVASZ")
