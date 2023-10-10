import numpy as np
import cvxpy as cp
import networkx as nx


def lovasz_theta_sdp(G: nx.Graph) -> cp.Problem:
    """
    Given a graph, construct the Lovasz Theta function SDP https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15859-f11/www/notes/lecture11.pdf
    The return of this program lies between the independence number and the chromatic number of the graph.
    :param G:
    :return:
    """
    n = len(G.nodes)
    X = cp.Variable((n, n), symmetric=True)
    obj = cp.Maximize(cp.trace(np.ones((n, n)) @ X))
    constraints = [cp.trace(X) == 1, X >> 0]
    for i, j in G.edges:
        constraints.append(X[i, j] == 0)
    return cp.Problem(obj, constraints)


def maxcut_sdp(G: nx.Graph) -> cp.Problem:
    """
    Given a weighted graph G, construct the semidefinite relaxation of the MAXCUT program https://www.cs.cmu.edu/~anupamg/adv-approx/lecture14.pdf to provide a lower bound.
    :param G:
    :return:
    """
    n = len(G.nodes)
    X = cp.Variable((n, n), symmetric=True)
    obj = cp.Maximize(cp.trace(nx.adjacency_matrix(G, weight="weight") @ X))
    constraints = [X >> 0]
    constraints += [X[i, i] == 1 for i in range(n)]
    return cp.Problem(obj, constraints)
