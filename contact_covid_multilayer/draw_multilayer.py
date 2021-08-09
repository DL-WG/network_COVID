# -*- coding: utf-8 -*-
# draw the graph

from simulation_graph import *

from DA_method import *

import networkx as nx

import scipy

from scipy.sparse import diags

from scipy.sparse.linalg import inv

import copy


A_true = np.load('network/Adjacency_multilayer_5clusters.npy')


A = A_true[0,:]


G = nx.from_numpy_matrix(A)



def community_layout(g, partition):

    pos_communities = _position_communities(g, partition, scale=3.)

    pos_nodes = _position_nodes(g, partition, scale=1.0)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos

def test():
    # to install networkx 2.0 compatible version of python-louvain use:
    # pip install -U git+https://github.com/taynaud/python-louvain.git@networkx2
    from community import community_louvain

    g = nx.from_numpy_matrix(A)
    partition = community_louvain.best_partition(g)
    
    for i in range(1000):
        
        partition[i] = int(i/200)
        
    #partition2 = nx.algorithms.community.asyn_fluid.asyn_fluidc(g,10)
    pos = community_layout(g, partition)

    nx.draw(g, pos, node_size=[70]*1000, node_color=list(partition.values())); 
    plt.savefig("figures/SIR_multilayer_5cluster.pdf")
    plt.show()
    return


test()