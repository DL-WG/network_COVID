# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 23:15:53 2020

@author: siboc
"""
import numpy as np

import matplotlib.pyplot as plt

import networkx as nx

from networkx.algorithms import community

import itertools

import community as community_louvain



def inter_community_edges(G, partition):

    #
    MG = nx.MultiDiGraph if G.is_directed() else nx.MultiGraph
    return nx.quotient_graph(G, partition, create_using=MG).size()

graph_global = np.load('network/graph_global.npy')

G_state=nx.from_numpy_matrix(graph_global)

#

#

performance_list = []

for i in range(1,4):
    community_list_x = []
    communities_generator = community.asyn_fluidc(G_state,i)

    for communities in itertools.islice(communities_generator, i):

        community_list_x.append(list(communities))
#    
    
    performance_list.append(nx.algorithms.community.quality.performance(G_state,community_list_x ))
  

performance_list = np.array(performance_list)
performance_vect = performance_list[1:10]

performance_vect_tmp = performance_list[1:9]
performance_vect_decale = performance_list[2:10]

performance_increment = performance_vect_decale - performance_vect_tmp

plt.plot(range(1,10),performance_vect,'--',linewidth = 2)
plt.plot(range(1,10),performance_vect,'b^',label = "performance value")
plt.bar(range(2,10),performance_increment,width=0.3,align='center',label = "performance increment")
#plt.plot(range(2,9),performance_increment,'r*')
plt.xticks(range(10))
plt.xlabel("number of cluster")
plt.ylabel("performance rate")
plt.xlim(0,10)
plt.grid(True)
plt.ylim(-0.1,1.)
plt.legend(loc='upper center', bbox_to_anchor=(0.7, .7), ncol=1)    

intra_edges = nx.algorithms.community.quality.intra_community_edges(G_state, partion)

inter_edges = nx.algorithms.community.quality.inter_community_edges(G_state, partion)
