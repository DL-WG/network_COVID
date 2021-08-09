# -*- coding: utf-8 -*-


import numpy as np

import scipy

from scipy.sparse import diags

import networkx as nx

import random

from scipy.sparse import diags

from DA_method import *


adjacency_per_contact = []

for i in range(0,100):
    
    print(i)

    G_gloabl = np.random.choice((0, 1), size=(1000, 1000), p=(.995, .005))
    
    for k in range(5):
        
        
        G = nx.scale_free_graph(100,  alpha=0.41, beta=0.54, gamma=0.05)
        
        G = nx.generators.random_graphs.barabasi_albert_graph(200,2)
        
        A = nx.adjacency_matrix(G)
        
        A = A.todense()
        
        order = list(np.random.permutation(200))
        
        A = A[order,:]
        
        A = A[:,order]
        
        
        G_gloabl[(200*k):(200*(k+1)),(200*k):(200*(k+1))] = np.copy(A)
        
    adjacency_per_contact.append(G_gloabl)
    
np.save('network/Adjacency_multilayer_5clusters.npy', adjacency_per_contact) 

plt.imshow(G_gloabl)


adjacency = np.load('network/Adjacency_true.npy')


def adjacency_per_contact (A_list,n): #A_list: adj matrix time list, n: number per time steps
    adjacency_per_contact = {}
    
    for i in range(0,len(A_list)-n,n):
        adj_overlap = np.zeros((A_list[0].shape[0],A_list[0].shape[0]))
        
        for j in range(i,i+n):
            
            adj_overlap += A_list[j]            

        p_error = 0.
        adj_overlap = partial_graph(adj_overlap, p_error)
        adjacency_per_contact["time_"+str(i)+"_to_"+str(i+n)] = adj_overlap
    return adjacency_per_contact
#
#
Adjacency_per1 = adjacency_per_contact (adjacency,1) 
#Adjacency_per100 = adjacency_per_contact (Adjacency_time,100) 
#
np.save('network/Adjacency_background_0pc.npy', Adjacency_per1) 

community_list = []

for i in range(5):
    community_list.append(list(range(i*200,i*200 + 200)))

np.save('network/clusters_5.npy', community_list)


