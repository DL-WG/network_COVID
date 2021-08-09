# -*- coding: utf-8 -*-
import numpy as np

import matplotlib.pyplot as plt

import networkx as nx

from networkx.algorithms import community

import itertools

Adjacency_time = np.load('network/Adjacency_time.npy')

graph_global = np.load('network/graph_global.npy')


###############################################################################
##Adjacency matrix per 50 contacts
#
#
def adjacency_per_contact (A_list,n): #A_list: adj matrix time list, n: number per time steps
    adjacency_per_contact = {}
    
    for i in range(0,len(A_list)-n,n):
        adj_overlap = np.zeros((A_list[0].shape[0],A_list[0].shape[0]))
        for j in range(i,i+n):

            adj_overlap += A_list[j]
            
        adjacency_per_contact["time_"+str(i)+"_to_"+str(i+n)] = adj_overlap
    return adjacency_per_contact

###############################################################################
###observed Adjacency matrix per 50 contacts, suppose the graph is partially observed
##
##
def adjacency_per_contact (A_list,n): #A_list: adj matrix time list, n: number per time steps
    adjacency_per_contact = {}
    
    for i in range(0,len(A_list)-n,n):
        adj_overlap = np.zeros((A_list[0].shape[0],A_list[0].shape[0]))
        for j in range(i,i+n):
            
            p_error = 0.4 #part of unobserved edges
            
            observe = np.random.choice(a=[0, 1], size=(Adjacency_time[0].shape[0]**2,1), p=[p_error, 1-p_error])
            
            observe.shape = (Adjacency_time[0].shape[0],Adjacency_time[0].shape[0])
            
            A_observed = np.multiply(A_list[j], observe)
            
            adj_overlap += A_observed
            
        adjacency_per_contact["time_"+str(i)+"_to_"+str(i+n)] = adj_overlap
    return adjacency_per_contact
#
#
Adjacency_per50 = adjacency_per_contact (Adjacency_time,50) 
#Adjacency_per100 = adjacency_per_contact (Adjacency_time,100) 
#
np.save('network/Adjacency_per50_background_40pc.npy', Adjacency_per50) 



