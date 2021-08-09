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

from DA_method import *

Adjacency_time_steps = np.load('network/Adjacency_time.npy')

graph_global = np.load('network/graph_global.npy')

############################################################"
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
#
#
Adjacency_per100 = adjacency_per_contact (Adjacency_time_steps,100) 

 
np.save('network/Adjacency_per100.npy', Adjacency_per100) 
##
##
###############################################################################
###observed Adjacency matrix per 50 contacts, suppose the graph is partially observed
##
##
def adjacency_per_contact (A_list,n,unobserved): #A_list: adj matrix time list, n: number per time steps
    adjacency_per_contact = {}
    
    for i in range(0,len(A_list)-n,n):
        adj_overlap = np.zeros((A_list[0].shape[0],A_list[0].shape[0]))
        
        for j in range(i,i+n):

            adj_overlap += A_list[j]            

        p_error = 0.4
        
        #adj_overlap = partial_graph(adj_overlap, p_error)
        
        adj_overlap = partial_graph_fix(adj_overlap,  unobserved)
        
        adjacency_per_contact["time_"+str(i)+"_to_"+str(i+n)] = adj_overlap
    return adjacency_per_contact
#
#
p_error = 0.4

unobserved = random.sample(list(range(Adjacency_time_steps[0].shape[0])), int(p_error*Adjacency_time_steps[0].shape[0]))

Adjacency_per100 = adjacency_per_contact(Adjacency_time_steps,100,unobserved)

np.save('network/Adjacency_per100_fix_background_40pc.npy', Adjacency_per100) 

##########################################################################

#
t1 = 220
t2 = 133
t3 = 29
#
t1_apperance = []
t2_apperance = []
t3_apperance = []

for tt in range(2000):
    t1_apperance.append(sum(Adjacency_time_steps[tt][t1]))
    t2_apperance.append(sum(Adjacency_time_steps[tt][t2]))
    t3_apperance.append(sum(Adjacency_time_steps[tt][t3]))
    
plt.plot(t2_apperance,'r', label = 'student 1')    
plt.plot(t1_apperance, label = 'student 2')
plt.plot(t3_apperance, 'g',label = 'student 3')
plt.xlabel('time')
plt.ylabel('degree in dynmaic network')
plt.legend()

#plt.savefig('figures/student123.eps',format='eps')

(np.var(t1_apperance) - np.mean(t1_apperance))/(np.var(t1_apperance) + np.mean(t1_apperance))
#######################################################################################
# regroup every 50 networks together

A_per3000 = np.load('network/Adjacency_per3000.npy').item()
l_non_zero = []
l_back = []
l_3000 = []
for i in range(1,len(Adjacency_time)):
    A_overlap =  A_per50["time_"+str((i-1)*50)+"_to_"+str(i*50)]
    A_overlap_background = A_per50_background["time_"+str((i-1)*50)+"_to_"+str(i*50)]
    l_non_zero.append(sum(A_overlap>0))
    l_back.append(sum(A_overlap_background>0))

    


##################################################################################
    
community_list = np.load('network/clusters_3.npy')


A1 = graph_global[:, community_list[0] + community_list[1] + community_list[2]][community_list[0] + community_list[1] + community_list[2]]

community_three = np.zeros((3,3))

for i in range(3):
    for j in range(3):
        
        community_three[i,j] = np.sum(graph_global[:,community_list[i]][community_list
                       [j]] >= 1)/(np.size(graph_global[:,community_list[i]][community_list[j]])*1.)
    
    
f = plt.figure()
plt.imshow(np.ones((3,3))-community_three,vmax = 1.0, vmin = 0.7, cmap='gray')
plt.show()


############################################################################

classes = ["cluster 1", "cluster 2", "cluster 3"]
size = 3
data = np.arange(size * size).reshape((size, size))
values =  np.around(community_three, decimals=3)

# Limits for the extent
x_start = 3.0
x_end = 9.0
y_start = 6.0
y_end = 12.0

extent = [x_start, x_end, y_start, y_end]

# The normal figure
fig = plt.figure(figsize=(16, 12))



ax = fig.add_subplot(111)

im = ax.imshow(np.ones((3,3))-community_three, extent=extent, origin='lower', interpolation='None', cmap='gray')

ax.set_xticks([4,6,8])
ax.set_yticks([7,9,11])
ax.set_xticklabels(classes,fontsize = 18)
ax.set_yticklabels(classes,fontsize = 18)

# Add the text
jump_x = (x_end - x_start) / (2.0 * size)
jump_y = (y_end - y_start) / (2.0 * size)
x_positions = np.linspace(start=x_start, stop=x_end, num=size, endpoint=False)
y_positions = np.linspace(start=y_start, stop=y_end, num=size, endpoint=False)

for y_index, y in enumerate(y_positions):
    for x_index, x in enumerate(x_positions):
        label = values[y_index, x_index]
        text_x = x + jump_x
        text_y = y + jump_y
        ax.text(text_x, text_y, label, color='b', ha='center', va='center',fontsize = 24)

#fig.colorbar(im)
plt.savefig("figures/cluster_bw.eps",fmt = '.eps')
plt.show()


A_per500 = np.load('network/Adjacency_per500.npy',allow_pickle='TRUE').item()

deg1 = np.sum(A_per500['time_6500_to_7000'],axis = 0)
deg2 = np.sum(A_per500['time_0_to_500'],axis = 0)

s = list(deg1)

deg1 = deg1[sorted(range(len(s)), key=lambda k: s[k])]

deg2 = deg2[sorted(range(len(s)), key=lambda k: s[k])]

fig, ax1 = plt.subplots()

ax1.plot(deg1,linewidth = 3, drawstyle='steps')
ax1.set_ylabel('degree (6500 to 7000)',color = 'b',fontsize = 18)
ax2 = ax1.twinx() 

ax2.plot(deg2,'r',drawstyle='steps')
ax2.set_ylabel('degree (0 to 500)',color = 'r',fontsize = 18)

plt.savefig("figures/degree_distribution.eps",fmt = '.eps')
plt.legend()
plt.show()

###############################################################################

A_per50 = np.load('network/Adjacency_per50.npy',allow_pickle='TRUE').item()

average_density = 0

for i in range(len(A_per50)):
   
    A = np.copy(A_per50["time_" + str(i*50)+"_to_"+str(i*50 + 50)])
    
    average_density += np.sum(A>0.001)/(A.shape[0]**2)
    
community_list = np.load('network/clusters_3.npy',allow_pickle='TRUE')
