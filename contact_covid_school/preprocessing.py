# -*- coding: utf-8 -*-
-
# construct the adjacency matrix for dynamic networks
import numpy as np

school_contact =  np.loadtxt( 'contact/tij_Thiers13.dat' )

######################################################]
# contact time steps

times_steps = list(unique(school_contact[:,0]))
steps_nb = len(times_steps)
unique_time = list(unique(times_steps))



all_nodes = np.concatenate((school_contact[:,1],school_contact[:,2]))

number_nodes = unique(all_nodes).size

unique_nodes = list(unique(all_nodes))

Adjacency_time = [np.zeros((number_nodes,number_nodes))]

new_contact = np.copy(school_contact)

for i in range(new_contact.shape[0]):
    vect = school_contact[i]
    new_contact[i] = np.array((int(unique_time.index(vect[0])), 
               int(unique_nodes.index(vect[1])), 
               int(unique_nodes.index(vect[2]))))
 
current_line = 0    
for i in range(int(max(new_contact[:,0]))):
    A_current = np.zeros((number_nodes,number_nodes))
    for j in range(sum(new_contact[:,0] == i)):
        A_current [int(new_contact[current_line,1]), int(new_contact[current_line,2])] = 1
        
        current_line += 1
    
    Adjacency_time.append(A_current + A_current.T)
    
Adjacency_time.pop(0)

print(sum(Adjacency_time[14]))

np.save('network/Adjacency_time.npy', Adjacency_time)  

graph_global = np.zeros((number_nodes,number_nodes))

for i in range(len(Adjacency_time)):
    graph_global +=  Adjacency_time[i]
    
np.save('network/graph_global.npy', graph_global)    