# -*- coding: utf-8 -*-
"""
#vaccination strategy with DA where the degree is observed
"""

from simulation_graph import *

from DA_method import *

import networkx as nx

import scipy

from scipy.sparse import diags

from scipy.sparse.linalg import inv

Adjacency_time = np.load('network/Adjacency_time.npy')

Adjacency_time = Adjacency_time[:1000]


A_per50 = np.load('network/Adjacency_per50.npy',allow_pickle='TRUE').item()

A_per50_background = np.load('network/Adjacency_per50_background_10pcbis.npy',allow_pickle='TRUE').item()

L = []

p_ini = 0.2

p = 0.02 #vaccinaton rate

#with_DA = 0 to simulate free propagation of virus
with_DA = 1

O_I = np.load('data/O_I.npy')

case = contact_infect(O_I,I_time = [-1]*O_I.size ,R = [])

number_infect = [sum(O_I)]

number_infect = []

if with_DA == 1:
    
    B = diags([1] * (O_I.size)**2, 0)
    R = diags([1] * (O_I.size), 0)
    
    sHT = scipy.sparse.csr_matrix(np.transpose(H_deg_1D(O_I.size))) 
    
    sH = scipy.sparse.csr_matrix(H_deg_1D(O_I.size)) 
    
    K = kron(B,kron(sHT,inv(kron(sH,kron(B,sHT))+R)))

for i in range(Adjacency_time.shape[0]):
    
    if i%100 == 0:
        np.save('data/number_infect_DA_bc_1000_10pc_observation_SIR20.npy',number_infect)
    
    A = Adjacency_time[i]
    
    if i in range(50,len(Adjacency_time),50):
        
        if with_DA == 0:
        
            A_overlap =  A_per50["time_"+str(i-50)+"_to_"+str(i)]
                
            L = list(highest_bc (A_overlap,p)) 
            
        else:
            A_overlap_background = A_per50_background["time_"+str(i-50)+"_to_"+str(i)]
            
            A_overlap_background_bis = A_per50_background["time_"+str(i-50)+"_to_"+str(i)]
            
            A_overlap =  A_per50["time_"+str(i-50)+"_to_"+str(i)]
            
            Y = sum(A_overlap_background_bis, axis = 1)
            
            Y.shape = (Y.size,1)
            
            xb = A_overlap_background_bis.reshape(A_overlap_background.size,1)                  
            
            xa.shape = (O_I.size,O_I.size)
            
            xb.shape = (O_I.size,O_I.size)
            
            # if simulating with the assimilated data
            #L = list(highest_degree (xa,p,L)) 
            L = list(highest_degree (xb,p,L)) 
            
                
    case.propagation_SIR(A,L, R_period = 120)
        
    number_infect.append(sum(case.I))

plt.plot(number_infect,linewidth = 2)
plt.show()



