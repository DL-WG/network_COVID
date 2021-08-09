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

MC = 100

Adjacency_time = np.load('network/Adjacency_time.npy')

#Adjacency_time = np.copy(Adjacency_time[:2000])


A_per50 = np.load('network/Adjacency_per50.npy',allow_pickle='TRUE').item()

A_per50_background = np.load('network/Adjacency_per50_background_40pc.npy',allow_pickle='TRUE').item()

A_per50_background_bis = np.load('network/Adjacency_per50_background_40pcbis.npy',allow_pickle='TRUE').item()

#L = list(np.load('data/vaccinate_highest_degree.npy'))

#L = list(np.load('data/vaccinate_centrality.npy'))

#L = list(np.load('data/vaccinate_random.npy'))


p_ini = 0.2

p = 0.02 #vaccinaton rate each step

with_DA = 1

O_I = np.load('data/O_I.npy')

if with_DA == 1:
    
    B = diags([1] * (O_I.size)**2, 0)
    R = diags([1] * (O_I.size), 0)
    
    sHT = scipy.sparse.csr_matrix(np.transpose(H_deg_1D(O_I.size))) 
    
    sH = scipy.sparse.csr_matrix(H_deg_1D(O_I.size)) 
    
    K = kron(B,kron(sHT,inv(kron(sH,kron(B,sHT))+R))) #calculate the Kalman gain matrix

for it_number in range(1,MC):
    
    
    p_ini = 0.2
    
    p = 0.02 #vaccinaton rate each step
    
    with_DA = 1
    
    O_I = np.load('data/O_I.npy')
    
    L = []
    
    #O_I = np.random.choice(a=[1, 0], size=(Adjacency_time[0].shape[0],1), p=[p_ini, 1-p_ini]) # original infected

    #np.save('data/O_I.npy',O_I)
    
    
    
    case = contact_infect(O_I,I_time = [-1]*O_I.size ,R = [])
    
    number_infect = [sum(O_I)]
    
    number_infect = []
    
    print('MC',it_number )

    for i in range(Adjacency_time.shape[0]):
        
        if i%100 == 0:
            np.save('data_MC/number_infect_background_hd_1000_40pc_ytrue_SIR960_'+str(it_number)+'.npy',number_infect)
            print (i)
        
        A = np.copy(Adjacency_time[i])
        
        if i in range(50,len(Adjacency_time),50):
            
            if with_DA == 0:
            
                A_overlap =  A_per50["time_"+str(i-50)+"_to_"+str(i)]
                    
                L = list(highest_bc (A_overlap,p)) 
                
            else:
                A_overlap_background = A_per50_background["time_"+str(i-50)+"_to_"+str(i)]
                
                #A_overlap_background_bis = A_per50_background_bis["time_"+str(i-50)+"_to_"+str(i)]
                
                A_overlap =  A_per50["time_"+str(i-50)+"_to_"+str(i)]
                
                
                xb = A_overlap_background.reshape(A_overlap_background.size,1)
                
    #            vect = np.dot(H_deg_1D(O_I.size),xb)
    #            
    #            xa = np.copy(xb+np.dot(K.toarray(),(Y-vect)))       
                
                #xa.shape = (O_I.size,O_I.size)
                
                xb.shape = (O_I.size,O_I.size)
                
                L = list(highest_degree (xb,p,L)) 
                
        
        case.propagation_SIR(A,L, R_period = 960)
            
        number_infect.append(int(np.sum(case.I)))
    
    plt.plot(number_infect,linewidth = 2)
    
    plt.show()
    
    np.save('data_MC/number_infect_background_hd_1000_40pc_ytrue_p5_SIR960_'+str(it_number)+'.npy',number_infect)


