# -*- coding: utf-8 -*-
# assimilation shallow water
import numpy as np

import scipy

from scipy.sparse import diags

import networkx as nx

import random

from scipy.sparse import diags

def Kalman_gain(H,B,R):
    K=np.dot(B,np.dot(np.transpose(H),np.linalg.pinv(np.dot(H,np.dot(B,np.transpose(H)))+R)))
    return K

def Kalman_gain_sparse(B,R): #H,B,R cipy sparse matrix
    
    sHT = scipy.sparse.csr_matrix(np.transpose(H_deg_1D(R.shape[0]))) 
    
    sH = scipy.sparse.csr_matrix(H_deg_1D(R.shape[0])) 
    
    K = kron(B,kron(sHT,inv(kron(sH,kron(B,sHT))+R))) #calculate the Kalman gain matrix
    
    return K

def BLUE(xb,Y,H,B,R): #booleen=0 garde la trace
    dim_x = xb.size
    #dim_y = Y.size
    Y.shape = (Y.size,1)
    xb1=np.copy(xb)
    xb1.shape=(xb1.size,1)
    K=np.dot(B,np.dot(np.transpose(H),np.linalg.pinv(np.dot(H,np.dot(B,np.transpose(H)))+R))) #matrice de gain
    
    vect=np.dot(H,xb1)
    xa=np.copy(xb1+np.dot(K,(Y-vect)))

    
    return xa
    

def H_degree(X): # X: boolean vector represent a graph (Adjacency matrix)
    n = int(sqrt(X.size))
    if n**2 != X.size:
        print ("graph no square")
    A = X.reshape(n,n)
    if A != A.T:
        print ("undirected graph")
    return np.sum(A,axis = 1)
    

def H_deg_1D(n): #n number of nodes
    H = np.zeros((n,n**2))
    
    for i in range(n):
        H[i,(i*n):((i+1)*n)] = np.ones(n)
        
    return H

#some functions for network
    
def highest_degree (A,p, L_old): #A adj matrix, p: percentatage eg 10% highest, L_old:alreday vaccinate
    
    vect_degree = np.sum(A,axis = 1)
    
    for i in L_old:
        vect_degree[i] = -1 #not vaccinate people already vaccinated
    v_number = int(A.shape[0]*p)
    return list(np.argsort(vect_degree)[-v_number:])  + L_old


def highest_degree_multilayer (A,p, p_list, L_old): #A adj matrix, p: percentatage eg 10% highest, L_old:alreday vaccinate
    
    vect_degree = np.sum(A,axis = 1)
    
    p_list = np.array(p_list)
    
    p_list.shape = (1,p_list.size)
    
    vect_degree.shape = (1,vect_degree.size)
    
    vect_degree = vect_degree*p_list
    
    vect_degree.shape = (vect_degree.size,)
    
    for i in L_old:
        vect_degree[i] = -1 #not vaccinate people already vaccinated
    v_number = int(A.shape[0]*p)
    
    return list(np.argsort(vect_degree)[-v_number:])  + L_old

def highest_bc(A,p, L_old):
    v_number = int(A.shape[0]*p)
    
    G_state=nx.from_numpy_matrix(A)
    bc = nx.betweenness_centrality(G_state, normalized=False)
    bc_list = []
    for key, value in bc.items():
        temp = [key,value]
        bc_list.append(value)
    
    vect_bc = np.array(bc_list)
    
    for i in L_old:
        vect_bc[i] = -1 #not vaccinate people already vaccinated
    
    return list(np.argsort(vect_bc)[-v_number:]) + L_old


def highest_bc_multilayer (A,p, p_list, L_old): #A adj matrix, p: percentatage eg 10% highest, L_old:alreday vaccinate
    
    v_number = int(A.shape[0]*p)

    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            
            A[i,j] = A[i,j]*((p_list[i] + p_list[j])/2.)
            
    G_state=nx.from_numpy_matrix(A)
    
    bc = nx.betweenness_centrality(G_state, normalized=False)
    bc_list = []
    for key, value in bc.items():
        temp = [key,value]
        bc_list.append(value)
    
    vect_bc = np.array(bc_list)
    
    for i in L_old:
        vect_bc[i] = -1 #not vaccinate people already vaccinated
    
    return list(np.argsort(vect_bc)[-v_number:]) + L_old    


def partial_graph(A, p): #A adjacency matrix, p: unobservable rate
    
    unobserved = random.sample(list(range(A.shape[0])), int(p*A.shape[0]))
    
    for i in unobserved:
        A[i,:] = np.zeros((1,A.shape[0]))
        A[:,i] = np.zeros((1,A.shape[0]))
        
    return A


def partial_graph_fix(A, unobserved): #A adjacency matrix, p: unobservable rate
        
    for i in unobserved:
        A[i,:] = np.zeros((1,A.shape[0]))
        A[:,i] = np.zeros((1,A.shape[0]))
        
    return A

def sub_G_H(A, community_list): #A adjacency matrix, p: unobservable rate
    
    H = np.zeros((A.shape[0],A.shape[0]))
    for i in range(len(community_list)):
        H[np.ix_(community_list[i],community_list[i])] = np.ones((len(community_list[i]),len(community_list[i])))
#        
    H_2D = diags(H.ravel(), 0)
    return H_2D


def evolution_proba(p,delta_p):
    
    plus_p = random.uniform(-delta_p, delta_p)
    
    p += plus_p
    
    p = max(p,0)
    
    return p


def infected_community(I,community_list):
    
    I_per_com = []
    for i in range(len(community_list)):
        I_per_com.append(int(sum(I[community_list[i]])))
    
    return I_per_com
    
    
    
        
if __name__ == '__main__':   
    #plt.imshow(H_deg_1D(10))  
    #print(partial_graph(np.random.rand(6,6), 0.5))
    
#    tt =np.random.rand(5,5)
#    tt = tt + tt.T
    tt = np.zeros((327,327))
    x = np.ones((327,1))
    community_list = np.load('network/clusters_3.npy')
    sub_H = sub_G_H(tt, community_list)
    kron(sub_H,x)
    print(sub_H)
   
    
    

    
    

