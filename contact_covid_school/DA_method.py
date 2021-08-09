# -*- coding: utf-8 -*-
# assimilation shallow water
import numpy as np

import scipy

from scipy.sparse import diags

import networkx as nx

def Kalman_gain(H,B,R):
    K=np.dot(B,np.dot(np.transpose(H),np.linalg.pinv(np.dot(H,np.dot(B,np.transpose(H)))+R)))
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


        
if __name__ == '__main__':   
    #plt.imshow(H_deg_1D(10))  
    print(highest_degree(np.eye(10),0.5, [0,1]))
    
    

