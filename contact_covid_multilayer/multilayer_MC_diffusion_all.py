# -*- coding: utf-8 -*-
"""
#multilayer network
"""

from simulation_graph import *

from DA_method import *

import networkx as nx

import scipy

from scipy.sparse import diags

from scipy.sparse.linalg import inv

import copy

MC = 10

error_level = 50

A_true = np.load('network/Adjacency_multilayer_5clusters.npy')

A_true = A_true[:90]

A_background = np.load('network/Adjacency_background_'+str(error_level)+'pc.npy',allow_pickle='TRUE').item()


p_vaccinate = 0.02 #vaccinaton rate each step

with_DA = 1

delta_p = 0.004

Ip1 = 0.025

Ip2 = 0.010

Ip3 = 0.010

Ip4 = 0.010

Ip5 = 0.010


###############################################################

community_list = np.load('network/clusters_5.npy')

N = np.array([1]*200 + [0]*800)

N.shape = (1,1000)

N2 = np.array([0]*200 + [1]*200 + [0]*600)

N2.shape = (1,1000)

N3 = np.array([0]*400 + [1]*200 + [0]*400)

N3.shape = (1,1000)

N4 = np.array([0]*600 + [1]*200 + [0]*200)

N4.shape = (1,1000)

N5 = np.array([0]*800 + [1]*200)

N5.shape = (1,1000)

N = np.concatenate((N,N2),axis = 0)

N = np.concatenate((N,N3),axis = 0)

N = np.concatenate((N,N4),axis = 0)

N = np.concatenate((N,N5),axis = 0)



############################################################

def vac_in_layer(L):
    L_vec = np.array(L)
    
    return [sum(L_vec<200), sum(L_vec<400)-sum(L_vec<200), sum(L_vec<600)
            -sum(L_vec<400), sum(L_vec<800) - sum(L_vec<600), sum(L_vec<1000)
            -sum(L_vec<800)]


############################################################

true_rate = np.zeros((1,5))
DA_rate = np.zeros((1,5))
###############################################################

for it_number in range(0,MC):
    
    p1 = Ip1
    
    p2 = Ip2
    
    p3 = Ip3
    
    p4 = Ip4
    
    p5 = Ip5
    
    proba_list = [p1]*200 + [p2]*200 + [p3]*200 +[p4]*200 +[p5]*200
    
##################################################################"

    true_rate = np.array([p1,p2,p3,p4,p5])
    true_rate.shape = (1,true_rate.size)
    
    DA_rate = np.array([0.01,0.01,0.01,0.01,0.01])
    DA_rate .shape = (1,DA_rate .size)
    
################################################################    
    
    proba_background = [0.01]*200 + [0.01]*200 + [0.01]*200 +[0.01]*200 +[0.01]*200

    xb = np.array([p1,p2,p3,p4,p5])
    
    xb.shape = (xb.size,1)
    
    O_I = np.load('data/O_I_artificial.npy')
    
    L_random = []
    
    L_OI = []
    
    L_free = []
    
    L_background = []
    
    L_true = []
    
    L_DA_square = []
    
    case_free = contact_infect(O_I,[-1]*O_I.size ,[])
    
    case_random = contact_infect(O_I, [-1]*O_I.size , [])
    
    case_background = contact_infect(O_I, [-1]*O_I.size , [])
    
    case_true = contact_infect(O_I,I_time = [-1]*O_I.size ,R = [])
    
    case_DA_square = contact_infect(O_I, [-1]*O_I.size , [])
    
    infect_by_com = infected_community(case_DA_square.I,community_list)
    
    
    infect_rate_old = np.array(infect_by_com)/np.array([200]*5)
    #initial number of infected in each layer
    
#####################################################################"
    
    number_infect_free = [sum(O_I)]
    
    number_infect_random = [sum(O_I)]
    
    number_infect_background = [sum(O_I)]
    
    number_infect_true = [sum(O_I)]
    
    number_infect_DA_square = [sum(O_I)]
    
    #number_infect = []
    
    print('MC',it_number )

    for i in range(A_true.shape[0]-1):
        
        if i%5 == 0:
            
            p1 = evolution_proba(p1,delta_p)
            
            p2 = evolution_proba(p2,delta_p)
            
            p3 = evolution_proba(p3,delta_p)
            
            p4 = evolution_proba(p4,delta_p)
            
            p5 = evolution_proba(p5,delta_p)
            
            print(p1,p2,p3,p4,p5)
            
            proba_list = [p1]*200 + [p2]*200 + [p3]*200 +[p4]*200 +[p5]*200
        
        A = np.copy(A_true[i,:])
        
        if i in range(A_true.shape[0]-1):
            
#        if i in range(40):
            
            print(i)
            if i == 1:
                
                L_free = []
    
                L_background = []
                
                L_true = []
                
                L_DA_square = []
                

            
            if with_DA == 0:
            
                #A_overlap =  A_per500["time_"+str(i-50)+"_to_"+str(i)]
                    
                L = list(highest_bc (A_overlap,p)) 
                
            if with_DA == 1:
                
                
                infect_by_com = infected_community(case_DA_square.I,community_list)
                
                
                
                infect_rate = np.array(infect_by_com)/np.array([200]*5)
                # all the infect number in each layer
                
                y = infect_rate - infect_rate_old
                
                y = y.clip(min = 0.001)
                #incremental infect number
                
                infect_rate_old = np.copy(infect_rate)
                
                xb.shape = (xb.size,1)
                
                I_current = np.copy(case_background.I)
                
                p_infect = np.dot(A,I_current)
                
                for ii in L_background:
                    
                    p_infect[ii] = 0
                    
                p_infect.shape = (p_infect.size,1)
                
                ee = np.dot(N, p_infect)
                
                ee.shape = (5,)
                
                H = np.diag(ee)
                
                B = np.eye(5)
                
                R = 4 * np.eye(5)
                
                xa = BLUE(xb,y,H,B,R)
                
                print('list(xa)',list(xa))
                
                print('true',p1,p2,p3,p4,p5)
                
                A_original =  np.copy(A_true[i,:])
                                                     
                pa = xa.ravel() #analyzed probability
                
                estimation_proba = [pa[0]]*200 + [pa[1]]*200 + [pa[2]]*200 + [pa[3]]*200 + [pa[4]]*200
                
                
                
                L_DA_square = list(highest_degree_multilayer (A_original,p_vaccinate,estimation_proba,L_DA_square ))
                
                print ('y',y)
                print ('infect_by_com',infect_by_com)
                print ('L_DA_square',vac_in_layer(L_DA_square))
   #             L_true = list(highest_degree_multilayer (A_original, p_vaccinate, proba_list, L_true ))
                
                xb = np.copy(xa)
                
                true_rate = np.concatenate((true_rate,np.array([[p1,p2,p3,p4,p5]])),axis = 0)
                
                xa.shape = (1,xa.size)
                
                DA_rate = np.concatenate((DA_rate,xa),axis = 0)
                
            
              
        
#        case_free.propagation_SIR(A,L_free, R_period = 120 + random.randint(-10,10))
#        case_random.propagation_SIR(A,L_random, R_period = 60 + random.randint(-10,10))
        
#        case_background.propagation_SIR_multilayer(A,L_background, proba_list, R_period = 60 + random.randint(-10,10))
        
#        case_true.propagation_SIR(A,L_true, R_period = 60 + random.randint(-10,10))
#        
        case_DA_square.propagation_SIR_multilayer(A,L_DA_square, proba_list, R_period = 60 + random.randint(-10,10))
#            
#        number_infect_free.append(int(np.sum(case_free.I)))
        
#        number_infect_random.append(int(np.sum(case_random.I)))
#        
#        number_infect_background.append(int(np.sum(case_background.I)))
        
#        number_infect_true.append(int(np.sum(case_true.I)))
#        
        number_infect_DA_square.append(int(np.sum(case_DA_square.I)))
    
#    plt.plot(number_infect_free,'b',linewidth = 2)
    
#    print ('L_background',L_background)
    
#    plt.plot(number_infect_random,'y',linewidth = 2)
    
#    plt.plot(number_infect_background,'r',linewidth = 2)
#    
#    plt.plot(number_infect_true,'k',linewidth = 2)
#    
    plt.plot(number_infect_DA_square,'g',linewidth = 2)
    
    plt.show()
    
    #print('len(L_true)',len(L_true))
    
#    np.save('data_final/free_per50_SIR120_'+str(it_number)+'.npy',number_infect_free)
    
#    np.save('data_multilayer/random_025_'+str(error_level)+'_SIR60'+str(it_number)+'.npy',number_infect_random)
#            
#    np.save('data_multilayer/hd_background_'+str(Ip1)+'_'+str(Ip1)+'_'+str(Ip1)+
#            '_'+str(Ip1)+'_'+str(Ip1)+'_'+ 'vac_' + str(p_vaccinate) + str(error_level)+'_SIR60_'+str(it_number)+'.npy',number_infect_background)
#            
#    np.save('data_final/bc_true_vac001_per50_'+str(error_level)+'_SIR120_'+str(it_number)+'.npy',number_infect_true)
#        
#######################    
    np.save('data_multilayer/hd_DA_'+str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
            '_'+str(Ip4)+'_'+str(Ip5)+'_'+ 'vac_' + str(p_vaccinate) + str(error_level)+'_SIR60_'+str(it_number)+'.npy',number_infect_DA_square)
    

    np.save('data_multilayer/DA_rate_'+str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
            '_'+str(Ip4)+'_'+str(Ip5)+'_'+ 'vac_'+str(it_number)+'.npy',DA_rate)
    
    np.save('data_multilayer/true_rate_'+str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
            '_'+str(Ip4)+'_'+str(Ip5)+'_'+ 'vac_'+str(it_number)+'.npy',true_rate)
    
        
##############################################   
