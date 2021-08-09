# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 17:56:24 2020

@author: siboc
"""

import time
from pylab import *
import matplotlib.gridspec as gridspec
import numpy as np
import random

def transmition(p):
    return random.random() < p


class contact_infect(object):
    
    time = 0
    
    def __init__(self,I, I_time, R ):# p: transmission rate, R: list of recovered person,
        #I_time : list of time being infected, otherwise -1
        

        self.I = I #vector of infected person
        self.R = R #vector of infected person
        self.I_time = I_time 
        
    def propagation_SI(self, A, L , p=0.01): #d: days of simulation? L: vaccinated person
        
        (self.I).shape = ((self.I).size,1)
        
        if (self.I).size != A.shape[0]:
            print ("not same size")
            
        
        for i in range((self.I).size):
            for j in range((self.I).size):

                A[i,j] =  transmition(A[i,j]*p)
            
        I_new = np.dot(A,self.I) 
        
        for i in L:
            I_new[i] = 0
        
        I_new += self.I
        
        I_new = [1 if a_ > 1 else a_ for a_ in I_new]
        
        self.I = np.array(I_new).astype(int)
        
        return self.I
    
    def propagation_SIR(self, A, L , R_period,p=0.01): #d: days of simulation? 
        #L: vaccinated person, R_period: time to be recovered after infected
        
        (self.I).shape = ((self.I).size,1)
        
        if (self.I).size != A.shape[0]:
            print ("not same size")
            
        
        for i in range((self.I).size):
            for j in range((self.I).size):

#                A[i,j] = (A[i,j]>0) * transmition(p)
                
                A[i,j] =  transmition((A[i,j]*p))
            
        I_new = np.dot(A,self.I) 
        
        
        for i in L:
            I_new[i] = 0
            
        I_new += self.I
            
        #recovered nodes will not be infected    
        for i in self.R:
            I_new[i] = 0
        
        I_new = [1 if a_ > 1 else a_ for a_ in I_new]
        
        # update I_time
        for i in range(len(self.I_time)):
            if I_new[i] >= 1:
                self.I_time[i] += 1
        
        #delete recovered nodes
        
        self.R += [i for i,v in enumerate(self.I_time) if v > R_period]
        
        self.I_time = [-2 if a_ > R_period else a_ for a_ in self.I_time]
        
        
        self.I = np.array(I_new).astype(int)
        
        return self.I, self.R, self.I_time


    def propagation_SIR_multilayer(self, A, L , proba_list, R_period,): #d: days of simulation? 
        # each layer has a different infectious probability
        #L: vaccinated person, R_period: time to be recovered after infected
        
        (self.I).shape = ((self.I).size,1)
        
        if (self.I).size != A.shape[0]:
            print ("not same size")
            
        
        for i in range((self.I).size):
            for j in range((self.I).size):

#◘                A[i,j] = (A[i,j]>0) * transmition(p)
                
                A[i,j] =  transmition((A[i,j]*proba_list[i]))
                #print (j,proba_list[j])
            
        I_new = np.dot(A,self.I) 
        
        
        for i in L:
            I_new[i] = 0
            
        I_new += self.I
            
        #recovered nodes will not be infected    
        for i in self.R:
            I_new[i] = 0
        
        I_new = [1 if a_ > 1 else a_ for a_ in I_new]
        
        # update I_time
        for i in range(len(self.I_time)):
            if I_new[i] >= 1:
                self.I_time[i] += 1
        
        #delete recovered nodes
        
        self.R += [i for i,v in enumerate(self.I_time) if v > R_period]
        
        self.I_time = [-2 if a_ > R_period else a_ for a_ in self.I_time]
        
        
        self.I = np.array(I_new).astype(int)
        
        return self.I, self.R, self.I_time
    
    
    
    def vaccinate(self, L): #L: list of vaccinated nodes
        
        for i in L:
            self.I[i] = 0
            
        return self.I
        

        

    
if __name__ == '__main__':    
    
    case = contact_infect(I = np.array([0,1,0,0,1]), I_time = [-1]*5 ,R = []) 
    
    A = np.random.randint(2, size=(5, 5))
    
    for i in range(6):
        
        case.propagation_SIR(A,  L = [], R_period = 2)
        print ("I",case.I)
        print ("I_time",case.I_time)
        print ("R",case.R)
        
        
    