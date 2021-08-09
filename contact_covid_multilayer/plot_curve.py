# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 21:34:39 2020

@author: siboc
"""

import time
from pylab import *
import matplotlib.gridspec as gridspec
import numpy as np
import random


number_infect_free = np.load('data/number_infect_free.npy')

number_infect_hd = np.load('data/number_infect_highest_degree.npy')

number_infect_bc = np.load('data/number_infect_bc.npy')


number_infect_random = np.load('data/number_infect_random.npy')


plt.plot(number_infect_free*0.9, linewidth = 2, label='free')

plt.plot(number_infect_hd,'r', linewidth = 2, label='high degree')

plt.plot(number_infect_bc,'g', linewidth = 2, label='centrality')

plt.plot(number_infect_random,'y', linewidth = 2, label='centrality')

plt.ylabel('infected',fontsize = 18)

plt.xlabel('time steps',fontsize = 18)

plt.legend()
plt.show()
plt.close()



##########################################################################

number_infect_free = np.load('data/number_infect_free_1000.npy')

number_infect_hd_DA = np.load('data/number_infect_DA_hd_1000.npy')

number_infect_bc_DA = np.load('data/number_infect_DA_bc_1000.npy')



number_infect_free = np.load('data/number_infect_free_1000.npy')

number_infect_hd = np.load('data/number_infect_per50_hd_1000.npy')

number_infect_bc = np.load('data/number_infect_per50_bc_1000.npy')


plt.plot(number_infect_free[:1000], linewidth = 2, label='free')

plt.plot(number_infect_hd[:1700],'r', linewidth = 2, label='high degree')

plt.plot(number_infect_bc[:1700],'g', linewidth = 2, label='centrality')

plt.plot(number_infect_hd_DA[:1700],'r--', linewidth = 2, label='hd DA')

plt.plot(number_infect_bc_DA[:1700],'g--', linewidth = 2, label='bc DA')


plt.ylabel('infected',fontsize = 18)

plt.xlabel('time steps',fontsize = 18)

plt.legend()
plt.show()
plt.close()

##########################################################################

number_infect_free = np.load('data/number_infect_free_1000.npy')

number_infect_hd = np.load('data/number_infect_per50_hd_1000.npy')

number_infect_bc = np.load('data/number_infect_per50_bc_1000.npy')


plt.plot(number_infect_free[:1000], linewidth = 2, label='free')

plt.plot(number_infect_hd[:1700],'r', linewidth = 2, label='high degree')

plt.plot(number_infect_bc[:1700],'g', linewidth = 2, label='centrality')

plt.ylabel('infected',fontsize = 18)

plt.xlabel('time steps',fontsize = 18)

plt.legend()
plt.show()
plt.close()

#######################################################################################"
#SIR

#number_infect_vac_obs = np.load('data/number_infect_DA_hd_1000_25pc_observation_SIR20.npy')
                                
number_infect_vac_back = np.load('data/number_infect_DA_hd_1000_25pc_background_SIR20.npy')

number_infect_vac_DA = np.load('data/number_infect_DA_hd_1000_25pc_SIR20.npy')

#plt.plot(number_infect_vac_obs[:500], linewidth = 2, label='obs')

plt.plot(number_infect_vac_back,'r', linewidth = 2, label='background')

plt.plot(number_infect_vac_DA,'g', linewidth = 2, label='DA')

plt.ylabel('infected',fontsize = 18)

plt.xlabel('time steps',fontsize = 18)

plt.legend()
#plt.savefig("figures/SI_40.eps",fmt = '.eps')
plt.show()
plt.close()


#######################################################################################"
#SIR R = 120, p = 0.05

#number_infect_vac_obs = np.load('data/number_infect_DA_hd_1000_25pc_observation_SIR20.npy')
                                
number_infect_vac_back = np.load('data_MC/number_infect_background_hd_1000_25pc_ytrue_p5_SIR960'+str(0)+'.npy')

number_infect_vac_DA = np.load('data_MC/number_infect_DA_hd_1000_25pc_ytrue_p5_SIR960'+str(0)+'.npy')



#plt.plot(number_infect_vac_obs[:500], linewidth = 2, label='obs')

plt.plot(number_infect_vac_back,'r', linewidth = 2, label='background')

plt.plot(number_infect_vac_DA,'g', linewidth = 2, label='DA')

plt.ylabel('infected',fontsize = 18)

plt.xlabel('time steps',fontsize = 18)

plt.legend()
#plt.savefig("figures/SI_40.eps",fmt = '.eps')
plt.show()
plt.close()



########################################################################################

# plot average and std 40pc error
MC = 32

background_vac = []
for it_number in range(1,MC):
    background_vac.append(np.load('data_MC/number_infect_background_hd_allper50by1_40pc_ytrue_p5_SIR100_'+str(it_number)+'.npy'))
background_vac =  np.array(background_vac)

DA_vac_square = []
for it_number in range(1,MC):
    DA_vac_square.append(np.load('data_MC/number_infect_DA_hd_allper50by1_40pc_ytrue_p5_varsquare_reg_SIR100_'+str(it_number)+'.npy'))

DA_vac_square =  np.array(DA_vac_square)   
   
DA_vac = []
for it_number in range(1,MC):
    DA_vac.append(np.load('data_MC/number_infect_DA_hd_allper50by1_40pc_ytrue_p5_SIR100_'+str(it_number)+'.npy'))
DA_vac =  np.array(DA_vac)   

DA_true = []
for it_number in range(1,MC):
    DA_true.append(np.load('data_MC/number_infect_true_hd_allper50by1_25pc_ytrue_p5_SIR100_'+str(it_number)+'.npy'))
DA_true =  np.array(DA_true)   


plt.plot(np.mean(background_vac,axis = 0),'r',linewidth = 2, label='background')
plt.plot(np.mean(DA_vac,axis = 0),'g', linewidth = 2, label='DA')
plt.plot(np.mean(DA_vac_square,axis = 0),'b', linewidth = 2, label='DA_square')
plt.plot(np.mean(DA_true,axis = 0),'k', linewidth = 2, label='DA_true')
plt.legend()
#plt.savefig("figures/SIR_40_MC.eps",fmt = '.eps')
plt.show()
      

plt.plot(np.std(background_vac,axis = 0),'r',linewidth = 2, label='background')
plt.plot(np.std(DA_vac,axis = 0),'g', linewidth = 2, label='DA')
plt.plot(np.std(DA_vac_square,axis = 0),'b', linewidth = 2, label='DA_square')
plt.plot(np.std(DA_true,axis = 0),'k', linewidth = 2, label='DA_true')
plt.legend()
#plt.savefig("figures/SIR_40_MC.eps",fmt = '.eps')
plt.show()


##########################################################################
#
## plot average and std all according
#
#error_level = 40
#
#MC = 10
##
##random_vac = []
##for it_number in range(1,MC):
##    random_vac.append(np.load('data_MC_accord//hd_free_step1_'+str(60)+'pc_ytrue_p5_SIR100_'+str(it_number)+'.npy'))
##
##random_vac =  np.array(random_vac)
#
#free_vac = []
#for it_number in range(1,6):
#    free_vac.append(np.load('data_MC_accord/hd_free_step1_'+str(40)+'pc_ytrue_p5_SIR100_'+str(it_number)+'.npy'))
#
#free_vac =  np.array(free_vac)
#
#background_vac = []
#for it_number in range(1,MC):
#    background_vac.append(np.load('data_MC_accord/hd_background_prior_setp1_vac001_per50by1_'+str(error_level)+'pc_ytrue_p2_SIR100_'+str(it_number)+'.npy'))
#background_vac =  np.array(background_vac)
#
#DA_vac_square = []
#for it_number in range(1,MC):
#    DA_vac_square.append(np.load('data_MC_accord/hd_DA_square_prior_per50by1_vac001_'+str(error_level)+'pc_ytrue_p5_SIR100_'+str(it_number)+'.npy'))
#
#DA_vac_square =  np.array(DA_vac_square)   
#   
##DA_vac = []
##for it_number in range(1,MC):
##    DA_vac.append(np.load('data_MC/number_infect_DA_hd_allper50by1_40pc_ytrue_p5_SIR100_'+str(it_number)+'.npy'))
##DA_vac =  np.array(DA_vac)   
#
#DA_true = []
#for it_number in range(1,MC):
#    DA_true.append(np.load('data_MC_accord/hd_true_setp1_vac001_per50by1_'+str(error_level)+'pc_ytrue_p1_SIR100_'+str(it_number)+'.npy'))
#DA_true =  np.array(DA_true)   
##
##DA_true_bc = []
##for it_number in range(1,MC):
##    DA_true_bc.append(np.load('data_MC_accord/hd_true_setp1_per50by1_'+str(error_level)+'pc_ytrue_p1_SIR100_'+str(it_number)+'.npy'))
##DA_true_bc =  np.array(DA_true_bc)   
#
##plt.plot(np.mean(random_vac,axis = 0),'y',linewidth = 2, label='random')
#plt.plot(np.mean(background_vac,axis = 0),'r',linewidth = 2, label='background')
#plt.plot(np.mean(free_vac,axis = 0),'g', linewidth = 2, label='free')
#plt.plot(np.mean(DA_vac_square,axis = 0),'b', linewidth = 2, label='DA_square')
#plt.plot(np.mean(DA_true,axis = 0),'k', linewidth = 2, label='DA_true')
##plt.plot(np.mean(DA_true_bc,axis = 0),'y', linewidth = 2, label='DA_bc')
#plt.xlabel('time steps')
#plt.ylabel('infected')
#plt.legend()
#plt.savefig("figures/SIR_bc_60_MC_infect_by1_withfree.eps",fmt = '.eps')
#plt.show()
#      
#
#plt.plot(np.std(background_vac,axis = 0),'r',linewidth = 2, label='background')
#plt.plot(np.std(free_vac,axis = 0),'g', linewidth = 2, label='free')
#plt.plot(np.std(DA_vac_square,axis = 0),'b', linewidth = 2, label='DA_square')
##plt.plot(np.std(DA_true,axis = 0),'k', linewidth = 2, label='DA_true')
#plt.legend()
##plt.savefig("figures/SIR_bc_60_MC_std.eps",fmt = '.eps')
#plt.show()



#########################################################################

# final plot

error_level = 70

MC = 10
#
random_vac = []
for it_number in range(0,MC):
    #random_vac.append(np.load('data_final/random_vac001_per50_'+str(40)+'_SIR120_'+str(it_number)+'.npy'))
    random_vac.append(np.load('data_final/random_vac002_per100_'+str(50)+'_SIR60_'+str(it_number)+'.npy'))

random_vac =  np.array(random_vac)

free_vac = []
for it_number in range(0,MC):
    free_vac.append(np.load('data_final/free_per50_SIR120_'+str(it_number)+'.npy'))

free_vac =  np.array(free_vac)

background_vac = []
for it_number in range(0,MC):
    background_vac.append(np.load('data_final/hd_background_fix_vac002_per100_'+str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
background_vac =  np.array(background_vac)

DA_vac_square = []
for it_number in range(0,MC):
    DA_vac_square.append(np.load('data_final/hd_DA_square_fix_vac002_per100_'+str(error_level)+'_SIR60_'+str(it_number)+'.npy'))

DA_vac_square =  np.array(DA_vac_square)   
   

background_vac_bc = []
for it_number in range(0,MC):
    background_vac_bc.append(np.load('data_final/bc_background_fix_vac002_per100_'+str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
background_vac_bc =  np.array(background_vac_bc)

DA_vac_square_bc = []
for it_number in range(0,MC):
    DA_vac_square_bc.append(np.load('data_final/bc_DA_square_fix_vac002_per100_'+str(error_level)+'_SIR60_'+str(it_number)+'.npy'))

DA_vac_square_bc =  np.array(DA_vac_square_bc) 

#DA_vac = []
#for it_number in range(1,MC):
#    DA_vac.append(np.load('data_MC/number_infect_DA_hd_allper50by1_40pc_ytrue_p5_SIR100_'+str(it_number)+'.npy'))
#DA_vac =  np.array(DA_vac)   

#DA_true = []
#for it_number in range(1,MC):
#    DA_true.append(np.load('data_final/hd_true_vac001_per50_'+str(60)+'_SIR120_'+str(it_number)+'.npy'))
#DA_true =  np.array(DA_true)   
##
#DA_true_bc = []
#for it_number in range(1,MC):
#    DA_true_bc.append(np.load('data_MC_accord/hd_true_setp1_per50by1_'+str(error_level)+'pc_ytrue_p1_SIR100_'+str(it_number)+'.npy'))
#DA_true_bc =  np.array(DA_true_bc)   

#plt.plot(np.mean(random_vac,axis = 0),'y',linewidth = 2, label='random')
plt.plot(np.mean(background_vac,axis = 0),'r',linewidth = 2, label='back(hd)')
plt.plot(np.mean(background_vac_bc,axis = 0),'r--', linewidth = 2, label='back(bc)')
plt.fill_between( np.arange(0, 74, 1), np.mean(background_vac,axis = 0) - np.std(background_vac,axis = 0), 
                 np.mean(background_vac,axis = 0) + np.std(background_vac,axis = 0),alpha = 0.4,facecolor='r')
plt.plot(np.mean(random_vac,axis = 0),'y',linewidth = 2, label='random')
plt.fill_between( np.arange(0, 74, 1), np.mean(random_vac,axis = 0) - np.std(random_vac,axis = 0), 
                 np.mean(random_vac,axis = 0) + np.std(random_vac,axis = 0),alpha = 0.4,facecolor='y')
plt.plot(np.mean(free_vac,axis = 0),'g', linewidth = 2, label='free')
plt.fill_between( np.arange(0, 74, 1), np.mean(free_vac,axis = 0) - np.std(free_vac,axis = 0), 
                 np.mean(free_vac,axis = 0) + np.std(free_vac,axis = 0),alpha = 0.4,facecolor='g')
plt.plot(np.mean(DA_vac_square,axis = 0),'b', linewidth = 2, label='DA(hd)')
plt.plot(np.mean(DA_vac_square_bc,axis = 0),'b--', linewidth = 2, label='DA(bc)')
plt.fill_between( np.arange(0, 74, 1), np.mean(DA_vac_square,axis = 0) - np.std(DA_vac_square,axis = 0), 
                 np.mean(DA_vac_square,axis = 0) + np.std(DA_vac_square,axis = 0),alpha = 0.4)
#plt.plot(np.mean(DA_true,axis = 0),'k', linewidth = 2, label='DA_true')
#plt.plot(np.mean(DA_true_bc,axis = 0),'y', linewidth = 2, label='DA_bc')
plt.xlabel('time steps',fontsize=20)
plt.ylabel('infected',fontsize=20)
plt.legend()
#plt.savefig("figures/hd_fix_error"+str(error_level)+"_compare_by50_withfree.pdf",fmt = '.pdf')
plt.show()
      
#########################################################################################
#PNAS plot
import pylab as plot
params = {'legend.fontsize': 15,
          'legend.handlelength': 2}
plot.rcParams.update(params)

fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15, 4.5))

x = range(10)
y = range(10)

for count, row in enumerate(ax):
    for col in [row]:
        
        if count ==0:
            error_level = 50
            
        if count ==1:
            error_level = 60

        if count ==2:
            error_level = 70
            
        MC = 10
        random_vac = []
        for it_number in range(0,MC):
            #random_vac.append(np.load('data_final/random_vac001_per50_'+str(40)+'_SIR120_'+str(it_number)+'.npy'))
            random_vac.append(np.load('data_final/random_vac002_per100_'+str(50)+'_SIR60_'+str(it_number)+'.npy'))
        
        random_vac =  np.array(random_vac)
        
        free_vac = []
        for it_number in range(0,MC):
            free_vac.append(np.load('data_final/free_per50_SIR120_'+str(it_number)+'.npy'))
        
        free_vac =  np.array(free_vac)
        
        background_vac = []
        for it_number in range(0,MC):
            background_vac.append(np.load('data_final/hd_background_fix_vac002_per100_'+str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
        background_vac =  np.array(background_vac)
        
        DA_vac_square = []
        for it_number in range(0,MC):
            DA_vac_square.append(np.load('data_final/hd_DA_square_fix_vac002_per100_'+str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
        
        DA_vac_square =  np.array(DA_vac_square)   
           
        
        background_vac_bc = []
        for it_number in range(0,MC):
            background_vac_bc.append(np.load('data_final/bc_background_fix_vac002_per100_'+str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
        background_vac_bc =  np.array(background_vac_bc)
        
        DA_vac_square_bc = []
        for it_number in range(0,MC):
            DA_vac_square_bc.append(np.load('data_final/bc_DA_square_fix_vac002_per100_'+str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
        
        DA_vac_square_bc =  np.array(DA_vac_square_bc) 
            
        col.plot(np.mean(background_vac,axis = 0),'r',linewidth = 2, label='back(hd)')
        col.plot(np.mean(background_vac_bc,axis = 0),'r--', linewidth = 2, label='back(bc)')
        col.fill_between( np.arange(0, 74, 1), np.mean(background_vac,axis = 0) - np.std(background_vac,axis = 0), 
                         np.mean(background_vac,axis = 0) + np.std(background_vac,axis = 0),alpha = 0.4,facecolor='r')
        col.plot(np.mean(random_vac,axis = 0),'y',linewidth = 2, label='random')
        col.fill_between( np.arange(0, 74, 1), np.mean(random_vac,axis = 0) - np.std(random_vac,axis = 0), 
                         np.mean(random_vac,axis = 0) + np.std(random_vac,axis = 0),alpha = 0.4,facecolor='y')
        col.plot(np.mean(free_vac,axis = 0),'g', linewidth = 2, label='free')
        col.fill_between( np.arange(0, 74, 1), np.mean(free_vac,axis = 0) - np.std(free_vac,axis = 0), 
                         np.mean(free_vac,axis = 0) + np.std(free_vac,axis = 0),alpha = 0.4,facecolor='g')
        col.plot(np.mean(DA_vac_square,axis = 0),'b', linewidth = 2, label='DA(hd)')
        col.plot(np.mean(DA_vac_square_bc,axis = 0),'b--', linewidth = 2, label='DA(bc)')
        col.fill_between( np.arange(0, 74, 1), np.mean(DA_vac_square,axis = 0) - np.std(DA_vac_square,axis = 0), 
                         np.mean(DA_vac_square,axis = 0) + np.std(DA_vac_square,axis = 0),alpha = 0.4)
        #plt.plot(np.mean(DA_true,axis = 0),'k', linewidth = 2, label='DA_true')
        #plt.plot(np.mean(DA_true_bc,axis = 0),'y', linewidth = 2, label='DA_bc')
        #col.legend()
        #plt.savefig("figures/hd_fix_error"+str(error_level)+"_compare_by50_withfree.pdf",fmt = '.pdf')


fig.text(0.5, 0.04, 'time steps', ha='center',fontsize=20)
fig.text(0.04, 0.5, 'infected', va='center', rotation='vertical',fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(0.5, 1.2), ncol = 6)
plt.savefig("figures/school_infect.pdf",fmt = '.pdf')
plt.show()

################################################################################

print('DA',round(max(np.mean(DA_vac_square,axis = 0))/329.,2))

print('DA_bc',round(max(np.mean(DA_vac_square_bc,axis = 0))/329.,2))

print('background',round(max(np.mean(background_vac,axis = 0))/329.,2))

print('background_bc',round(max(np.mean(background_vac_bc,axis = 0))/329.,2))

print('free',round(max(np.mean(free_vac,axis = 0))/329.,2))

print('random',round(max(np.mean(random_vac,axis = 0))/329.,2))

#plt.plot(np.std(background_vac,axis = 0),'r',linewidth = 2, label='background')
#plt.plot(np.std(free_vac,axis = 0),'g', linewidth = 2, label='free')
#plt.plot(np.std(DA_vac_square,axis = 0),'b', linewidth = 2, label='DA')
#plt.plot(np.std(random_vac,axis = 0),'y', linewidth = 2, label='random')
##plt.plot(np.std(DA_true,axis = 0),'k', linewidth = 2, label='DA_true')
#plt.legend()
##plt.savefig("figures/SIR_bc_60_MC_std.eps",fmt = '.eps')
#plt.show()


##################################################################"



error_level = 60

MC = 10
#
background_vac = []
for it_number in range(1,MC):
    background_vac.append(np.load('data_multilayer/hd_background_vac003_'+str(error_level)+'_SIR120_'+str(it_number)+'.npy'))
    
    background_vac.append(np.load('data_multilayer/hd_backgrounde_025_vac002_'+str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
background_vac =  np.array(background_vac)

DA_vac_square = []
for it_number in range(0,MC):
    #DA_vac_square.append(np.load('data_multilayer/hd_DA_square_vac002_'+str(error_level)+'_SIR60_'+str(it_number)+'.npy'))

    DA_vac_square.append(np.load('data_multilayer/hd_DA_square_025_vac002_'+str(error_level)+'_SIR60_'+str(it_number)+'.npy'))

DA_vac_square =  np.array(DA_vac_square)   


DA_true = []
for it_number in range(0,MC):
    DA_true.append(np.load('data_multilayer/hd_DA_square_vac003_'+str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
    
    DA_true.append(np.load('data_multilayer/hd_true_045_'+str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
DA_true =  np.array(DA_true)



plt.plot(np.mean(background_vac,axis = 0),'r',linewidth = 2, label='back(hd)')
#plt.plot(np.mean(DA_true,axis = 0),'y', linewidth = 2, label='optimal')
plt.plot(np.mean(DA_vac_square,axis = 0),'g', linewidth = 2, label='DA')

plt.fill_between( np.arange(0, 90, 1), np.mean(background_vac,axis = 0) - np.std(background_vac,axis = 0), 
                 np.mean(background_vac,axis = 0) + np.std(background_vac,axis = 0),alpha = 0.4,facecolor='r')
#
plt.fill_between( np.arange(0, 90, 1), np.mean(DA_vac_square,axis = 0) - np.std(DA_vac_square,axis = 0), 
                 np.mean(DA_vac_square,axis = 0) + np.std(DA_vac_square,axis = 0),alpha = 0.4,facecolor='g')
#
#plt.fill_between( np.arange(0, 90, 1), np.mean(DA_true,axis = 0) - np.std(DA_true,axis = 0), 
#                  np.mean(DA_true,axis = 0) + np.std(DA_true,axis = 0),alpha = 0.4,facecolor='y')
#

plt.legend()
#plt.savefig("figures/SIR_multilayer_025.pdf",fmt = '.pdf')
plt.show()



###############################################################

true_rate_average = np.zeros((90,5))

DA_rate_average = np.zeros((90,5))

for it_number in range(1,MC):
    
    
    true_rate = np.load('data_multilayer/true_rate_025_025_001_0005_0005_'+str(it_number)+'.npy')
    
    l1 = true_rate.sum(axis=1)
    
    true_rate_norm = true_rate/l1.reshape(l1.size,1)
    
    true_rate_average += true_rate_norm/MC
    
    DA_rate = np.load('data_multilayer/DA_rate_025_025_001_0005_0005_'+str(it_number)+'.npy')
    
    l2 = DA_rate.sum(axis=1)
    
    DA_rate_norm = DA_rate/l2.reshape(l2.size,1)
    
    DA_rate_average += DA_rate_norm/MC
    
plt.plot(true_rate_average[:,0],label = 'true rate')

plt.plot(DA_rate_average[:,0],'r',label = 'DA rate')

plt.title('1st layer')
plt.legend()
plt.savefig("figures/rate_1_025.eps",fmt = '.eps')
plt.show()
plt.close()

plt.plot(true_rate_average[:,1],label = 'true rate')

plt.plot(DA_rate_average[:,1],'r',label = 'DA rate')

plt.title('2nd layer')
plt.legend()
plt.savefig("figures/rate_2_025.eps",fmt = '.eps')
plt.show()
plt.close()

plt.plot(true_rate_average[:,2],label = 'true rate')

plt.plot(DA_rate_average[:,2],'r',label = 'DA rate')

plt.title('3rd layer')
plt.legend()
plt.savefig("figures/rate_2_025.eps",fmt = '.eps')
plt.show()
plt.close()

plt.plot(true_rate_average[:,3],label = 'true rate')

plt.plot(DA_rate_average[:,3],'r',label = 'DA rate')

plt.title('4th layer')
plt.legend()
plt.savefig("figures/rate_4_025.eps",fmt = '.eps')
plt.show()
plt.close()

plt.plot(true_rate_average[:,4],label = 'true rate')

plt.plot(DA_rate_average[:,4],'r',label = 'DA rate')

plt.title('5th layer')
plt.legend()
plt.savefig("figures/rate_5_025.eps",fmt = '.eps')
plt.show()
plt.close()


################################################################

Ip1 = 0.025

Ip2 = 0.010

Ip3 = 0.010

Ip4 = 0.010

Ip5 = 0.010


error_level = 50

MC = 10

p_vaccinate = 0.02
#
background_vac = []
for it_number in range(0,MC):
    background_vac.append(np.load('data_multilayer/hd_background_vac003_'+str(error_level)+'_SIR120_'+str(it_number)+'.npy'))
    
    background_vac.append(np.load('data_multilayer/hd_background_'+str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
            '_'+str(Ip4)+'_'+str(Ip5)+'_'+ 'vac_' + str(p_vaccinate) + str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
background_vac =  np.array(background_vac)

DA_vac_square = []

for it_number in range(0,MC):
    #DA_vac_square.append(np.load('data_multilayer/hd_DA_square_vac002_'+str(error_level)+'_SIR60_'+str(it_number)+'.npy'))

    DA_vac_square.append(np.load('data_multilayer/hd_DA_'+str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
            '_'+str(Ip4)+'_'+str(Ip5)+'_'+ 'vac_' + str(p_vaccinate) + str(error_level)+'_SIR60_'+str(it_number)+'.npy'))

DA_vac_square =  np.array(DA_vac_square)   


DA_true = []
for it_number in range(0,MC):
#    DA_true.append(np.load('data_multilayer/hd_DA_square_vac003_'+str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
    
    DA_true.append(np.load('data_multilayer/hd_true_'+str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
            '_'+str(Ip4)+'_'+str(Ip5)+'_'+ 'vac_' + str(p_vaccinate) + str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
DA_true =  np.array(DA_true)


#######################################################################################

background_bc = []
for it_number in range(0,MC):
    
    background_bc.append(np.load('data_multilayer/bc_background_'+str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
            '_'+str(Ip4)+'_'+str(Ip5)+'_'+ 'vac_' + str(p_vaccinate) + str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
background_bc =  np.array(background_bc)

DA_bc = []

for it_number in range(0,MC):

    DA_bc.append(np.load('data_multilayer/bc_DA_'+str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
            '_'+str(Ip4)+'_'+str(Ip5)+'_'+ 'vac_' + str(p_vaccinate) + str(error_level)+'_SIR60_'+str(it_number)+'.npy'))

DA_bc =  np.array(DA_bc)   


DA_true_bc = []
for it_number in range(0,MC):
#    DA_true.append(np.load('data_multilayer/hd_DA_square_vac003_'+str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
    
    DA_true_bc.append(np.load('data_multilayer/bc_true_'+str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
            '_'+str(Ip4)+'_'+str(Ip5)+'_'+ 'vac_' + str(p_vaccinate) + str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
DA_true_bc =  np.array(DA_true_bc)

#######################################################################################

plt.plot(np.mean(background_vac,axis = 0),'r',linewidth = 2, label='back(hd)')
plt.plot(np.mean(DA_true,axis = 0),'y', linewidth = 2, label='optimal(hd)')
plt.plot(np.mean(DA_vac_square,axis = 0),'g', linewidth = 2, label='DA(hd)')

plt.plot(np.mean(background_bc,axis = 0),'r--',linewidth = 2, label='back(bc)')
plt.plot(np.mean(DA_true_bc,axis = 0),'y--', linewidth = 2, label='optimal(bc)')
plt.plot(np.mean(DA_bc,axis = 0),'g--', linewidth = 2, label='DA(bc)')

plt.fill_between( np.arange(0, 90, 1), np.mean(background_vac,axis = 0) - np.std(background_vac,axis = 0), 
                 np.mean(background_vac,axis = 0) + np.std(background_vac,axis = 0),alpha = 0.4,facecolor='r')
#
plt.fill_between( np.arange(0, 90, 1), np.mean(DA_vac_square,axis = 0) - np.std(DA_vac_square,axis = 0), 
                 np.mean(DA_vac_square,axis = 0) + np.std(DA_vac_square,axis = 0),alpha = 0.4,facecolor='g')
#
plt.fill_between( np.arange(0, 90, 1), np.mean(DA_true,axis = 0) - np.std(DA_true,axis = 0), 
                  np.mean(DA_true,axis = 0) + np.std(DA_true,axis = 0),alpha = 0.4,facecolor='y')


plt.legend()
#plt.savefig("figures/SIR_multilayer_" + str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
#            '_'+str(Ip4)+'_'+str(Ip5)+ ".pdf",fmt = '.pdf')
plt.show()

#########################################################################
print ('max background',np.max(np.mean(background_vac,axis = 0)))

print ('std background',np.mean(np.std(background_vac,axis = 0)))

print ('max DA',np.max(np.mean(DA_vac_square,axis = 0)))

print ('std DA',np.mean(np.std(DA_vac_square,axis = 0)))

print ('max true',np.max(np.mean(DA_true,axis = 0)))

print ('std true',np.mean(np.std(DA_true,axis = 0)))

print ('max background bc',np.max(np.mean(background_bc,axis = 0)))

print ('std background bc',np.mean(np.std(background_bc,axis = 0)))

print ('max DA',np.max(np.mean(DA_bc,axis = 0)))

print ('std DA bc',np.mean(np.std(DA_bc,axis = 0)))

print ('max true',np.max(np.mean(DA_true_bc,axis = 0)))

print ('std true bc',np.mean(np.std(DA_true_bc,axis = 0)))


print (round(np.max(np.mean(background_vac,axis = 0)),2), round(np.max(np.mean(DA_vac_square,axis = 0)),2),
       round(np.max(np.mean(DA_true,axis = 0)),2), round(np.mean(np.std(background_vac,axis = 0)),2),
       round(np.mean(np.std(DA_vac_square,axis = 0)),2), round(np.mean(np.std(DA_true,axis = 0)),2),
       round(np.max(np.mean(background_bc,axis = 0)),2), round(np.max(np.mean(DA_bc,axis = 0)),2),
       round(np.max(np.mean(DA_true_bc,axis = 0)),2), round(np.mean(np.std(background_bc,axis = 0)),2),
       round(np.mean(np.std(DA_bc,axis = 0)),2), round(np.mean(np.std(DA_true_bc,axis = 0)),2))

#########################################################################

MC = 1

true_rate_average = np.zeros((90,5))

DA_rate_average = np.zeros((90,5))

for it_number in range(0,MC):
    
    
    true_rate = np.load('data_multilayer/true_rate_'+str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
            '_'+str(Ip4)+'_'+str(Ip5)+'_'+ 'vac_'+str(it_number)+'.npy')
    
    l1 = true_rate.sum(axis=1)
    
    true_rate_norm = true_rate/l1.reshape(l1.size,1)
    
    true_rate_average += true_rate_norm/MC
    
    DA_rate = np.load('data_multilayer/DA_rate_'+str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
            '_'+str(Ip4)+'_'+str(Ip5)+'_'+ 'vac_'+str(it_number)+'.npy')
    
    l2 = DA_rate.sum(axis=1)
    
    DA_rate_norm = DA_rate/l2.reshape(l2.size,1)
    
    DA_rate_average += DA_rate_norm/MC
    
plt.plot(true_rate_average[:50,0],label = 'true rate')

plt.plot(DA_rate_average[:50,0],'r',label = 'DA rate')

plt.title('1st layer')
plt.legend()
plt.savefig("figures/rate_1"+ str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
            '_'+str(Ip4)+'_'+str(Ip5)+".eps",fmt = '.eps')
plt.show()
plt.close()

plt.plot(true_rate_average[:50,1],label = 'true rate')

plt.plot(DA_rate_average[:50,1],'r',label = 'DA rate')

plt.title('2nd layer')
plt.legend()
plt.savefig("figures/rate_2"+ str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
            '_'+str(Ip4)+'_'+str(Ip5)+".eps",fmt = '.eps')
plt.show()
plt.close()

plt.plot(true_rate_average[:50,2],label = 'true rate')

plt.plot(DA_rate_average[:50,2],'r',label = 'DA rate')

plt.title('3rd layer')
plt.legend()
plt.savefig("figures/rate_3"+ str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
            '_'+str(Ip4)+'_'+str(Ip5)+".eps",fmt = '.eps')
plt.show()
plt.close()

plt.plot(true_rate_average[:50,3],label = 'true rate')

plt.plot(DA_rate_average[:50,3],'r',label = 'DA rate')

plt.title('4th layer')
plt.legend()
plt.savefig("figures/rate_4"+ str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
            '_'+str(Ip4)+'_'+str(Ip5)+".eps",fmt = '.eps')
plt.show()
plt.close()

plt.plot(true_rate_average[:50,4],label = 'true rate')

plt.plot(DA_rate_average[:50,4],'r',label = 'DA rate')

plt.title('5th layer')
plt.legend()
plt.savefig("figures/rate_5"+ str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
            '_'+str(Ip4)+'_'+str(Ip5)+".eps",fmt = '.eps')
plt.show()
plt.close()



#########################################################################################
#PNAS plot
import pylab as plot
params = {'legend.fontsize': 15,
          'legend.handlelength': 2}
plot.rcParams.update(params)

fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(15, 9))

x = range(10)
y = range(10)

for count, row in enumerate(ax):
    for col in row:
        
        if count ==0:
            
            Ip1 = 0.025
        
            Ip2 = 0.010
            
            Ip3 = 0.010
            
            Ip4 = 0.010
            
            Ip5 = 0.010
            
        if count ==1:
            
            Ip1 = 0.035
        
            Ip2 = 0.015
            
            Ip3 = 0.010
            
            Ip4 = 0.005
            
            Ip5 = 0.005

        if count ==2:
            Ip1 = 0.025
        
            Ip2 = 0.025
            
            Ip3 = 0.025
            
            Ip4 = 0.005
            
            Ip5 = 0.005

        if count ==3:
            Ip1 = 0.045
        
            Ip2 = 0.015
            
            Ip3 = 0.010
            
            Ip4 = 0.005
            
            Ip5 = 0.005            
            
        if count ==4:
            Ip1 = 0.035
        
            Ip2 = 0.025
            
            Ip3 = 0.010
            
            Ip4 = 0.010
            
            Ip5 = 0.000  

        if count ==5:
            Ip1 = 0.020
        
            Ip2 = 0.020
            
            Ip3 = 0.015
            
            Ip4 = 0.010
            
            Ip5 = 0.010

        error_level = 50
        
        MC = 10
        
        p_vaccinate = 0.02
        #
        background_vac = []
        for it_number in range(0,MC):
            background_vac.append(np.load('data_multilayer/hd_background_vac003_'+str(error_level)+'_SIR120_'+str(it_number)+'.npy'))
            
            background_vac.append(np.load('data_multilayer/hd_background_'+str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
                    '_'+str(Ip4)+'_'+str(Ip5)+'_'+ 'vac_' + str(p_vaccinate) + str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
        background_vac =  np.array(background_vac)
        
        DA_vac_square = []
        
        for it_number in range(0,MC):
            #DA_vac_square.append(np.load('data_multilayer/hd_DA_square_vac002_'+str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
        
            DA_vac_square.append(np.load('data_multilayer/hd_DA_'+str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
                    '_'+str(Ip4)+'_'+str(Ip5)+'_'+ 'vac_' + str(p_vaccinate) + str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
        
        DA_vac_square =  np.array(DA_vac_square)   
        
        
        DA_true = []
        for it_number in range(0,MC):
        #    DA_true.append(np.load('data_multilayer/hd_DA_square_vac003_'+str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
            
            DA_true.append(np.load('data_multilayer/hd_true_'+str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
                    '_'+str(Ip4)+'_'+str(Ip5)+'_'+ 'vac_' + str(p_vaccinate) + str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
        DA_true =  np.array(DA_true)
        
        
        #######################################################################################
        
        background_bc = []
        for it_number in range(0,MC):
            
            background_bc.append(np.load('data_multilayer/bc_background_'+str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
                    '_'+str(Ip4)+'_'+str(Ip5)+'_'+ 'vac_' + str(p_vaccinate) + str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
        background_bc =  np.array(background_bc)
        
        DA_bc = []
        
        for it_number in range(0,MC):
        
            DA_bc.append(np.load('data_multilayer/bc_DA_'+str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
                    '_'+str(Ip4)+'_'+str(Ip5)+'_'+ 'vac_' + str(p_vaccinate) + str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
        
        DA_bc =  np.array(DA_bc)   
        
        
        DA_true_bc = []
        for it_number in range(0,MC):
        #    DA_true.append(np.load('data_multilayer/hd_DA_square_vac003_'+str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
            
            DA_true_bc.append(np.load('data_multilayer/bc_true_'+str(Ip1)+'_'+str(Ip2)+'_'+str(Ip3)+
                    '_'+str(Ip4)+'_'+str(Ip5)+'_'+ 'vac_' + str(p_vaccinate) + str(error_level)+'_SIR60_'+str(it_number)+'.npy'))
        DA_true_bc =  np.array(DA_true_bc)
        
        #######################################################################################
        
        col.plot(np.mean(background_vac,axis = 0),'r',linewidth = 2, label='back(hd)')
        col.plot(np.mean(DA_true,axis = 0),'y', linewidth = 2, label='optimal(hd)')
        col.plot(np.mean(DA_vac_square,axis = 0),'g', linewidth = 2, label='DA(hd)')
        
        col.plot(np.mean(background_bc,axis = 0),'r--',linewidth = 2, label='back(bc)')
        col.plot(np.mean(DA_true_bc,axis = 0),'y--', linewidth = 2, label='optimal(bc)')
        col.plot(np.mean(DA_bc,axis = 0),'g--', linewidth = 2, label='DA(bc)')
        
        col.fill_between( np.arange(0, 90, 1), np.mean(background_vac,axis = 0) - np.std(background_vac,axis = 0), 
                         np.mean(background_vac,axis = 0) + np.std(background_vac,axis = 0),alpha = 0.4,facecolor='r')
        #
        col.fill_between( np.arange(0, 90, 1), np.mean(DA_vac_square,axis = 0) - np.std(DA_vac_square,axis = 0), 
                         np.mean(DA_vac_square,axis = 0) + np.std(DA_vac_square,axis = 0),alpha = 0.4,facecolor='g')
        #
        col.fill_between( np.arange(0, 90, 1), np.mean(DA_true,axis = 0) - np.std(DA_true,axis = 0), 
                          np.mean(DA_true,axis = 0) + np.std(DA_true,axis = 0),alpha = 0.4,facecolor='y')
           #plt.savefig("figures/hd_fix_error"+str(error_level)+"_compare_by50_withfree.pdf",fmt = '.pdf')
        

fig.text(0.5, 0.04, 'time steps', ha='center',fontsize=20)
fig.text(0.04, 0.5, 'infected', va='center', rotation='vertical',fontsize=20)
plt.legend(loc='upper right', bbox_to_anchor=(0.5, 2.4), ncol = 6)
plt.savefig("figures/multilayer_infect.pdf",fmt = '.pdf')
plt.show()

################################################################################
