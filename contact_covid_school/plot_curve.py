# -*- coding: utf-8 -*-

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


#########################################################################

# plot average and std all according

error_level = 40

MC = 10

free_vac = []
for it_number in range(1,MC):
    free_vac.append(np.load('data_MC_accord/hd_free_per500by1_'+str(error_level)+'pc_ytrue_p5_SIR100_'+str(it_number)+'.npy'))

free_vac =  np.array(free_vac)

background_vac = []
for it_number in range(1,MC):
    background_vac.append(np.load('data_MC_accord/hd_background_per500by1_'+str(error_level)+'pc_ytrue_p5_SIR100_'+str(it_number)+'.npy'))
background_vac =  np.array(background_vac)

DA_vac_square = []
for it_number in range(1,MC):
    DA_vac_square.append(np.load('data_MC_accord/hd_DA_square_per500by1_'+str(error_level)+'pc_ytrue_p5_SIR100_'+str(it_number)+'.npy'))

DA_vac_square =  np.array(DA_vac_square)   
   
#DA_vac = []
#for it_number in range(1,MC):
#    DA_vac.append(np.load('data_MC/number_infect_DA_hd_allper50by1_40pc_ytrue_p5_SIR100_'+str(it_number)+'.npy'))
#DA_vac =  np.array(DA_vac)   

DA_true = []
for it_number in range(1,MC):
    DA_true.append(np.load('data_MC_accord/hd_true_per500by1_'+str(error_level)+'pc_ytrue_p5_SIR100_'+str(it_number)+'.npy'))
DA_true =  np.array(DA_true)   

DA_true_bc = []
for it_number in range(1,MC):
    DA_true_bc.append(np.load('data_MC_accord/bc_true_per500by1_'+str(error_level)+'pc_ytrue_p5_SIR100_'+str(it_number)+'.npy'))
DA_true_bc =  np.array(DA_true_bc)   

plt.plot(np.mean(background_vac,axis = 0),'r',linewidth = 2, label='background')
plt.plot(np.mean(free_vac,axis = 0),'g', linewidth = 2, label='free')
plt.plot(np.mean(DA_vac_square,axis = 0),'b', linewidth = 2, label='DA_square')
plt.plot(np.mean(DA_true,axis = 0),'k', linewidth = 2, label='DA_true')
plt.plot(np.mean(DA_true_bc,axis = 0),'y', linewidth = 2, label='DA_bc')
plt.legend()
#plt.savefig("figures/SIR_40_MC.eps",fmt = '.eps')
plt.show()
      

plt.plot(np.std(background_vac,axis = 0),'r',linewidth = 2, label='background')
plt.plot(np.std(free_vac,axis = 0),'g', linewidth = 2, label='free')
plt.plot(np.std(DA_vac_square,axis = 0),'b', linewidth = 2, label='DA_square')
plt.plot(np.std(DA_true,axis = 0),'k', linewidth = 2, label='DA_true')
plt.legend()
#plt.savefig("figures/SIR_40_MC.eps",fmt = '.eps')
plt.show()

