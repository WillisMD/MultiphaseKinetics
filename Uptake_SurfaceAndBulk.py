#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uptake coefficient calculations, according to expressions derived in 
    Wilson, Prophet & Willis, A Kinetic Model for Predicting Trace Gas Uptake and Reaction JPCA, 2022
Uses a range of input values for droplet size and concentration to produce Figure 10 in
     Wilson, Prophet & Willis, JPCA, 2022 for trans-aconitic acid (AA)
     
@author: meganwillis
"""

####################
import numpy as np
import matplotlib.pylab as plt
####################

############# ---- FUNCTION DEFINITIONS

def calc_gamma(radius, ConcY0):
    ############# ---- surface volume fraction
    Vsurf_frac=(radius**3-(radius-delta)**3)/(radius**3)
    ############# ---- calculate k_transport
    Rweight = radius/(radius + rcrit_sb + rcrit_gs)
    k_diffusion = (2*D_X)/((radius/3)**2)
    k_trans_sb = ((1/(k_diffusion*Rweight)+1/(k_solv_X*(1-Rweight))))**-1
    k_trans_gs = ((1/(k_diffusion*Rweight)+1/(k_des_X*(1-Rweight))))**-1
    k_transport = (k_trans_sb+k_trans_gs)/2
    ############# ---- calculate gamma
    gamma_first = (4*radius*ConcY0)/(3*c_X) #first term in gamma expression
    gamma_surf = Vsurf_frac*gamma_first*((k_srxn*Hcc_gs*Y_maxsurf*Keq_Y)/(1+Keq_Y*ConcY0)) #uptake coefficient for surface reactions only, Equation 30a in Wilson, Prophet & Willis, JPCA, 2022
    gamma_bulk = gamma_first*((k_brxn*k_transport*Hcc_gb)/((k_brxn*ConcY0)+k_transport)) #uptake coefficient for bulk reactions only, Equation 18 in Wilson, Prophet & Willis, JPCA, 2022
    gamma_total = gamma_surf + gamma_bulk #uptake coefficient for both surface & bulk reactions, Equation 30 in Wilson, Prophet & Willis, JPCA, 2022
    
    return [gamma_surf, gamma_bulk, gamma_total]
    

############# ---- constants
N_A = 6.022e23 #molecules/mole
delta = 1e-7 #interface depth 1nm, given in cm for calculations, see Willis & Wilson, JPCA, 2022

############# ---- define variables for X = O_3
rcrit_sb = 0.263*(1e-4) #0.263um, conversion to cm (from Wilson, Prophet & Willis, JPCA, 2022)
rcrit_gs = 0.0763*(1e-4) #0.0763um, conversion to cm (from Wilson, Prophet & Willis, JPCA, 2022)
k_solv_X = 4.58e5 #s^-1 (from Willis & Wilson, JPCA, 2022)
k_des_X = 5.44e6#s^-1 (from Willis & Wilson, JPCA, 2022)
Hcc_gb = 0.27 #unitless Henry's Law Constant for O3 in water (Sander, ACP, 2015)
Hcc_gs = 8.9 #unitless gas-surface Henry's Law Constant for O3 (derived in Willis & Wilson, JPCA, 2022)
c_X = 36000 #cm s^-1, mean molecular speed of O3 (exptl temperature of Willis & Wilson, JPCA, 2022)
D_X = 1.76e-5 # cm^2 s^-1, bulk diffusion coefficient for O3 in water

############# ---- define variables for Y = aconitic acid (AA, according to experiment in Willis & Wilson, JPCA, 2022)
k_brxn = 1.40e-17 #cm^3 molec^-1 s^-1, based on fumaric acid, see Willis & Wilson, JPCA, 2022
k_srxn = k_brxn #cm^3 molec^-1 s^-1, in the absence of information on the surface reaction rate coefficient, assume it equals the bulk rate coefficient
k_solv_Y = 90 #s^-1, see Willis & Wilson, JPCA, 2022
k_desolv_Y = 1.20e-20 #cm^3 molec^-1 s^-1, see Willis & Wilson, JPCA, 2022
Keq_Y = k_desolv_Y/k_solv_Y #Langmuir surface partitioning equilibrium constant
Y_maxsurf = 1.50e21 # molec cm^-3, maximum surface concentration, based on estimated molecular area for Y and interface depth 1nm
ConcY0_expt = 1.90e21 #molecules/cm3, estimated with AIOMFAC for exptl relative humidity
radius_expt = 9.1*(1e-4) #9.1um radius, conversion to cm

############# ---- define range inputs for radius and droplet concentration
radius_um_rng = np.geomspace(0.01, 100, num = 100, endpoint=True) #range of droplet radii in micrometers
radius_cm_rng = radius_um_rng*(1e-4) #convert to cm for calculations


ConcY0_M_rng = np.geomspace(1e-6, 10, num = 100, endpoint=True) #range of droplet molar (M) concentrations
ConcY0_rng = ConcY0_M_rng*N_A*(1/1000) #convert initial droplet concentration to molec cm^-3 for calculations


############# ---- calculate uptake coefficients over a range of sizes, at [AA] from experiment
result_radius = calc_gamma(radius_cm_rng, ConcY0_expt)

############# ---- calculate uptake coefficients over a range of initial AA concentration, at experimental radius
result_concY = calc_gamma(radius_expt, ConcY0_rng)

############# ---- plot uptake coefficient as a function of droplet concentration, at fixed (experimental) radius
width = 15/2.54
height = width
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width,height))
ax.loglog(ConcY0_M_rng, result_concY[2], color = "black", label = "Surface + Bulk") #plot bulk and surface reaction components
ax.loglog(ConcY0_M_rng, result_concY[1], color = "lightblue", linestyle = "dashed", label = "Bulk") #plot bulk reaction component
ax.loglog(ConcY0_M_rng, result_concY[0], color = 'darkred', linestyle = "dashed", label = "Surface") #plot surface reaction component
ax.set_xlabel("Concentration (M)")
ax.set_ylabel("Uptake Coefficient")
ax.legend()
plt.tight_layout()
plt.show()
plt.clf()

############# ---- plot uptake coefficient as a function of droplet radius, at fixed (experimental) droplet concentration
width = 15/2.54
height = width
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width,height))
ax.loglog(radius_um_rng, result_radius[2], color = "black", label = "Surface + Bulk") #plot bulk and surface reaction components
ax.loglog(radius_um_rng, result_radius[1], color = "lightblue", linestyle = "dashed", label = "Bulk") #plot bulk reaction component
ax.loglog(radius_um_rng, result_radius[0], color = 'darkred', linestyle = "dashed", label = "Surface") #plot surface reaction component
ax.set_xlabel("Radius (um)")
ax.set_ylabel("Uptake Coefficient")
ax.legend()
plt.tight_layout()
plt.show()
plt.clf()
