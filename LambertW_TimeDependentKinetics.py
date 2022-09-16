#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time dependent expressions for multiphase kinetics, using the Lambert W function, according to expressions derived in 
    Wilson, Prophet & Willis, A Kinetic Model for Predicting Trace Gas Uptake and Reaction JPCA, 2022
Uses single input values for droplet concentration and droplet size, corresponding to experimental conditions 
    (Corresponding to expt #1 in Willis & Wilson, Coupled Interfacial and Bulk Kinetics Govern the Timescales of Multiphase Ozonolysis Reactions, https://doi.org/10.1021/acs.jpca.2c03059, JPCA, 2022)

@author: meganwillis
"""

####################
import numpy as np
import matplotlib.pylab as plt
from scipy.special import lambertw
####################

############# ---- constants
N_A = 6.022e23 #molecules/mole
T = 297 # experimental temperature in Kelvin
P = 1 # experimental pressure in atmospheres
R_LatmmolK = 0.08206 # gas constant, L atm mol^-1 K^-1
R_JmolK = 8.314# gas constant, J mol^-1 K^-1
delta = 1e-7 #interface depth 1nm, given in cm for calculations, see Willis & Wilson, JPCA, 2022


############# ---- define variables for X_g = O_3
rcrit_sb = 0.263*(1e-4) #0.263um, conversion to cm (from Wilson, Prophet & Willis, JPCA, 2022)
rcrit_gs = 0.0763*(1e-4) #0.0763um, conversion to cm (from Wilson, Prophet & Willis, JPCA, 2022)
k_solv_X = 4.58e5 #s^-1 (from Willis & Wilson, JPCA, 2022)
k_des_X = 5.44e6#s^-1 (from Willis & Wilson, JPCA, 2022)
Hcc_gb = 0.27 #unitless Henry's Law Constant for O3 in water (Sander, ACP, 2015)
Hcc_gs = 8.9 #unitless gas-surface Henry's Law Constant for O3 (derived in Willis & Wilson, JPCA, 2022)
MW = 16*3 #g/mol
c_X = (np.sqrt(8*R_JmolK*T/(np.pi*MW/1000)))*100 #cm s^-1, mean molecular speed of O3 (exptl temperature of Willis & Wilson, JPCA, 2022)
D_X = 1.76e-5 # cm^2 s^-1, bulk diffusion coefficient for O3 in water
ConcX_g = (58.4*(1e-6))*((P*(1e-3)*N_A)/(R_LatmmolK*T)) # molecules cm^-3, Expt #1 from Willis & Wilson, JPCA, 2022 average O3 mixing ratio was 58.4 ppmv

############# ---- define variables for Y = aconitic acid (AA, according to experiment in Willis & Wilson, JPCA, 2022)
radius = 9.2*(1e-4) #9.1um radius, conversion to cm
Vsurf_frac=(radius**3-(radius-delta)**3)/(radius**3) #fractional volume contribution from the interfacial volume
k_brxn = 1.40e-17 #cm^3 molec^-1 s^-1, based on fumaric acid, see Willis & Wilson, JPCA, 2022
k_srxn = k_brxn #cm^3 molec^-1 s^-1, in the absence of information on the surface reaction rate coefficient, assume it equals the bulk rate coefficient
ConcY_0 = 1.90e21 #molecules/cm3, estimated with AIOMFAC for exptl relative humidity
k_solv_Y = 90 #s^-1, see Willis & Wilson, JPCA, 2022
k_desolv_Y = 1.20e-20 #cm^3 molec^-1 s^-1, see Willis & Wilson, JPCA, 2022
Keq_Y = k_desolv_Y/k_solv_Y #Langmuir surface partitioning equilibrium constant
Y_maxsurf = 1.50e21 # molec cm^-3 (Gamma_max (molec cm^-2)/delta (cm)), maximum surface concentration from Y molecular area  and interface depth

############# ---- calculate k_transport
Rweight = radius/(radius + rcrit_sb + rcrit_gs)
k_diffusion = (2*D_X)/((radius/3)**2)
k_trans_sb = ((1/(k_diffusion*Rweight)+1/(k_solv_X*(1-Rweight))))**-1
k_trans_gs = ((1/(k_diffusion*Rweight)+1/(k_des_X*(1-Rweight))))**-1
k_transport = (k_trans_sb+k_trans_gs)/2

############# ---- create time array
rxn_time = np.linspace(0, 16000, num = 1600, endpoint=True) #reaction time in seconds


############# ---- Kinetic expression for [Y(b)]t that includes only bulk reactions (Eq. 22a in Wilson, Prophet & Willis, JPCA, 2022)
exp_term_b = (k_brxn*ConcY_0/k_transport)-(k_brxn*Hcc_gb*ConcX_g*rxn_time)
ConcYb_brxn = (k_transport/k_brxn)*lambertw((k_brxn*ConcY_0/k_transport)*np.exp(exp_term_b))
#plt.plot(rxn_time, ConcYb_brxn)

############# ---- Kinetic expression for [X(b)]t that includes only bulk reactions (Eq. 23a in Wilson, Prophet & Willis, JPCA, 2022)
ConcXb_brxn = (k_transport*Hcc_gb*ConcX_g)/(k_brxn*ConcYb_brxn+k_transport)
#plt.semilogy(rxn_time, ConcXb_brxn)

############# ---- Kinetic expression for [Y(b)]t that includes only surface reactions (Eq. 35 in Wilson, Prophet & Willis, JPCA, 2022)
exp_term_b = ((Keq_Y*ConcY_0)-k_srxn*Hcc_gs*ConcX_g*Y_maxsurf*Keq_Y*Vsurf_frac*rxn_time)
ConcYb_srxn = (1/Keq_Y)*lambertw((Keq_Y*ConcY_0)*np.exp(exp_term_b))
#plt.plot(rxn_time, ConcYb_srxn)

############# ---- Kinetic expression for [Y(b)]t that includes both surface and bulk reactions (Eq. 36 in Wilson, Prophet & Willis, JPCA, 2022)
exp_term_sb = ((Keq_Y*ConcYb_brxn)-k_srxn*Hcc_gs*ConcX_g*Y_maxsurf*Keq_Y*Vsurf_frac*rxn_time)
ConcYb_sbrxn = (1/Keq_Y)*lambertw((Keq_Y*ConcYb_brxn)*np.exp(exp_term_sb))

############# ---- plot multiphase time-dependent kinetics
width = 12/2.54
height = width*0.8
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width,height))
ax.plot(rxn_time/60, ConcYb_brxn/ConcYb_brxn[0], color = "mediumblue", linestyle = "dashed", label = "Bulk") #plot bulk reaction component
ax.plot(rxn_time/60, ConcYb_srxn/ConcYb_srxn[0], color = 'darkred', linestyle = "dashed", label = "Surface") #plot surface reaction component
ax.plot(rxn_time/60, ConcYb_sbrxn/ConcYb_sbrxn[0], color = "black", label = "Surface + Bulk") #plot bulk and surface reaction components
ax.set_xlabel("Reaction Time (min)")
ax.set_ylabel("[Y]$_{t}$/[Y]$_{0}$")
ax.legend()
plt.tight_layout()
plt.show()



