"""
This file includes changes made to the original script generating a local Miller equilibrium
to include squareness (zeta).
"""

#!/usr/bin/env python3
"""
The purpose of this script is to generate a local Miller equilibrium and compare various parameters of interest with 
eiktest(old routine on GS2) for the same equilibrium. In some ways, this script is the pythonized version of eiktest.
The derivatives are calculated usign a central-difference method. The integrals are performed using a trapezoidal sum.

Author: Rahul Gaur (rgaur@terpmail.umd.edu)
Created: Jan 2022
"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as linspl
from scipy.interpolate import CubicSpline as cubspl
from scipy.integrate import cumulative_trapezoid as ctrap
from utils import *
import os
import sys

# Add toml directory to module search path
parent_dir = os.path.abspath(os.path.dirname(__file__))
toml_dir = os.path.join(parent_dir, 'toml')

sys.path.append(toml_dir)
import toml

# definitions of the following variables can be obtained from 
# https://gyrokinetics.gitlab.io/gs2/page/namelists/#theta_grid_eik_knobs

# Further information can be found from Miller's paper and the references 
# provided  in the repo

# Which theta grid do you want? If all of the options below are 0
# the default theta will be a geometric one. Recommended: staight, eqarc or collocation
# EQARC IS EQUISPACED, straight and collocation are not.
want_eqarc = 1
want_straight = 0
want_collocation = 0


# choose this factor(>0) to control the number of lambda points in the GS2 grid.out file
# increasing fac decreases the number of lambda points
fac = 0.5

#If you want to see the lambda grid set lambda_knob = 1
lambda_knob = 1

file_idx = 42 #random number to identify your output file

# read parameters from input file
input_file = sys.argv[1]
if len(sys.argv) > 2:
    stem = input_file[:-3]
    eikfile = sys.argv[2]
    eiknc = eikfile[-8:] + ".eiknc.nc"
else:
    stem = input_file[:-3]
    eikfile = stem + ".eik.out"
    eiknc = stem + ".eiknc.nc"

f = toml.load(input_file)

# note: this script assumes irho=2, so rhoc = r/a
ntgrid = int(f['Dimensions']['ntheta']/2+1)
nperiod = f['Dimensions']['nperiod']
rhoc = f['Geometry']['rhoc']
qinp = f['Geometry']['qinp']
s_hat_input = f['Geometry']['shat']
if s_hat_input == 0.0:
   s_hat_input = 1.e-6
try:
   zero_shat_threshold = f['Domain']['zero_shat_threshold']
   if abs(s_hat_input) < zero_shat_threshold:
      s_hat_input = 1.e-6
except:
   pass
Rmaj =  f['Geometry']['Rmaj']
R_geo = f['Geometry']['R_geo']
shift = f['Geometry']['shift']
akappa = f['Geometry']['akappa']
tri = f['Geometry']['tri']
zeta = f['Geometry']['zeta']
akappri = f['Geometry']['akappri']
tripri = f['Geometry']['tripri']
zetapri = f['Geometry']['zetapri']
beta_prime_input = f['Geometry']['betaprim']

# YOU MUST set Bunit = 0 if you have the R_geo value else 
# the Bunit calc will replace R_geo
Bunit = 0

# The results may be sensitive to delrho! Choose carefully.
delrho = 0.001

print('For a valid calculation all the errors you see < 1E-2\n')

#######################################################################################################################
########################------------ primary lowest level calculations---------------------############################
#######################################################################################################################
# no need to change this
no_of_surfs = 3

# note that this theta is neither geometric nor flux. It's just used to generate the surfaces. We will call it colocation theta
# GS2 uses this theta for the grad par calculation
theta = np.linspace(0, np.pi, ntgrid)

# position of the magnetic axes of the three surfaces
R_0 = np.array([Rmaj+np.abs(shift)*delrho, Rmaj, Rmaj-np.abs(shift)*delrho])

rho = np.array([rhoc - delrho, rhoc, rhoc + delrho])

qfac = np.array([qinp-s_hat_input*(qinp/rhoc)*delrho, qinp, qinp+s_hat_input*(qinp/rhoc)*delrho]) 
kappa = np.array([akappa-akappri*delrho, akappa, akappa+akappri*delrho])
delta = np.array([tri-tripri*delrho, tri, tri+tripri*delrho])
zetas = np.array([zeta-zetapri*delrho, zeta, zeta+zetapri*delrho])

#R_mag_ax can be anything as long as it's inside the annulus. 
R_mag_ax = Rmaj

dpdrho = beta_prime_input/2  #This definiton with a factor of 2 has been taken directly from geometry.f90.

R= np.array([R_0[i] + (rho[i])*np.cos(theta +np.arcsin(delta[i])*np.sin(theta)) for i in range(no_of_surfs)])
Z = np.array([kappa[i]*(rho[i])*np.sin(theta + zetas[i]*np.sin(2*theta)) for i in range(no_of_surfs)])

R0 = R.copy()
Z0 = Z.copy()

## Quick intersection check. If shift is large, surfaces can intersect
# To check if the equilibrium sufaces intersect with each other
if intersection_chk(R, Z, R_mag_ax) != 0:
	print("WARNING! Surfaces intersect...\n")
else:
	print("curve intersection check passed... surfaces do not intersect\n")

# theta array with a common magnetic axis
theta_comn_mag_ax = np.array([np.arctan2(Z[i], R[i]-R_mag_ax) for i in range(no_of_surfs)])
        
dRj      = np.zeros((no_of_surfs, ntgrid))
dZj      = np.zeros((no_of_surfs, ntgrid))
L        = np.zeros((no_of_surfs, ntgrid))
L_st     = np.zeros((no_of_surfs, ntgrid))
dt       = np.zeros((no_of_surfs, ntgrid))
dBr_ML   = np.zeros((no_of_surfs, ntgrid))
theta_st = np.zeros((no_of_surfs, ntgrid))
phi_n    = np.zeros((no_of_surfs, ntgrid))
u_ML     = np.zeros((no_of_surfs, ntgrid))


###################################################################################################################
######################------------------GRADIENTS ON GEOMETRIC THETA GRID------------------########################
###################################################################################################################
#TODO: Replace all instances of derm with np.gradient
dl = np.sqrt(derm(R,'l','e')**2 + derm(Z,'l','o')**2)
for i in range(no_of_surfs):
    L[i, 1:] = np.cumsum(np.sqrt(np.diff(R[i])**2 + np.diff(Z[i])**2))

dt = derm(theta_comn_mag_ax, 'l', 'o')
rho_diff = derm(rho, 'r')
# partial derivatives of R and Z on the exact rho and theta_geometric grid
dR_drho  = derm(R, 'r')/rho_diff

# should be second order accurate
dR_dt   = dermv(R, theta_comn_mag_ax, 'l', 'e')
dZ_drho = derm(Z, 'r')/rho_diff

# should be second order accurate
dZ_dt = dermv(Z, theta_comn_mag_ax, 'l', 'o')

jac   = dR_drho*dZ_dt - dZ_drho*dR_dt

# partial derivatives of psi and theta_geometric on the cartesian grid
drhodR = dZ_dt/jac
drhodZ = -dR_dt/jac
dt_dR  = -dZ_drho/jac
dt_dZ  =  dR_drho/jac

test_diff = (dt_dR[1]*drhodZ[1] - dt_dZ[1]*drhodR[1])/np.sqrt(drhodR[1]**2 + drhodZ[1]**2) \
            + 1/dermv(L, theta_comn_mag_ax, 'l', 'o')[1]

if np.max(np.abs(test_diff)) > 3E-5:
	print("grad theta_geo along l don't match...error = %.4E \n"%(np.max(np.abs(test_diff))))
else:
	print("grad theta_geo along the surface test passed...\n")

if Bunit != 0:
    grho0 = np.sqrt(drhodR**2 + drhodZ**2)
    R_geo = 1/ctrap(1/(R[1]*grho0[1]), L[1], initial=0)[-1]*rhoc # R_geo = F/(a*Bunit).Note the missing a_N goes into grho0 since its already normalized

# determining dpsidrho from the safety factor relation
#dpsidrho_arr = -R_geo/np.abs(2*np.pi*qfac/(2*ctrap(jac/R, theta_comn_mag_ax)[:, -1]))
dpsidrho_arr = -(R_geo/np.abs(2*np.pi*qfac))*np.abs(2*ctrap(jac/R, theta_comn_mag_ax)[:, -1])
dpsidrho = dpsidrho_arr[1]

#Normalized F is R_geo
F = np.ones((3,))*R_geo
drhodpsi = 1/dpsidrho

dpdpsi = dpdrho*drhodpsi

psi = np.array([1-delrho/drhodpsi, 1, 1+delrho/drhodpsi])
psi_diff = derm(psi, 'r')

# partial theta/partial rho (radial component of grad theta)
dtdr_geo = np.sign(psi_diff)*(dt_dR*drhodR + dt_dZ*drhodZ)/np.sqrt(drhodR**2 + drhodZ**2)

B_p = np.abs(dpsidrho)*np.array([np.sqrt(drhodR[i]**2 + drhodZ[i]**2)/R[i] for i in range(no_of_surfs)])
B_t = np.array([np.reshape(F, (-1,1))[i]/R[i] for i in range(no_of_surfs)])
B2 = np.array([B_p[i]**2 + B_t[i]**2 for i in range(no_of_surfs)])
B = np.sqrt(B2)

# grad psi from the cartesian grid 
grad_psi_cart = dpsidrho*np.sqrt(drhodR**2 + drhodZ**2)

# gradpar_0 is b.grad(theta) where theta = collocation theta
# should be second order accurate
gradpar_0 = 1/(R*B)*np.array([np.abs(dpsidrho_arr[i])*np.sqrt(drhodR[i]**2 + drhodZ[i]**2)\
            for i in range(no_of_surfs)])*(1/dermv(L, np.ones((no_of_surfs, ntgrid))*theta, 'l', 'o')) 


# To reiterate, this theta is neither the geometric nor flux theta
# This calculation of gradpar_0 is only meaningful on the central surface as theta = collocation theta is only known as a
# function of geometric theta on the central surface.
#####################################################################################################################
#######################------------------GRADIENTS ON FLUX THETA GRID------------------------########################
#####################################################################################################################

# Calculating theta_f or theta_st from the cartesian derivatives.
# Note that this theta_st is only meaningful for the central surface. 
# This happens because we only know the exact value of F on the central surface.

for i in range(no_of_surfs):
	 theta_st[i, 1:] = ctrap(np.abs(np.reshape(F,(-1,1))[i]*(1/dpsidrho_arr[i])*jac[i]/R[i]), theta_comn_mag_ax[i])
	 theta_st[i, 1:] = theta_st[i, 1:]/theta_st[i, -1]
	 theta_st[i, 1:] = np.pi*theta_st[i, 1:]


# splining here before we interpolate variables onto a uniform theta_st 
#spline object b/w flux theta and collocation theta
spl1 = linspl(theta_st[1], theta)
#spline object b/w geometric theta and flux theta
th_geo_st_spl = linspl(theta_comn_mag_ax[1], theta_st[1], k = 1)

#Before we take gradients on the theta_st grid we interpolate all the important variables on to a uniform theta_st grid.
# Not done in this code since it increases the F_chk error significantly
theta_st_new = np.linspace(0, np.pi, ntgrid)*np.reshape(np.ones((no_of_surfs,)),(-1,1))
theta_st_new = theta_st
theta_comn_mag_ax_new = np.zeros((no_of_surfs, ntgrid))
B1 = np.zeros((1, ntgrid))
B1 = B[1].copy()

# gradpar1 is b.grad(theta_st) where we use straight field line theta
gradpar1 = 1/(B1)*(B_p[1])*(derm(theta_st[1], 'l', 'o')/dl[1])

for i in range(no_of_surfs):
        R[i] = np.interp(theta_st_new[i], theta_st[i], R[i])
        Z[i] = np.interp(theta_st_new[i], theta_st[i], Z[i])
        B[i] = np.interp(theta_st_new[i], theta_st[i], B[i])
        B_p[i] = np.interp(theta_st_new[i], theta_st[i], B_p[i])
        gradpar_0[i] = np.interp(theta_st_new[i], theta_st[i], gradpar_0[i])
        dtdr_geo[i] = np.interp(theta_st_new[i], theta_st[i], dtdr_geo[i])
        #dtdr_st[i] = np.interp(theta_st_new[i], theta_st[i], dtdr_st[i])
        theta_comn_mag_ax_new[i] = np.arctan2(Z[i], R[i]-R_mag_ax)


# partial derivatives of R and Z on the exact psi and theta_f grid
# We don't have to use dermv to retain second-order accuracy since the variables are still on a uniform theta_st grid

for i in range(no_of_surfs):
    L_st[i, 1:] = np.cumsum(np.sqrt(np.diff(R[i])**2 + np.diff(Z[i])**2))
dt_st_l = derm(theta_st_new, 'l', 'o')
dR_dpsi = derm(R, 'r')/psi_diff
dR_dt   = dermv(R, theta_st_new, 'l', 'e')
dZ_dpsi = derm(Z, 'r')/psi_diff
dZ_dt   = dermv(Z, theta_st_new, 'l', 'o')

jac = dR_dpsi*dZ_dt - dZ_dpsi*dR_dt

# partial derivatives of psi and theta_f on the cartesian grid
dpsidR = dZ_dt/jac
dpsidZ = -dR_dt/jac
dt_dR  = -dZ_dpsi/jac
dt_dZ  =  dR_dpsi/jac

dtdr_st0 = (dt_dR*dpsidR + dt_dZ*dpsidZ)/np.sqrt(dpsidR**2 + dpsidZ**2)

# Recalculate dl on the new grid
dl = np.sqrt(derm(R,'l', 'e')**2 + derm(Z,'l', 'o')**2)

dt = derm(theta_comn_mag_ax_new, 'l', 'o')
for i in range(no_of_surfs):
	 dRj[i, :]  = derm(R[i,:], 'l', 'e')
	 dZj[i, :]  = derm(Z[i,:], 'l', 'o') 
	 phi        = np.arctan2(dZj[i,:], dRj[i,:])
	 phi        = np.concatenate((phi[phi>=0]-np.pi/2, phi[phi<0]+3*np.pi/2)) 
	 phi_n[i,:] = phi

u_ML = np.arctan2(derm(Z, 'l', 'o'), derm(R, 'l', 'e'))

# du_ML/dl is negative and dphi = -du_ML so R_c = -du_ML/dl > 0
#R_c = dl/(2*np.concatenate((np.diff(phi_n, axis=1), np.reshape(np.diff(phi_n)[:, -1],(-1,1))), axis=1))
R_c = dl/derm(phi_n, 'l', 'o')

gradpar2 = 1/(B[1])*(B_p[1])*(derm(theta_st_new[1], 'l', 'o')/dl[1]) # gradpar is b.grad(theta)

gradpar_geo    = 1/(B[1])*(B_p[1])*(derm(theta_comn_mag_ax[1], 'l', 'o')/dl[1]) # gradpar is b.grad(theta)
gradpar_geo_ex = nperiod_data_extend(gradpar_geo, nperiod, istheta=1)


B_p_ex                   = nperiod_data_extend(np.abs(B_p[1]), nperiod, istheta = 0, par = 'e')
B_ex                     = nperiod_data_extend(B[1], nperiod, istheta = 0, par = 'e')
R_ex                     = nperiod_data_extend(R[1], nperiod, istheta = 0, par = 'e')
Z_ex                     = nperiod_data_extend(Z[1], nperiod, istheta = 0, par = 'o')
theta_col                = spl1(theta_st_new[1])
theta_col_ex             = nperiod_data_extend(theta_col, nperiod, istheta=1)
theta_st_new_ex          = nperiod_data_extend(theta_st_new[1], nperiod, istheta=1)
theta_comn_mag_ax_new_ex = nperiod_data_extend(theta_comn_mag_ax_new[1], nperiod, istheta=1)

u_ML_ex = nperiod_data_extend(u_ML[1], nperiod)
R_c_ex  = nperiod_data_extend(R_c[1], nperiod)
dl_ex   = nperiod_data_extend(dl[1], nperiod)
L_st_ex = np.concatenate((np.array([0.]), np.cumsum(np.sqrt(np.diff(R_ex)**2 + np.diff(Z_ex)**2))))

diffrho = derm(rho, 'r')

###################################################################################################################
############################------------------------- BISHOP'S METHOD--------------------------####################
###################################################################################################################

# Since we are calculating these coefficients in straight field line theta, we can use the fact that F[1]*jac[1]/R[1] = qfac[1]

a_s = -(2*qfac[1]/F[1]*theta_st_new_ex + 2*F[1]*qfac[1]*ctrap(1/(R_ex**2*B_p_ex**2), theta_st_new_ex, initial=0))  
b_s = -(2*qfac[1]*ctrap(1/(B_p_ex**2), theta_st_new_ex, initial=0))
c_s =  (2*qfac[1]*ctrap((2*np.sin(u_ML_ex)/R_ex - 2/R_c_ex)*1/(R_ex*B_p_ex), theta_st_new_ex, initial=0))

# calculating the exact dFdpsi on the surface from relation 21 in Miller's paper.
dFdpsi = (-s_hat_input/(rho[1]*(psi_diff[1]/diffrho[1])*(1/(2*np.pi*qfac[1]*(2*nperiod-1))))-(b_s[-1]*dpdpsi - c_s[-1]))/a_s[-1]

# psi_diff[1]/2 is essential
dF = dFdpsi*(psi_diff[1]/2) 
F[0], F[1], F[2] = F[1] - dF.item(), F[1], F[1] + dF.item()

# Calculating the current from the relation (21) in Miller's paper(involving shat) and comparing it with F = q*R^2/J, 
# where J = R*jac is the flux theta jacobian 
F_chk = np.array([np.abs(np.mean(qfac[i]*R[i]/jac[i])) for i in range(no_of_surfs)])

#print("F_chk error(self_consistency_chk) = %.4E\n"%((F_chk[1]-F[1])*(a_N*B_N)))


### A bunch of basic sanity checks
test_diff_st = (dt_dR[1]*dpsidZ[1] - dt_dZ[1]*dpsidR[1])/np.sqrt(dpsidR[1]**2 + dpsidZ[1]**2)\
                - 1/dermv(L_st, theta_st_new, 'l', 'o')[1]
if np.max(np.abs(test_diff_st)) > 1.2E-2:
	print("grad theta_st along l doesn't match...error = %.4E\n"%(np.max(np.abs(test_diff_st))))
else:
	print("grad theta_st along the surface test passed...\n")

# Dual relation check
if  np.abs(np.max((-dt_dR[1]*dpsidZ[1] + dpsidR[1]*dt_dZ[1])*jac[1]) - 1.0) > 1E-11:
	print("theta hat dot grad theta = 1 test failed... difference > 1E-11 \n")
else:
	print("theta hat dot grad theta = 1 test passed...\n")


dpsi_dr = np.zeros((no_of_surfs, ntgrid))
dpsi_dr = np.sign(psi_diff)*np.sqrt(dpsidR**2 + dpsidZ**2)
B_p1 = np.array([np.sqrt(dpsidR[i]**2 + dpsidZ[i]**2)/R[i] for i in range(no_of_surfs)])
B_p1_ex = nperiod_data_extend(B_p1[1], nperiod, istheta = 0, par = 'e')

B_p = np.abs(dpsi_dr)/R
B_t = np.array([np.reshape(F, (-1,1))[i]/R[i] for i in range(no_of_surfs)])

B2 = B_p**2 + B_t**2
B = np.sqrt(B2)

B_p_ex = nperiod_data_extend(B_p[1], nperiod, istheta = 0, par = 'e')
B_ex = nperiod_data_extend(B[1], nperiod, istheta = 0, par = 'e')
B2_ex = nperiod_data_extend(B2[1], nperiod, istheta = 0, par = 'e')


dB2l = derm(B2, 'l', par = 'e')
dBl = derm(B, 'l', par = 'e')
diffq = derm(qfac, 'r')


dB2l_ex = derm(B_ex**2, 'l')[0] # not dB[1]2l zero because the higher dimensional array
dB2l_dl_ex = dermv(B_ex**2, L_st_ex, 'l', par = 'e')
dBl_ex =  derm(B_ex, 'l')[0]
dBl_dl_ex = dermv(B_ex, L_st_ex, 'l', par = 'e')


dpsi_dr_ex = nperiod_data_extend(dpsi_dr[1], nperiod)
gds22 = (diffq/diffrho)**2*np.abs(dpsi_dr_ex)**2  
alpha = -np.reshape(qfac,(-1,1))*theta_st_new_ex
grho = drhodpsi*dpsi_dr_ex

dqdr = diffq*dpsi_dr_ex/psi_diff
dpdr = dpdpsi*dpsi_dr_ex

dpsidR_ex = nperiod_data_extend(dpsidR[1], nperiod, istheta = 0, par = 'e')
dt_dR_ex = nperiod_data_extend(dt_dR[1], nperiod, istheta = 0, par = 'o')
dt_dZ_ex = nperiod_data_extend(dt_dZ[1], nperiod, istheta = 0, par = 'e')
dpsidZ_ex = nperiod_data_extend(dpsidZ[1], nperiod, istheta=0, par = 'o')

dt_st_l_ex = nperiod_data_extend(dt_st_l[1], nperiod, istheta=0, par='e')
dt_st_l_dl_ex = nperiod_data_extend(1/dermv(L_st, theta_st_new, 'l', par = 'o')[1], nperiod, istheta = 0, par = 'e')


# gradpar = b.grad(theta) with st field line theta
#gradpar_ex = -1/(R_ex*B_ex)*(dpsi_dr_ex)*(dt_st_l_ex/dl_ex) 
gradpar_ex = -1/(R_ex*B_ex)*(dpsi_dr_ex)*(dt_st_l_dl_ex) 

#gradpar with theta = colocation theta
gradpar_col_ex = -1/(R_ex*B_ex)*(dpsi_dr_ex)*(nperiod_data_extend(derm(theta_col, 'l', 'o')[0], nperiod)/dl_ex) 

aprime_bish = -R_ex*B_p_ex*(a_s*dFdpsi +b_s*dpdpsi - c_s)/(2*np.abs(drhodpsi))
#plt.plot(theta, np.interp(theta_comn_mag_ax[1], theta_comn_mag_ax_new[1],aprime_bish)); plt.show()
#dtdr_st = diffrho/psi_diff*(aprime_bish - dqdr*theta_st_new)/np.reshape(qfac, (-1,1))

gds21 = diffq/diffrho*(-dpsi_dr_ex)*aprime_bish

dtdr_st_ex = (aprime_bish*drhodpsi - dqdr*theta_st_new_ex)/np.reshape(qfac, (-1,1))

#plt.plot(theta, np.interp(theta_comn_mag_ax[1], theta_comn_mag_ax_new[1],dtdr_st[1]))

gds2 =  (psi_diff/diffrho)**2*(1/R_ex**2 + (dqdr*theta_st_new_ex)**2 + \
        (np.reshape(qfac,(-1,1)))**2*(dtdr_st_ex**2 + (dt_st_l_dl_ex)**2)+ 2*np.reshape(qfac,(-1,1))*dqdr*theta_st_new_ex*dtdr_st_ex)

#plt.plot(theta, np.interp(theta_comn_mag_ax[1], theta_comn_mag_ax_new[1], gds2[1]))

#plt.figure()
gbdrift0 =  1/(B2_ex**2)*dpsidrho*F[1]/R_ex*(dqdr[1]*dB2l_ex/dl_ex)

#############################################################################################################
######################-----------------------dBr CALCULATION-------------------------########################
#############################################################################################################
#We use Miller's equations to find dBdr using the information given on the middle surface.
# Miller and Bishop subscripts have been used interchangeably
# dBdr_bish = (B_p**2/B*(1/R_c + dpdpsi*R/(B_p) + F*dFdpsi/dpsi_dr) + B_t**2/(R*B)*(np.sin(u_ML) - dFdpsi/F*R*dpsi_dr))
dBdr_bish = B_p_ex/B_ex*(-B_p_ex/R_c_ex + dpdpsi*R_ex - F[1]**2*np.sin(u_ML_ex)/(R_ex**3*B_p_ex))
#dBdr_bish_2 = B_p_ex/B_ex*(B_p_ex/R_c_ex + dpdpsi*R_ex - F[1]**2*np.sin(u_ML_ex)/(R_ex**3*B_p_ex))
dBdr = dBdr_bish

gbdrift = 1/np.abs(drhodpsi*B_ex**3)*(2*B2_ex*dBdr/dpsi_dr_ex + aprime_bish*drhodpsi*F[1]/R_ex*dB2l_ex/dl_ex*1/B_ex)
#gbdrift = dpsidrho*(-2/B_ex*dBdr_bish/dpsi_dr_ex + 2*aprime*F/R_ex*1/B_ex**3*dBl_ex/dl_ex)

cvdrift = 1/np.abs(drhodpsi*B_ex**3)*(2*B_ex*dpdpsi) + gbdrift 


####################################################################################################################
#####################---------------------EQUAL_ARC THETA CALCULATION-------------------------######################
#################################################################################################################### 
#equal-arc theta calculation from straight field line gradpar
gradpar_lim   = gradpar_ex[theta_st_new_ex <= np.pi]
B_lim         = B_ex[theta_st_new_ex <= np.pi]
B_p_lim       = B_p_ex[theta_st_new_ex <= np.pi]
theta_lim     = theta_st_new_ex[theta_st_new_ex <= np.pi]
L_eqarc       = ctrap(B_p_lim/(B_lim*gradpar_lim), theta_lim, initial=0)
gradpar_eqarc = np.pi/ctrap(1/(gradpar_lim), theta_lim, initial=0)[-1]


theta_eqarc        = ctrap(B_lim/B_p_lim*gradpar_eqarc, L_eqarc, initial=0)
theta_eqarc_new    = np.linspace(0, np.pi, ntgrid)
theta_eqarc_ex     = nperiod_data_extend(theta_eqarc, nperiod, istheta=1)
theta_eqarc_new_ex = nperiod_data_extend(theta_eqarc_new, nperiod, istheta=1)

gradpar_eqarc_new_ex  = np.interp(theta_eqarc_new_ex, theta_eqarc_ex, gradpar_eqarc*np.ones((len(theta_eqarc_ex,))))
R_eqarc_new_ex        = np.interp(theta_eqarc_new_ex, theta_eqarc_ex, R_ex)
gds21_eqarc_new_ex    = np.interp(theta_eqarc_new_ex, theta_eqarc_ex, gds21[1])
gds22_eqarc_new_ex    = np.interp(theta_eqarc_new_ex, theta_eqarc_ex, gds22[1])
gds2_eqarc_new_ex     = np.interp(theta_eqarc_new_ex, theta_eqarc_ex, gds2[1])
grho_eqarc_new_ex     = np.interp(theta_eqarc_new_ex, theta_eqarc_ex, grho)
gbdrift0_eqarc_new_ex = np.interp(theta_eqarc_new_ex, theta_eqarc_ex, gbdrift0)
B_eqarc_new_ex        = np.interp(theta_eqarc_new_ex, theta_eqarc_ex, B_ex)
cvdrift_eqarc_new_ex  = np.interp(theta_eqarc_new_ex, theta_eqarc_ex, cvdrift)
gbdrift_eqarc_new_ex  = np.interp(theta_eqarc_new_ex, theta_eqarc_ex, gbdrift)

###########################################################################################################
################---------------PACKING EIKCOEFS INTO A DICTIONARY------------------########################
##########################################################################################################

#pdb.set_trace()
if want_eqarc == 1:
    eikcoefs_dict = {'theta_ex':theta_eqarc_new_ex, 'nperiod':nperiod, 'gradpar_ex':gradpar_eqarc_new_ex, 'R_ex':R_eqarc_new_ex,\
                    'B_ex':B_eqarc_new_ex, 'gds21_ex':gds21_eqarc_new_ex, 'gds22_ex':gds22_eqarc_new_ex, 'gds2_ex':gds2_eqarc_new_ex,\
                    'grho_ex':grho_eqarc_new_ex, 'gbdrift_ex':gbdrift_eqarc_new_ex, 'cvdrift_ex':cvdrift_eqarc_new_ex,\
                    'gbdrift0_ex':gbdrift0_eqarc_new_ex, 'cvdrift0_ex':gbdrift0_eqarc_new_ex, 'qfac':qfac[1], 'shat':s_hat_input,\
                    'dpsidrho':dpsidrho, 'Z_ex': Z_ex, 'aplot':alpha, 'aprime':aprime_bish, 'fac':fac, 'file_idx':file_idx,\
                    'lambda_knob':lambda_knob, 'u_ML':u_ML_ex}

elif want_straight == 1:
    eikcoefs_dict = {'theta_ex':theta_st_new_ex,  'nperiod':nperiod,'gradpar_ex':gradpar_ex, 'R_ex':R_ex, 'B_ex':B_ex, 'gds21_ex':gds21[1],\
                    'gds22_ex':gds22[1], 'gds2_ex':gds2[1], 'grho_ex':grho, 'gbdrift_ex':gbdrift, 'cvdrift_ex':cvdrift, 'gbdrift0_ex':gbdrift0,\
                    'cvdrift0_ex':gbdrift0, 'qfac':qfac[1], 'shat':s_hat_input, 'dpsidrho':dpsidrho,'Z_ex':Z_ex, 'aplot':alpha,\
                    'aprime':aprime_bish, 'fac':fac, 'file_idx':file_idx,'lambda_knob':lambda_knob, 'u_ML':u_ML_ex}

elif want_collocation == 1:
    eikcoefs_dict = {'theta_ex':theta_col_ex, 'nperiod':nperiod, 'gradpar_ex':gradpar_col_ex, 'R_ex':R_ex, 'B_ex':B_ex, 'gds21_ex':gds21[1],\
                    'gds22_ex':gds22[1], 'gds2_ex':gds2[1], 'grho_ex':grho, 'gbdrift_ex':gbdrift, 'cvdrift_ex':cvdrift, 'gbdrift0_ex':gbdrift0,\
                    'cvdrift0_ex':gbdrift0, 'qfac':qfac[1], 'shat':s_hat_input, 'dpsidrho':dpsidrho,'Z_ex':Z_ex, 'aplot':alpha,\
                    'aprime':aprime_bish, 'fac':fac, 'file_idx':file_idx,'lambda_knob':lambda_knob, 'u_ML':u_ML_ex}

else:# theta geometric
    eikcoefs_dict = {'theta_ex':theta_comn_mag_ax_new_ex, 'nperiod':nperiod, 'gradpar_ex':gradpar_geo_ex[0], 'R_ex':R_ex, 'B_ex':B_ex, 'gds21_ex':gds21[1],\
                    'gds22_ex':gds22[1], 'gds2_ex':gds2[1], 'grho_ex':grho, 'gbdrift_ex':gbdrift, 'cvdrift_ex':cvdrift, 'gbdrift0_ex':gbdrift0, \
                    'cvdrift0_ex':gbdrift0, 'qfac':qfac[1], 'shat':s_hat_input, 'dpsidrho':dpsidrho,'Z_ex':Z_ex, 'aplot':alpha, 'aprime':aprime_bish,\
                    'fac':fac, 'file_idx':file_idx, 'lambda_knob':lambda_knob,'u_ML':u_ML_ex}

theta       = eikcoefs_dict['theta_ex']
nperiod     = eikcoefs_dict['nperiod']
qfac        = eikcoefs_dict['qfac']
shat        = eikcoefs_dict['shat']
dpsidrho    = eikcoefs_dict['dpsidrho']
gradpar     = eikcoefs_dict['gradpar_ex']
R           = eikcoefs_dict['R_ex']
Z           = eikcoefs_dict['Z_ex']
B           = eikcoefs_dict['B_ex']
gds21       = eikcoefs_dict['gds21_ex']
gds22       = eikcoefs_dict['gds22_ex']
gds2        = eikcoefs_dict['gds2_ex']
grho        = eikcoefs_dict['grho_ex']
gbdrift     = eikcoefs_dict['gbdrift_ex']
cvdrift     = eikcoefs_dict['cvdrift_ex']
gbdrift0    = eikcoefs_dict['gbdrift0_ex']
cvdrift0    = gbdrift0
aplot       = eikcoefs_dict['aplot']
aprime      = eikcoefs_dict['aprime']
fac         = eikcoefs_dict['fac']
file_idx    = eikcoefs_dict['file_idx']
lambda_knob = eikcoefs_dict['lambda_knob']
u_ML        = eikcoefs_dict['u_ML']

gradpar_ball   = reflect_n_append(gradpar, 'e')
theta_ball     = reflect_n_append(theta, 'o')
cvdrift_ball   = reflect_n_append(cvdrift, 'e')
gbdrift_ball   = reflect_n_append(gbdrift, 'e')
gbdrift0_ball  = reflect_n_append(gbdrift0, 'o')
B_ball         = reflect_n_append(B, 'e')
gds2_ball      = reflect_n_append(gds2, 'e')
gds21_ball     = reflect_n_append(gds21, 'o')
gds22_ball     = reflect_n_append(gds22 , 'e')
grho_ball      = reflect_n_append(grho , 'e')

jacob_ball     = 1./abs(drhodpsi*gradpar_ball*B_ball)

Rplot_ball     = reflect_n_append(R, 'e')
Rprime_ball    = reflect_n_append(nperiod_data_extend(np.sin(u_ML[theta <= np.pi]), nperiod, istheta=0, par='e'), 'e')

Zplot_ball     = reflect_n_append(Z, 'o')
Zprime_ball    = -reflect_n_append(nperiod_data_extend(np.cos(u_ML[theta <= np.pi]), nperiod, istheta=0, par='o'), 'o')

aplot_ball     = reflect_n_append(aplot[1], 'o')
aprime_ball    = reflect_n_append(aprime, 'o')


ntheta   = len(theta_ball)


##################################################################################################################
###########################---------------------GX SAVE FORMAT1---------------------------########################
##################################################################################################################

A1 = []
A2 = []
A3 = []
A4 = []
A5 = []
A6 = []
A7 = []
A8 = [] 

for i in range(ntheta):
	A2.append('    %.9e    %.9e    %.9e    %.9e\n'%(gbdrift_ball[i], gradpar_ball[i], grho_ball[i], theta_ball[i]))
	A3.append('    %.9e    %.9e    %.12e    %.9e\n'%(cvdrift_ball[i], gds2_ball[i], B_ball[i], theta_ball[i]))
	A4.append('    %.9e    %.9e    %.9e\n'%(gds21_ball[i], gds22_ball[i], theta_ball[i]))
	A5.append('    %.9e    %.9e    %.9e\n'%(gbdrift0_ball[i], gbdrift0_ball[i], theta_ball[i]))
	A6.append('    %.9e    %.9e    %.9e\n'%(Rplot_ball[i], Rprime_ball[i], theta_ball[i]))
	A7.append('    %.9e    %.9e    %.9e\n'%(Zplot_ball[i], Zprime_ball[i], theta_ball[i]))
	A8.append('    %.9e    %.9e    %.9e\n'%(aplot_ball[i], aprime_ball[i], theta_ball[i]))


A1.append([A2, A3, A4, A5, A6, A7, A8])
A1 = A1[0]

print("Writing eikfile", eikfile)
g = open(eikfile, 'w')

headings = ['ntgrid nperiod ntheta drhodpsi rmaj shat kxfac q\n', 'gbdrift gradpar grho tgrid\n', 'cvdrift gds2 bmag tgrid\n',\
            'gds21 gds22 tgrid\n', 'cvdrift0 gbdrift0 tgrid\n']

g.writelines(headings[0])

g.writelines('  %d    %d    %d   %0.3f   %0.1f    %.7e   %.1f   %.4f\n'%((ntheta-1)/2, 1, (ntheta-1), np.abs(1/dpsidrho),\
             (np.max(Rplot_ball)+np.min(Rplot_ball))/2., shat, abs(qfac/rhoc*dpsidrho), qfac))

for i in np.arange(1, len(headings)):
    g.writelines(headings[i])
    for j in range(ntheta):
            g.write(A1[i-1][j])
g.close()

##################################################################################################################
###########################---------------------GX SAVE FORMAT2---------------------------########################
##################################################################################################################
try:
	import netCDF4 as nc
	eikfile_nc = stem + ".eiknc.nc"

	print('Writing eikfile in netCDF format\n')

	ds = nc.Dataset(eikfile_nc, 'w')

	# The netCDF input file to GX doesn't take the last(repeated) element
	ntheta2       = ntheta - 1

	z_nc = ds.createDimension('z', ntheta2)

	theta_nc    = ds.createVariable('theta', 'f8', ('z',))
	bmag_nc     = ds.createVariable('bmag', 'f8', ('z',))
	gradpar_nc  = ds.createVariable('gradpar', 'f8', ('z',))
	grho_nc     = ds.createVariable('grho', 'f8', ('z',))
	gds2_nc     = ds.createVariable('gds2', 'f8', ('z',))
	gds21_nc    = ds.createVariable('gds21', 'f8', ('z',))
	gds22_nc    = ds.createVariable('gds22', 'f8', ('z',))
	gbdrift_nc  = ds.createVariable('gbdrift', 'f8', ('z',))
	gbdrift0_nc = ds.createVariable('gbdrift0', 'f8', ('z',))
	cvdrift_nc  = ds.createVariable('cvdrift', 'f8', ('z',))
	cvdrift0_nc = ds.createVariable('cvdrift0', 'f8', ('z',))
	jacob_nc    = ds.createVariable('jacob', 'f8', ('z',))

	Rplot_nc    = ds.createVariable('Rplot', 'f8', ('z',))
	Zplot_nc    = ds.createVariable('Zplot', 'f8', ('z',))
	aplot_nc    = ds.createVariable('aplot', 'f8', ('z',))
	Rprime_nc   = ds.createVariable('Rprime', 'f8', ('z',))
	Zprime_nc   = ds.createVariable('Zprime', 'f8', ('z',))
	aprime_nc   = ds.createVariable('aprime', 'f8', ('z',))

	drhodpsi_nc = ds.createVariable('drhodpsi', 'f8', )
	kxfac_nc    = ds.createVariable('kxfac', 'f8', )
	Rmaj_nc     = ds.createVariable('Rmaj', 'f8', )
	q           = ds.createVariable('q', 'f8', )
	shat        = ds.createVariable('shat', 'f8', )  

	theta_nc[:]    = theta_ball[:-1]
	bmag_nc[:]     = B_ball[:-1]
	gradpar_nc[:]  = gradpar_ball[:-1]
	grho_nc[:]     = grho_ball[:-1]
	gds2_nc[:]     = gds2_ball[:-1]
	gds21_nc[:]    = gds21_ball[:-1]
	gds22_nc[:]    = gds22_ball[:-1]
	gbdrift_nc[:]  = gbdrift_ball[:-1]
	gbdrift0_nc[:] = gbdrift0_ball[:-1]
	cvdrift_nc[:]  = cvdrift_ball[:-1]
	cvdrift0_nc[:] = gbdrift0_ball[:-1]
	jacob_nc[:]    = jacob_ball[:-1]

	Rplot_nc[:]    = Rplot_ball[:-1]
	Zplot_nc[:]    = Zplot_ball[:-1]
	aplot_nc[:]    = aplot_ball[:-1]

	Rprime_nc[:]   = Rprime_ball[:-1]
	Zprime_nc[:]   = Zprime_ball[:-1]
	aprime_nc[:]   = aprime_ball[:-1]

	drhodpsi_nc[0] = abs(1/dpsidrho)
	kxfac_nc[0]    = abs(qfac/rhoc*dpsidrho)
	Rmaj_nc[0]     = (np.max(Rplot_nc) + np.min(Rplot_nc))/2
	q[0]           = qfac
	shat[0]        = s_hat_input

	ds.close()
except ModuleNotFoundError:
	print("No netCDF4 package in your Python environment...Not saving a netCDf input file")
	pass





