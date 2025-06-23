"""
This script generates input configuration dictionaries for gyrokinetic simulations using the Miller equilibrium parameterization.

Key features:
- Randomly generates physically valid Miller geometry and simulation parameters for each run, ensuring that generated flux surfaces do not intersect (using intersection checks).
- Uses a scan index (scan_idx) as a random seed for reproducibility: the same scan_idx will always generate the same input parameters.
- Provides utility functions for truncated random sampling, geometry calculations, and file output.
- Main functions:
    * return_input_dict: Generates a random, valid input configuration for a given scan_idx, tprim, and fprim.
    * return_input_dict_geom_input: Generates an input configuration from explicitly provided geometry parameters.
- Outputs configuration dictionaries suitable for saving as TOML files for use in GX simulation codes.
"""

import  toml
import  numpy           as np
import  scipy           as sp




#### The following section is utils from the GXdatabase with some edits ####
def check_intersect(ntheta,rhoc,Rmaj,shift,akappa,akappri,tri,tripri,zeta,zetapri,delrho=1e-3,plot=False):

    # The results may be sensitive to delrho! Choose carefully.
    #######################################################################################################################
    ########################------------ primary lowest level calculations---------------------############################
    #######################################################################################################################
    # Note that tri in gs2 is actually the sin(delta).
    #tri = np.sin(tri_gs2) # tri is tri Miller
    #tripri = (np.sin(tri_gs2+tripri_gs2*delrho) - np.sin(tri_gs2-tripri_gs2*delrho))/(2*delrho)
    ntgrid=ntheta+1

    # no need to change this
    no_of_surfs = 3

    # note that this theta is neither geometric nor flux. It's just used to generate the surfaces. We will call it colocation theta
    # GS2 uses this theta for the grad par calculation
    theta = np.linspace(0, 2*np.pi, ntgrid)

    # position of the magnetic axes of the three surfaces
    R_0 = np.array([Rmaj+np.abs(shift)*delrho, Rmaj, Rmaj-np.abs(shift)*delrho])

    rho = np.array([rhoc - delrho, rhoc, rhoc + delrho])
    kappa = np.array([akappa-akappri*delrho, akappa, akappa+akappri*delrho])
    delta = np.array([tri-tripri*delrho, tri, tri+tripri*delrho])
    zeta = np.array([zeta-zetapri*delrho, zeta, zeta+zetapri*delrho])

    #R_mag_ax can be anything as long as it's inside the annulus. 
    R_mag_ax = Rmaj

    R= np.array([R_0[i] + (rho[i])*np.cos(theta +np.arcsin(delta[i])*np.sin(theta)) for i in range(no_of_surfs)])
    Z = np.array([kappa[i]*(rho[i])*np.sin(theta + zeta[i]*np.sin(2*theta)) for i in range(no_of_surfs)])

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i in range(no_of_surfs):
            ax.plot(R[i], Z[i])
        ax.set_aspect('equal')
        plt.show()

    ## Quick intersection check. If shift is large, surfaces can intersect
    # To check if the equilibrium sufaces intersect with each other
    intersect = intersection_chk(R, Z)

    return intersect

def intersection_chk(R, Z):
    # check intersection for the given R and Z paths
    R_path_inner = R[0,:]
    Z_path_inner = Z[0,:]
    R_path = R[1,:]
    Z_path = Z[1,:]
    R_path_outer = R[2,:]
    Z_path_outer = Z[2,:]
    
    # now check cross-intersection. Self-intersection implies cross-intersection.
    cross_intersects = False
    for i in range(len(R_path_inner)-1):
        for j in range(len(R_path)-1):
            if intersect(R_path_inner[i], Z_path_inner[i], R_path_inner[i+1], Z_path_inner[i+1], R_path[j], Z_path[j], R_path[j+1], Z_path[j+1]):
                cross_intersects = True
                # print('cross-intersects at:',i,j)
                break
    for i in range(len(R_path_outer)-1):
        for j in range(len(R_path)-1):
            if intersect(R_path_outer[i], Z_path_outer[i], R_path_outer[i+1], Z_path_outer[i+1], R_path[j], Z_path[j], R_path[j+1], Z_path[j+1]):
                cross_intersects = True
                # print('cross-intersects at:',i,j)
                break

    return cross_intersects
            
def intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    # Returns True if the line segments AB and CD intersect
    # https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    def ccw(A,B,C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    return ccw((x1,y1), (x3,y3), (x4,y4)) != ccw((x2,y2), (x3,y3), (x4,y4)) and ccw((x1,y1), (x2,y2), (x3,y3)) != ccw((x1,y1), (x2,y2), (x4,y4))

################################################################################################




#### The following section are utils for the input file generation ####
def trunc_normal(loc,scale,trunc_lower,trunc_upper):
    val = np.random.normal(loc,scale)
    while (val < trunc_lower) or (val > trunc_upper):
        val = np.random.normal(loc,scale)
    return val

def trunc_power(a,trunc_lower,trunc_upper):
    val = np.random.power(a)
    while (val < trunc_lower) or (val > trunc_upper):
        val = np.random.power(a)
    return val

def get_shat(lower,upper,discard):
    shat = np.random.uniform(lower,upper)
    while np.abs(shat) < discard:
        shat = np.random.uniform(lower,upper)
    return shat


def jtwist_rounder(shat, max_jtwist=20):
    val = np.round(2*np.pi*shat)
    if val == 0:
        val = np.sign(shat)
    val = int(val)
    if np.abs(val) > max_jtwist:
        val = max_jtwist * np.sign(val)
    return int(val)
    
def average_flux(Rmaj,rhoc,kappa,delta,zeta):
    # use miller parameterisation
    R_param = lambda theta: Rmaj + rhoc * np.cos(theta + np.arcsin(delta)*np.sin(theta))
    Z_param = lambda theta: rhoc * kappa * np.sin(theta + zeta * np.sin(2*theta))
    dRdtheta= lambda theta: -rhoc * (1.0 + np.arcsin(delta) * np.cos(theta)) * np.sin(theta + np.arcsin(delta) * np.sin(theta))

    # B_T = B_ref/R so that the toroidal flux is
    # d psi_T = (B_ref/R) * 2*Z*dR = 2*Z(theta)/R(theta) * dR/dtheta * dtheta
    # furthermore, the area is simply
    # d A = 2*Z*dR = 2*Z(theta)*dR/dtheta * dtheta
    # do the integral over theta
    integrand_flux = lambda theta: 2*Z_param(theta)*dRdtheta(theta)/R_param(theta)
    integrand_area = lambda theta: 2*Z_param(theta)*dRdtheta(theta)
    toroidal_flux = -sp.integrate.quad(integrand_flux,0,np.pi)[0]
    area = -sp.integrate.quad(integrand_area,0,np.pi)[0]
    return toroidal_flux/area

def file_creator(config_dict, file_name):
    # Save the dictionary to a toml file
    with open(file_name, 'w') as toml_file:
        toml.dump(config_dict, toml_file)
############################################################################


# The following function is the main function to generate the input files
def return_input_dict(scan_idx,tprim,fprim,**kwargs):

    # set kwargs to default values.
    # useful to change these for convergence studies
    kwargs.setdefault('ntheta',24)
    kwargs.setdefault('nperiod',1)
    kwargs.setdefault('ny',96)
    kwargs.setdefault('nx',193)
    kwargs.setdefault('nhermite',12)
    kwargs.setdefault('nlaguerre',4)
    kwargs.setdefault('y0',50.0)
    kwargs.setdefault('t_max',2500.0)
    kwargs.setdefault('D_hyper',0.1)
    kwargs.setdefault('vnewk',0.01)
    kwargs.setdefault('jmult',1)
    kwargs.setdefault('beta',0.0)
    kwargs.setdefault('cfl',0.6)
    
    # # highly converged values
    # kwargs.setdefault('ntheta',48)
    # kwargs.setdefault('nperiod',1)
    # kwargs.setdefault('ny',192)
    # kwargs.setdefault('nx',386)
    # kwargs.setdefault('nhermite',12)
    # kwargs.setdefault('nlaguerre',4)
    # kwargs.setdefault('y0',60.0)
    # kwargs.setdefault('t_max',3000.0)
    # kwargs.setdefault('D_hyper',0.1)
    # kwargs.setdefault('vnewk',0.01)
    # kwargs.setdefault('jmult',1)
    # kwargs.setdefault('beta',0.0)
    # kwargs.setdefault('cfl',0.6)




    # set random seed to scan_idx. If one ever needs to more closely investigate a data-point, they may 
    # readily regenerate it by setting the seed equal to the data index
    if scan_idx != None:
        np.random.seed(scan_idx)

    intersection = 1

    while intersection == 1:
        # initialise random variables
        Rmaj        = np.random.uniform(1.1,10.0)
        q_r         = np.random.uniform(1.0,10.0) # realistic q values
        q_l         = np.random.uniform(10.0,100.0) # long connection length
        qinp        = np.random.choice([q_r,q_l],p=[1.0,0.0])
        shat        = get_shat(-1,2,0.05)
        shift       = np.random.uniform(0.0,0.2)
        rhoc        = trunc_power(2,0.01,1.0)
        akappa      = np.random.uniform(1.0,3.0)
        akappapri   = np.random.normal(akappa,akappa)
        delta_edge  = trunc_normal(0.0,0.4,-0.9,0.9)
        tri         = rhoc * delta_edge
        tripri      = trunc_normal(np.abs(delta_edge),2*np.abs(delta_edge),0.0,np.inf)*np.sign(delta_edge)
        zeta_edge   = trunc_normal(0.0,0.2,-0.45,0.9) # at zeta=-0.5 a sign change happens in the derivative
        zeta        = rhoc**2 * zeta_edge
        zetapri     = trunc_normal(2*rhoc*np.abs(zeta_edge),4*rhoc*np.abs(zeta_edge),0.0,np.inf)*np.sign(zeta)
        betaprim    =-np.random.uniform(0.0,0.01)*(tprim+fprim) #check sign

        # R_geo is such that B_ref = B_T(R_geo), i.e. B_T = R_geo/R B_ref
        # Since average_flux = int(1/R)dA / int dA = 1/R_geo -> R_geo = 1.0/average_flux
        R_geo       = 1.0/average_flux(Rmaj,rhoc,akappa,tri,zeta)

        intersection = check_intersect(kwargs['ntheta'],rhoc,Rmaj,shift,akappa,akappapri,tri,tripri,zeta,zetapri)
        if intersection == 1:
            print('Initial Miller parameterisation invalid for scan_idx {}. Running again.'.format(scan_idx))
            # print miller parameters
            # print('Rmaj =',Rmaj)
            # print('qinp =',qinp)
            # print('shat =',shat)
            # print('shift =', shift)
            # print('rhoc =', rhoc)
            # print('kappa =', akappa)
            # print('akappapri =', akappapri)
            # print('tri =', tri)
            # print('tripri =', tripri)
            # print('zeta =', zeta)
            # print('zetapri =', zetapri)
            # print('betaprim =', betaprim)



    # print('Relative diff Rmaj Rgeo: ',np.abs((R_geo-Rmaj)/Rmaj)*100, '%')
    # print('Rmaj =',Rmaj)
    # print('qinp =',qinp)
    # print('shat =',shat)
    # print('shift =', shift)
    # print('rhoc =', rhoc)
    # print('kappa =', akappa)
    # print('akappapri =', akappapri)
    # print('tri =', tri)
    # print('tripri =', tripri)
    # print('zeta =', zeta)
    # print('zetapri =', zetapri)
    # print('betaprim =', betaprim)

    
    config = {
        'debug': False,
        'Dimensions': {
            'ntheta': kwargs['ntheta'],
            'nperiod': kwargs['nperiod'],
            'ny': kwargs['ny'],
            'nx': kwargs['nx'],
            'nhermite': kwargs['nhermite'],
            'nlaguerre': kwargs['nlaguerre'],
            'nspecies': 1
        },
        'Domain': {
            'y0': kwargs['y0'],
            'jtwist': kwargs['jmult']*jtwist_rounder(shat),
            'boundary': 'linked'
        },
        'Physics': {
            'beta': kwargs['beta'],
            'nonlinear_mode': True
        },
        'Time': {
            't_max': kwargs['t_max'],
            'cfl': kwargs['cfl'],
            'scheme': 'rk3'
        },
        'Initialization': {
            'ikpar_init': 0,
            'init_field': 'density',
            'init_amp': 1.0e-2
        },
        'Geometry': {
            'geo_option': 'miller',
            'Rmaj': float(Rmaj),
            'R_geo': float(R_geo),
            'qinp': float(qinp),
            'shat': float(shat),
            'shift': float(shift),
            'rhoc': float(rhoc),
            'akappa': float(akappa),
            'akappri': float(akappapri),
            'tri': float(tri),
            'tripri': float(tripri),
            'zeta': float(zeta),
            'zetapri': float(zetapri),
            'betaprim': float(betaprim)
        },
        'species': {
            'z': [1.0, -1.0],
            'mass': [1.0, 2.7e-4],
            'dens': [1.0, 1.0],
            'temp': [1.0, 1.0],
            'tprim': [tprim, 0.0],
            'fprim': [fprim, 0.0],
            'vnewk': [kwargs['vnewk'], 0.0],
            'type': ['ion', 'electron']
        },
        'Boltzmann': {
            'add_Boltzmann_species': True,
            'Boltzmann_type': 'electrons',
            'tau_fac': 1.0
        },
        'Dissipation': {
            'closure_model': 'none',
            'hypercollisions': True,
            'hyper': True,
            'D_hyper': kwargs['D_hyper'],
            'p_hyper': 2
        },
        'Restart': {
            'restart': False,
            'save_for_restart': True,
            'nsave': 10000
        },
        'Diagnostics': {
            'nwrite': 200,
            'nwrite_big': 10000,
            'free_energy': True,
            'fluxes': True,
            'fields': True,
            'moments': True
        }
    }

    return config, kwargs




def return_input_dict_geom_input(tprim,fprim,Rmaj,shift,rhoc,akappa,akappapri,tri,tripri,zeta,zetapri,betaprim,qinp,shat,**kwargs):

    # set kwargs to default values.
    # useful to change these for convergence studies
    kwargs.setdefault('ntheta',24)
    kwargs.setdefault('nperiod',1)
    kwargs.setdefault('ny',96)
    kwargs.setdefault('nx',193)
    kwargs.setdefault('nhermite',12)
    kwargs.setdefault('nlaguerre',4)
    kwargs.setdefault('y0',50.0)
    kwargs.setdefault('t_max',2500.0)
    kwargs.setdefault('D_hyper',0.1)
    kwargs.setdefault('vnewk',0.01)
    kwargs.setdefault('jmult',1)
    kwargs.setdefault('beta',0.0)
    kwargs.setdefault('cfl',0.6)
    
    # # highly converged values
    # kwargs.setdefault('ntheta',48)
    # kwargs.setdefault('nperiod',1)
    # kwargs.setdefault('ny',192)
    # kwargs.setdefault('nx',386)
    # kwargs.setdefault('nhermite',12)
    # kwargs.setdefault('nlaguerre',4)
    # kwargs.setdefault('y0',60.0)
    # kwargs.setdefault('t_max',3000.0)
    # kwargs.setdefault('D_hyper',0.1)
    # kwargs.setdefault('vnewk',0.01)
    # kwargs.setdefault('jmult',1)
    # kwargs.setdefault('beta',0.0)
    # kwargs.setdefault('cfl',0.6)
    
    # set R_geo
    R_geo       = 1.0/average_flux(Rmaj,rhoc,akappa,tri,zeta)

    config = {
        'debug': False,
        'Dimensions': {
            'ntheta': kwargs['ntheta'],
            'nperiod': kwargs['nperiod'],
            'ny': kwargs['ny'],
            'nx': kwargs['nx'],
            'nhermite': kwargs['nhermite'],
            'nlaguerre': kwargs['nlaguerre'],
            'nspecies': 1
        },
        'Domain': {
            'y0': kwargs['y0'],
            'jtwist': kwargs['jmult']*jtwist_rounder(shat),
            'boundary': 'linked'
        },
        'Physics': {
            'beta': kwargs['beta'],
            'nonlinear_mode': True
        },
        'Time': {
            't_max': kwargs['t_max'],
            'cfl': kwargs['cfl'],
            'scheme': 'rk3'
        },
        'Initialization': {
            'ikpar_init': 0,
            'init_field': 'density',
            'init_amp': 1.0e-2
        },
        'Geometry': {
            'geo_option': 'miller',
            'Rmaj': float(Rmaj),
            'R_geo': float(R_geo),
            'qinp': float(qinp),
            'shat': float(shat),
            'shift': float(shift),
            'rhoc': float(rhoc),
            'akappa': float(akappa),
            'akappri': float(akappapri),
            'tri': float(tri),
            'tripri': float(tripri),
            'zeta': float(zeta),
            'zetapri': float(zetapri),
            'betaprim': float(betaprim)
        },
        'species': {
            'z': [1.0, -1.0],
            'mass': [1.0, 2.7e-4],
            'dens': [1.0, 1.0],
            'temp': [1.0, 1.0],
            'tprim': [tprim, 0.0],
            'fprim': [fprim, 0.0],
            'vnewk': [kwargs['vnewk'], 0.0],
            'type': ['ion', 'electron']
        },
        'Boltzmann': {
            'add_Boltzmann_species': True,
            'Boltzmann_type': 'electrons',
            'tau_fac': 1.0
        },
        'Dissipation': {
            'closure_model': 'none',
            'hypercollisions': True,
            'hyper': True,
            'D_hyper': kwargs['D_hyper'],
            'p_hyper': 2
        },
        'Restart': {
            'restart': False,
            'save_for_restart': True,
            'nsave': 10000
        },
        'Diagnostics': {
            'nwrite': 200,
            'nwrite_big': 10000,
            'free_energy': True,
            'fluxes': True,
            'fields': True,
            'moments': True
        }
    }

    return config, kwargs