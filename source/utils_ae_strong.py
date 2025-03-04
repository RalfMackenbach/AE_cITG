# These are the functions used to calculate the available energy in the strong-gradient limit
import numpy as np
import scipy.special as sp
import scipy.integrate as spi
import scipy.optimize as spo
import source.utils_integral as sui
import source.utils_solver as sus
import source.utils_ae_full as suaf
import source.utils_ae_iso as suai

def v_0_squared_strong(v_perp_squared,w_n,w_T):
    """
    Returns the function v_0_squared
    """
    return 3/2 - w_n/w_T - v_perp_squared

def I_1(v0_2):
    """
    Returns the function I_1
    """
    return v0_2-1/2

def equation_isodynamic_strong(kappa_alpha_tilde,w_alpha,w_n,w_T):
    """
    Returns the equation for the strongly driven limit.
    """
    # define the integrals
    integrand = lambda v_perp2: suai.I_iso(v_0_squared_strong(v_perp2,w_n,w_T)) * np.exp(-v_perp2)
    # calculate the LHS
    LHS = -spi.quad(integrand,0,kappa_alpha_tilde)[0] + spi.quad(integrand,kappa_alpha_tilde,np.inf)[0]
    RHS = (w_n)* np.sign(w_alpha) /np.abs(w_T)
    return LHS - RHS

def solve_iso_strong(w_alpha,w_n,w_T,method='fsolve',**kwargs):
    """
    Returns the solution for k_alpha in the strongly driven limit.
    """
    eq = lambda x: equation_isodynamic_strong(x,w_alpha,w_n,w_T)
    solution = sus.solver(eq,0.0,method=method,**kwargs)
    return -w_alpha * solution[0]


def equations_k_strong(k_psi,k_alpha,w_alpha,w_psi,w_n,w_T):
    """
    Returns the first solution for k_alpha.
    """
    w_star_fun = lambda v_perp2,v_par2: suaf.w_star(w_n,w_T,v_perp2,v_par2)
    denom = lambda v_perp2,v_par2: np.sqrt((v_perp2 * w_alpha + k_alpha)**2 + (v_perp2 * w_psi - k_psi)**2)
    int_1 = lambda v_perp2,v_par2: w_star_fun(v_perp2,v_par2) + np.abs(w_star_fun(v_perp2,v_par2)) * (v_perp2 * w_alpha) / denom(v_perp2,v_par2)
    int_2 = lambda v_perp2,v_par2: np.abs(w_star_fun(v_perp2,v_par2)) / denom(v_perp2,v_par2)
    int_3 = lambda v_perp2,v_par2: np.abs(w_star_fun(v_perp2,v_par2)) * (v_perp2 * w_psi) / denom(v_perp2,v_par2)
    k_alpha_new = - sui.integrate_2D_AE_weighting(int_1) / sui.integrate_2D_AE_weighting(int_2)
    k_psi_new = sui.integrate_2D_AE_weighting(int_3) / sui.integrate_2D_AE_weighting(int_2)
    return np.asarray([k_psi_new,k_alpha_new])

def equation_k_strong_1D(kappa_psi,kappa_alpha,w_alpha,w_psi,w_n,w_T):
    """
    Returns the equation for the strongly driven limit.
    """
    # define the integrals
    v0_2 = lambda v_perp2:  v_0_squared_strong(v_perp2,w_n,w_T)
    denom = lambda v_perp2: np.sqrt( (kappa_alpha + v_perp2 * w_alpha)**2 + (v_perp2 * w_psi - kappa_psi)**2 )
    integrand_k_alpha_1 = lambda v_perp2: np.exp(-v_perp2) * ( -I_1(v0_2(v_perp2))*np.sign(w_T) - suai.I_iso(v0_2(v_perp2)) * v_perp2 * w_alpha/denom(v_perp2) )
    integrand_k_alpha_2 = lambda v_perp2: np.exp(-v_perp2) * ( suai.I_iso(v0_2(v_perp2)) / denom(v_perp2) )
    integrand_k_psi_1 = lambda v_perp2: np.exp(-v_perp2) * ( suai.I_iso(v0_2(v_perp2)) * v_perp2 * w_psi/denom(v_perp2) )
    denominator = spi.quad(integrand_k_alpha_2,0,np.inf)[0]
    kappa_alpha_new = (spi.quad(integrand_k_alpha_1,0,np.inf)[0])/denominator
    kappa_psi_new = (spi.quad(integrand_k_psi_1,0,np.inf)[0])/denominator
    return np.asarray([kappa_psi_new,kappa_alpha_new])


def solve_k_strong(w_alpha,w_psi,w_n,w_T,method='iterative',eq='2D',**kwargs):
    """
    Returns the solution for k_psi and k_alpha.
    """
    # initial guess, use pure density gradient solution
    k_psi = 0.0
    k_alpha = solve_iso_strong(w_alpha,w_n,w_T)
    # solve the equations
    if method == 'iterative':
        if eq == '1D':
            eq_solve = lambda x: equation_k_strong_1D(x[0],x[1],w_alpha,w_psi,w_n,w_T)
        elif eq == '2D':
            eq_solve = lambda x: equations_k_strong(x[0],x[1],w_alpha,w_psi,w_n,w_T)
        k_psi, k_alpha = sus.solver(eq_solve, np.array([k_alpha,k_psi]), method=method, **kwargs)
    else:
        if eq == '1D':
            eq = lambda x: equation_k_strong_1D(x[0],x[1],w_alpha,w_psi,w_n,w_T) - x
        elif eq == '2D':
            eq = lambda x: equations_k_strong(x[0],x[1],w_alpha,w_psi,w_n,w_T) - x
        k_psi, k_alpha = sus.solver(eq, np.array([k_psi,k_alpha]), method=method, **kwargs)
        


    return k_psi, k_alpha


def available_energy_strong_integrand(v_perp2,w_alpha,w_psi,w_n,w_T,k_alpha,k_psi, **kwargs):
    """
    Returns the integrand for the available energy in the strongly driven limit.
    """

    # define the integrand
    v0_2 = v_0_squared_strong(v_perp2,w_n,w_T)
    I_1_val = I_1(v0_2)
    I1_fac = (k_alpha + v_perp2 * w_alpha) * np.sign(w_T)
    I_val = suai.I_iso(v0_2)
    sqrt_fac = np.sqrt( (k_alpha + v_perp2 * w_alpha)**2 + (v_perp2 * w_psi - k_psi)**2 )
    full_integrand = I_1_val*I1_fac + I_val * sqrt_fac
    return np.abs(w_T) * full_integrand * np.exp(-v_perp2)/12

def AE_local_strong(w_n,w_T,w_alpha,w_psi,k_alpha,k_psi):
    """
    Returns the AE for a given w_n, w_T, w_alpha, w_psi, k_psi, k_alpha.
    """
    int_AE = lambda v_perp2: available_energy_strong_integrand(v_perp2,w_alpha,w_psi,w_n,w_T,k_alpha,k_psi)
    # integrate using adaptive quadrature
    integral = spi.quad(int_AE,0,np.inf)[0]
    return integral