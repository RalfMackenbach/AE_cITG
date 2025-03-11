# These are the functions used to calculate the available energy in the weak-gradient limit. NOT TESTED
import numpy as np
import scipy.special as sp
import scipy.integrate as spi
import scipy.optimize as spo
import source.utils_integral as sui
import source.utils_solver as sus
import source.utils_ae_full as suaf
import source.utils_ae_iso as suai


def equations_k_weak(k_psi,k_alpha,w_alpha,w_psi,w_n,w_T):
    """
    Returns k_psi and k_alpha, when equation are written in the form:
    k_psi = f(k_psi,k_alpha)
    k_alpha = g(k_psi,k_alpha)
    Useful for iterative solver.
    """
    # define denominator
    w_star =lambda v_perp2,v_par2: suaf.w_star(w_n,w_T,v_perp2,v_par2)
    denom = lambda v_perp2,v_par2: (v_perp2 * w_alpha)**2 + (v_perp2 * w_psi)**2
    int_1 = lambda v_perp2,v_par2: ((v_perp2**2 * w_alpha * w_psi) * k_alpha + w_star(v_perp2,v_par2) * w_alpha * w_psi * v_perp2**2)/denom(v_perp2,v_par2)
    int_2 = lambda v_perp2,v_par2: ((v_perp2**2 * w_alpha * w_psi) * k_psi   + w_star(v_perp2,v_par2) * w_psi   * w_psi * v_perp2**2)/denom(v_perp2,v_par2)
    int_3 = lambda v_perp2,v_par2: w_alpha**2 * v_perp2**2/denom(v_perp2,v_par2)
    int_4 = lambda v_perp2,v_par2: w_psi**2 * v_perp2**2/denom(v_perp2,v_par2)
    kappa_psi   = -sui.integrate_2D_AE_weighting(int_1)/sui.integrate_2D_AE_weighting(int_3)
    kappa_alpha = -sui.integrate_2D_AE_weighting(int_2)/sui.integrate_2D_AE_weighting(int_4)
    return np.asarray([kappa_psi,kappa_alpha])


def AE_integrand_weak(v_perp2,v_par2,w_alpha,w_psi,w_n,w_T,k_psi,k_alpha):
    """
    Returns the integrand for the AE.
    """
    w_star_val = suaf.w_star(w_n,w_T,v_perp2,v_par2)
    integrand = (k_psi * v_perp2 * w_alpha + (k_alpha + w_star_val) * v_perp2 * w_psi)**2.0/(2*((v_perp2 * w_alpha)**2.0 + (v_perp2 * w_psi)**2.0))
    return integrand


def solve_k_weak(w_alpha,w_n,w_T,method='fsolve',**kwargs):
    """
    Returns the solution for k_psi and k_alpha in the weakly driven limit.
    """
    # initial guess, use pure density gradient solution
    k_psi = 0.0
    k_alpha = suai.solve_tilde_k_alpha_iso(w_alpha,w_n,w_T)[0]
    # solve the equations
    if method == 'iterative':
        # iterative solver
        k_psi,k_alpha = equations_k_weak(k_psi,k_alpha,w_alpha,w_alpha,w_n,w_T)
    else:
        # define the equations
        eq = lambda x: equations_k_weak(x[0],x[1],w_alpha,w_alpha,w_n,w_T) - x
        # solve the equations
        solution = sus.solver(eq,[k_psi,k_alpha],method=method,**kwargs)
        k_psi = solution[0]
        k_alpha = solution[1]

    return np.asarray([k_psi,k_alpha])


def AE_local_weak(k_psi,k_alpha,w_alpha,w_psi,w_n,w_T):
    """
    Returns the local AE in the weakly driven limit.
    """
    # define the integrand
    int = lambda v_perp2,v_par2: AE_integrand_weak(v_perp2,v_par2,w_alpha,w_psi,w_n,w_T,k_psi,k_alpha)
    # calculate the integral
    AE = sui.integrate_2D_AE_weighting(int)
    return AE

