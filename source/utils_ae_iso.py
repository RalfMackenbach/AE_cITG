# These are the functions used to calculate the available energy in the isodynamic limit
import numpy as np
import scipy.special as sp
import scipy.integrate as spi
import scipy.optimize as spo
import source.utils_integral as sui
import source.utils_solver as sus

# surpress warnings from fsolve
import warnings
warnings.filterwarnings("ignore")

def _ramp(x):
    """
    Returns the ramp function.
    """
    return x * np.heaviside(x, 0.0)


# def _special_erf(x,b):
#     if b < -1:
#         return 2*sp.erfi(np.sqrt(-1-b)*x)/np.sqrt(-1-b)
#     if b > -1:
#         return 2*sp.erf(np.sqrt(1+b)*x)/np.sqrt(1+b)
#     if b == -1:
#         return 4*x/np.sqrt(np.pi)
#     else:
#         # raise an error if b is not a number
#         raise ValueError("b must be a number")
    

def v_0_squared(v_perp_squared,w_n,w_T,w_alpha):
    """
    Returns the function v_0_squared
    """
    return 3/2 - w_n/w_T - (w_alpha/w_T + 1)*v_perp_squared


def v_perp_squared(v_0_squared,a,b):
    """
    Returns the function v_perp_squared
    """
    return -a + b * v_0_squared



def I_iso(v_0_squared):
    """
    Returns the function I
    """
    # if v_0_squared is not an array, make it an array
    if not isinstance(v_0_squared, np.ndarray):
        v_0_squared = np.array([v_0_squared])
    ans = 0.5 - v_0_squared
    # check where v_0_squared is positive and add the corresponding terms, keeping things vectorized
    mask = v_0_squared > 0
    ans[mask] = ans[mask] + (2/np.sqrt(np.pi)) * np.sqrt(v_0_squared[mask]) * np.exp(-v_0_squared[mask]) + (2 * v_0_squared[mask] - 1) * sp.erf(np.sqrt(v_0_squared[mask]))
    # v_par0 = np.sqrt(np.abs(v_0_squared))
    # ans = np.where(mask, ans + (2/np.sqrt(np.pi)) * v_par0* np.exp(-v_0_squared) + (2 * v_0_squared - 1) * sp.erf(v_par0), ans)
    return ans


def J_iso(v_0_squared,Omega):
    """
    Returns the function J
    """
    if not isinstance(v_0_squared, np.ndarray):
        v_0_squared = np.array([v_0_squared])
    if not isinstance(Omega, np.ndarray):
        Omega = np.array([Omega])
    ans = _ramp(Omega)* (0.5 - v_0_squared)
    # check where v_0_squared is positive and add the corresponding terms, keeping things vectorized
    mask = v_0_squared > 0
    ans[mask] = ans[mask] + np.abs(Omega[mask]) * ( np.sqrt(v_0_squared[mask]) * np.exp(-v_0_squared[mask]) / np.sqrt(np.pi) - ( 0.5 - v_0_squared[mask]) * sp.erf(np.sqrt(v_0_squared[mask])) )
    return ans


def equation_tilde_k_alpha_iso(tilde_k_alpha,w_alpha,w_n,w_T):
    """
    Returns the equation for the isodynamic limit.
    """
    # define the integrals
    integrand = lambda v_perp_squared:  I_iso(v_0_squared(v_perp_squared,w_n,w_T,w_alpha)) * np.exp(-v_perp_squared)
    # calculate the LHS
    LHS = -spi.quad(integrand,0,tilde_k_alpha)[0] + spi.quad(integrand,tilde_k_alpha,np.inf)[0]
    RHS = np.sign(w_alpha) *(w_n + w_alpha)/np.abs(w_T)
    return LHS - RHS


def solve_tilde_k_alpha_iso(w_alpha,w_n,w_T):
    """
    Solves the equation for the isodynamic limit.
    """
    eq = lambda tilde_k_alpha: equation_tilde_k_alpha_iso(np.abs(tilde_k_alpha),w_alpha,w_n,w_T)
    solution = sus.solver(eq,0.0,method='fsolve')
    k_alpha_iso = -w_alpha * solution
    return k_alpha_iso


def AE_local_iso(w_alpha,w_n,w_T,k_alpha_iso, **kwargs):
    """
    Returns the available energy for the isodynamic case.
    """
    # solve integral using quad
    tilde_k_alpha = -k_alpha_iso/w_alpha
    # define the integrand
    v_0_squared = lambda v_perp_squared: 3/2 - w_n/w_T - (w_alpha/w_T + 1)*v_perp_squared
    Omega = lambda v_perp_squared: - w_T * w_alpha * ( v_perp_squared - tilde_k_alpha)
    integrand = lambda v_perp_squared: J_iso(v_0_squared(v_perp_squared),Omega(v_perp_squared)) * np.exp(-v_perp_squared)/6
    # calculate the integral
    integral = spi.quad(integrand,0,np.inf)[0]
    return integral