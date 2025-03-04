# These are the functions used to calculate the available energy.
import numpy as np
import source.utils_integral as sui
import source.utils_solver as sus
import source.utils_ae_iso as suai


def w_star(w_n,w_T,v_perp2,v_par2):
    """
    Returns the diamagnetic drift frequency w_star^T.
    """
    return - w_n - w_T*(v_perp2 + v_par2 - 3/2)



def G(k_psi,k_alpha,w_star,w_alpha,w_psi,v_perp2):
    """
    Returns the function G.
    """
    numerator = np.sqrt((v_perp2 * w_alpha - w_star)**2 + (v_perp2 * w_psi)**2)
    denominator = np.sqrt((v_perp2 * w_alpha + k_alpha)**2 + (v_perp2 * w_psi - k_psi)**2)
    return numerator/denominator



def AE_integrand(v_perp2,v_par2,w_alpha,w_psi,w_n,w_T,k_psi,k_alpha):
    """
    Returns the integrand for the AE.
    """
    w_star_val = w_star(w_n,w_T,v_perp2,v_par2)
    factor1 = np.sqrt( ( v_perp2 * w_alpha + k_alpha )**2 + ( v_perp2 * w_psi - k_psi )**2 )
    factor2 = np.sqrt( ( v_perp2 * w_alpha - w_star_val )**2 + ( v_perp2 * w_psi )**2 )
    factor3 = (v_perp2 * w_alpha + k_alpha) * (v_perp2 * w_alpha - w_star_val)
    factor4 = (v_perp2 * w_psi - k_psi) * (v_perp2 * w_psi)
    return factor1 * factor2 - factor3 - factor4


def equations_k(k_psi,k_alpha,w_alpha,w_psi,w_n,w_T):
    """
    Returns k_psi and k_alpha, when equation are written in the form:
    k_psi = f(k_psi,k_alpha)
    k_alpha = g(k_psi,k_alpha)
    Useful for iterative solver.
    """
    # define G function as function of v_perp and v_par
    G_func = lambda v_perp2,v_par2: G(k_psi,k_alpha,w_star(w_n,w_T,v_perp2,v_par2),w_alpha,w_psi,v_perp2)
    # first define 2D function for the integrand of k_psi
    int_k_psi_1 = lambda v_perp2,v_par2: G_func(v_perp2,v_par2)*v_perp2*w_psi
    int_k_psi_2 = lambda v_perp2,v_par2: G_func(v_perp2,v_par2) 
    # calculate the integral
    kappa_psi = (sui.integrate_2D_AE_weighting(int_k_psi_1)/np.sqrt(np.pi) - w_psi)/(sui.integrate_2D_AE_weighting(int_k_psi_2)/np.sqrt(np.pi))
    # now define 2D function for the integrand of k_alpha
    int_k_alpha_1 = lambda v_perp2,v_par2: G_func(v_perp2,v_par2)*v_perp2*w_alpha
    int_k_alpha_2 = lambda v_perp2,v_par2: G_func(v_perp2,v_par2)
    kappa_alpha = (w_n + w_alpha - sui.integrate_2D_AE_weighting(int_k_alpha_1)/np.sqrt(np.pi))/(sui.integrate_2D_AE_weighting(int_k_alpha_2)/np.sqrt(np.pi))
    return np.asarray([kappa_psi, kappa_alpha])


def solve_k(w_alpha,w_psi,w_n,w_T,method='iterative',**kwargs):
    """
    Returns the solution for k_psi and k_alpha.
    """
    # initial guess at isodynamic solution
    k_psi = 0.0
    k_alpha = suai.solve_tilde_k_alpha_iso(w_alpha,w_n,w_T)[0]
    # solve the equations
    if method == 'iterative':
        k_psi, k_alpha = sus.solver(lambda x: equations_k(x[0],x[1],w_alpha,w_psi,w_n,w_T), np.array([k_psi,k_alpha]), method=method, **kwargs)
    else:
        # set up equations
        eqs = lambda x: equations_k(x[0],x[1],w_alpha,w_psi,w_n,w_T) - x
        # solve the equations
        k_psi, k_alpha = sus.solver(eqs, np.array([k_psi,k_alpha]), method=method)

    return k_psi, k_alpha


def AE_local(w_n,w_T,w_alpha,w_psi,k_psi,k_alpha):
    """
    Returns the AE for a given w_n, w_T, w_alpha, w_psi, k_psi, k_alpha.
    """
    int_AE = lambda v_perp2,v_par2: AE_integrand(v_perp2,v_par2,w_alpha,w_psi,w_n,w_T,k_psi,k_alpha)
    # calculate the integral
    return sui.integrate_2D_AE_weighting(int_AE)/np.sqrt(np.pi)/12