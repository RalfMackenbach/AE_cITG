# Here we define all functions relevant to the calculation of the AE for a given magnetic field.
import numpy as np
import scipy.special as sp
import scipy.integrate as spi
import scipy.optimize as spo


def _ramp(x):
    """
    Returns the ramp function.
    """
    return x * np.heaviside(x, 0.0)



def integrate_2D_AE_weighting(f, method='simpson', **kwargs):
    """
    Integrate a function over a 2D domain. 

    This function integrates a given function over a 2D domain using a specified method. 
    Assumes that

    Parameters:
    f (function): The integrand without the exponential factor exp(-x)*exp(-y)/sqrt(y). Order of arguments is (x, y).
    method (str): The method to be used for integration. Default is 'gauss_hermite'.

    Returns:
    float: The result of the integration.
    """

    # set standard values for the optional keyword arguments
    kwargs.setdefault('n_gauss_laguerre', 20)
    kwargs.setdefault('x_max_bounded', 10)
    kwargs.setdefault('y_max_bounded', 10)
    kwargs.setdefault('n_trapz',   30 * int((kwargs['x_max_bounded']+kwargs['y_max_bounded'])/2))
    kwargs.setdefault('n_simpson', 10 * int((kwargs['x_max_bounded']+kwargs['y_max_bounded'])/2))



    if method == 'gauss_laguerre':
        # Define the number of points for the Gauss-Laguerre. Mixed with general Gauss-Laguerre for y (alpha=-1/2)
        n = kwargs.get('n_gauss_laguerre')
        # Create the 2D grid
        # get the roots and weights in 1D
        x, x_w = sp.roots_genlaguerre(n, 0.0)
        y, y_w = sp.roots_genlaguerre(n, -0.5)
        # create the 2D grid
        X, Y = np.meshgrid(x, y)
        weights = np.outer(x_w, y_w)
        function = f(X, Y)

        # Calculate the integral
        integral = np.sum(weights * function)
        return integral
    
    if method == "dblquad":
        # Define the limits of integration
        x0, x1 = 0, np.inf
        y0, y1 = 0, np.inf
        # first do the integral over y 
        f_int_y = lambda x: spi.quad(lambda y: f(x, y)*np.exp(-x-y)/np.sqrt(y), y0, y1)[0]
        # then do the integral over x
        integral = spi.quad(f_int_y, x0, x1)[0]
        return integral
    
    if method == "trapz":
        # Define the limits of integration
        x0, x1 = 0, kwargs.get('x_max_bounded')
        y0, y1 = 0, kwargs.get('y_max_bounded')
        # in order to avoid the singularity at y=0, we change the integration variable to theta = sqrt(y)
        # create the 2D grid if not provided
        x = np.linspace(x0, x1, kwargs.get('n_trapz'))
        theta = np.linspace(0, np.sqrt(y1), kwargs.get('n_trapz'))
        X, Theta = np.meshgrid(x, theta)
        # make the function to integrate
        f_int = lambda x, theta: f(x, theta**2)*np.exp(-x)*np.exp(-theta**2)*2
        # calculate the integral
        integral = np.trapezoid(np.trapezoid(f_int(X, Theta), theta, axis=0), x, axis=0)
        # Calculate the integral
        return integral
    
    if method == "x_trapz_y_quad":
        # Define the limits of integration
        x0, x1 = 0, kwargs.get('x_max_bounded')

        # define integral over y
        f_int_y = lambda x: spi.quad(lambda y: f(x, y)*np.exp(-x)*np.exp(-y)/np.sqrt(y), 0, np.inf)[0]
        # calculate the integral
        integral = np.trapezoid([f_int_y(x) for x in np.linspace(x0, x1, kwargs.get('n_trapz'))], np.linspace(x0, x1, kwargs.get('n_trapz')))
        return integral
    
    if method == "romberg":
        # Define the limits of integration
        x0, x1 = 0, kwargs.get('x_max_bounded')
        y0, y1 = 0, kwargs.get('y_max_bounded')
        # first do the integral over y.
        # change the integration variable to theta = sqrt(y)
        f_int_y = lambda x: spi.romberg(lambda theta: f(x, theta**2)*np.exp(-x)*np.exp(-theta**2)*2, 0, np.sqrt(y1))
        # then do the integral over x
        integral = spi.romberg(f_int_y, x0, x1)
        return integral

    if method=='simpson':
        # Define the limits of integration
        x0, x1 = 0, kwargs.get('x_max_bounded')
        y0, y1 = 0, kwargs.get('y_max_bounded')
        # in order to avoid the singularity at y=0, we change the integration variable to theta = sqrt(y)
        # create the 2D grid if not provided
        x = np.linspace(x0, x1, kwargs.get('n_simpson'))
        theta = np.linspace(0, np.sqrt(y1), kwargs.get('n_simpson'))
        X, Theta = np.meshgrid(x, theta)
        # make the function to integrate
        f_int = lambda x, theta: f(x, theta**2)*np.exp(-x)*np.exp(-theta**2)*2
        # calculate the integral
        integral = spi.simpson(spi.simpson(f_int(X, Theta), x=theta, axis=0), x=x, axis=0)
        # Calculate the integral
        return integral



def solver(f, x0, method='fsolve', **kwargs):
    """
    Solve a system of equations using a specified method.

    This function solves a system of equations using a specified method.

    Parameters:
    f (function): The function to be solved.
    x0 (array): The initial guess for the solution.
    method (str): The method to be used for solving the system of equations. Default is 'least_squares'.

    Returns:
    array: The solution of the system of equations.
    """
    # set standard values for the optional keyword arguments
    kwargs.setdefault('maxfev', 10000)
    kwargs.setdefault('x_tol',  1e-8)
    kwargs.setdefault('f_tol',  None)
    kwargs.setdefault('g_tol',  None)
    kwargs.setdefault('fprime', None)
    kwargs.setdefault('fprime2', None)
    kwargs.setdefault('epsfcn', 1e-12)
    kwargs.setdefault('bounds', (-np.inf, np.inf))
    kwargs.setdefault('n_iter', 10000)
    kwargs.setdefault('iter_abs_tol', 1e-8)
    kwargs.setdefault('iter_rel_tol', 1e-2)



    if method == 'least_squares':
        # Define the bounds for the solution
        bounds = kwargs.get('bounds', (-np.inf, np.inf))
        # Solve the system of equations
        res = spo.least_squares(f, x0, bounds=kwargs.get('bounds'),method='dogbox', x_scale=kwargs.get('x_scale'), xtol=kwargs.get('x_tol'), ftol=kwargs.get('f_tol'), jac=kwargs.get('fprime'), max_nfev=kwargs.get('maxfev'), gtol=kwargs.get('g_tol'))
        return res.x
    
    if method == 'fsolve':
        # Solve the system of equations
        res = spo.fsolve(f, x0, maxfev=kwargs.get('maxfev'), xtol=kwargs.get('x_tol'), fprime=kwargs.get('fprime'), epsfcn=kwargs.get('epsfcn'))
        return res
    
    if method == 'newton':
        # Solve the system of equations
        res = spo.newton(f, x0, fprime=kwargs.get('fprime'), maxiter=kwargs.get('maxfev'), tol=kwargs.get('x_tol'))
        return res
    
    if method == 'minimize_scalar':
        # Solves the 1D problem
        res = spo.minimize_scalar(f)
        return res.x

    if method == 'brenth':
        # Solves the 1D problem
        res = spo.brenth(f, x0[0], x0[1])
        return res
    
    if method == 'root_scalar':
        # Solves the 1D problem
        res = spo.root_scalar(f, fprime=kwargs.get('fprime'), x0=x0, fprime2=kwargs.get('fprime2'))
        return res.root
    
    if method == 'brute':
        # Define the bounds for the solution
        # Solve the system of equations
        res = spo.brute(f, bounds=kwargs.get('bounds'), Ns=kwargs.get('Ns'))
        return res
    
    if method == 'iterative':
        # Solve the system of equations
        x = x0
        abs_err = 1.0
        rel_err = 1.0
        idx = 0
        while abs_err > kwargs.get('iter_abs_tol') and rel_err > kwargs.get('iter_rel_tol') and idx < kwargs.get('n_iter'):
            x_new = f(x)
            abs_err = np.linalg.norm(x_new - x,ord=np.inf)
            rel_err = np.linalg.norm(2*(x_new - x)/(np.abs(x_new) + np.abs(x)),ord=np.inf)
            x = x_new
            idx += 1
        if idx == kwargs.get('n_iter'):
            print("Maximum number of iterations reached. Maximum relative and absolute error: ", rel_err, abs_err)
        return x


##################################################################################
######### Here we have functions relevant for the general calculation ############
##################################################################################
def w_star(w_n,w_T,v_perp2,v_par2):
    """
    Returns the diamagnetic drift frequency w_star
    """
    return - w_n - w_T*(v_perp2 + v_par2 - 3/2)



def G(k_psi,k_alpha,w_star,w_alpha,w_psi,v_perp2):
    """
    Returns the function G
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



def integrand_k_psi(k_psi,k_alpha,w_alpha,w_psi,w_n,w_T,v_perp2,v_par2):
    """
    Returns the integrand for the k_psi integral
    """
    w_star_val = w_star(w_n,w_T,v_perp2,v_par2)
    return G(k_psi,k_alpha,w_star_val,w_alpha,w_psi,v_perp2)*(v_perp2 * w_psi - k_psi)



def integrand_k_alpha(k_psi,k_alpha,w_alpha,w_psi,w_n,w_T,v_perp2,v_par2):
    """
    Returns the integrand for the k_alpha integral
    """
    w_star_val = w_star(w_n,w_T,v_perp2,v_par2)
    return G(k_psi,k_alpha,w_star_val,w_alpha,w_psi,v_perp2)*(v_perp2 * w_alpha + k_alpha)



def equations_k(k_psi,k_alpha,w_alpha,w_psi,w_n,w_T):
    """
    Returns the LHS of the equations for k_psi and k_alpha.
    When the LHS is zero, the equations are satisfied.
    """
    # first define 2D function for the integrand of k_psi
    int_k_psi = lambda v_perp2,v_par2: integrand_k_psi(k_psi,k_alpha,w_alpha,w_psi,w_n,w_T,v_perp2,v_par2)
    # calculate the integral
    kappa_psi_LHS = integrate_2D_AE_weighting(int_k_psi)/np.sqrt(np.pi) - w_psi
    # now define 2D function for the integrand of k_alpha
    int_k_alpha = lambda v_perp2,v_par2: integrand_k_alpha(k_psi,k_alpha,w_alpha,w_psi,w_n,w_T,v_perp2,v_par2)
    # calculate the integral
    kappa_alpha_LHS = integrate_2D_AE_weighting(int_k_alpha)/np.sqrt(np.pi) - w_alpha - w_n
    return kappa_psi_LHS, kappa_alpha_LHS



def equations_k_iter(k_psi,k_alpha,w_alpha,w_psi,w_n,w_T):
    """
    Returns k_psi and k_alpha from the equations for k_psi and k_alpha.
    """
    # define G function as function of v_perp and v_par
    G_func = lambda v_perp2,v_par2: G(k_psi,k_alpha,w_star(w_n,w_T,v_perp2,v_par2),w_alpha,w_psi,v_perp2)
    # first define 2D function for the integrand of k_psi
    int_k_psi_1 = lambda v_perp2,v_par2: G_func(v_perp2,v_par2)*v_perp2*w_psi
    int_k_psi_2 = lambda v_perp2,v_par2: G_func(v_perp2,v_par2) 
    # calculate the integral
    kappa_psi = (integrate_2D_AE_weighting(int_k_psi_1)/np.sqrt(np.pi) - w_psi)/(integrate_2D_AE_weighting(int_k_psi_2)/np.sqrt(np.pi))
    # now define 2D function for the integrand of k_alpha
    int_k_alpha_1 = lambda v_perp2,v_par2: G_func(v_perp2,v_par2)*v_perp2*w_alpha
    int_k_alpha_2 = lambda v_perp2,v_par2: G_func(v_perp2,v_par2)
    kappa_alpha = (w_n + w_alpha - integrate_2D_AE_weighting(int_k_alpha_1)/np.sqrt(np.pi))/(integrate_2D_AE_weighting(int_k_alpha_2)/np.sqrt(np.pi))
    return kappa_psi, kappa_alpha



def solve_k(w_alpha,w_psi,w_n,w_T,method="iterative",**kwargs):
    """
    Returns the solutions for k_psi and k_alpha.
    """
    # set standard values for the optional keyword arguments
    kwargs.setdefault('t_res_homotopy', 10)
    kwargs.setdefault('plot_traj', False)
    kwargs.setdefault('isodynamic', 'iso')
    kwargs.setdefault('verbose', False)


    if w_psi == 0.0:
        k_psi = 0.0
        k_alpha = solve_k_alpha_iso(w_alpha,w_n,w_T)

    elif w_psi != 0.0:
        if method == "homotopy":
            # initial guess, solution to the homotopy equations for t = 0
            # we now slowly increase t to 1, to find the solution to the original equations
            t_res_homotopy = kwargs.get('t_res_homotopy')
            # choose the simple equation we wish to solve (either isodynamic or density)
            # here's the isodynamic solution
            if kwargs.get('homotopy_initial_guess') == 'isodynamic':
                eq_solve = lambda k,t: equations_k(k[0],k[1],w_alpha,w_psi*t,w_n,w_T)
                t_arr = np.linspace(0,1,t_res_homotopy+1)
                # store trajectory of k_psi and k_alpha
                k_alpha_traj = np.zeros_like(t_arr)
                k_psi_traj = np.zeros_like(t_arr)
                # set the initial guess, the isodynamic solution
                k_alpha_traj[0] = solve_k_alpha_iso(w_alpha,w_n,w_T, stability_return=False)
                k_psi_traj[0] = 0.0
            # here's the density solution
            elif kwargs.get('homotopy_initial_guess') == 'density':
                eq_solve = lambda k,t: equations_k(k[0],k[1],w_alpha,w_psi,w_n,w_T*t)
                t_arr = np.linspace(0,1,t_res_homotopy+1)
                # store trajectory of k_psi and k_alpha
                k_alpha_traj = np.zeros_like(t_arr)
                k_psi_traj = np.zeros_like(t_arr)
                # set the initial guess, density solution
                k_alpha_traj[0] = w_n
                k_psi_traj[0] = 0.0
            for i,t in enumerate(t_arr):
                if i == 0:
                    continue
                # solve the equations
                k = solver(lambda k: eq_solve(k,t),[k_psi_traj[i-1],k_alpha_traj[i-1]])
                k_psi_traj[i] = k[0]
                k_alpha_traj[i] = k[1]
            if kwargs.get('plot_traj'):
                # plot the trajectory
                import matplotlib.pyplot as plt
                # enable latex
                plt.rc('text', usetex=True)
                plt.plot(t_arr,k_alpha_traj,label=r'$\kappa_\alpha$')
                plt.plot(t_arr,k_psi_traj,label=r'$\kappa_\psi$')
                plt.legend()
                plt.xlabel('t')
                plt.ylabel('k')
                plt.xlim(0,1)
                plt.show()
            k_psi = k[0]
            k_alpha = k[1]

        if method=="direct":
            # initial guess at isodynamic solution, and w_psi as initial guess for k_psi
            k_psi_0 = w_psi
            k_alpha_0 = solve_k_alpha_iso(w_alpha,w_n,w_T, stability_return=False)[0]
            print(k_psi_0, k_alpha_0)

            # solve the equations
            k0 = np.asarray([k_psi_0,k_alpha_0])
            k = solver(lambda k: equations_k(k[0],k[1],w_alpha,w_psi,w_n,w_T),k0, method='fsolve')
            k_psi = k[0]
            k_alpha = k[1]

        if method=='iterative':
            # initial guess at drifts
            k_psi_0 = w_psi*np.sqrt(2)
            k_alpha_0 = w_alpha
            k0 = np.asarray([k_psi_0,k_alpha_0])
            # solve the equations
            eq = lambda k: np.asarray(equations_k_iter(k[0],k[1],w_alpha,w_psi,w_n,w_T))
            k = solver(eq,k0,method='iterative')
            k_psi = k[0]
            k_alpha = k[1]





    return k_psi, k_alpha



def stability_boolean(w_alpha,w_psi,w_n,w_T):
    """
    Returns a boolean indicating stability.
    """
    if w_T == 0.0:
        return True 
    elif w_psi != 0.0:
        return False
    elif w_alpha == 0.0 and w_psi == 0.0:
        return True
    # eta between 0 and 2/3, and eta > eta_B in isodynamic limit
    # construct eta
    elif w_n == 0.0:
        return False  
    elif (0 <=  w_T/w_n <= 2/3) and (w_T/w_n >= -w_alpha/w_n):
        return True
    else:
        return False



def calculate_AE(w_alpha,w_psi,w_n,w_T, **kwargs):
    """
    Returns the AE for a given magnetic field.
    """
    # set standard values for the optional keyword arguments
    kwargs.setdefault('method', 'trapz')
    kwargs.setdefault('n_gauss_laguerre', 20)
    kwargs.setdefault('x_max_bounded', 10)
    kwargs.setdefault('y_max_bounded', 10)
    kwargs.setdefault('n_trapz',   100* int((kwargs['x_max_bounded']+kwargs['y_max_bounded'])/2))
    kwargs.setdefault('n_simpson', 100* int((kwargs['x_max_bounded']+kwargs['y_max_bounded'])/2))
    kwargs.setdefault('k_psi', None)
    kwargs.setdefault('k_alpha', None)

    # check stability
    if stability_boolean(w_alpha,w_psi,w_n,w_T):
        dict_AE = {'AE': 0.0, 'k_alpha': np.nan, 'k_psi': np.nan}
    else:
        if w_psi==0.0:
            k_psi = 0.0
            k_alpha = solve_k_alpha_iso(w_alpha,w_n,w_T)
        else:
            k_psi, k_alpha = solve_k(w_alpha,w_psi,w_n,w_T)
        # define the integrand
        integrand = lambda v_perp2,v_par2: AE_integrand(v_perp2,v_par2,w_alpha,w_psi,w_n,w_T,k_psi,k_alpha)/(12*np.sqrt(np.pi))
        # calculate the integral
        integral = integrate_2D_AE_weighting(integrand)
        dict_AE = {'AE': integral, 'k_alpha': k_alpha, 'k_psi': k_psi}
    return dict_AE


##################################################################################
######### Here we have functions relevant for the isodynamic calculation #########
##################################################################################
def _special_erf(x,b):
    if b < -1:
        return 2*sp.erfi(np.sqrt(-1-b)*x)/np.sqrt(-1-b)
    if b > -1:
        return 2*sp.erf(np.sqrt(1+b)*x)/np.sqrt(1+b)
    if b == -1:
        return 4*x/np.sqrt(np.pi)
    else:
        # raise an error if b is not a number
        raise ValueError("b must be a number")



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



def H_iso(y,b, warnings=False):
    """
    Returns the function H
    """
    # this one will often raise warnings when, which we can suppress
    if not warnings:
        np.seterr(all='ignore')
    # if y is not an array, make it an array
    if not isinstance(y, np.ndarray):
        y = np.array([y])
    # Different from the paper by the definition of H_paper(log(y),b)=H_iso(y,b)
    # This is done to avoid overflow errors with np.exp(x) when x is large.
    # mathematica FortranForm: (-2 + b + (2 - b + 2*b*Log(y))/y**b)/2.
    ans = (-2 + b + (2 - b + 2*b*np.log(y))/y**b)/2
    mask = y > 1
    # mathematica FortranForm: (2*Erf(Sqrt((1 + b)*Log(y))))/Sqrt(1 + b) + ((-2*b*Sqrt(Log(y)))/(Sqrt(Pi)*y)                        + Erf(Sqrt(Log(y)))*(-2 + b - 2*b*Log(y)))/y**b
    ans[mask] = ans[mask] + (_special_erf(np.sqrt(np.log(y[mask])),b) + ( (-2*b*np.sqrt(np.log(y[mask])))/(np.sqrt(np.pi)*y[mask]) + sp.erf(np.sqrt(np.log(y[mask])))*(-2 + b - 2*b*np.log(y[mask])) )/y[mask]**b )
    return ans



def H_inf(b):
    """
    Returns the function H_iso evaluated at v_perp = infinity
    """
    if b < 0:
        ans = -1 + b/2 
    if b> 0:
        ans = -1 + b/2 + 2 / np.sqrt(1+b)
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



def equation_tilde_k_alpha_iso_II(kappa_alpha_tilde,w_alpha,w_n,w_T):
    """
    Returns the equation for the isodynamic limit.
    """
    eta = w_T/w_n
    eta_B = -w_alpha/w_n
    b = 1/(eta_B/eta - 1)
    a = (3/2 - 1/eta)*b
    lower_bound = v_0_squared(0,w_n,w_T,w_alpha)
    lower_val   = H_iso(np.exp(lower_bound),b)[0]
    upper_val   = H_inf(b)
    mid_val     = v_0_squared(kappa_alpha_tilde,w_n,w_T,w_alpha)
    varying_part   = H_iso(np.exp(mid_val),b)[0]
    LHS =  (- 2 * varying_part + lower_val + upper_val)*np.exp(a)/b
    RHS = np.sign(w_alpha*w_T) * 0.5 * ( -2 - 2 * a + b )/b
    return LHS - RHS



def equation_tilde_k_alpha_iso_III(x,w_alpha,w_n,w_T):
    """
    Returns the equation for the isodynamic limit.
    """
    eta             = w_T/w_n
    eta_B           = -w_alpha/w_n
    b               = 1/(eta_B/eta - 1)
    a               = (3/2 - 1/eta)*b
    lower_bound     = v_0_squared(0,w_n,w_T,w_alpha)
    lower_val       = H_iso(np.exp(lower_bound),b)[0]
    upper_val       = H_inf(b)
    varying_part    = H_iso(np.exp(x),b)[0]
    S               = np.sign(w_alpha*w_T) * (-1 + b/2 - a) * np.exp(-a)
    LHS = varying_part
    RHS = (lower_val + upper_val - S) /2
    return (LHS - RHS)



def solve_k_alpha_iso(w_alpha,w_n,w_T,stability_return=True, **kwargs):
    """
    Returns the solution for k_alpha.
    """
    # set standard values for the optional keyword arguments
    kwargs.setdefault('equation', 'I')

    if stability_boolean(w_alpha,0.0,w_n,w_T) and stability_return:
        k_alpha = np.nan

    # initial guess at -w_n/w_alpha
    elif kwargs.get('equation') == 'I':
        k_alpha_0 = np.max([-w_n/w_alpha,0.0])
        # solve the equation
        b = 1/(w_alpha/w_T + 1)
        a = (3/2 - w_n/w_T)*b
        fprime = lambda k: - 2 * np.exp(-k) * I_iso(a/b + k/b)
        k_alpha = solver(lambda k: equation_tilde_k_alpha_iso(np.abs(k),w_alpha,w_n,w_T),k_alpha_0, method='fsolve', fprime=None)
        # abs to avoid negative values
        k_alpha = np.abs(k_alpha)
        k_alpha = - k_alpha * w_alpha

    elif kwargs.get('equation') == 'II':
        # solve equation
        b = 1/(w_alpha/w_T + 1)
        a = (3/2 - w_n/w_T)*b
        k_alpha_0 = -w_n/w_alpha
        k_alpha_0 = np.max([-w_n/w_alpha,0.0])
        y0 = 1.0
        #fprime = lambda y: (- 2 * b * np.exp(-b*y) * I_iso(a/b + y/b))*np.exp(a)
        y = solver(lambda y: equation_tilde_k_alpha_iso_II(np.abs(y),w_alpha,w_n,w_T),x0=y0, method='fsolve')
        k_alpha_tilde = np.abs(y)
        k_alpha = - k_alpha_tilde * w_alpha


    elif kwargs.get('equation') == 'III':
        # solve equation
        eta             = w_T/w_n
        eta_B           = -w_alpha/w_n
        b               = 1/(eta_B/eta - 1)
        a               = (3/2 - 1/eta)*b
        S = np.sign(w_alpha*w_T) * (-1 + b/2 - a) * np.exp(-a)
        lower_bound     = v_0_squared(0,w_n,w_T,w_alpha)
        lower_val       = H_iso(np.exp(lower_bound),b)[0]
        upper_val       = H_inf(b)
        x_bounds = [lower_bound, np.sign(b)*np.inf]
        # sort the bounds
        x_bounds.sort()
        # two first guesses according to asymptotic behavior
        y = (lower_val + upper_val - S)/2
        x_guess = np.asarray(v_0_squared(np.geomspace(1e-3,1e3,100),w_n,w_T,w_alpha))
        # check which guess is better
        sols = [equation_tilde_k_alpha_iso_III(x,w_alpha,w_n,w_T) for x in x_guess]
        # choose the best guess
        x0 = x_guess[np.nanargmin(np.abs(sols))]
        # solve the equation
        fprime = lambda x: b**2 * np.exp(-b*x) * I_iso(x)
        x = solver(lambda x: equation_tilde_k_alpha_iso_III(x,w_alpha,w_n,w_T),x0=x0, method='least_squares', bounds=(x_bounds[0],x_bounds[1]), x_scale=1/b**2, fprime=fprime)
        # convert x to k_alpha_tilde
        k_alpha_tilde = v_perp_squared(x,a,b)
        k_alpha = - k_alpha_tilde * w_alpha


    return k_alpha



def calculate_AE_iso(w_alpha,w_n,w_T, **kwargs):
    """
    Returns the available energy for the isodynamic case.
    """
    # set standard values for the optional keyword arguments
    kwargs.setdefault('method', 'trapz')

    # check stability
    if stability_boolean(w_alpha,0.0,w_n,w_T):
        ae_dict = {'AE': 0.0, 'k_alpha': np.nan}


    else:
        k_alpha_iso = solve_k_alpha_iso(w_alpha,w_n,w_T)
        tilde_k_alpha = -k_alpha_iso/w_alpha
        # define the integrand
        v_0_squared = lambda v_perp_squared: 3/2 - w_n/w_T - (w_alpha/w_T + 1)*v_perp_squared
        Omega = lambda v_perp_squared: - w_T * w_alpha * ( v_perp_squared - tilde_k_alpha)
        integrand = lambda v_perp_squared: J_iso(v_0_squared(v_perp_squared),Omega(v_perp_squared)) * np.exp(-v_perp_squared)/6
        # calculate the integral
        if kwargs.get('method') == "trapz":
            # define the grid
            v_perp_squared = np.linspace(0,30,1000*30)
            # calculate the integral
            integral = np.trapezoid(integrand(v_perp_squared),v_perp_squared)

        elif kwargs.get('method') == "quad":
            integral = spi.quad(integrand,0,np.inf)[0]

        ae_dict = {'AE': integral, 'k_alpha': k_alpha_iso}

    # we may encounter catastrophic cancellation, where AE is negative due to numerical errors. Here
    # we set it to np.abs(AE) if it is negative
    if ae_dict['AE'] < 0:
        ae_dict['AE'] = np.abs(ae_dict['AE'])
    
    return ae_dict


##################################################################################
######### Here we have functions relevant for the strongly driven limit #########
##################################################################################
def v_0_squared_strong(v_perp_squared,w_n,w_T):
    """
    Returns the function v_0_squared
    """
    return 3/2 - w_n/w_T - v_perp_squared



def stability_boolean_strong(w_alpha,w_psi,w_n,w_T):
    # only is eta is between 0 and 2/3 
    if (0 <=  w_T/w_n <= 2/3):
        return True
    else:
        return False



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
    integrand = lambda v_perp2:  I_iso(v_0_squared_strong(v_perp2,w_n,w_T)) * np.exp(-v_perp2)
    # calculate the LHS
    LHS = -spi.quad(integrand,0,kappa_alpha_tilde)[0] + spi.quad(integrand,kappa_alpha_tilde,np.inf)[0]
    RHS = (w_n)* np.sign(w_alpha) /np.abs(w_T)
    return LHS - RHS



def equation_k_iter_strong(kappa_psi,kappa_alpha,w_alpha,w_psi,w_n,w_T):
    """
    Returns k_psi and k_alpha from the equations for k_psi and k_alpha.
    """
    # set the kwarg method to trapz
    kwargs = {'method': 'simpson'}


    v0_2 = lambda v_perp2:  v_0_squared_strong(v_perp2,w_n,w_T)
    denom = lambda v_perp2: np.sqrt( (kappa_alpha + v_perp2 * w_alpha)**2 + (v_perp2 * w_psi - kappa_psi)**2 )
    integrand_k_alpha_1 = lambda v_perp2: np.exp(-v_perp2) * ( -I_1(v0_2(v_perp2))*np.sign(w_T) - I_iso(v0_2(v_perp2)) * v_perp2 * w_alpha/denom(v_perp2) )
    integrand_k_alpha_2 = lambda v_perp2: np.exp(-v_perp2) * ( I_iso(v0_2(v_perp2)) / denom(v_perp2) )
    integrand_k_psi_1 = lambda v_perp2: np.exp(-v_perp2) * ( I_iso(v0_2(v_perp2)) * v_perp2 * w_psi/denom(v_perp2) )
    # calculate the integral
    if kwargs.get('method') == "trapz":
        # define the grid
        v_perp2 = np.linspace(0,30,1000*30)
        # calculate the integral
        denominator = np.trapezoid(integrand_k_alpha_2(v_perp2),v_perp2)
        kappa_alpha_new = np.trapezoid(integrand_k_alpha_1(v_perp2),v_perp2)/denominator
        kappa_psi_new = np.trapezoid(integrand_k_psi_1(v_perp2),v_perp2)/denominator

    elif kwargs.get('method') == "quad":
        denominator = spi.quad(integrand_k_alpha_2,0,np.inf)[0]
        kappa_alpha_new = (spi.quad(integrand_k_alpha_1,0,np.inf)[0])/denominator
        kappa_psi_new = (spi.quad(integrand_k_psi_1,0,np.inf)[0])/denominator

    elif kwargs.get('method') == "simpson":
        # define the grid
        v_perp2 = np.linspace(0,30,200*30)
        # calculate the integral
        denominator = spi.simpson(integrand_k_alpha_2(v_perp2),x=v_perp2)
        kappa_alpha_new = spi.simpson(integrand_k_alpha_1(v_perp2),x=v_perp2)/denominator
        kappa_psi_new = spi.simpson(integrand_k_psi_1(v_perp2),x=v_perp2)/denominator
    return kappa_psi_new, kappa_alpha_new



def solve_k_strong(w_alpha,w_psi,w_n,w_T, **kwargs):
    """
    Returns the solutions for k_psi and k_alpha.
    """
    # set standard values for the optional keyword arguments
    kwargs.setdefault('verbose', False)
    
    if w_psi == 0.0:
        k_psi = 0.0
        # use kappa_alpha_tilde equation to find kappa_alpha in strongly driven limit
        eq = lambda kappa_alpha_tilde: equation_isodynamic_strong(np.abs(kappa_alpha_tilde),w_alpha,w_n,w_T)
        k_alpha = solver(eq,0.0, method='fsolve')
        # convert to k_alpha
        k_alpha = -np.abs(k_alpha) * w_alpha
    
    elif w_psi != 0.0:
        
    
        # initial guess at drifts
        k_psi_0 = w_psi
        k_alpha0 = w_alpha
        # solve the equations
        k_0 = np.asarray([k_psi_0,k_alpha0])
        # solve with iterative method
        eq = lambda k: np.asarray(equation_k_iter_strong(k[0],k[1],w_alpha,w_psi,w_n,w_T))
        k = solver(eq,k_0,method='iterative')
        k_psi = k[0]
        k_alpha = k[1]

    return k_psi, k_alpha



def available_energy_strong_integrand(v_perp2,w_alpha,w_psi,w_n,w_T,k_alpha,k_psi, **kwargs):
    """
    Returns the integrand for the available energy in the strongly driven limit.
    """

    # define the integrand
    v0_2 = v_0_squared_strong(v_perp2,w_n,w_T)
    I_1_val = I_1(v0_2)
    I1_fac = (k_alpha + v_perp2 * w_alpha) * np.sign(w_T)
    I_val = I_iso(v0_2)
    sqrt_fac = np.sqrt( (k_alpha + v_perp2 * w_alpha)**2 + (v_perp2 * w_psi - k_psi)**2 )
    full_integrand = I_1_val*I1_fac + I_val * sqrt_fac
    return np.abs(w_T) * full_integrand * np.exp(-v_perp2)/12



def calculate_AE_strong(w_alpha,w_psi,w_n,w_T, **kwargs):
    """
    Returns the available energy for the strongly driven case.
    """
    # set standard values for the optional keyword arguments
    kwargs.setdefault('method', 'simpson')

    # check stability
    if stability_boolean_strong(w_alpha,w_psi,w_n,w_T):
        ae_dict = {'AE': 0.0, 'k_alpha': np.nan, 'k_psi': np.nan}
    else:
        k_psi, k_alpha = solve_k_strong(w_alpha,w_psi,w_n,w_T)
        # define the integrand
        integrand = lambda v_perp2: available_energy_strong_integrand(v_perp2,w_alpha,w_psi,w_n,w_T,k_alpha,k_psi)
        # calculate the integral
        if kwargs.get('method') == "trapz":
            # define the grid
            v_perp2 = np.linspace(0,30,1000*30)
            # calculate the integral
            integral = np.trapezoid(integrand(v_perp2),v_perp2)

        elif kwargs.get('method') == "quad":
            integral = spi.quad(integrand,0,np.inf)[0]

        elif kwargs.get('method') == "simpson":
            # define the grid
            v_perp2 = np.linspace(0,30,200*30)
            # calculate the integral
            integral = spi.simpson(integrand(v_perp2),x=v_perp2)

        ae_dict = {'AE': integral, 'k_alpha': k_alpha, 'k_psi': k_psi}

    # in isodynamic limit, one may encounter catastrophic cancellation resulting in AE < 0.
    # in such cases, we replace AE with its absolute value
    if ae_dict['AE'] < 0:
        ae_dict['AE'] = np.abs(ae_dict['AE'])

    return ae_dict

