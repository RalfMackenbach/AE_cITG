import numpy as np
import scipy.special as sp
import scipy.optimize as spo


def max_rel_error(x_new, x):
    """
    Calculate the maximum relative error between two arrays.

    This function calculates the maximum relative error between two arrays.

    Parameters:
    x_new (array): The new array.
    x (array): The old array.

    Returns:
    float: The maximum relative error between the two arrays.
    """
    if np.any(np.abs(x) == 0):
        return np.inf
    else:
        return np.max(np.abs(x_new - x) / np.abs(x))
    

def solver(f, x0, method='iterative', **kwargs):
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
    kwargs.setdefault('abs_tol', 1.0e-5)
    kwargs.setdefault('rel_tol', 1.0e-5)
    kwargs.setdefault('max_iter', 10000)

    # convert x0 to numpy array
    x0 = np.asarray(x0)

    if method == 'iterative':
        # iterative solver x = f(x)
        abs_tol = kwargs.get('abs_tol')
        rel_tol = kwargs.get('rel_tol')
        x = x0
        x_new = f(x)
        max_err = np.max(np.abs(x_new - x))
        max_rel = max_rel_error(x_new, x)
        iter_count = 0
        while max_err > abs_tol and max_rel > rel_tol:
            x = x_new
            x_new = f(x)
            max_err  = np.max(np.abs(x_new - x))
            max_rel = max_rel_error(x_new, x)
            iter_count += 1
            if iter_count > kwargs.get('max_iter'):
                # print warning and break
                print('Warning: Maximum number of iterations reached. Current maximal relative and absolute error: {}, {}'.format(max_rel, max_err))
                break
        x_sol = x_new

    elif method == 'fsolve':
        # use scipy.optimize.fsolve
        x_sol = spo.fsolve(f, x0, xtol=kwargs.get('abs_tol'))
    
    elif method == 'least_squares':
        # use scipy.optimize.least_squares
        x_sol = spo.least_squares(f, x0, xtol=kwargs.get('abs_tol')).x

    elif method == 'newton_krylov':
        # use scipy.optimize.newton_krylov
        x_sol = spo.newton_krylov(f, x0, f_tol=kwargs.get('abs_tol'), f_rtol=kwargs.get('rel_tol'))

    elif method == 'broyden1':
        # use scipy.optimize.broyden1
        x_sol = spo.broyden1(f, x0, f_tol=kwargs.get('abs_tol'), f_rtol=kwargs.get('rel_tol'))

    elif method == 'broyden2':
        # use scipy.optimize.broyden2
        x_sol = spo.broyden2(f, x0, f_tol=kwargs.get('abs_tol'), f_rtol=kwargs.get('rel_tol'))

    else:
        raise ValueError('Invalid method: {}'.format(method))

    return np.asarray(x_sol)