# Here we define all functions relevant to the calculation of the AE for a given magnetic field.
import numpy as np
import scipy.special as sp
import scipy.integrate as spi
import scipy.optimize as spo
from numpy.polynomial.chebyshev import chebgauss



def newton_cotes_weights_uniform(N, order, avoid_edges = True):
    """
    Compute the weights for Newton-Cotes quadrature on a uniform grid.
    
    Parameters:
        N (int): number of grid points for uniform grid. If Simpson or Boole round to the right 4*prder+1 value
        order (int): Order of the Newton-Cotes rule (1 to higher orders).
        
    Returns:
        weights (array-like): The corresponding weights for the given order.
    """
   
    if order == 0:  # Midpoint Rule (order 0)
        if avoid_edges:
            x = np.linspace(0, 1, N+2)
            x = x[1:-1]
        else:
            x = np.linspace(0, 1, N)
        h = x[1] - x[0]  # uniform grid spacing

        w = np.zeros(N)
        w[:] = h

    elif order == 1:  # Trapezoidal Rule (order 1)
        if avoid_edges:
            x = np.linspace(0, 1, N+2)
            x = x[1:-1]
        else:
            x = np.linspace(0, 1, N)
        h = x[1] - x[0]  # uniform grid spacing
        
        w = np.zeros(N)
        w[0] = h / 2
        w[-1] = h / 2
        w[1:-1] = h
    
    elif order == 2:  # Simpson's Rule (order 2)
        # Need to make sure that N is made odd
        N -= 1 - (N % 2)
        if avoid_edges:
            x = np.linspace(0, 1, N+2)
            x = x[1:-1]
        else:
            x = np.linspace(0, 1, N)
        h = x[1] - x[0]  # uniform grid spacing
        
        w = np.zeros(N)
        w[0] = w[-1] = h / 3
        w[1:-1:2] = 4 * h / 3  # Odd-index points
        w[2:-1:2] = 2 * h / 3  # Even-index points
    
    elif order == 4:  # Boole's Rule (order 4)
        # Need to make sure that N is made odd
        N -= 1 + (N % 4)
        if avoid_edges:
            x = np.linspace(0, 1, N+2)
            x = x[1:-1]
        else:
            x = np.linspace(0, 1, N)
        h = x[1] - x[0]  # uniform grid spacing

        w = np.zeros(N)
        w[0] = w[-1] = 14 * h / 45
        w[1::2] = 64 * h / 45
        w[2::4] = 8 * h / 15
        w[4::4] = 28 * h / 45
    else:
        raise ValueError("Only orders, 0, 1, 2, and 4 are implemented for uniform grid.")

    return x, w

def make_clenshaw_curtis_nodes_and_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of the Clenshaw-Curtis quadrature. From https://github.com/inducer/modepy/blob/main/modepy/quadrature/clenshaw_curtis.py """
    if n < 1:
        raise ValueError(f"Clenshaw-Curtis order must be at least 1: n = {n}")

    if n == 1:
        return np.array([-1, 1]), np.array([1, 1])

    N = np.arange(1, n, 2)  # noqa: N806
    r = len(N)
    m = n - r

    # Clenshaw-Curtis nodes
    x = np.cos(np.arange(0, n + 1) * np.pi / n)

    # Clenshaw-Curtis weights
    w = np.concatenate([2 / N / (N - 2), 1 / N[-1:], np.zeros(m)])
    w = 0 - w[:-1] - w[-1:0:-1]
    g0: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = -np.ones(n)
    g0[r] = g0[r] + n
    g0[m] = g0[m] + n
    g0 = g0 / (n**2 - 1 + (n % 2))
    w = np.fft.ifft(w + g0)
    assert np.allclose(w.imag, 0)

    wr = w.real
    return x, np.concatenate([wr, wr[:1]])

def chebgauss1(N):
    """
    Gauss-Chebyshev quadrature.

    Returns quadrature points xₖ and weights wₖ for the approximate evaluation
    of the integral ∫₋₁¹ f(x)/sqrt(1-x^2) dx ≈ ∑ₖ wₖ f(xₖ).

    Parameters
    ----------
    N : int
        Number of quadrature points.

    Returns
    -------
    x, w : tuple[np.ndarray]
        Shape (N, ).
        Quadrature points and weights.

    """
    x, w = chebgauss(N)         
    return x, w

def gaussjacobi(N, alpha, beta):
    """
    Gauss-Jacobi quadrature.

    Returns quadrature points xₖ and weights wₖ for the approximate evaluation
    of the integral ∫₋₁¹ f(x) (1-x)^α (1+x)^β dx ≈ ∑ₖ wₖ f(xₖ).

    Parameters
    ----------
    N : int
        Number of quadrature points.
    alpha : float
        Shape parameter.
    beta : float
        Shape parameter.

    Returns
    -------
    x, w : tuple[np.ndarray]
        Shape (N, ).
        Quadrature points and weights.

    """
    x, w = sp.roots_jacobi(N, alpha, beta)
    return x, w





def integrate_2D_AE_weighting(f, method='gaussian', **kwargs):
    """
    Integrate a function over a 2D domain. 

    This function integrates a given function over a 2D domain using a specified method. 
    Assumes that

    Parameters:
    f (function): The integrand *without* the exponential factor exp(-x)*exp(-y)/sqrt(y). Order of arguments is (x, y).
    method (str): The method to be used for integration. Default is 'guassian': best performance on most examples 
                  for lowest computational cost of the fixed-quadrature methods. For the adaptive quadrature methods, 
                  'dbltanhsinh_uv' is the best choice.

    Returns:
    float: The result of the integration.
    """

    # set standard values for the optional keyword arguments
    kwargs.setdefault('n_quad', 100)
    kwargs.setdefault('abs_tol', 1e-8)
    kwargs.setdefault('rel_tol', 1e-8)


    if method == 'gauss_laguerre':
        # Define the number of points for the Gauss-Laguerre. Mixed with general Gauss-Laguerre for y (alpha=-1/2)
        n = kwargs.get('n_quad')
        # Create the 2D grid
        # get the roots and weights in 1D
        x, x_w = sp.roots_genlaguerre(n, 0.0)
        y, y_w = sp.roots_genlaguerre(n, -0.5)
        # create the 2D grid
        X, Y = np.meshgrid(x, y)
        weights = np.outer(y_w, x_w)
        function = f(X, Y)

        # Calculate the integral
        integral = np.sum(weights * function)
    
    elif method == 'hermite_laguerre':
        # hermite-laguerre quadrature
        # by setting y = z^2, we have the weight exp(-x -y)/sqrt(y) f dx dy = 2 * exp(-x) * exp(-z^2) f dx dz
        # get the roots and weights in 1D
        n = kwargs.get('n_quad')
        x, x_w = sp.roots_laguerre(n)
        z, z_w = sp.roots_hermite(n)
        # create the 2D grid
        X, Z = np.meshgrid(x, z)
        weights = np.outer(z_w, x_w)
        # evaluate the function at the grid points
        function = f(X, Z**2) 

        # Calculate the integral
        integral = np.sum(weights * function)

    elif method == 'dblquad':
        # construct the integrand
        def integrand(x, y):
            return f(x, y) * np.exp(-x) * np.exp(-y) / np.sqrt(y)
        # integrate the function
        integral, _ = spi.dblquad(integrand, 0, np.inf, lambda x: 0, lambda x: np.inf, epsabs=kwargs.get('abs_tol'), epsrel=kwargs.get('rel_tol'))


    elif method == 'gaussian':
        # we map x [0,inf] to [0,1] using the transformation x = -log(1-u)
        # setting z = y**2, we have the weight exp(-y)/sqrt(y) dy = 2 * exp(-z**2) dz
        # then set z = erfinv(v), so that 2 * exp(-z**2) dz = sqrt(pi) dv
        # finally y = z**2 = erfinv(v)**2
        # the integral is then sqrt(pi) f(-log(1-u), erfinv(v)**2) du dv over [0,1]x[0,1]
        # we use a 2D Gaussian quadrature to evaluate the integral
        n = kwargs.get('n_quad')
        # get the roots and weights in 1D
        u, u_w = sp.roots_sh_legendre(n)
        v, v_w = u.copy(), u_w.copy()
        # create the 2D grid
        U, V = np.meshgrid(u, v)
        weights = np.outer(u_w, v_w)
        # evaluate the function at the grid points
        X = -np.log(1-U)
        Y = sp.erfinv(V)**2
        function = f(X, Y)
        # Calculate the integral
        integral = np.sqrt(np.pi) * np.sum(weights * function)

    elif method == 'midpoint':
        # use newton_cotes with order 0
        n = kwargs.get('n_quad')
        u, u_w = newton_cotes_weights_uniform(n, 0)
        v, v_w = u.copy(), u_w.copy()
        # create the 2D grid
        U, V = np.meshgrid(u, v)
        weights = np.outer(u_w, v_w)
        # evaluate the function at the grid points
        X = -np.log(1-U)
        Y = sp.erfinv(V)**2
        function = f(X, Y)
        # Calculate the integral
        integral = np.sqrt(np.pi) * np.sum(weights * function)


    elif method == 'trapz':
        # use newton_cotes with order 1
        n = kwargs.get('n_quad')
        u, u_w = newton_cotes_weights_uniform(n, 1)
        v, v_w = u.copy(), u_w.copy()
        # create the 2D grid
        U, V = np.meshgrid(u, v)
        weights = np.outer(u_w, v_w)
        # evaluate the function at the grid points
        X = -np.log(1-U)
        Y = sp.erfinv(V)**2
        function = f(X, Y)
        # Calculate the integral
        integral = np.sqrt(np.pi) * np.sum(weights * function)

    
    elif method == 'simpson':
        # use newton_cotes with order 2
        n = kwargs.get('n_quad')
        u, u_w = newton_cotes_weights_uniform(n, 2)
        v, v_w = u.copy(), u_w.copy()
        # create the 2D grid
        U, V = np.meshgrid(u, v)
        weights = np.outer(u_w, v_w)
        # evaluate the function at the grid points
        X = -np.log(1-U)
        Y = sp.erfinv(V)**2
        function = f(X, Y)
        # Calculate the integral
        integral = np.sqrt(np.pi) * np.sum(weights * function)


    elif method == 'boole':
        # use newton_cotes with order 4
        n = kwargs.get('n_quad')
        u, u_w = newton_cotes_weights_uniform(n, 4)
        v, v_w = u.copy(), u_w.copy()
        # create the 2D grid
        U, V = np.meshgrid(u, v)
        weights = np.outer(u_w, v_w)
        # evaluate the function at the grid points
        X = -np.log(1-U)
        Y = sp.erfinv(V)**2
        function = f(X, Y)
        # Calculate the integral
        integral = np.sqrt(np.pi) * np.sum(weights * function)
                                           

    elif method == 'dblquad_uv':
        # construct the integrand on the unit square
        def integrand(u, v):
            x = -np.log(1-u)
            y = sp.erfinv(v)**2
            return np.sqrt(np.pi) * f(x,y)
        # integrate the function
        integral, _ = spi.dblquad(integrand, 0, 1, lambda u: 0, lambda u: 1, epsabs=kwargs.get('abs_tol'), epsrel=kwargs.get('rel_tol'))


    elif method == 'dbltanhsinh_uv':
        # construct the integrand on the unit square
        def integrand(u, v):
            x = -np.log(1-u)
            y = sp.erfinv(v)**2
            return np.sqrt(np.pi) * f(x,y)
        # integrate the function using nested spi.tanhsinh (res = tanhsinh(f, 0, 1).integral)
        def integrand_u(u):
            integrals = np.zeros_like(u)
            for i, u_val in np.ndenumerate(u):
                res = spi.tanhsinh(lambda v: integrand(u_val, v), 0, 1, atol=kwargs.get('abs_tol'), rtol=kwargs.get('rel_tol'), minlevel=1)
                integrals[i] = res.integral
            return integrals
                
        res = spi.tanhsinh(integrand_u, 0, 1, atol=kwargs.get('abs_tol'), rtol=kwargs.get('rel_tol'), minlevel=1)
        integral = res.integral



    elif method == 'clenshaw_curtis':
        # construct the integrand on interval [-1,1]x[-1,1]
        # set a = 2u-1, b = 2v-1
        # then x = -log(1-(a+1)/2), y = erfinv((b+1)/2)**2
        # the Jacobian is dudv = 1/4 dxdy
        def integrand(a, b):
            x = -np.log(1-(a+1)/2)
            y = sp.erfinv((b+1)/2)**2
            return np.sqrt(np.pi) * f(x,y) / 4
        # integrate the function
        n = kwargs.get('n_quad')
        a, a_w = make_clenshaw_curtis_nodes_and_weights(n+2)
        b, b_w = a.copy(), a_w.copy()
        # exclude the edges
        a = a[1:-1]
        b = b[1:-1]
        a_w = a_w[1:-1]
        b_w = b_w[1:-1]
        # create the 2D grid
        A, B = np.meshgrid(a, b)
        weights = np.outer(a_w, b_w)
        function = integrand(A, B)
        # Calculate the integral
        integral = np.sum(weights * function)
        
    elif method == 'cheb_gauss':
        # construct the integrand on interval [-1,1]x[-1,1]
        # set a = 2u-1, b = 2v-1
        # then x = -log(1-(a+1)/2), y = erfinv((b+1)/2)**2
        # the Jacobian is dudv = 1/4 dxdy
        # we redefine f(x,y) = f(x,y) * sqrt(1-a^2)*sqrt(1-b^2) / sqrt(1-a^2)*sqrt(1-b^2)
        def integrand(a, b):
            x = -np.log(1-(a+1)/2)
            y = sp.erfinv((b+1)/2)**2
            return np.sqrt(np.pi) * f(x,y) * np.sqrt(1-a**2) * np.sqrt(1-b**2) / 4
        # integrate the function
        n = kwargs.get('n_quad')
        a, a_w = chebgauss1(n)
        b, b_w = a.copy(), a_w.copy()
        # create the 2D grid
        A, B = np.meshgrid(a, b)
        weights = np.outer(a_w, b_w)
        function = integrand(A, B)
        # Calculate the integral
        integral = np.sum(weights * function)
    

    elif method == 'gauss_jacobi':
        alpha = 0.05
        # construct the integrand on interval [-1,1]x[-1,1]
        # set a = 2u-1, b = 2v-1
        # then x = -log(1-(a+1)/2), y = erfinv((b+1)/2)**2
        # the Jacobian is dudv = 1/4 dxdy
        # we redefine f(x,y) = f(x,y) * (1 - a)^alpha * (1 - b)^alpha / (1 - a)^alpha * (1 - b)^alpha
        def integrand(a, b):
            x = -np.log(1-(a+1)/2)
            y = sp.erfinv((b+1)/2)**2
            return np.sqrt(np.pi) * f(x,y) * (1 - a)**alpha * (1 - b)**alpha / 4
        # integrate the function
        n = kwargs.get('n_quad')
        a, a_w = gaussjacobi(n, -alpha, 0.0)
        b, b_w = a.copy(), a_w.copy()
        # create the 2D grid
        A, B = np.meshgrid(a, b)
        weights = np.outer(a_w, b_w)
        function = integrand(A, B)
        # Calculate the integral
        integral = np.sum(weights * function)

    else:
        raise ValueError("Invalid integration method")


    return integral