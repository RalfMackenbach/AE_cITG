import numpy as np
import source.utils_ae_full as suaf
import source.utils_ae_iso as suai
import source.utils_ae_strong as suas

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
    

def stability_boolean_strong(w_alpha,w_psi,w_n,w_T):
    """
    Returns a boolean indicating stability.
    """
    if w_n ==0.0 and w_T == 0.0:
        return True
    elif w_n == 0.0:
        return False
    elif w_T == 0.0:
        return True 
    elif w_T/w_n >= 0.0 and w_T/w_n <= 2/3:
        return True
    else:
        return False


def calculate_AE_arr(w_T, w_n, w_alpha, w_psi):
    """
    Returns the AE, k_psi, and k_alpha arrays, given arrays of w_alpha, B, Delta_psi, Delta_alpha.
    """
    # check if w_alpha and w_psi have the same shape
    if w_alpha.shape != w_psi.shape:
        raise ValueError("All input arrays must be of the same shape.")
    # initialize arrays
    AE = np.ones_like(w_alpha)
    k_psi = np.ones_like(w_alpha)
    k_alpha = np.ones_like(w_alpha)
    # if w_T = 0, we know the solution is k_psi = 0 k_alpha = w_n and AE = 0
    if w_T == 0.0:
        AE = 0.0 * AE 
        k_psi = 0.0 * k_psi
        k_alpha = w_n * k_alpha
    # else we need to calculate the AE, k_psi, and k_alpha per point
    else:
        for idx, _ in np.ndenumerate(w_alpha):
            # extract values
            w_alpha_val = w_alpha[idx]
            w_psi_val = w_psi[idx]
            # check if w_psi is zero
            if w_psi_val == 0.0:
                k_psi_val = 0.0
                # solve the isodynamic problem
                # first check stability
                if stability_boolean(w_alpha_val, w_psi_val, w_n, w_T):
                    # if w_alpha is zero, k_alpha is zero
                    if w_alpha_val == 0.0:
                        k_alpha_val = 0.0
                    # otherwise, k_alpha is ill-defined
                    # elif w_T * w_alpha_val < 0.0, kappa_alpha is np.inf
                    elif w_T * w_alpha_val < 0.0:
                        k_alpha_val = np.inf
                    # otherwise, kappa_alpha is ill-defined
                    else:
                        k_alpha_val = np.nan
                    AE_val = 0.0
                else:
                    k_alpha_val = suai.solve_tilde_k_alpha_iso(w_alpha_val, w_n, w_T)
                    # if array, take first element
                    if isinstance(k_alpha_val, np.ndarray):
                        k_alpha_val = k_alpha_val[0]
                    # calculate AE
                    AE_val = suai.AE_local_iso(w_alpha_val, w_n, w_T, k_alpha_val)
            # otherwise solve the full problem
            else:
                k_psi_val, k_alpha_val = suaf.solve_k(w_alpha_val, w_psi_val, w_n, w_T)
                # calculate AE
                AE_val = suaf.AE_local(w_n, w_T, w_alpha_val, w_psi_val, k_psi_val, k_alpha_val)
            # store values
            AE[idx] = AE_val
            k_psi[idx] = k_psi_val
            k_alpha[idx] = k_alpha_val
    
    # check if any AE is negative
    if np.any(AE < 0.0):
        # print warning and set to zero
        negative_AE = AE < 0.0
        print(f"Warning: Negative AE values at {np.sum(negative_AE)} points, with values {AE[negative_AE]}")
        print(f"Setting negative AE values to zero.")
        AE[AE < 0.0] = 0.0

    # make a dictionary
    AE_dict = {'AE': AE, 'k_psi': k_psi, 'k_alpha': k_alpha}

    return AE_dict


def calculate_AE_strong_arr(w_T, w_n, w_alpha, w_psi):
    """
    Returns the AE, k_psi, and k_alpha arrays, given arrays of w_alpha, B, Delta_psi, Delta_alpha.
    """
    # check if w_alpha and w_psi have the same shape
    if w_alpha.shape != w_psi.shape:
        raise ValueError("All input arrays must be of the same shape.")
    # initialize arrays
    AE = np.ones_like(w_alpha)
    k_psi = np.ones_like(w_alpha)
    k_alpha = np.ones_like(w_alpha)
    # check stability
    if stability_boolean_strong(w_alpha, w_psi, w_n, w_T):
        # we know k_psi = 0, k_alpha = nan, and AE = 0
        AE = 0.0 * AE
        k_psi = 0.0 * k_psi
        k_alpha = np.nan * k_alpha
    else:
        for idx, _ in np.ndenumerate(w_alpha):
            # extract values
            w_alpha_val = w_alpha[idx]
            w_psi_val = w_psi[idx]
            # solve the strong problem
            k_psi_val, k_alpha_val = suas.solve_k_strong(w_alpha_val, w_psi_val, w_n, w_T)
            # calculate AE
            AE_val = suas.AE_local_strong(w_n, w_T, w_alpha_val, w_psi_val, k_alpha_val, k_psi_val)
            # store values
            AE[idx] = AE_val
            k_psi[idx] = k_psi_val
            k_alpha[idx] = k_alpha_val

    # check if any AE is negative
    if np.any(AE < 0.0):
        # print warning and set to zero
        negative_AE = AE < 0.0
        print(f"Warning: Negative AE values at {np.sum(negative_AE)} points, with values {AE[negative_AE]}")
        print(f"Setting negative AE values to zero.")
        AE[AE < 0.0] = 0.0

    # make a dictionary
    AE_dict = {'AE': AE, 'k_psi': k_psi, 'k_alpha': k_alpha}

    return AE_dict


def calculate_AE_iso_arr(w_T, w_n, w_alpha, w_psi):
    """
    Returns the AE, k_psi, and k_alpha arrays, given arrays of w_alpha, B, Delta_psi, Delta_alpha.
    """
    # initialize arrays
    AE = np.ones_like(w_alpha)
    k_psi = np.zeros_like(w_alpha)
    k_alpha = np.ones_like(w_alpha)
    # check stability
    for idx, _ in np.ndenumerate(w_alpha):
        # extract values
        w_alpha_val = w_alpha[idx]
        if stability_boolean(w_alpha_val, 0.0, w_n, w_T):
            # we know k_psi = 0, k_alpha = nan, and AE = 0
            AE = 0.0 * AE
            k_alpha = np.nan * k_alpha
        else:
            # solve the isodynamic problem
            k_alpha_val = suai.solve_tilde_k_alpha_iso(w_alpha_val, w_n, w_T)
            # if array, take first element
            if isinstance(k_alpha_val, np.ndarray):
                k_alpha_val = k_alpha_val[0]
            # calculate AE
            AE_val = suai.AE_local_iso(w_alpha_val, w_n, w_T, k_alpha_val)
            # store values
            AE[idx] = AE_val
            k_psi[idx] = 0.0
            k_alpha[idx] = k_alpha_val

    # check if any AE is negative
    if np.any(AE < 0.0):
        # print warning and set to zero
        negative_AE = AE < 0.0
        print(f"Warning: Negative AE values at {np.sum(negative_AE)} points, with values {AE[negative_AE]}")
        print(f"Setting negative AE values to zero.")
        AE[AE < 0.0] = 0.0

    # make a dictionary
    AE_dict = {'AE': AE, 'k_psi': k_psi, 'k_alpha': k_alpha}

    return AE_dict