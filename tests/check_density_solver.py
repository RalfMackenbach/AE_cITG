from source.AE_ITG import *

# test solve_k
print("="*50)
print("Testing solve_k",end="\n\n")

# define formatting, scientific notation
def num_trunc(x, end=" "):
    return "{:.5e}".format(x)+end

n_samples = 10
n=0
np.random.seed(0)
while n < n_samples:
    print("Sample",n+1)
    # generate random values for w_alpha, w_psi, w_n, w_T (fixed seed for reproducibility)
    # generate random number between -1 and 1
    w_alpha = np.random.uniform(-1,1)
    w_psi   = 0.1*np.random.uniform(-1,1)
    w_n     = np.random.uniform(-1,1)
    w_T     = np.random.uniform(-1,1)
    
    print('The following values are used for the test:')
    print('w_alpha:',num_trunc(w_alpha), 'w_psi:',num_trunc(w_psi), 'w_n:',num_trunc(w_n), 'w_T:',num_trunc(w_T), 'eta:',num_trunc(w_T/w_n), end="\n\n")




    # test solve_k
    k_psi, k_alpha = solve_k(w_alpha,w_psi,w_n,w_T)
    if isinstance(k_alpha,np.ndarray):
        k_alpha = k_alpha[0]
    if isinstance(k_psi,np.ndarray):
        k_psi = k_psi[0]
    print("Solution for k_psi and k_alpha:",num_trunc(k_psi),num_trunc(k_alpha), end="\n")
    sol1,sol2=equations_k(k_psi,k_alpha,w_alpha,w_psi,w_n,w_T)
    print("Solution for the equations:",num_trunc(sol1),num_trunc(sol2), end="\n\n")

    # check against isodynamic limit
    k_alpha_iso = solve_k_alpha_iso(w_alpha,w_n,w_T)
    # check if k_alpha_iso is an array, if so, take the first element
    if isinstance(k_alpha_iso,np.ndarray):
        k_alpha_iso = k_alpha_iso[0]
    kappa_psi_iso = 0.0
    print("Solution for k_alpha in the isodynamic limit:",num_trunc(k_alpha_iso), end="\n")
    print("Solution for the equation in the isodynamic limit:",num_trunc(equation_tilde_k_alpha_iso(-k_alpha_iso/w_alpha,w_alpha,w_n,w_T)), end="\n\n")

    # print difference between general and isodynamic solution
    print("Relative and absolute difference between general and isodynamic solution:")
    print("Abs diff k_alpha sols:",num_trunc(np.abs(k_alpha-k_alpha_iso)))
    print("Rel diff k_alpha sols:",num_trunc(np.abs((k_alpha-k_alpha_iso)/np.abs(k_alpha))))
    print("Abs diff kappa_psi sols:",num_trunc(np.abs(k_psi)),end="\n\n\n\n")

    n += 1
print("="*50)