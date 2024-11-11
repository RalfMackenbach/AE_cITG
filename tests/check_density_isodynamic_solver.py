from source.AE_ITG import *

# test solve_k

w_alpha = -1.0
w_n = 0.0
w_T = 2.0

k_sol = solve_k_alpha_iso(w_alpha,w_n,w_T)
print(k_sol)
print(equation_tilde_k_alpha_iso_II(k_sol,w_alpha,w_n,w_T))