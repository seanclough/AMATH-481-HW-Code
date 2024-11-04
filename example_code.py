# final version 1025midnight
# A5-A8
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define our rhsfunc2, similar to (a)
def rhsfunc2(x, y, k, gamma, eps):
    f1 = y[1] 
    f2 = (gamma * (abs(y[0])**2) + k * x**2 - eps) * y[0]
    dydt = [f1, f2]
    return dydt

# Variables
L = 2
xspan = np.arange(-L, L + 0.1, 0.1) # timestep
k = 1
tol = 1e-4

# Compute eigenstuffs for \gamma
def compute_eigen(gamma):

    A = 0.1 # launch angle
    eps_start = 0

    eig_vals = []
    eig_vecs = []

    for modes in range(1, 3): # mode 1 and mode 2
        eps = eps_start
        deps = 0.1 # step

        # Inner loop for epsilon shooting
        for i in range(1000):
            yinit = [A, A * np.sqrt(k * L**2 - eps)] # Initial conditions for y1(-L), y2(-L). later one refers to hw2

            # Use py's solver 
            sol = solve_ivp(rhsfunc2, [xspan[0], xspan[-1]], yinit, t_eval=xspan, args=(k, gamma, eps))
            y1 = sol.y[0] # y1 = phi 
            y2 = sol.y[1] # y2 = phi'

            # normalize y1 by trapz
            ynorm = np.trapz(y1**2, xspan)

            A /= np.sqrt(ynorm)                

            # Epsilon shooting adjustment based on boundary condition
            temp = y2[-1] + np.sqrt(L**2 - eps) * y1[-1] # Boundary condition

            if abs(temp) < tol and abs(ynorm - 1) < tol:
                print('epsilon = ' + str(eps))
                #print('A = ' + str(A))
                print('norm = ' + str(ynorm))
                break

            if (-1)**(modes + 1) * temp > 0:
                eps += deps
            else:
                eps -= deps / 2
                deps /= 2

        eig_vals.append(eps)
        eig_vecs.append(np.abs(y1))
        eps_start = eps + 0.1

    return np.column_stack(eig_vecs), np.array(eig_vals)

# Compute results for both gamma values
A5, A6 = compute_eigen(gamma=0.05) # For gamma = 0.05
A7, A8 = compute_eigen(gamma=-0.05) # For gamma = -0.05

# Print results
"""
print("A5 (eigenfunctions for gamma = 0.05):", A5)
print("A6 (eigenvalues for gamma = 0.05):", A6)
print("A7 (eigenfunctions for gamma = -0.05):", A7)
print("A8 (eigenvalues for gamma = -0.05):", A8)
"""

# Plot y1 against xspan for gamma = 0.05
plt.figure(figsize=(12, 6))
for i in range(A5.shape[1]):
    plt.plot(xspan, A5[:, i], label=f'Mode {i+1}')
plt.title('y1 vs xspan for gamma = 0.05')
plt.xlabel('xspan')
plt.ylabel('y1')
plt.legend()
plt.grid(True)
plt.show()

# Plot y1 against xspan for gamma = -0.05
plt.figure(figsize=(12, 6))
for i in range(A7.shape[1]):
    plt.plot(xspan, A7[:, i], label=f'Mode {i+1}')
plt.title('y1 vs xspan for gamma = -0.05')
plt.xlabel('xspan')
plt.ylabel('y1')
plt.legend()
plt.grid(True)
plt.show()