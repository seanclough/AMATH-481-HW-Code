import numpy as np
from scipy.integrate import odeint
from scipy.integrate import RK45
import matplotlib.pyplot as plt

def shoot2(x_vect, x, K, epsilon):
    return [x_vect[1], (K*x**2-epsilon) * x_vect[0]]

tol = 1e-4  # define a tolerance level 
col = ['r', 'b', 'g', 'c', 'm']  # eigenfunc colors
eps_start = 0.5
eps_list = [] 
eig_func_list = []
A = 1
K = 1 
L = 4
xp = [-L, L] 
xspan = np.linspace(-L, L, int((2 * L) / 0.1) + 1)
#x0 = [A, A*np.sqrt(K*L**2)] #initial conditions
for modes in range(1, 6):  # begin mode loop
    eps = eps_start  # initial value of eigenvalue beta
    deps = eps_start / 100  # default step size in beta
    for _ in range(1000):  # begin convergence loop for beta
        x0 = [A, A*np.sqrt(K*L**2-eps)] #initial conditions
        y = odeint(shoot2, x0, xspan, args=(K,eps)) 
        # y = RK45(shoot2, xp[0], x0, xp[1], args=(n0,beta)) 

        if abs(y[-1, 1] + np.sqrt(K*L**2-eps)*y[-1,0]) < tol:  # final condition
            #print(eps)  # write out eigenvalue 
            eps_list.append(eps)
            #print(_)
            break  # get out of convergence loop

        if (-1) ** (modes + 1) * (y[-1, 1] + np.sqrt(K*L**2-eps)*y[-1,0]) > 0:
            eps += deps
        else:
            eps -= deps / 2
            deps /= 2

    eps_start = eps + 0.01  # after finding eigenvalue, pick new start
    norm = np.trapz(y[:, 0] * y[:, 0], xspan)  # calculate the normalization
    plt.plot(xspan, y[:, 0] / np.sqrt(norm), col[modes - 1])  # plot modes
    eig_func_list.append(abs(y[:, 0] / np.sqrt(norm)))

plt.show()
A1 = eig_func_list
A2 = eps_list
#print('A1 = ' + str(A1))
print('A2 = ' + str(A2))