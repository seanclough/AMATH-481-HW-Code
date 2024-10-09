import numpy as np
from scipy.integrate import odeint
from scipy.integrate import RK45
import matplotlib.pyplot as plt

def shoot2(x_vect, x, K, epsilon):
    return [x_vect[1], (K*x**2-epsilon) * x_vect[0]]

tol = 1e-4  # define a tolerance level 
col = ['r', 'b', 'g', 'c', 'm']  # eigenfunc colors
eps_start = 100
eps_list = [] 
eig_func_list = []
A = 1
K = 1 
L = 4
xp = [-L, L] 
xspan = np.linspace(-L, L, int((2 * L) / 0.1) + 1)
x0 = [A, A] #initial condition needs to be changed

for modes in range(1, 5):  # begin mode loop
    eps = eps_start  # initial value of eigenvalue beta
    deps = eps_start / 100  # default step size in beta
    for _ in range(1000):  # begin convergence loop for beta
        y = odeint(shoot2, x0, xspan, args=(K,eps)) 
        # y = RK45(shoot2, xp[0], x0, xp[1], args=(n0,beta)) 

        #print('y[-1, 1]= ' + str(y[-1, 1]))
        #print('y[-1, 0]= '+ str(y[-1, 0]))
        if abs(y[-1, 1] + y[-1,0]) < tol:  # final condition EDIT THIS
            print(eps)  # write out eigenvalue 
            eps_list.append(eps)
            break  # get out of convergence loop

        if (-1) ** (modes + 1) * y[-1, 0] > 0: # EDIT THIS TOO
            eps -= deps
        else:
            eps += deps / 2
            deps /= 2

    eps_start = eps  # after finding eigenvalue, pick new start
    norm = np.trapz(y[:, 0] * y[:, 0], xspan)  # calculate the normalization
    plt.plot(xspan, y[:, 0] / np.sqrt(norm), col[modes - 1])  # plot modes
    eig_func_list.append(abs(y[:, 0] / np.sqrt(norm)))

plt.show()
A1 = eig_func_list
A2 = eps_list