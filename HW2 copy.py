import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.integrate import RK45
import matplotlib.pyplot as plt

def shoot2(x, y, K, epsilon,gamma):
    return [y[1], (gamma*(abs(y[0]))**2+K*x**2-epsilon) * y[0]]

tol = 1e-4  # define a tolerance level 
col = ['r', 'b', 'g', 'c', 'm']  # eigenfunc colors
eps_start = 0.01
eps_list = [] 
eig_func_list = []
A_start = 0.1
K = 1 
L = 2
gamma = 0.05
xp = [-L, L] 
#xspan = np.linspace(-L, L, int((2 * L) / 0.1) + 1)
xspan = np.arange(-L,L+.1,.1)
#x0 = [A, A*np.sqrt(K*L**2)] #initial conditions
plt.figure(1)
for modes in range(1, 3):           # begin mode loop
    A = A_start
    dA = .01
    eps = eps_start                 # initial value of eigenvalue beta
              # default step size in beta
    for __ in range(100):            # loops A here

        # notice: put (d)eps above in order to make sure
        # in each mode's loop, we use the same initial
        deps = 0.1
        for _ in range(1000):       # begin convergence loop for beta
            x0 = [A, A * np.sqrt(K * L ** 2 - eps)]                         #initial conditions
            sol = solve_ivp(shoot2, [xspan[0], xspan[-1]], x0, t_eval=xspan, args=(K, eps, gamma))
            y1 = sol.y[0]  
            y2 = sol.y[1]   
            # y = odeint(shoot2, x0, xspan, args=(K,eps,gamma)) 
            # y = RK45(shoot2, xp[0], x0, xp[1], args=(n0,beta)) 

            if abs(y2[-1] + np.sqrt(K * L ** 2 - eps) * y1[-1]) < tol:      # final condition
                #print(eps)  # write out eigenvalue 
                #eps_list.append(eps)
                #print(_)
                break  # get out of convergence loop

            if (-1) ** (modes + 1) * (y2[-1] + np.sqrt(K * L ** 2 - eps) * y1[-1]) > 0:
                eps += deps
            else:
                eps -= deps
                deps /= 2
                #print(deps)
                #print(eps)

            if _ == 999 or deps < 1e-20:
                print('Did not converge')
                print(__)
                print('deps = ' + str(deps))
                break

        norm = np.trapz(y1 ** 2, xspan)  # calculate the normalization
        # print("norm = " + str(norm))
        # print("A = " + str(A))
        # A += dA
        if abs(norm - 1) < tol:
            break
        else:
            A /= np.sqrt(norm)   
        
        """
        elif norm - 1 < 0:
            A += dA
        else:
            A -= dA
            dA /= 2
        if A < 0:
            A = 0
        """

    #print(norm)
    eps_start = eps + 0.01  # after finding eigenvalue, pick new start
    A_start = A + 0.01

    # 存储并绘制结果
    eps_list.append(eps)
    eig_func_list.append(abs(y1 / np.sqrt(norm)))
    plt.plot(xspan, abs(y1 / np.sqrt(norm)), col[modes - 1], label=f"Mode {modes}") # plot modes

plt.show()
A5 = np.transpose(eig_func_list)
A6 = eps_list

tol = 1e-4  # define a tolerance level 
col = ['r', 'b', 'g', 'c', 'm']  # eigenfunc colors
eps_start = 0.1
eps_list = [] 
eig_func_list = []
A_start = 0.1
K = 1 
L = 2
gamma = -0.05
xp = [-L, L] 
#xspan = np.linspace(-L, L, int((2 * L) / 0.1) + 1)
xspan = np.arange(-L,L+.1,.1)
#x0 = [A, A*np.sqrt(K*L**2)] #initial conditions
plt.figure(2)
for modes in range(1, 3):           # begin mode loop
    A = A_start
    dA = .01
    eps = eps_start                 # initial value of eigenvalue beta
              # default step size in beta
    for __ in range(100):            # loops A here

        # notice: put (d)eps above in order to make sure
        # in each mode's loop, we use the same initial
        deps = 0.1
        for _ in range(1000):       # begin convergence loop for beta
            x0 = [A, A * np.sqrt(K * L ** 2 - eps)]                         #initial conditions
            sol = solve_ivp(shoot2, [xspan[0], xspan[-1]], x0, t_eval=xspan, args=(K, eps, gamma))
            y1 = sol.y[0]  
            y2 = sol.y[1]   
            # y = odeint(shoot2, x0, xspan, args=(K,eps,gamma)) 
            # y = RK45(shoot2, xp[0], x0, xp[1], args=(n0,beta)) 

            if abs(y2[-1] + np.sqrt(K * L ** 2 - eps) * y1[-1]) < tol:      # final condition
                #print(eps)  # write out eigenvalue 
                #eps_list.append(eps)
                #print(_)
                break  # get out of convergence loop

            if (-1) ** (modes + 1) * (y2[-1] + np.sqrt(K * L ** 2 - eps) * y1[-1]) > 0:
                eps += deps
            else:
                eps -= deps
                deps /= 2
                #print(deps)
                #print(eps)

            if _ == 999 or deps < 1e-20:
                print('Did not converge')
                print(__)
                print('deps = ' + str(deps))
                break

        norm = np.trapz(y1 ** 2, xspan)  # calculate the normalization
        # print("norm = " + str(norm))
        # print("A = " + str(A))
        # A += dA
        if abs(norm - 1) < tol:
            break
        else:
            A /= np.sqrt(norm)   
        
        """
        elif norm - 1 < 0:
            A += dA
        else:
            A -= dA
            dA /= 2
        if A < 0:
            A = 0
        """

    #print(norm)
    eps_start = eps + 0.01  # after finding eigenvalue, pick new start
    A_start = A + 0.01

    # 存储并绘制结果
    eps_list.append(eps)
    eig_func_list.append(abs(y1 / np.sqrt(norm)))
    plt.plot(xspan, abs(y1 / np.sqrt(norm)), col[modes - 1], label=f"Mode {modes}") # plot modes

plt.show()
A7 = np.transpose(eig_func_list)
A8 = eps_list

#"""
other_A1 = np.genfromtxt('AMATH 481\AMATH-481-HW-Code\A5.csv', delimiter=',')
error = A5-other_A1
y_min = -0.0007
y_max = 0.0007
y_ticks = np.linspace(y_min, y_max, 3)
fig, axs = plt.subplots(2, 1, figsize=(10, 15))
for i in range(6,8):
    axs[i-6].plot(xspan, error[:,i-6])
    #axs[i-6].set_ylim(y_min, y_max)  # Set the same y-axis limits for each subplot
    #axs[i-6].set_yticks(y_ticks)
    axs[i-6].set_title(f'Plot {i-6+1}')  # Optionally set a title for each subplot
plt.subplots_adjust(hspace=0.5) 
plt.show()
other_A1 = np.genfromtxt('AMATH 481\AMATH-481-HW-Code\A7.csv', delimiter=',')
error = A7-other_A1
y_min = -0.0007
y_max = 0.0007
y_ticks = np.linspace(y_min, y_max, 3)
fig, axs = plt.subplots(2, 1, figsize=(10, 15))
for i in range(6,8):
    axs[i-6].plot(xspan, error[:,i-6])
    #axs[i-6].set_ylim(y_min, y_max)  # Set the same y-axis limits for each subplot
    #axs[i-6].set_yticks(y_ticks)
    axs[i-6].set_title(f'Plot {i-6+1}')  # Optionally set a title for each subplot
plt.subplots_adjust(hspace=0.5) 
plt.show()
#"""    

#print(error)
print('A6 = ' + str(A6))