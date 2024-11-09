import numpy as np
from scipy.integrate import odeint
from scipy.sparse import diags
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
from scipy.integrate import solve_ivp

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
#plt.figure(1)
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
    #plt.figure(modes)
    #plt.plot(xspan, y[:, 0] / np.sqrt(norm), col[modes - 1])  # plot modes
    eig_func_list.append(abs(y[:, 0] / np.sqrt(norm)))
#plt.show()
A1 = abs(np.transpose(eig_func_list))
A2 = eps_list
"""
from scipy.signal import find_peaks
for n in range(5):
    peaks, _ = find_peaks(eig_func_list[n])
    for __ in range(len(peaks)):
        print('mode ' + str(n) + ' peaks = ' + str(eig_func_list[n][peaks[__]]))
"""

L = 4
K = 1
tol = 1e-4  # define a tolerance level 
col = ['r', 'b', 'g', 'c', 'm']  # eigenfunc colors
A = 1
xp = [-L, L] 
xstep = 0.1
xspan = np.linspace(-L, L, int((2 * L) / xstep) + 1)

e1 = np.ones(len(xspan))
diagonals = [e1, -2 * e1, e1]
offsets = [-1, 0, 1]
A = diags(diagonals, offsets, shape = (len(xspan)-2, len(xspan)-2), format = 'csr')
A[0,0] = -2/3
A[-1,-1] = -2/3
A[0,1] = 2/3
A[-1,-2] = 2/3
A /= (xstep)**2

B = diags([xspan[1:-1]**2], [0], shape = (len(xspan)-2, len(xspan)-2), format = 'csr')

"""
# Print the matrices
print("Matrix A:")
print(A.toarray())
print("Matrix B:")
print(B.toarray())

# Print the size of the matrices
print("Size of matrix A:", A.shape)
print("Size of matrix B:", B.shape)
"""

#print(len(xspan))
# Solve for the eigenvalues and eigenvectors
num_eigenvalues = 5  # Number of eigenvalues and eigenvectors to compute
eigenvalues, eigenvectors = eigs(B-A, k=num_eigenvalues, which='SM', tol=tol)

#print(eigenvalues)

# Filter out complex eigenvalues and corresponding eigenvectors
real_idx = np.isreal(eigenvalues)
eigenvalues = eigenvalues[real_idx].real
eigenvectors = eigenvectors[:, real_idx]

# Filter out negative eigenvalues and corresponding eigenvectors
positive_idx = eigenvalues > 0
eigenvalues = eigenvalues[positive_idx]
eigenvectors = eigenvectors[:, positive_idx]

# Sort eigenvalues and corresponding eigenvectors
idx = eigenvalues.argsort()
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Append a value at the front of each eigenvector
eigenvectors_with_value = np.zeros((eigenvectors.shape[0] + 2, eigenvectors.shape[1]))
eigenvectors_with_value[1:-1, :] = eigenvectors
for n in range(num_eigenvalues):
    value_to_front = 4/3*eigenvectors[0, n] - 1/3*eigenvectors[1, n]
    value_to_end = 4/3*eigenvectors[-1, n] - 1/3*eigenvectors[-2, n]
    eigenvectors_with_value[0, n] = value_to_front
    eigenvectors_with_value[-1, n] = value_to_end
    if eigenvectors[1,n]-eigenvectors[0,n] < 0:
        eigenvectors_with_value[:, n] *= -1
eigenvectors = eigenvectors_with_value

for n in range(num_eigenvalues):
    eigenvectors[:, n] /= np.sqrt(np.trapz(eigenvectors[:, n]**2, xspan))
    #print ('norm = ' + str(np.trapz(eigenvectors[:, n]**2, xspan)))

A3 = abs(eigenvectors)
A4 = eigenvalues

# Print the eigenvalues
#print("Eigenvalues:")
#print(eigenvalues)
"""
# Plot the eigenvectors
plt.figure(figsize=(10, 6))
for i in range(num_eigenvalues):
    plt.plot(xspan, eigenvectors[:, i], label=f'Eigenvector {i+1}', color=col[i % len(col)])

plt.title('Eigenvectors of the Matrix A')
plt.xlabel('x')
plt.ylabel('Eigenvector')
plt.legend()
plt.grid(True)
plt.show()

from scipy.signal import find_peaks
for n in range(5):
    peaks, _ = find_peaks(eigenvectors[:, n])
    for m in range(len(peaks)):
        print('mode ' + str(n) + ' peaks = ' + str(eigenvectors[peaks[m]][n]))
"""

def shoot2_1(x, y, K, epsilon,gamma):
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
#plt.figure(1)
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
            sol = solve_ivp(shoot2_1, [xspan[0], xspan[-1]], x0, t_eval=xspan, args=(K, eps, gamma))
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
    #plt.plot(xspan, abs(y1 / np.sqrt(norm)), col[modes - 1], label=f"Mode {modes}") # plot modes

#plt.show()
A5 = abs(np.transpose(eig_func_list))
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
#plt.figure(2)
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
            sol = solve_ivp(shoot2_1, [xspan[0], xspan[-1]], x0, t_eval=xspan, args=(K, eps, gamma))
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
    #plt.plot(xspan, abs(y1 / np.sqrt(norm)), col[modes - 1], label=f"Mode {modes}") # plot modes

#plt.show()
A7 = abs(np.transpose(eig_func_list))
A8 = eps_list

eps = 1
L = 2
K = 1 
A = 1
def shoot2_2(x, y, K, epsilon):
    return [y[1], (K*x**2-epsilon) * y[0]]
xspan = [-L, L]
TOL = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
methods = ['RK45', 'RK23', 'Radau', 'BDF']
results = np.zeros((len(TOL), len(methods)))
for method in methods:
    for tol in TOL:
        x0 = [A, A*np.sqrt(K*L**2-eps)] #initial conditions
        options = {'rtol': tol, 'atol': tol}
        y = solve_ivp(shoot2_2, xspan, x0, method=method, args=(K,eps), **options)
        step_size = np.mean(np.diff(y.t))
        #print(f"Method: {method}, Tolerance: {tol}, Step Size: {step_size}")
        results[TOL.index(tol), methods.index(method)] = step_size



# Plot the results
plt.figure()
slopes = np.zeros(len(methods))
for i, method in enumerate(methods):
    #plt.plot(TOL, results[:, i], label=method, marker='o')
    plt.plot(results[:, i], TOL, label=method, marker='o', color=col[i])

    
    # Fit a linear model to the log-transformed data
    log_tol = np.log(TOL)
    log_step_size = np.log(results[:, i])
    slope, intercept = np.polyfit(log_step_size, log_tol, 1)
    slopes[i] = slope

    # Plot the fitted line
    plt.plot(log_step_size, (slope * log_step_size + intercept), linestyle='--', color=col[i])

print(slopes)
A9 = slopes

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Tolerance')
plt.ylabel('Average Step Size')
plt.title('Average Step Size vs Tolerance for Different Methods')
plt.legend()
plt.grid(True)
plt.show()
