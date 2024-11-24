import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
#import pandas as pd
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
from scipy.linalg import solve
from scipy.linalg import lu, solve_triangular

m = 64    # N value in x and y directions
n = m * m  # total size of matrix
xrange = 20
yrange = 20
dx = xrange / m
dy = yrange / m
x = np.linspace(-xrange/2, xrange/2, m+1)
x = x[0:m]
y = np.linspace(-yrange/2, yrange/2, m+1)
y = y[0:m]
X, Y = np.meshgrid(x, y)
w0 = np.exp(-X**2 - (1 / 20) * (Y**2))

kx = (2 * np.pi / xrange) * np.concatenate((np.arange(0, m/2), np.arange(-m/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / yrange) * np.concatenate((np.arange(0, m/2), np.arange(-m/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

e0 = np.zeros((n, 1))  # vector of zeros
e1 = np.ones((n, 1))   # vector of ones
e2 = np.copy(e1)    # copy the one vector
e4 = np.copy(e0)    # copy the zero vector

for j in range(1, m+1):
    e2[m*j-1] = 0  # overwrite every m^th value with zero
    e4[m*j-1] = 1  # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

# Place diagonal elements
diagonals = [e1.flatten(), e1.flatten(), e5.flatten(), 
             e2.flatten(), -4 * e1.flatten(), e3.flatten(), 
             e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]

matA = spdiags(diagonals, offsets, n, n).toarray()
matA = matA/(dx*dy) # assuming dx = dy
mod_matA = np.copy(matA)
mod_matA[0,0] = 2
#A1 = matA

# Convert to Pandas DataFrame for better formatting
#df_matA = pd.DataFrame(matA)

# Print the matrix matA using Pandas
#print("Matrix matA:")
#print(df_matA)

"""
# Plot matrix structure
plt.figure(5)
plt.spy(matA)
plt.title('Matrix Structure')
plt.show()
"""

e1 = np.ones((n, 1))
diagonals = [e5.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets = [-(m-1), -1, 1, m-1]
matC = spdiags(diagonals, offsets, n, n).toarray()
matC = matC/(2*dx)
#df_matC = pd.DataFrame(matB)
#print("Matrix matC:")
#print(df_matC)
#A3 = matC

e1 = np.ones((n, 1))
diagonals = [e1.flatten(), -e1.flatten(), e1.flatten(), -e1.flatten()]
offsets = [-(n-m), -m, m, n-m]
matB = spdiags(diagonals, offsets, n, n).toarray()
matB = matB/(2*dy)
#df_matB = pd.DataFrame(matB)
#print("Matrix matB:")
#print(df_matB)
#A2 = matB

def rhs1(t,w_1d, m, K, matA, matB, matC):
    w_2d = w_1d.reshape((m, m))
    psi = np.real(ifft2(-fft2(w_2d)/K))
    w_t_1d = 0.001*np.dot(matA,w_1d)-np.dot(matB,psi.flatten())*np.dot(matC,w_1d)+np.dot(matC,psi.flatten())*np.dot(matB,w_1d)
    return w_t_1d 

t_eval = np.arange(0, 4.1, 0.5)
tspan = [0, 4]
result1 = solve_ivp(rhs1, t_span=tspan, y0=w0.flatten(), args=(m, K, matA, matB, matC), method='RK45', t_eval=t_eval)
A1 = result1.y
print(A1.shape)
print(A1[0][0])
print(A1[-1][-1])

def rhs2(t,w_1d, m, matA, matB, matC):
    psi = solve(matA,w_1d)
    psi = psi.reshape((m,m))
    w_t_1d = 0.001*np.dot(matA,w_1d)-np.dot(matB,psi.flatten())*np.dot(matC,w_1d)+np.dot(matC,psi.flatten())*np.dot(matB,w_1d)
    return w_t_1d 

result2 = solve_ivp(rhs2, t_span=tspan, y0=w0.flatten(), args=(m, mod_matA, matB, matC), method='RK45', t_eval=t_eval)
A2 = result2.y
#print(A2)

P, L, U = lu(mod_matA)
def rhs3(t,w_1d, m, P, L, U, matA, matB, matC):
    Pb = np.dot(P, w_1d) 
    y = solve_triangular(L, Pb, lower=True) 
    psi = solve_triangular(U, y)
    psi = psi.reshape((m,m))
    w_t_1d = 0.001*np.dot(matA,w_1d)-np.dot(matB,psi.flatten())*np.dot(matC,w_1d)+np.dot(matC,psi.flatten())*np.dot(matB,w_1d)
    return w_t_1d 

result3 = solve_ivp(rhs3, t_span=tspan, y0=w0.flatten(), args=(m, P, L, U, mod_matA, matB, matC), method='RK45', t_eval=t_eval)
A3 = result3.y
#print(A3)