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
w0 = fft2(np.exp(-X**2 - (1 / 20) * (Y**2)))
w0 = w0.reshape(n)
w0 = np.hstack((np.real(w0), np.imag(w0)))

kx = (2 * np.pi / xrange) * np.concatenate((np.arange(0, m/2), np.arange(-m/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / yrange) * np.concatenate((np.arange(0, m/2), np.arange(-m/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

def rhs1(t,w_f_1d, m, n, K):
    w_f_1d = w_f_1d[0:n]+1j*w_f_1d[n:]
    w_f = w_f_1d.reshape((m, m))
    psi_f = -w_f/K
    w_t_f = 0.001*-1*K*w_f-fft2(ifft2(1j*KX*psi_f)*ifft2(1j*KY*w_f))+fft2(ifft2(1j*KY*psi_f)*ifft2(1j*KX*w_f))
    w_t_f_1d = w_t_f.reshape(n)
    w_t_f_1d = np.hstack((np.real(w_t_f_1d), np.imag(w_t_f_1d)))
    return w_t_f_1d

t_eval = np.arange(0, 4.1, 0.5)
tspan = [0, 4]
result1 = solve_ivp(rhs1, t_span=tspan, y0=w0, args=(m, n, K), method='RK45', t_eval=t_eval)
A1 = result1.y
print(A1.shape) # why is it not 8192 by 9? >:(
A1 = A1[0:n]+1j*A1[n:]
A1 = A1.reshape((m, m))
A1 = np.real(ifft2(A1))
plt.figure()
plt.contourf(X, Y, A1, 100, cmap='jet')
plt.show()
