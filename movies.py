import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

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
w0=np.exp(-X**2 - (1 / 20) * (Y**2))
w0_f_2d = fft2(w0)
w0_f_1d = w0_f_2d.reshape(n)
w0_f_1d = np.hstack((np.real(w0_f_1d), np.imag(w0_f_1d)))

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
    w_t_f = 0.001*-1*K*w_f-fft2(np.real(ifft2(1j*KX*psi_f))*np.real(ifft2(1j*KY*w_f)))+fft2(np.real(ifft2(1j*KY*psi_f))*np.real(ifft2(1j*KX*w_f)))
    w_t_f_1d = w_t_f.reshape(n)
    w_t_f_1d = np.hstack((np.real(w_t_f_1d), np.imag(w_t_f_1d)))
    return w_t_f_1d

run_time = 60
fps = 30
t_eval = np.linspace(0, run_time, run_time*fps)
#t_eval = np.arange(0, 4.1, 0.5)
tspan = [0, run_time]
result1 = solve_ivp(rhs1, t_span=tspan, y0=w0_f_1d, args=(m, n, K), method='RK45', t_eval=t_eval)
result2 = []
for i in range(len(t_eval)):
    result_f_1d_i = result1.y[:,i][0:n]+1j*result1.y[:,i][n:]
    result_f_2d_i =result_f_1d_i.reshape((m, m))
    result_2d_i = np.real(ifft2(result_f_2d_i))
    result_1d_i = result_2d_i.reshape(n)
    for j in range(n):
        if abs(result_1d_i[j]) < 1e-12:
            result_1d_i[j] = 0
    result_2d_i = result_1d_i.reshape((m, m))
    result2.append(result_2d_i)
#plt.figure()
#plt.contourf(X, Y, result2[2], 100, cmap='jet')
#plt.show()
#'''
# Create the figure and axis for the animation
fig, ax = plt.subplots()
contour = ax.contourf(X, Y, result2[0], 100, vmax = 1, vmin = 0, cmap='jet')
plt.colorbar(contour)

# Update function for the animation
def update(frame):
    ax.clear()
    contour = ax.contourf(X, Y, result2[frame], 100, cmap='jet')
    ax.set_title(f'Time = {t_eval[frame]:.1f}')
    return contour

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t_eval), blit=False)

# Save the animation as a movie file (e.g., MP4)
ani.save('AMATH 481\\AMATH-481-HW-Code\\anime.gif', writer='pillow', fps=fps)

# Display the animation
#plt.show()
#'''
