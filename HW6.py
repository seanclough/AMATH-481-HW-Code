import numpy as np
from scipy.integrate import solve_ivp
from scipy.fftpack import fft2, ifft2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import kron

#'''
n = 64
x = np.linspace(-10, 10, n+1)
x = x[0:n]
y = np.linspace(-10, 10, n+1)
y = y[0:n]
X, Y = np.meshgrid(x, y)
m = 1
u0 = np.tanh(np.sqrt(X**2 + Y**2)) * np.cos(m * np.angle(X + 1j * Y) - np.sqrt(X**2 + Y**2))
v0 = np.tanh(np.sqrt(X**2 + Y**2)) * np.sin(m * np.angle(X + 1j * Y) - np.sqrt(X**2 + Y**2))
u0_f = fft2(u0)
v0_f = fft2(v0)
u0_f_1d = u0_f.flatten()
v0_f_1d = v0_f.flatten()
f0 = np.hstack([u0_f_1d, v0_f_1d])
beta = 1
D_1 = 0.1
D_2 = 0.1
t_eval = np.arange(0,4.1,0.5)

kx = (2 * np.pi / 20) * np.concatenate((np.arange(0, n/2), np.arange(-n/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / 20) * np.concatenate((np.arange(0, n/2), np.arange(-n/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

def rhs(t, f, D_1, D_2, beta, K, n):
    u_f_1d = f[0:n**2]
    v_f_1d = f[n**2:]
    u_f = u_f_1d.reshape(n, n)
    v_f = v_f_1d.reshape(n, n)
    u = np.real(ifft2(u_f))
    v = np.real(ifft2(v_f))
    Asq = u**2 + v**2
    lamb = np.ones([n,n]) - Asq
    omega = -beta * Asq
    u_t_f = fft2(lamb * u - omega * v) + D_1*(-K * u_f)
    v_t_f = fft2(omega * u + lamb * v) + D_2*(-K * v_f)
    u_t_f_1d = u_t_f.flatten()
    v_t_f_1d = v_t_f.flatten()
    return np.hstack([u_t_f_1d, v_t_f_1d])

raw_result = solve_ivp(rhs, t_span=[0,t_eval[-1]], y0=f0, t_eval=t_eval, args=(D_1, D_2, beta, K, n), method='RK45')
A1 = raw_result.y
#'''

# from movies.py
def plot(raw_result, domain, X, Y, n, t_eval, fps, file_name):
    m = n 
    n = m * m
    result = []
    for i in range(len(t_eval)):
        result_f_1d_i = raw_result.y[:,i][0:n]+1j*raw_result.y[:,i][n:]
        result_f_2d_i =result_f_1d_i.reshape((m, m))
        if domain == 'f':
            result_2d_i = np.real(ifft2(result_f_2d_i))
        else:
            result_2d_i = result_f_2d_i
        result_1d_i = result_2d_i.reshape(n)
        # hot fix to remove graphing anomalies
        # a better understanding of the animation function would allow a more elegant solution
        for j in range(n):
            if abs(result_1d_i[j]) < 1e-12:
                result_1d_i[j] = 0
        result_2d_i = result_1d_i.reshape((m, m))
        result.append(result_2d_i)
    # PLOTTING!!!
    #plt.figure()
    #plt.contourf(X, Y, result2[2], 100, cmap='jet')
    #plt.show()
    #'''
    # Create the figure and axis for the animation
    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, result[0], 100, vmax = 1, vmin = 0, cmap='jet')
    plt.colorbar(contour)

    # Update function for the animation
    def update(frame):
        ax.clear()
        contour = ax.contourf(X, Y, result[frame], 100, cmap='jet')
        ax.set_title(f'Time = {t_eval[frame]:.1f}')
        return contour

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(t_eval), blit=False)

    # Save the animation as a movie file (e.g., MP4)
    ani.save('AMATH 481\\AMATH-481-HW-Code\\'+file_name, writer='pillow', fps=fps)

    # Display the animation
    #plt.show()
    #'''

# from lecture
def cheb(N):
    if N == 0:
        D = 0.0
        x = 1.0
    else:
        n = np.arange(0, N + 1)
        x = np.cos(np.pi * n / N).reshape(N + 1, 1)  # Chebyshev points
        c = np.hstack(([2], np.ones(N - 1), [2])) * ((-1) ** n)  # Scale factors
        X = np.tile(x, (1, N + 1))
        dX = X - X.T
        D = np.dot(c.reshape(N + 1, 1), (1 / c).reshape(1, N + 1)) / (dX + np.eye(N + 1))
        D -= np.diag(np.sum(D.T, axis=0))
    return D, x.reshape(N + 1)

n = 31
t_eval = np.arange(0,4.1,0.5)
#run_time = 5 #seconds
#fps = 1 #frames per second
#t_eval = np.linspace(0, run_time, run_time*fps)
#n = 256
m = 1
beta = 1
D_1 = 0.1
D_2 = 0.1
D0, x0 = cheb(n-1)
D = .1 * D0
x = 10. * x0
D[n-1,:] = 0
D[0,:] = 0 
D2 = np.dot(D, D)
X, Y = np.meshgrid(x, x)
u0 = np.tanh(np.sqrt(X**2 + Y**2)) * np.cos(m * np.angle(X + 1j * Y) - np.sqrt(X**2 + Y**2))
v0 = np.tanh(np.sqrt(X**2 + Y**2)) * np.sin(m * np.angle(X + 1j * Y) - np.sqrt(X**2 + Y**2))
f0 = np.hstack([u0.flatten(), v0.flatten()])
I = np.eye(n)
L = kron(I, D2) + kron(D2, I)
def rhs2(t, f, D_1, D_2, beta, L,n):
    u = f[:n**2]
    v = f[n**2:]
    Asq = u**2 + v**2
    lamb = 1 - Asq
    omega = -beta * Asq
    u_t = lamb * u - omega * v + D_1 * (L @ u)
    v_t = omega * u + lamb * v + D_2 * (L @ v)
    return np.hstack([u_t, v_t])

raw_result = solve_ivp(rhs2, t_span=[0,4], y0=f0, t_eval=t_eval, args=(D_1, D_2, beta, L, n), method='RK45')
A2 = raw_result.y
#plot(raw_result,'r', X, Y, n, t_eval, fps, 'hw_6_pretty.gif')

