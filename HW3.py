import numpy as np
from scipy.integrate import odeint
from scipy.sparse import diags
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs

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
A = diags(diagonals, offsets, shape = (len(xspan), len(xspan)), format = 'csr')
A[0,0] = -2/3
A[-1,-1] = -2/3
A[0,1] = 2/3
A[-1,-2] = 2/3
A /= (xstep)**2

B = diags([xspan**2], [0], shape = (len(xspan), len(xspan)), format = 'csr')

"""
# Print the matrices
print("Matrix A:")
print(A.toarray())
print("Matrix B:")
print(B.toarray())
"""

print(len(xspan))
# Solve for the eigenvalues and eigenvectors
num_eigenvalues = 5  # Number of eigenvalues and eigenvectors to compute
eigenvalues, eigenvectors = eigs(B-A, k=num_eigenvalues, which='SM', tol=tol)

print(eigenvalues)

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

for n in range(num_eigenvalues):
    eigenvectors[:, n] /= np.trapz(eigenvectors[:, n]**2, xspan)

# Print the eigenvalues
print("Eigenvalues:")
print(eigenvalues)

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