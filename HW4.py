import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
import pandas as pd

m = 8    # N value in x and y directions
n = m * m  # total size of matrix
xrange = 20
yrange = 20
dx = xrange / (m - 1)
dy = yrange / (m - 1)

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
A1 = matA

# Convert to Pandas DataFrame for better formatting
df_matA = pd.DataFrame(matA)

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
diagonals = [e1.flatten(), -e1.flatten(), e1.flatten(), -e1.flatten()]
offsets = [-(n-1), -1, 1, n-1]
matB = spdiags(diagonals, offsets, n, n).toarray()
matB = matB/(2*dx)
df_matB = pd.DataFrame(matB)
print("Matrix matB:")
print(df_matB)
A2 = matB

e1 = np.ones((n, 1))
diagonals = [e1.flatten(), -e1.flatten(), e1.flatten(), -e1.flatten()]
offsets = [-(n-m), -m, m, n-m]
matC = spdiags(diagonals, offsets, n, n).toarray()
matC = matC/(2*dy)
df_matC = pd.DataFrame(matC)
print("Matrix matC:")
print(df_matC)
A2 = matC
