import matplotlib.pyplot as plt
import numpy as np

# Close all existing plots
plt.close('all')

x = np.array([0]); y = np.array([0]); z = np.array([0])
for j in range(100):
    x = np.append(x, (7 + y[j] - z[j]) / 4)
    y = np.append(y, (21 + 4*x[j+1] + z[j]) / 8)
    z = np.append(z, (15 + 2*x[j+1] - y[j+1]) / 5)
    if j > 0 and abs(x[j] - x[j - 1]) < 1e-6:
        break
print(j)
print(x[-1], y[-1], z[-1])
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the coordinates
ax.plot(x, y, z, label='3D Line')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Plot of Coordinates')
ax.legend()

plt.show()

# Initialize the 3x3 matrix
A = np.array([[4, -1, 1],
              [4, -8, 1],
              [-2, 1, 5]])

# Initialize the 1x3 vector
b = np.array([7, -21, 15])

# Solve the linear system Ax = b
solution = np.linalg.solve(A, b)

print("Solution:", solution)