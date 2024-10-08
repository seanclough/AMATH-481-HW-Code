import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd


# Define the function and its derivative
def f(x):
    return x * np.sin(3 * x) - np.exp(x)

def df(x):
    return np.sin(3 * x) + 3 * x * np.cos(3 * x) - np.exp(x)

# Newton-Raphson method
def newton_raphson(x0, tol=1e-6, max_iter=100):
    x_values = [x0]
    for _ in range(max_iter):
        x1 = x0 - f(x0) / df(x0)
        x_values.append(x1)
        if abs(f(x0)) < tol:
            break
        x0 = x1
    return x_values

# Initial guess
x0 = -1.6
A1 = newton_raphson(x0)

"""
# Plotting
x = np.linspace(-2, 4, 1000)
y = f(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='f(x) = x*sin(3x) - exp(x)')
plt.scatter(A1, [f(xi) for xi in A1], color='red', zorder=5, label='Newton-Raphson Iterations')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Newton-Raphson Method')
plt.legend()
plt.grid(True)
plt.show()
"""

# Bisection method
def bisection(a, b, tol=1e-6, max_iter=100):
    mid_values = []
    for _ in range(max_iter):
        c = (a + b) / 2
        mid_values.append(c)
        if abs(f(c)) < tol:
            break
        if f(c) * f(a) > 0:
            a = c
        else:
            b = c
    return mid_values

# Initial endpoints
a = -0.7
b = -0.4
A2 = bisection(a, b)

"""
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='f(x) = x*sin(3x) - exp(x)')
plt.scatter(A2, [f(xi) for xi in A2], color='blue', zorder=5, label='Bisection Midpoints')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Bisection Method')
plt.legend()
plt.grid(True)
plt.show()
"""
"""
# Create dataframes for A1 and A2
df_A1 = pd.DataFrame({'Iteration': range(1, len(A1) + 1), 'x-values': A1})
df_A2 = pd.DataFrame({'Iteration': range(1, len(A2) + 1), 'Mid-point values': A2})

# Print the tables
print("Newton-Raphson Method (A1):")
print(df_A1.to_string(index=False))

print("\nBisection Method (A2):")
print(df_A2.to_string(index=False))
"""
A3 = [len(A1) - 1, len(A2)]
print("Number of iterations for Newton-Raphson and Bisection:", A3)

A = np.array([[1,2], [-1,1]])
B = np.array([[2,0],[0,2]])
C = np.array([[2,0,-3],[0,0,-1]])
D = np.array([[1,2],[2,3],[-1,0]])
x = np.array([1,0])
y = np.array([0,1])
z = np.array([1,2,-1])
A4 = np.add(A,B)
A5 = np.add(3*x,-4*y)
A6 = np.dot(A,x)
A7 = np.dot(B,np.add(x,-y))
A8 = np.dot(D,x)
A9 = np.add(np.dot(D,y),z)
A10 = np.dot(A,B)
A11 = np.dot(B,C)
A12 = np.dot(C,D)

"""
print("\nA4:", A4)
print("A5:", A5)
print("A6:", A6)
print("A7:", A7)
print("A8:", A8)
print("A9:", A9)
print("A10:", A10)
print("A11:", A11)
print("A12:", A12)
"""