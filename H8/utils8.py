import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def numerical_gradient(func, x, h=1e-5):
    """
    Calculează gradientul numeric al unei funcții într-un punct dat.
    """
    grad = np.zeros_like(x, dtype=float)
    n = len(x)
    for i in range(n):
        x_plus_2h = np.copy(x)
        x_plus_2h[i] += 2 * h
        x_plus_h = np.copy(x)
        x_plus_h[i] += h
        x_minus_h = np.copy(x)
        x_minus_h[i] -= h
        x_minus_2h = np.copy(x)
        x_minus_2h[i] -= 2 * h
        grad[i] = (-func(x_plus_2h) + 8 * func(x_plus_h) - 8 * func(x_minus_h) + func(x_minus_2h)) / (12 * h)
    return grad


# written by deepseek
def plot_3d_function(objective_function, x_range=None, y_range=None, min_point=None, points=None):
    """
    Plot a 3D surface for a given objective function.
    """
    # Default ranges if not provided
    if x_range is None:
        x_range = (-5, 5)
    if y_range is None:
        y_range = (-5, 5)

    z_limits = (-10, 10)
    
    # Create grid
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    
    # Vectorized function evaluation
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = objective_function([X[i,j], Y[i,j]])
    
    # Create plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                          vmin=z_limits[0], vmax=z_limits[1])
        
    # Mark minimum point if provided
    if min_point is not None:
        min_z = objective_function(min_point)
        ax.scatter(min_point[0], min_point[1], min_z, 
                  color='red', s=100, label=f'Minimum at {min_point}')
        ax.legend()
    
        # Mark additional points if provided
    if points is not None:
        for i, point in enumerate(points):
            point_z = objective_function(point)
            ax.scatter(point[0], point[1], point_z,
                      color='blue', s=50, label=f'Point {i+1}: {point}')
        ax.legend()
    
    # Labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Plot of Objective Function')
    
    ax.set_zlim(z_limits)
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.show()