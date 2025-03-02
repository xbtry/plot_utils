import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def vec_dir_plot(vectors):
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    origin = np.array([[0,0,0]])
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    for v in vectors:
        ax.quiver(*origin[0], *v, color='b', linewidth=2, arrow_length_ratio=0.1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([-1.5,1.5])
    ax.set_zlim([-1.5,1.5])
    ax.set_title("3D Vector Direction Plot")
    plt.show()

"""
Example:
f = lambda t: np.sin(t) 
plot_function(f)
"""
def plot_func(func, t_min=0, t_max=2*np.pi, num_points=1000):
    """
    Plot a function defined by a lambda function.
    
    Parameters:
        func: A closure function that defines the function to be plotted.
        t_min: Minimum value of the parameter t.
        t_max: Maximum value of the parameter t.
        num_points: Number of points to plot.
    """
    t_values = np.linspace(t_min, t_max, num_points)
    
    y_values = np.vectorize(func)(t_values)
    
    plt.plot(t_values, y_values)
    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.title("Plot of f(t)")
    plt.grid(True)
    plt.show()

"""
Example:
x = lambda t: np.cos(t)
y = lambda t: np.sin(t)
parametric_plot(x, y)
"""
def parametric_plot(x_func, y_func, t_min=0, t_max=2*np.pi, num_points=1000):
    t_values = np.linspace(t_min, t_max, num_points)
    
    x_values = np.vectorize(x_func)(t_values)
    y_values = np.vectorize(y_func)(t_values)
    
    plt.plot(x_values, y_values)
    plt.xlabel("x(t)")
    plt.ylabel("y(t)")
    plt.title("Parametric Plot")
    plt.grid(True)
    plt.show()
def parametric_plot_3d(x_func, y_func, z_func, t_min=0, t_max=2*np.pi, num_points=1000):
    """
    Plot a 3D parametric curve defined by three closure functions x(t), y(t), and z(t).
    
    Parameters:
        x_func: A closure function that defines x(t).
        y_func: A closure function that defines y(t).
        z_func: A closure function that defines z(t).
        t_min: Minimum value of the parameter t.
        t_max: Maximum value of the parameter t.
        num_points: Number of points to plot.
    """
    t_values = np.linspace(t_min, t_max, num_points)
    x_values = np.vectorize(x_func)(t_values)
    y_values = np.vectorize(y_func)(t_values)
    z_values = np.vectorize(z_func)(t_values)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_values, y_values, z_values)
    ax.set_xlabel("x(t)")
    ax.set_ylabel("y(t)")
    ax.set_zlabel("z(t)")
    ax.set_title("3D Parametric Plot")
    plt.show()
