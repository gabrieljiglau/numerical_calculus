import numpy as np
import numdifftools as nd
from utils8 import numerical_gradient, plot_3d_function

def gradient_descent(func, grad_func, initial_x, learning_rate=0.01, epsilon=1e-5, max_iterations=300000):
    """
    Implementează metoda gradientului descendent cu rată de învățare fixă.
    """
    x = np.array(initial_x, dtype=float)
    history = [func(x)]
    for k in range(max_iterations):
        gradient = grad_func(x)
        x_new = x - learning_rate * gradient
        history.append(func(x_new))
        if np.linalg.norm(x_new - x) < epsilon:
            return x_new, history
        x = x_new
    print(f"Avertisment (Rată Fixă): Gradientul descendent nu a convergit după {max_iterations} iterații.")
    print(f"Returnez ultima valoare calculată: {x_new}, în {len(history)} iterații")
    return x_new, history 
    return None, history

def gradient_descent_backtracking(func, grad_func, initial_x, beta=0.8, epsilon=1e-5, max_iterations=300000):
    """
    Implementează metoda gradientului descendent cu backtracking line search.
    """
    x = np.array(initial_x, dtype=float)
    history = [func(x)]
    for k in range(max_iterations):
        gradient = grad_func(x)
        eta = 0.1
        p = 1.0
        while func(x - eta * gradient) > func(x) - (eta / 2) * np.linalg.norm(gradient)**2 and p < 8:
            eta *= beta
            p += 1
        x_new = x - eta * gradient
        history.append(func(x_new))
        if np.linalg.norm(x_new - x) < epsilon:
            return x_new, history
        x = x_new    
    print(f"Avertisment (Backtracking): Gradientul descendent nu a convergit după {max_iterations} iterații.")
    return x_new, history
    return None, history


def newton_method(func, initial_x, epsilon=1e-5, max_iterations=300000):
    """
    Implementează metoda Newton pentru minimizare numerică găsită în fișierul de la ML.
    """
    x = np.array(initial_x, dtype=float)
    history = [func(x)]

    grad_func = nd.Gradient(func)
    hess_func = nd.Hessian(func)

    for k in range(max_iterations):
        grad = numerical_gradient(func, x)
        hess = hess_func(x)

        hessian_inverse = np.linalg.inv(hess)

        delta_x = np.dot(hessian_inverse, grad)

        x_new = x - delta_x
        history.append(func(x_new))

        if np.linalg.norm(x_new - x) < epsilon:
            return x_new, history

        x = x_new

    print(f"Avertisment (Newton): Nu a convergit după {max_iterations} iterații.")
    return x, history

# Exemplu de utilizare (trebuie să definești funcția)
if __name__ == '__main__':
    def objective_function(x):
        #return x[0]**2 + x[1]**2 - 2 * x[0] - 4 * x[1] - 1  # works well, minim in 1, 2
        #return 3 * x[0]**2 - 12 * x[0] + 2 * x[1]**2 + 16 * x[1] - 10  # works well, minim in 2, -4
        #return x[0]**2 - 4 * x[0] * x[1] + 5 * x[1] ** 2 - 4 * x[1] + 3   # works well, minim in 4, 2
        return x[0] * x[0] * x[1] - 2 * x[0] * x[1] * x[1] + 3 * x[0] * x[1] + 4  # minim local in -1, 0.5

    # Gradientul analitic al funcției (dacă îl cunoști)
    def analytic_gradient(x):
        #return np.array([2 * x[0] - 2, 2 * x[1] - 4])
        #return np.array([6 * x[0] - 12, 4 * x[1] + 16])
        #return np.array([2 * x[0] - 4 * x[1], -4 * x[0] + 10 * x[1] - 4])
        return np.array([2 * x[0] * x [1] - 2 * x[1] * x[1] + 3 * x[1], x[0] * x[0] - 4 * x[0] * x[1] + 3 * x[0]])

    # Punct de start aleator
    #initial_point = (np.random.rand(2) - 0.5) * 5
    initial_point = np.array([-1.2, 0.4])
    print(f"Inițiat cu numarul : {initial_point}\n\n")
    print(f"Gradientul la punctul initial: {analytic_gradient(initial_point)}")

    mins = []

    print("Metoda gradientului descendent cu rată de învățare fixă (gradient analitic):")
    min_point_fixed_analytic, history_fixed_analytic = gradient_descent(
        objective_function, analytic_gradient, initial_point, learning_rate=0.1
    )
    if min_point_fixed_analytic is not None:
        print(f"Punct de minim aproximat: {min_point_fixed_analytic}")
        print(f"Valoarea funcției la minim: {objective_function(min_point_fixed_analytic)}")
        print(f"Număr de iterații: {len(history_fixed_analytic) - 1}")
        mins.append(min_point_fixed_analytic)
    else:
        print("Nu s-a găsit un punct de minim.")

    print("\nMetoda gradientului descendent cu backtracking line search (gradient analitic):")
    min_point_backtracking_analytic, history_backtracking_analytic = gradient_descent_backtracking(
        objective_function, analytic_gradient, initial_point
    )
    if min_point_backtracking_analytic is not None:
        print(f"Punct de minim aproximat: {min_point_backtracking_analytic}")
        print(f"Valoarea funcției la minim: {objective_function(min_point_backtracking_analytic)}")
        print(f"Număr de iterații: {len(history_backtracking_analytic) - 1}")
        mins.append(min_point_backtracking_analytic)
    else:
        print("Nu s-a găsit un punct de minim.")

    print("\nMetoda gradientului descendent cu rată de învățare fixă (gradient numeric):")
    min_point_fixed_numeric, history_fixed_numeric = gradient_descent(
        objective_function, lambda x: numerical_gradient(objective_function, x), initial_point, learning_rate=0.1
    )
    if min_point_fixed_numeric is not None:
        print(f"Punct de minim aproximat: {min_point_fixed_numeric}")
        print(f"Valoarea funcției la minim: {objective_function(min_point_fixed_numeric)}")
        print(f"Număr de iterații: {len(history_fixed_numeric) - 1}")
        mins.append(min_point_fixed_numeric)
    else:
        print("Nu s-a găsit un punct de minim.")

    print("\nMetoda gradientului descendent cu backtracking line search (gradient numeric):")
    min_point_backtracking_numeric, history_backtracking_numeric = gradient_descent_backtracking(
        objective_function, lambda x: numerical_gradient(objective_function, x), initial_point
    )
    if min_point_backtracking_numeric is not None:
        print(f"Punct de minim aproximat: {min_point_backtracking_numeric}")
        print(f"Valoarea funcției la minim: {objective_function(min_point_backtracking_numeric)}")
        print(f"Număr de iterații: {len(history_backtracking_numeric) - 1}")
        mins.append(min_point_backtracking_numeric)
    else:
        print("Nu s-a găsit un punct de minim.")


    print("\nMetoda lui Newton (gradient și hessian numeric):")
    min_point_newton, history_newton = newton_method(objective_function, initial_point)
    if min_point_newton is not None:
        print(f"Punct de minim aproximat: {min_point_newton}")
        print(f"Valoarea funcției la minim: {objective_function(min_point_newton)}")
        print(f"Număr de iterații: {len(history_newton) - 1}")
        mins.append(min_point_newton)
    else:
        print("Nu s-a găsit un punct de minim.")

    plot_3d_function(objective_function, points=mins)


