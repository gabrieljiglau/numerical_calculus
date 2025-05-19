import math
import matplotlib.pyplot as plt
import numpy as np


def interval_radacini(coef):
    """Calculează intervalul [-R, R] unde se află rădăcinile reale."""
    A = max(abs(c) for c in coef[1:])  # pentru primul: 11
    R = (abs(coef[0]) + A) / abs(coef[0])  # pentru primul: (1 + 11) / 1
    return (-R, R)

def horner(coef, v):
    """Calculează valoarea unui polinom folosind schema lui Horner."""
    b = coef[0]
    for i in range(1, len(coef)):
        b = coef[i] + b * v
    return b

def derivata(coef):
    """Calculează derivata de ordinul n a unui polinom."""
    if len(coef) <= 1:
        return [0]

    derived_coeffs = [coef[i] * (len(coef) - 1 - i) for i in range(len(coef) - 1)]
    return derived_coeffs

def radacini_distincte(radacini, epsilon):
    """Returnează rădăcinile distincte dintr-o listă."""
    if radacini != []:
        radacini.sort()
        distincte = [radacini[0]]
        for i in range(1, len(radacini)):
            if abs(radacini[i] - distincte[-1]) > epsilon:
                distincte.append(radacini[i])
        return distincte
    else:
        return []
    
    # the plotting was made by DeepSeek
def plot_polynomial_with_roots(coef, roots_halley, roots_newton4, roots_newton5, epsilon=1e-10):
    """Plot the polynomial and mark found roots"""
    # Create x values for plotting
    interval = interval_radacini(coef)
    x_vals = np.linspace(interval[0], interval[1], 1000)
    y_vals = [horner(coef, x) for x in x_vals]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot polynomial
    plt.plot(x_vals, y_vals, label=f'P(x) = {format_polynomial(coef)}', color='blue')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
    
    # Plot roots
    if roots_halley:
        y_halley = [horner(coef, r) for r in roots_halley]
        plt.scatter(roots_halley, y_halley, color='red', marker='s', 
                   s=100, label=f'Halley roots (n={len(roots_halley)})')
    
    if roots_newton4:
        y_newton4 = [horner(coef, r) for r in roots_newton4]
        plt.scatter(roots_newton4, y_newton4, color='green', marker='o', 
                   s=80, label=f'Newton4 roots (n={len(roots_newton4)})')
    
    if roots_newton5:
        y_newton5 = [horner(coef, r) for r in roots_newton5]
        plt.scatter(roots_newton5, y_newton5, color='purple', marker='^', 
                   s=80, label=f'Newton5 roots (n={len(roots_newton5)})')
    
    # Formatting
    plt.title('Polynomial with Found Roots')
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.legend()
    plt.grid(True)
    
    # Adjust y-axis limits to show roots clearly
    y_min, y_max = min(y_vals), max(y_vals)
    plt.ylim(y_min - 0.5, y_max + 0.5)
    
    plt.tight_layout()
    plt.show()

def format_polynomial(coef):
    """Format polynomial coefficients as a string"""
    terms = []
    degree = len(coef) - 1
    for i, c in enumerate(coef):
        power = degree - i
        if power == 0:
            terms.append(f"{c:.2f}")
        elif power == 1:
            terms.append(f"{c:.2f}x")
        else:
            terms.append(f"{c:.2f}x^{power}")
    return " + ".join(terms).replace(" + -", " - ")