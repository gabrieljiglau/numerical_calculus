import sys
import numpy as np
import math
from utils7 import horner, derivata, radacini_distincte

original_stdout = sys.stdout

with open('output.txt', 'w', encoding='utf-8') as f:
    sys.stdout = f  # Change the standard output to the file we created
    def interval_radacini(coef):
        """Calculează intervalul [-R, R] unde se află rădăcinile reale."""
        A = max(abs(c) for c in coef[1:])  # pentru primul: 11
        R = (abs(coef[0]) + A) / abs(coef[0])  # pentru primul: (1 + 11) / 1
        return (-R, R)

    def halley(coef, x, epsilon, kmax):
        """Implementează metoda lui Halley pentru aproximarea rădăcinilor."""
        dP_coef = derivata(coef)
        ddP_coef = derivata(dP_coef)
        for k in range(kmax):
            P = horner(coef, x)
            dP = horner(dP_coef, x)
            ddP = horner(ddP_coef, x)
            A = 2 * (dP**2) - P * ddP
            if abs(A) < epsilon:
                #print("Posibila eroare, A este prea mic")
                #print(f"Halley1: {x} la iteratia {k}")
                return "esuata"
            delta = 2 * P * dP / A
            x = x - delta
            if abs(delta) < epsilon:
                #print(f"Halley2: {x} la iteratia {k}")
                return x
            if abs(delta) > 10 ** 8:
                return "Divergenta"
        return "Divergenta"

    def newton4(coef, x0, epsilon=1e-10, kmax=100):
        """Newton fourth-order method (N4)"""
        for _ in range(kmax):
            P = horner(coef, x0)
            dP = horner(derivata(coef), x0)
            
            if abs(P) < epsilon:
                return x0

            if abs(dP) < epsilon:
                return "Divergenta"
            
            y = x0 - P / dP
            Py = horner(coef, y)
            
            numerator = P**2 + Py**2
            denominator = dP * (P - Py)
            
            if abs(denominator) < epsilon:
                return "Divergenta"
            
            delta = numerator / denominator
            x_new = x0 - delta
            
            if abs(x_new - x0) < epsilon * max(1, abs(x_new)):
                return x_new
            
            if abs(delta) > 1e12 or not math.isfinite(x_new):
                return "Divergenta"
            
            x0 = x_new
        
        return  "Divergenta"

    def newton5(coef, x0, epsilon=1e-10, kmax=100):
        """Newton-type fifth-order method (N5)"""
        for _ in range(kmax):
            P = horner(coef, x0)
            dP = horner(derivata(coef), x0)
            
            if abs(P) < epsilon:
                return x0

            if abs(dP) < epsilon:
                return "Divergenta"
            
            y = x0 - P / dP
            Py = horner(coef, y)
            
            numerator_z = P**2 + Py**2
            denominator_z = dP * (P - Py)
            
            if abs(denominator_z) < epsilon:
                return "Divergenta"
            
            z = x0 - numerator_z / denominator_z
            Pz = horner(coef, z)
            
            x_new = z - Pz / dP
            
            if abs(x_new - x0) < epsilon * max(1, abs(x_new)):
                return x_new
            
            if abs(x_new - x0) > 1e12 or not math.isfinite(x_new):
                return "Divergenta"
            
            x0 = x_new
        return "Divergenta"


    # Exemple de polinoame
    polinoame = [
        [1, -6, 11, -6],
        [42/42, -55/42, -42/42, 49/42, -6/42],
        [8/8, -38/8, 49/8, -22/8, 3/8],
        [1, -6, 13, -12, 4]
    ]

        # Parametri
    epsilon = 1e-10
    kmax = 5000
    step = 0.1
    #fixed_test_values = [1, 2, 3, 2/3, 1/7, -1, 1.5, 0.5, 0.25]
    fixed_test_values = [0]


        # Calculul rădăcinilor pentru fiecare polinom
    for coef in polinoame:
        interval = interval_radacini(coef)
        print(f"\nIntervalul rădăcinilor pentru {coef}: {interval}")
        radacini_halley = []
        radacini_newton4 = []
        radacini_newton5 = []

        for x0 in np.concatenate([np.arange(interval[0], interval[1] + step, step), np.array(fixed_test_values)]):
            radacina = halley(coef, x0, epsilon, kmax)
            if isinstance(radacina, float):
                radacini_halley.append(radacina)
            radacina = newton4(coef, x0, epsilon, kmax)
            if isinstance(radacina, float):
                radacini_newton4.append(radacina)
            radacina = newton5(coef, x0, epsilon, kmax)
            if isinstance(radacina, float):
                radacini_newton5.append(radacina)
        radacini_dist_halley = radacini_distincte(radacini_halley, epsilon)
        radacini_dist_newton4 = radacini_distincte(radacini_newton4, epsilon)
        radacini_dist_newton5 = radacini_distincte(radacini_newton5, epsilon)

        print(f"\nRădăcini distincte pentru metoda Halley: {coef}: {radacini_dist_halley}")

        for r in radacini_dist_halley:
            valoare = horner(coef, r)
            print(f"P({r:.12f}) = {valoare:.2e}")


        print(f"Rădăcini distincte pentru metoda Newton4: {coef}: {radacini_dist_newton4}")

        for r in radacini_dist_newton4:
            valoare = horner(coef, r)
            print(f"P({r:.12f}) = {valoare:.2e}")

        print(f"Rădăcini distincte pentru metoda Newton5: {coef}: {radacini_dist_newton5}")

        for r in radacini_dist_newton5:
            valoare = horner(coef, r)
            print(f"P({r:.12f}) = {valoare:.2e}")

    sys.stdout = original_stdout  # Reset the standard output to its original value