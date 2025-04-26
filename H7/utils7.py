import math

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