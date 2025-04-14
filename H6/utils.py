import math


def f_poly(x_in):
    return x_in**4  - 12 * x_in**3 + 30 * x_in**2 + 12

def f1_trig(x_in):
    return sin(x_in) - cos(x_in)

def f2_trig(x_in):
    return sin(2 * x_in) + sin(x_in) + cos(3 * x_in)

def f3_trig(x_in):
    return sin(x_in) ** 2 - cos(x_in) ** 2


if __name__ == '__main__':
    print(f_poly(1))