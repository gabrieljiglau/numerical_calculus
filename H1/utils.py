import random
import time

c1 = 0.16666666666666666666666666666667
c2 = 0.00833333333333333333333333333333
c3 = 1.984126984126984126984126984127e-4
c4 = 2.7557319223985890652557319223986e-6
c5 = 2.5052108385441718775052108385442e-8
c6 = 1.6059043836821614599392377170155e-10


def double_between(lower, upper):
    return lower + (random.uniform(0, 1)) * (upper - lower)


def get_associative_factors(lower, upper, epsilon):
    first, second, third = 0, epsilon, epsilon

    for i in range(100000000000):
        a = (first * second) * third
        b = first * (second * third)
        if a != b:
            print(f"Found the solution in {i} iterations")
            print(f"a = {a}, b = {b}")
            return first, second, third
        else:
            first = double_between(lower, upper)
            print(f"first = {first}")


def get_p1_value(num, num_squared):
    return num * (1 + num_squared * (-c1 + num_squared * (c2 - c3)))


def get_p2_value(num, num_squared):
    return num * (1 + num_squared * (-c1 + num_squared * (c2 - c3 * num_squared)))


def get_p3_value(num, num_squared):
    return num * (1 + num_squared * (-c1 + num_squared * (c2 + num_squared * (-c3 + c4 * num_squared))))


def get_p4_value(num, num_squared):
    return num * (1 + num_squared * (-0.166 + num_squared * (0.00833 + num_squared
                                                             * (-c3 + c4 * num_squared))))


def get_p5_value(num, num_squared):
    return num * (1 + num_squared * (-0.1666 + num_squared * (0.008333 + num_squared
                                                              * (-c3 + c4 * num_squared))))


def get_p6_value(num, num_squared):
    return num * (1 + num_squared * (-0.16666 + num_squared * (0.0083333 + num_squared * (-c3 + c4 * num_squared))))


def get_p7_value(num, num_squared):
    return num * (
                1 + num_squared * (-c1 + num_squared * (c2 + num_squared * (-c3 + num_squared * (c4 - c5 * num_squared)))))


def get_p8_value(num, num_squared):
    return num * (1 + num_squared * (-c1 + num_squared * (c2 + num_squared * (-c3 + num_squared * (c4 - num_squared * (c5 + c6 * num_squared))))))

def calculate_ps(num, squared_num):

    """
    aici gepeto
    """

    p = []
    times = []

    for get_p_value in [get_p1_value, get_p2_value, get_p3_value, get_p4_value,
                        get_p5_value, get_p6_value, get_p7_value, get_p8_value]:
        poly_times = []
        start_time = time.perf_counter()
        p.append(get_p_value(num, squared_num))
        poly_times.append(time.perf_counter() - start_time)
        times.append(poly_times)

    return p, times

def compute_error(polynomial_results, current_output):

    losses = {i: 0 for i in range(len(polynomial_results))}
    for i in range(len(polynomial_results)):
        losses[i] = abs(polynomial_results[i] - current_output)

    return losses
