import numpy as np
import random
from copy import copy
import sys

# Using magic square/cube as initial problem

# Using dimension 2 (magic square) as initial problem
dim = 2

if dim == 2:
    n = 12
elif dim == 3:
    n = 7
else:
    raise ValueError('Invalid number of dimensions, use 2 or 3')

solution_length = n**dim

employed = 50
onlookers = 100
scouts = 5

# initial_food_sources = []
# for _ in range(employed):
#     initial_food_sources.append([random.randint(0,1) for _ in range(solution_length)])

max_iterations = 100
alpha = 1 # initial value of penalty parameterfor jth agent?
EC_length = 5 # length  of ejection chain neigbourhood?


# Pseudo code for the algorithm
def ABC_algorithm():
    pass
    # configure params
    # initialize bees
    # memorize best source? (initialize sources?)
    # while (not(stopping condition)):
    #    send employed bees
    #    calculate probabilities
    #    send onlooker bees
    #    memorize best source
    #    send scout bees
    #    increase cycle

#############
# FUNCTIONS #
#############

# Generate new random solutions by a Bernoulli process according to https://arxiv.org/pdf/2003.11641.pdf
# (this is to make it possible to apply ABC to binary optimization problems)
def new_random_solution_bernoulli(x_i):
    for d in range(len(x_i)):
        new_value = 1
        if random.random() < 0.5:
            new_value = 0
        x_i[d] = new_value
    return x_i

# Probability function according to https://arxiv.org/pdf/2003.11641.pdf
def probability(x, x_i):
    fit_sum = 0
    for i in range(len(x)):
        fit_sum += Fit(x[i])
    return Fit(x_i)/fit_sum

# Fit function according to https://arxiv.org/pdf/2003.11641.pdf
def Fit(x_i):
    if x_i > 0:
        return 1/(1+f(x_i))
    else:
        return 1 + abs(f(x_i))

# TODO
def f(x):
    return x