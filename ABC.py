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
population_size = 20



employed = np.floor(population_size/2)
onlookers = np.ceil(population_size/2)
scouts = 0

initial_food_sources = []
for _ in range(employed):
    initial_food_sources.append([random.randint(0,1) for _ in range(solution_length)])