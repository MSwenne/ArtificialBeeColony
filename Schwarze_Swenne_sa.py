import sys
sys.path.append('../')
#import printFile as mn
import numpy as np
import math
import matplotlib.pyplot as plt
from copy import copy

def Schwarze_Swenne_sa(dim, eval_budget, fitness_func, do_plot=True, return_stats=False):
    """
    extended skeleton for simulated anealing algorithm. 

    :param dim:          problem dimension, which should be either 2 (square) or 3 (cube)
    :param eval_budget:  int, the function evaluation budget
    :param fitness_func: function handle, you should use one of the evaluation function provided
    :param do_plot:      should interactive plots be drawn during the optimization
    :param return_stats: should the convergence history be returned too

    :return:
       xopt : array, the final solution vector found by the algorithm
       fopt : double, the corresponding fitness value of xopt

    Author: Koen van der Blom, Hao Wang, Sander van Rijn
    Edited by: Yenebeb Schwarze, Martijn Swenne
    Last modified: 2018-12-23
    """

    # Initialize static parameters
    pm = 2        # mutation rate
    alpha = 0.6      # temperature decaying parameter
    k = 100        # number of evaluations per iteration
    num_iterations = int (np.ceil(eval_budget / k))

    # problem size: 12 for square and 7 for cube
    if dim == 2:
        n = 12
    elif dim == 3:
        n = 7
    else:
        raise ValueError('Invalid number of dimensions, use 2 or 3')

    # Set initial temperature
    T = 1000

    # Statistics data
    evalcount = 0
    itercount = 0
    fopt = np.inf
    xopt = np.array([np.nan] * n**dim)
    hist_best_f = np.array([np.nan] * eval_budget)
    hist_iter_f = np.array([np.nan] * num_iterations)
    hist_temperature = np.array([np.nan] * num_iterations)

    # Generate initial solution and evaluate
    x = make_random(n, dim)
    f = fitness_func(x)         # evaluate the solution using fitness_func

    fopt = f
    xopt = copy(x)

    evalcount += 1             # Increase evaluation counter
    hist_best_f[evalcount] = fopt

    if do_plot:
        plt.ion()
        fig = plt.figure()

        ax1 = plt.subplot(131)
        line1 = ax1.plot(hist_best_f[:evalcount])[0]
        ax1.set_title('minimal global error')
        ax1.set_ylabel('error')
        ax1.set_xlabel('evaluations')
        ax1.set_ylim([0, np.max(hist_best_f[:evalcount])])

        ax2 = plt.subplot(132)
        line2 = ax2.plot(np.arange(itercount), hist_temperature[:itercount])[0]
        ax2.set_title('temperature')
        ax2.set_ylabel('T')
        ax2.set_xlabel('iteration')
        ax2.set_ylim([0, T])

        ax3 = plt.subplot(133)
        bars3 = ax3.bar(np.arange(len(xopt)), xopt)
        ax3.set_title('best representation')
        ax3.set_ylabel('value')
        ax3.set_xlabel('representation index')

        plt.show(block=False)

    #mn.printOut(xopt, n, 0)
    restart_best = f
    restart_bestx = x
    restart_bestt = T
    # ----------------------- Evolution loop ------------------------------
    while fopt > 0 and evalcount < eval_budget and itercount < num_iterations:

        hist_temperature[itercount] = T

        k = min(k, eval_budget-evalcount)
        for j in range(k):

            # Generate a new solution by the permutation of x
            x_new = copy(x)
            for p in range(pm):
                x_new = make_permutation(x_new)
            f_new = fitness_func(x_new)   # evaluate the new solution
            if f_new < f:
                # accept the new solution when it is better
                x = x_new
                f = f_new
            elif T > 0.0:
            	if round(0, 1) < math.exp(-(f_new - f)/T):
            	    # choose to accept or reject the new solution
            	    # probabilistically based on the current temperature
            	    x = x_new
            	    f = f_new

            # update the best solution found so far
            if f < fopt:
                fopt = f
                xopt = copy(x)
                #mn.printOut(xopt, n, 1)

            hist_best_f[evalcount] = fopt   # tracking the best fitness
            # ever found

            # Generation best statistics
            hist_iter_f[itercount] = f

            # Plot statistics
            if do_plot:
                line1.set_data(np.arange(evalcount), hist_best_f[:evalcount])
                ax1.set_xlim([0, evalcount])
                ax1.set_ylim([0, np.max(hist_best_f[:evalcount])])

                line2.set_data(np.arange(itercount), hist_temperature[:itercount])
                ax2.set_xlim([0, itercount])

                for bar, h in zip(bars3, xopt):
                    bar.set_height(h)

                plt.pause(0.00001)
                plt.draw()
        
            evalcount += 1   # Increase evaluation counter

        # Temperature update
        T = round(alpha*T, 10)
        
        itercount += 1   # Increase iteration counter
    if return_stats:
        #mn.printOut(xopt, n, 10)
        return xopt, fopt, hist_best_f
    else:
        #mn.printOut(xopt, n, 10)
        return fopt


def make_random(n, dim):
    m = n**dim
    numbers = list(range(1,m+1))
    x = [1 for i in range(m)]
    # fill the matrix conf_arr random with numbers
    for i in range(m):
        index = np.random.randint(0, len(numbers))
        x[i] = numbers[index-1]
        numbers.remove(numbers[index-1])
    return x


def make_permutation(lst):
    same = True
    while same:
        index_one = np.random.randint(0, len(lst))
        index_two = np.random.randint(0, len(lst))
        if index_one != index_two:
            same = False
    lst[index_one], lst[index_two] = lst[index_two], lst[index_one]
    return lst
