import sys
sys.path.append('../')
#import printFile as mn
import numpy as np
import matplotlib.pyplot as plt
from copy import copy


def Schwarze_Swenne_ga(dim, eval_budget, fitness_func, do_plot=False, return_stats=False):
    """
    extended skeleton for the genetic algorithm. 
    
    :param dim:          problem dimension, which should be either 2 (square) or 3 (cube)
    :param eval_budget:  int, the function evaluation budget
    :param fitness_func: function handle, you should use one of the
                         evaluation function provided
    :param do_plot:      should interactive plots be drawn during the optimization
    :param return_stats: should the convergence history be returned too

    :return:
        xopt : array, the final solution vector found by the algorithm
        fopt : double, the corresponding fitness value of xopt

    Author: Koen van der Blom, Hao Wang, Sander van Rijn
    Edited by: Yenebeb Schwarze, Martijn Swenne
    Last modified: 2018-12-23
    """

    # ----------------- general setting of variables ----------------------

    # static variables
    # problem size: 12 for square and 7 for cube
    if dim == 2:
        n = 12
    elif dim == 3:
        n = 7
    else:
        raise ValueError('Invalid number of dimensions, use 2 or 3')

    # the pheno type of the solution is the permutation of integers
    # from 1 to n ^ dim
    pheno_len = n**dim

    # At this point, you should think which geno type representation you
    # would like to use and thus determine the length of the geno type
    # solution vector. Example of geno encoding: bit-string converted from
    # array of integers. This is up to you :)
    geno_len = n**dim  # we use permutation representation

    # endogenous parameters setting
    mu = 10               # population size
    pc = 0               # crossover rate
    pm = 1/2               # mutation rate

    # internal counter variable
    evalcount = 0     # count function evaluations
    gencount = 0      # count generation/iterations

    # historical information
    hist_best_f = np.zeros(eval_budget)
    hist_gen_f = np.zeros(int (np.ceil(eval_budget/mu)))


    # ------------------- population initialization -----------------------
    # row vector representation is used throughout this code
    # you need to keep pheno type population updated with the geno types
    # for function evaluation

    # population
    pop_pheno = np.zeros((mu, pheno_len))   # pheno type
    pop_geno = np.zeros((mu, geno_len))     # geno type
    fitness = np.zeros(mu)                  # fitness values

    for i in range(mu):
        pop_pheno[i, :] = make_random(n, dim)  # generate pheno type individual uniformly
        pop_geno[i, :] = pop_pheno[i, :]   # convert them to geno type solution
        fitness[i] = fitness_func(pop_pheno[i, :])   # and evaluate the
                                                     # solution...

    index = np.argmin(fitness)
    fopt = fitness[index]
    xopt = copy(pop_geno[index, :])
    x_opt_curr_gen = copy(pop_geno[index, :])
    xopt_pheno = copy(pop_pheno[index, :])

    # increase the evalcount by mu
    hist_best_f[evalcount:evalcount+mu] = fopt
    evalcount += mu

    if do_plot:
        plt.ion()
        fig = plt.figure()

        ax1 = plt.subplot(131)
        line1, = ax1.plot(hist_best_f[:evalcount])
        ax1.set_title('minimal global error')
        ax1.set_ylabel('error')
        ax1.set_xlabel('evaluations')
        ax1.set_ylim([0, np.max(hist_best_f)])

        ax2 = plt.subplot(132)
        line2, = ax1.plot(hist_gen_f[:gencount])
        ax2.set_title('minimal error in the current generation')
        ax2.set_ylabel('error')
        ax2.set_xlabel('generation')
        ax2.set_ylim([0, np.max(hist_gen_f)])

        ax3 = plt.subplot(133)
        bar3, = ax2.bar(np.arange(pheno_len), xopt_pheno)
        ax3.set_title('best chromosome')
        ax3.set_ylabel('value')
        ax3.set_xlabel('phenotype index')

        plt.show(block=False)

    #mn.printOut(xopt, n, 0)
    # ----------------------- Evolution loop ------------------------------
    while evalcount < eval_budget:

        # generate the a new population using crossover and mutation
        pop_new_geno = np.zeros((mu, geno_len))
        for i in range(mu):

            # implement the selection operator.
            p1 = copy(xopt)               # select the first parent from pop_geno
            if np.random.randn() < pc:
                p2 = copy(xopt)           # select the second parent from pop_geno
                # implement the crossover operator
                # crossover p1 and p2
                pop_new_geno[i] = copy(order1crossover(p1,p2,geno_len))

            else:

                # No crossover, copy the parent chromosome
                pop_new_geno[i] = copy(p1)


            # implement the muation operator
                # apply the mutation and then
                # store it in pop_new_geno
            pop_new_geno[i, :] = copy(swap(pop_new_geno[i, :], pm))
            pop_new_geno[i, :] = copy(inversion(pop_new_geno[i, :], pm))

        # Replace old population by the newly generated population
        pop_geno = copy(pop_new_geno)

        for i in range(mu):
            pop_pheno[i, :] = copy(pop_geno[i, :])   # decode the geno type solution to
            fitness[i] = fitness_func(pop_pheno[i, :])

        # optimal solution in each iteration
        index = np.argmin(fitness)
        x_opt_curr_gen = pop_geno[index, :]
        x_opt_pheno_curr_gen = pop_pheno[index, :]
        fopt_curr_gen = fitness[index]

        # keep track of the best solution ever found
        if fopt_curr_gen < fopt:
            fopt = fopt_curr_gen
            xopt = x_opt_curr_gen
            xopt_pheno = x_opt_pheno_curr_gen
            #mn.printOut(xopt, n, 1)

        # record historical information
        hist_best_f[evalcount:evalcount+mu] = fopt
        hist_gen_f[gencount] = fopt_curr_gen

        # internal counters increment
        gencount += 1
        evalcount += mu

        # Plot statistics
        if do_plot:
            line1.set_data(np.arange(evalcount), hist_best_f[:evalcount])
            ax1.set_xlim([0, evalcount])
            ax1.set_ylim([0, np.max(hist_best_f)])

            line2.set_data(np.arange(gencount), hist_gen_f[:gencount])
            ax2.set_xlim([0, gencount])
            ax2.set_ylim([0, np.max(hist_gen_f)])

            bar3.set_ydata(xopt_pheno)

            plt.draw()

    if return_stats:
        #mn.printOut(xopt, n, 10)
        return xopt, fopt, hist_best_f
    else:
        #mn.printOut(xopt, n, 10)
        return xopt, fopt


def make_random(n, dim):
    m = n**dim
    numbers = list(range(1,m+1))
    x = [1 for i in range(m)]
    # fill the matrix conf_arr random with numbers
    for i in range(m):
        index = np.random.randint(0, len(numbers))
        x[i] = numbers[index-1]
        numbers.remove(numbers[index-1])
    numbers = list(range(1,m+1))
    return x

def order1crossover(p1, p2, geno_len):
    child = np.zeros(geno_len)
    visited = np.zeros(geno_len)
    same = True
    while same:
        i1 = np.random.randint(0, geno_len)
        i2 = np.random.randint(0, geno_len)
        if i1 > i2:
            i1, i2 = i2, i1
        if i1 != i2:
            same = False
    for i in range(i1, i2):
        child[i] = copy(p1[i])
        visited[int (p1[i])-1] = True
    i_child = i2
    for i in range(i2, geno_len):
        if not(visited[int (p2[i])-1]):
            child[i_child] = p2[i]
            visited[int (p2[i])-1] = True
            i_child += 1
            if i_child >= geno_len:
                i_child = 0
    for i in range(0, i2):
        if not(visited[int (p2[i])-1]):
            child[i_child] = p2[i]
            visited[int (p2[i])-1] = True
            i_child += 1
            if i_child >= geno_len:
                i_child = 0
    return child

def swap(lst, pm):
    same = True
    x = np.random.randint(0,100)
    x = x/100
    if(x < pm):
        while same:
            i1 = np.random.randint(0, len(lst)-1)
            i2 = np.random.randint(0, len(lst)-1)
            if i1 != i2:
                same = False
        lst[i1], lst[i2] = lst[i2], lst[i1]
    return lst

def inversion(lst, pm):
    same = True
    x = np.random.randint(0,100)
    x = x/100
    if(x < pm):
        while same:
            i1 = np.random.randint(0, len(lst))
            i2 = np.random.randint(0, len(lst))
            if i1 > i2:
                i1, i2 = i2, i1
            if i1 != i2:
                same = False
        for i in range(i1, int ((i1+i2)/2)):
            lst[i1], lst[i2] = lst[i2], lst[i1]
            i1 += 1
            i2 -= 1
    return lst
