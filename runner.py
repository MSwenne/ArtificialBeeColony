from IOHexperimenter import IOH_function, IOH_logger, IOHexperimenter
from IOHexperimenter.IOHprofiler import IOHprofiler_Problem_double
from numpy.core.numeric import ComplexWarning
from algorithm import ArtificialBeeColony
from multiprocessing import Pool, cpu_count
import time
import CustomFunctions
start = time.time()

# Generates results for a singular configuration
def experiment(configuration):
    custom = {
        -1: CustomFunctions.IOH_Griewank,
        -2: CustomFunctions.IOH_Rastrigin,
        -3: CustomFunctions.IOH_Rosenbrock,
        -4: CustomFunctions.IOH_Ackley,
        -5: CustomFunctions.IOH_Schwefel
    }

    iterations = 30
    # problem_id = [-1, -2, -3, -4, -5] # Griewank, Rastrigin, Rosenbrock, Ackley, Schwefel (as used in the paper)
    problem_id = range(1, 25)
    instance_id = range(1, iterations + 1)

    logger = IOH_logger("./", f"result-{', '.join(map(str, configuration))}", f"abc-{', '.join(map(str, configuration))}", f"abc-{', '.join(map(str, configuration))}")
    for p_id in problem_id:
        print(f"configuration: {', '.join(map(str, configuration))}, problem: {p_id}, dim: {configuration[-1]}")
        for i_id in instance_id:
            # Getting the problem with corresponding id, dimension, and instance.
            if p_id >= 0:
                f = IOH_function(p_id, configuration[-1], i_id, suite="BBOB")
                domain = (-5, 5)
            else:
                f = custom[p_id](configuration[-1], i_id)
                domain = f.get_init_range()

            f.add_logger(logger)
            abc = ArtificialBeeColony(configuration[:-1], domain)
            try:
                xopt, fopt = abc.optimize(f)
            except:
                pass
            
            print(f'\tProblem: {p_id} dim: {configuration[-1]} iteration: {i_id}/{iterations} after: {f.evaluations}\n\t\tBest fitness: {fopt}')
        # print(f"Finished configuration {', '.join(map(str, configuration))}, problem: {p_id}, dim: {configuration[-1]} at:", time.time() - start)
    logger.clear_logger()


if __name__ == '__main__':
    # Launch all configurations in parallel
# Launch all configurations in parallel
    iterations = [500, 750, 1000, 1000, 1500, 2000]
    dimensions = [10, 20, 30, 10, 20, 30]

    population = 125
    n_scouts = 1
    n_employed = n_onlookers = int((125 - n_scouts) / 2)

    # nr_evals, nr_employed, nr_onlookers, nr_scouts, limit, dim
    configurations = [[iter * population, n_employed, n_onlookers, n_scouts, dimension * population * 4 , dimension] for iter, dimension in zip(iterations, dimensions)]
    for c in configurations:
        print(c)

    with Pool(cpu_count()) as p:
        p.map(experiment, configurations)
    print('Finished at:', time.time() - start)

