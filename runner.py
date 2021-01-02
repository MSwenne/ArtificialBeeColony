from IOHexperimenter import IOH_function, IOH_logger, IOHexperimenter
from IOHexperimenter.IOHprofiler import IOHprofiler_Problem_double
from algorithm import ArtificialBeeColony
from multiprocessing import Pool, cpu_count
import time
import Ackly
start = time.time()

# Generates results for a singular configuration
def experiment(configuration):
    iterations = 30
    problem_id = [19, 24, 9, -1, 20] # Griewank, Rastrigin, Rosenbrock, Ackley, Schwefel (as used in the paper)
    instance_id = range(1, iterations + 1)

    logger = IOH_logger("./", f"result-{', '.join(map(str, configuration))}", f"abc-{', '.join(map(str, configuration))}", f"abc-{', '.join(map(str, configuration))}")
    for p_id in problem_id:
        print(f"configuration: {', '.join(map(str, configuration))}, problem: {p_id}, dim: {configuration[-1]}")
        for i_id in instance_id:
            # Getting the problem with corresponding id, dimension, and instance.
            if p_id != -1:
                f = IOH_function(p_id, configuration[-1], i_id, suite="BBOB")
            else:
                f = Ackly.IOH_Ackley(configuration[-1], i_id)

            f.add_logger(logger)
            abc = ArtificialBeeColony(configuration[:-1])
            xopt, fopt = abc.optimize(f)
            print(f'\tProblem: {p_id} dim: {configuration[-1]} iteration: {i_id}/{iterations} after: {f.evaluations}\n\t\tBest fitness: {fopt}')
        # print(f"Finished configuration {', '.join(map(str, configuration))}, problem: {p_id}, dim: {configuration[-1]} at:", time.time() - start)
    logger.clear_logger()


if __name__ == '__main__':
    # Launch all configurations in parallel
# Launch all configurations in parallel
    iterations = [500, 750, 1000]
    dimensions = [10, 20, 30]

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