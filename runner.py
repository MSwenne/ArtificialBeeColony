from IOHexperimenter import IOH_function, IOH_logger, IOHexperimenter
from IOHexperimenter.IOHprofiler import IOHprofiler_Problem_double
from algorithm import ArtificialBeeColony
from multiprocessing import Pool, cpu_count
import time
import Ackly
start = time.time()

# Generates results for a singular configuration
def experiment(configuration):
    iterations = 1
    problem_id = [19, 24, 9, -1, 20] # Griewank, Rastrigin, Rosenbrock, Ackley, Schwefel (as used in the paper)
    # problem_id = range(1, 2)
    instance_id = range(1, iterations + 1)
    dimension = [10]

    logger = IOH_logger("./", f"result-{', '.join(map(str, configuration))}", f"abc-{', '.join(map(str, configuration))}", f"abc-{', '.join(map(str, configuration))}")
    for p_id in problem_id:
        for d in dimension:
            print(f"configuration: {', '.join(map(str, configuration))}, problem: {p_id}, dim: {d}")
            for i_id in instance_id:
                # Getting the problem with corresponding id,dimension, and instance.
                if p_id != -1:
                    f = IOH_function(p_id, d, i_id, suite="BBOB")
                else:
                    f = Ackly.IOH_Ackley().func(d, i_id)

                f.add_logger(logger)
                abc = ArtificialBeeColony(configuration)
                xopt, fopt = abc.optimize(f)
                print(f'\tProblem: {p_id} dim: {d} iteration: {i_id}/{iterations} after: {f.evaluations}\n\t\tBest fitness: {fopt}')
            # print(f"Finished configuration {', '.join(map(str, configuration))}, problem: {p_id}, dim: {d} at:", time.time() - start)
    logger.clear_logger()


if __name__ == '__main__':
    # Launch all configurations in parallel
    iterations = [500]
    population = 125
    limit = 5000
    # nr_evals, nr_employed, nr_onlookers, nr_scouts, limit
    configurations = [[iter*population, int((population-1)/2), int((population-1)/2), 1, limit] for iter in iterations]

    with Pool(cpu_count()) as p:
        p.map(experiment, configurations)
    print('Finished at:', time.time() - start)