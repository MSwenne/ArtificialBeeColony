from IOHexperimenter import IOH_function, IOH_logger, IOHexperimenter
from algorithm import ArtificialBeeColony
from multiprocessing import Pool


# Generates results for a singular configuration
def experiment(configuration):
    problem_id = range(1, 25)
    instance_id = range(1, 26)
    dimension = [2, 5, 20]

    logger = IOH_logger("./", f"result-{', '.join(map(str, configuration))}", f"abc-{', '.join(map(str, configuration))}", f"abc-{', '.join(map(str, configuration))}")
    for p_id in problem_id:
        for d in dimension:
            print(f"configuration: {', '.join(map(str, configuration))}, problem: {p_id}, dim: {d}")
            for i_id in instance_id:
                # Getting the problem with corresponding id,dimension, and instance.
                f = IOH_function(p_id, d, i_id, suite="BBOB")
                f.add_logger(logger)
                abc = ArtificialBeeColony(configuration)
                xopt, fopt = abc.optimize(f)
    logger.clear_logger()


if __name__ == '__main__':
    # Launch all configurations in parallel
    # with Pool() as p:
    #     p.map(experiment, configurations)
    configuration = [10000, 50, 50, 2, 10]
    experiment(configuration)