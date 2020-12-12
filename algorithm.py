import numpy as np
import sys

class ArtificialBeeColony:
    def __init__(self, hyperparameters):
        self.budget = hyperparameters[0]
        self.n_employed = hyperparameters[1]
        self.n_onlookers = hyperparameters[2]
        self.n_scouts = hyperparameters[3]
        self.n_solutions = hyperparameters[4]

    # Main function call, optimized a given IOH profiler problem
    def optimize(self, problem):
        self.dimension = problem.number_of_variables
        self.global_best = None
        self.global_best_f = -sys.maxsize

        # Generate initial solutions
        solutions = np.array([self.scout_solution(problem) for _ in range(self.n_employed)])
        print(solutions.shape)

        while not problem.final_target_hit and problem.evaluations < self.budget:
            # send employed bees (exploit existing solutions)
            solutions = np.array([self.exploit(problem, solution) for solution in solutions])

            # For every onlooker, pick a weighted random source
            onlooker_picks = np.array([self.pick_source_index(solutions) for _ in range(self.n_onlookers)])
            # Some indices will remain unchanged as these were appearantly not that interesting
            # Some will get multiple changes, which is why this has to be a synchronous process
            for i in onlooker_picks:
                solutions[i] = self.exploit(problem, solutions[i])

            # TODO: Analyze stagnated food sources
            # TODO: Send scout bees to stagnated food sources
            # TODO: Swap out stagnated foodsouce with 
        
        return self.global_best, self.global_best_f

    # Evaluate a single solution
    def evaluate(self, problem, solution):
        solution[-1] = problem(solution[:-1])
        if solution[-1] < self.global_best_f:
            self.global_best_f = solution[-1]
            self.global_best = solution[:-1]
        return solution

    # Exploit a solution and return it.
    def exploit(self, problem, solution):
        # Create copy for array
        mutated_solution = solution.copy()        
        # Mutate solution (this was not defined in the paper)
        mutation_array = np.random.uniform(low=-0.1, high=0.1, size=problem.number_of_variables + 1)
        mutated_solution += mutation_array
        mutated_solution = self.evaluate(problem, mutated_solution)
        
        # Compare solution with previous solution
        # TODO: Measure whether or not this solution is "stagnating", probably by incrementing a counter somewhere some place.
        if mutated_solution[-1] < solution[-1]:
            return mutated_solution
        return solution

    # Probibalistically select an id for a solution
    def pick_source_index(self, solutions):

        # This method requries all function values to be > 0 and thus the lowest value is found to try and create a 0 .. n result space
        f_x = solutions[:, -1]
        f_x_scaled = f_x + abs(np.min(f_x))

        # Calculate the weights of every solution
        total = np.sum(f_x_scaled)
        weights = f_x_scaled / total

        # Pick a random solution based on the weights
        return np.random.choice(np.arange(start=0, stop=len(solutions)), p=weights)

    # Randomly find a new solution
    def scout_solution(self, problem):
        solution = np.random.uniform(low=-5, high=5, size=problem.number_of_variables + 1)
        return self.evaluate(problem, solution)