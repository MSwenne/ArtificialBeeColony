import numpy as np
import sys

class ArtificialBeeColony:
    def __init__(self, hyperparameters):
        self.budget = hyperparameters[0]
        self.n_employed = hyperparameters[1]
        self.n_onlookers = hyperparameters[2]
        self.n_scouts = hyperparameters[3]
        self.limit = hyperparameters[4]

    # Main function call, optimized a given IOH profiler problem
    def optimize(self, problem):
        self.dimension = problem.number_of_variables
        self.global_best = None
        self.global_best_f = sys.maxsize

        # Generate initial solutions
        solutions = np.array([self.scout_solution(problem) for _ in range(self.n_employed)])
        while not problem.final_target_hit and problem.evaluations < self.budget:
            # send employed bees (exploit existing solutions)
            solutions = np.array([self.exploit(problem, solution, solutions) for solution in solutions])
            # For every onlooker, pick a weighted random source
            onlooker_picks = np.array([self.pick_source_index(solutions) for _ in range(self.n_onlookers)])
            # Some indices will remain unchanged as these were appearantly not that interesting
            # Some will get multiple changes, which is why this has to be a synchronous process
            for i in onlooker_picks:
                solutions[i] = self.exploit(problem, solutions[i], solutions)

            for _ in range(self.n_scouts):
                # Find the stalest solution in the space
                max_i = np.argmax(solutions[:, -2])
                # Replace this problem with a new one
                if solutions[max_i, -2] >= self.limit:
                    solutions[max_i] = self.scout_solution(problem)
        return self.global_best, self.global_best_f

    # Randomly find a new solution
    def scout_solution(self, problem):
        # + 2 for staleness and performance of the solution
        solution = np.random.uniform(low=-5, high=5, size=problem.number_of_variables + 2)
        # Set staleness to 0
        solution[-2] = 0
        return self.evaluate(problem, solution)

    # Evaluate a single solution
    def evaluate(self, problem, solution):
        solution[-1] = problem(solution[:-2])
        solution[-1] -= problem.get_target()
        if solution[-1] < self.global_best_f:
            self.global_best_f = solution[-1]
            self.global_best = solution[:-2]
        return solution

    # Exploit a solution and return it.
    # For new solution generation we use the existing list of solutions
    def exploit(self, problem, solution, solutions):
        # Create copy for array
        mutated_solution = solution.copy()

        # vij = xij + φij(xij − xkj),
        # With xi = solution  
        # k in {0 .. employeed bees}
        # j in {0 .. dimension}
        # vij = solution[j] + U(-1, 1)(solution[j] − solutions[k][j])
        mutate_j = np.random.randint(low=0, high=len(solution) - 2) # -2 because we don't want performance / staleness to be mutated
        mutate_k = np.random.randint(low=0, high=len(solutions)) # Random sample from existing mutations
        mutation_factor = np.random.uniform(low=-1, high=1)
        mutation = solution[mutate_j] + mutation_factor * (solution[mutate_j] - solutions[mutate_k][mutate_j])
        # Make sure the new variable is between -5 and 5
        mutated_solution[mutate_j] = max(-5, min(5, mutated_solution[mutate_j] + mutation))
        mutated_solution = self.evaluate(problem, mutated_solution)
        # Compare solution with previous solution
        if mutated_solution[-1] <= solution[-1]:
            # Reset staleness counter since this is a new solution
            mutated_solution[-2] = 0
            return mutated_solution
        # Increment staleness counter
        solution[-2] += 1
        return solution

    # Probibalistically select an id for a solution
    def pick_source_index(self, solutions):
        # This method requries all function values to be > 0 and thus the lowest value is found to try and create a 0 .. n result space
        f_x = solutions[:, -1]
        f_x_scaled = 1.0 / f_x
        # Calculate the weights of every solution
        total = np.sum(f_x_scaled)
        weights = f_x_scaled / total # Weights = percentage that a solution is picked, these should however be inverted

        # Pick a random solution based on the weights
        return np.random.choice(np.arange(start=0, stop=len(solutions)), p=weights)
