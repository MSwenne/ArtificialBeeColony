from IOHexperimenter import custom_IOH_function
import math

# Function 1 in the paper
class IOH_Griewank(custom_IOH_function):
    def __init__(self, dim, iid) -> None:  
        super().__init__(self.griewank, "Griewank function (f1)", dim, 50001, iid)

    def griewank(*args):
        values = args[1]
        left = 1.0 / 4000 * sum([x**2 for x in values])
        right = math.prod([math.cos(x / math.sqrt(i + 1)) for i, x in enumerate(values)])
        return left - right + 1

    def get_target(self, raw=False):
        return 0

    def get_init_range(self):
        return -600, 600

# Function 2 in paper
class IOH_Rastrigin(custom_IOH_function):
    def __init__(self, dim, iid) -> None:  
        super().__init__(self.rastrigin, "Rastrigin function (f2)", dim, 50002, iid)

    def rastrigin(*args):
        values = args[1]
        inner = [x**2 - 10 * math.cos(2 * math.pi * x) + 10 for x in values]
        return sum(inner)

    def get_target(self, raw=False):
        return 0

    def get_init_range(self):
        return -15, 15

# Function 3 in paper
class IOH_Rosenbrock(custom_IOH_function):
    def __init__(self, dim, iid) -> None:  
        super().__init__(self.rosenbrock, "Rosenbrock function (f3)", dim, 50003, iid)

    def rosenbrock(*args):
        values = args[1]
        inner = [100 * (x**2 - values[i + 1] )**2 + (1 - x)**2 for i, x in enumerate(values[:-1])]
        # Add last element aswell
        last = values[-1]
        inner.append(100 * (last**2 - 0)**2 + (1 - last)**2)
        return sum(inner)

    def get_target(self, raw=False):
        return 0

    def get_init_range(self):
        return -15, 15


# Function 4 in paper
class IOH_Ackley(custom_IOH_function):
    def __init__(self, dim, iid) -> None:  
        super().__init__(self.ackley, "Ackley function (f4)", dim, 50004, iid)

    def ackley(*args):
        values = args[1]
        exponent = -0.2 * math.sqrt(sum([parameter ** 2 for parameter in values]) / float(len(values)))
        exponent2 = sum([math.cos(2.0 * math.pi * param) for param in values]) / float(len(values))
        return 20 + math.e - 20 * math.exp(exponent) - math.exp(exponent2)

    def get_target(self, raw=False):
        return 0

    def get_init_range(self):
        return -32.768, 32.768

# Function 5 in paper
class IOH_Schwefel(custom_IOH_function):
    def __init__(self, dim, iid) -> None:  
        super().__init__(self.schwefel, "Schwefel function (f5)", dim, 50005, iid)

    def schwefel(*args):
        values = args[1]
        inner = [-x * math.sin(math.sqrt(abs(x))) for x in values]
        return len(values) * 418.9829 + sum(inner)

    def get_target(self, raw=False):
        return 0

    def get_init_range(self):
        return -500, 500