from IOHexperimenter import custom_IOH_function
import math

class IOH_Ackley():
    def __init__(self) -> None:
        pass

    def func(self, dim, iid) -> custom_IOH_function:
        return custom_IOH_function(self.ackley, "Ackley function", dim, 50001, iid)

    def ackley(*args):
        values = args[1]
        exponent = -0.2 * math.sqrt(sum([parameter ** 2 for parameter in values]) / float(len(values)))
        exponent2 = sum([math.cos(2.0 * math.pi * param) for param in values]) / float(len(values))
        return 20 + math.e - 20 * math.exp(exponent) - math.exp(exponent2)