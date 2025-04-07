from enum import Enum

class ConstraintType(Enum):
    EQ = 0
    INEQ = 1

class Constraint:
    def __init__(self, func, type: ConstraintType, name: str = ""):
        self.func = func
        self.type = type
        self.name = name

    def evaluate(self, variables: dict) -> float:
        return self.func(variables)
