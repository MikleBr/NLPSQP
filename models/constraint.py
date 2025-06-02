from enum import Enum

class ConstraintType(Enum):
    EQ = 0
    INEQ = 1

class Constraint:
    def __init__(self, func, type: ConstraintType, name: str = "", penalty_gain: float = 1.0):
        self.func = func
        self.type = type
        self.name = name
        self.penalty_gain = penalty_gain # Используется для приведения ограничений в merit-функции к одному порядку

    def evaluate(self, variables: dict) -> float:
        return self.func(variables)
