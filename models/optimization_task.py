from typing import List
from models.optimization_config import OptimizationConfig
from models.target_function import TargetFunction
from models.constraint import Constraint
from models.design_variable import DesignVariable
import numpy as np

class OptimizationTask:
    def __init__(
        self,
        variables: List[DesignVariable],
        target: TargetFunction,
        constraints: List[Constraint],
        config: OptimizationConfig
    ):
        self.variables = variables
        self.target = target
        self.constraints = constraints
        self.config = config

    def get_variable_dict(self) -> dict:
        return {v.name: v.value for v in self.variables}

    def get_variable_values(self) -> np.ndarray:
        return np.array([v.value for v in self.variables])
    
    def get_variable_names(self) -> np.ndarray:
        return np.array([v.name for v in self.variables])

    def set_variable_values(self, values: np.ndarray):
        for v, val in zip(self.variables, values):
            v.value = val
            v.clip()
