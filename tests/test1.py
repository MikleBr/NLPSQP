from models.optimization_task import OptimizationTask
from models.optimization_config import OptimizationConfig
from models.constraint import Constraint, ConstraintType
from models.target_function import TargetFunction
from models.design_variable import DesignVariable

from managers.optimizer import Optimizer

def run_example():
    x = DesignVariable("x", 100)
    y = DesignVariable("y", 100)

    def func(vars): return vars["x"]**2 + vars["y"]**2 # x^2 + y^2
    def constraint1(vars): return - (vars["x"] + vars["y"] - 1)  # x + y >= 1 → - (x + y - 1) <= 0

    tf = TargetFunction(func, "x^2 + y^2")
    c1 = Constraint(constraint1, ConstraintType.INEQ, "x + y >= 1")

    config = OptimizationConfig(max_iter=10000, tol=1e-6)
    task = OptimizationTask([x, y], tf, [c1], config)

    solver = Optimizer(task)
    solver.optimize()

run_example()
