from models.optimization_task import OptimizationTask
from models.optimization_config import OptimizationConfig
from models.constraint import Constraint, ConstraintType
from models.target_function import TargetFunction
from models.design_variable import DesignVariable

from managers.optimizer import Optimizer

def run_example():
    x = DesignVariable("x", value=100)

    def func(vars): return vars["x"]**2 # x^2
    def constraint1(vars): return (vars["x"] - 2)  # x - 2 <= 0 

    tf = TargetFunction(func, "x^2")
    c1 = Constraint(constraint1, ConstraintType.INEQ, "x -2 <= 0")

    config = OptimizationConfig(max_iter=10000, tol=1e-6)
    task = OptimizationTask([x], tf, [c1], config)

    solver = Optimizer(task)
    solver.optimize()

run_example()
