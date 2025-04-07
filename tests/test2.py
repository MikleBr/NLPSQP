from models.optimization_task import OptimizationTask
from models.optimization_config import OptimizationConfig
from models.constraint import Constraint, ConstraintType
from models.target_function import TargetFunction
from models.design_variable import DesignVariable
from managers.optimizer import Optimizer


def run_example():
    x = DesignVariable("x", value=2.9, lower=1.0, upper=15.0)
    y = DesignVariable("y", value=2.1, lower=0.0, upper=3.0)

    def tf(vars): return (vars["x"] - 3)**2 + (vars["y"] - 2)**2 # (x-3)^2 + (y-2)^2
    def constraint_func(vars): return vars["x"] + vars["y"] - 5  # x + y <= 5 → x + y - 5 <= 0

    target_function = TargetFunction(tf, "(x-3)^2 + (y-2)^2")
    constraints = [Constraint(constraint_func, ConstraintType.INEQ, "x + y <= 5")]

    config = OptimizationConfig(max_iter=50, tol=1e-8)

    # Построение задачи
    task = OptimizationTask(
        variables=[x, y],
        target=target_function,
        constraints=constraints,
        config=config
    )

    solver = Optimizer(task)
    solver.optimize()


run_example()
