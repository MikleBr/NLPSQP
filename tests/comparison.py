from prettytable import PrettyTable
import unittest
import numpy as np
from managers.problem_json_parser import ProblemJSONParser
from managers.optimizer import Optimizer
from scipy.optimize import minimize
from models.constraint import ConstraintType
from models.optimization_task import OptimizationTask
from autograd import grad

class OptimizationComparisonTest(unittest.TestCase):
    def load_problem(self, path):
        parser = ProblemJSONParser()
        return parser.createProblem(path)
    
    def _print_comparison_table(self, var_names, result_sqp, result_scipy):
        table = PrettyTable()
        table.field_names = ["Переменная", "SQP", "SciPy", "Разница (%)"]
        
        for name, sqp_val, scipy_val in zip(var_names, result_sqp, result_scipy):
            percent_diff = abs(sqp_val - scipy_val) / (abs(scipy_val) + 1e-8) * 100
            table.add_row([
                name,
                f"{sqp_val:.6f}",
                f"{scipy_val:.6f}",
                f"{percent_diff:.3f}"
            ])
        
        print("\nСравнение результатов оптимизации:")
        print(table)

    def scipy_solve(self, problem: OptimizationTask):
        x0 = problem.get_variable_values()
        names = problem.get_variable_names()

        def f(x):
            return problem.target.evaluate(dict(zip(names, x)))

        def jac(x):
            from autograd import grad
            grad_f = grad(lambda x: problem.target.evaluate(dict(zip(names, x))))
            return grad_f(x)

        constraints = []
        for con in problem.constraints:
            expr = lambda x: con.evaluate(dict(zip(names, x)))
            grad_expr = grad(lambda x: con.evaluate(dict(zip(names, x))))
            if con.type == ConstraintType.INEQ:
                constraints.append({'type': 'ineq', 'fun': expr, 'jac': grad_expr})
            elif con.type == ConstraintType.EQ:
                constraints.append({'type': 'eq', 'fun': expr, 'jac': grad_expr})
            else :
                raise ValueError(f"Неподдерживаемый тип ограничения: {con.type}")

        bounds = [(v.lower, v.upper) for v in problem.variables]

        res = minimize(f, x0, jac=jac, constraints=constraints, bounds=bounds, method='Nelder-Mead')
        print(problem.target.evaluate(dict(zip(names, res.x))))
        return res.x

    def test_sqp_vs_scipy(self):
        problem_path = "tests/bowl.json"
        task = self.load_problem(problem_path)

        optimizer = Optimizer(task)
        result_sqp = optimizer.optimize()
        result_sqp_vals = np.array([result_sqp[var.name] for var in task.variables])

        task2 = self.load_problem(problem_path)
        result_scipy = self.scipy_solve(task2)

        self._print_comparison_table(task.get_variable_names(), result_sqp_vals, result_scipy)
        

