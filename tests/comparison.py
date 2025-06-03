from prettytable import PrettyTable
import unittest
import numpy as np
from managers.problem_json_parser import ProblemJSONParser
from managers.optimizer import Optimizer
from scipy.optimize import minimize
from models.constraint import ConstraintType
from models.optimization_task import OptimizationTask
from autograd import grad
import matplotlib.pyplot as plt
from typing import List

def plot_convergence(values):
    """
    Строит график сходимости алгоритма оптимизации.

    :param values: Список значений для оси Y.
    """
    # Создаем массив индексов для оси X
    x = list(range(len(values)))

    # Создаем график
    plt.plot(x, values)

    # Добавляем подписи осей
    plt.xlabel('Итерация')
    plt.ylabel('Значение функции')

    # Добавляем заголовок
    # plt.title(title)

    # Отображаем график
    plt.show()

def plot_convergence_2(arrays: List[np.ndarray], names: List[str]):
    """
    Строит график сходимости алгоритма оптимизации для нескольких линий.

    :param arrays: Массив массивов NumPy, где каждый внутренний массив представляет собой значения для одной линии.
    """
    # Определяем количество шагов (итераций)
    num_steps = len(arrays)

    # Создаем массив индексов для оси X
    x = np.arange(num_steps)

    first_step_values = arrays[0]

    # Создаем график для каждого массива
    for i in range(first_step_values.shape[0]):
        y_values = [arr[i] for arr in arrays]
        plt.plot(x, y_values, label=f'{names[i]}')

    # Добавляем подписи осей
    plt.xlabel('Итерация')
    plt.ylabel('Значение функции')

    # Добавляем заголовок
    # plt.title('График сходимости алгоритма оптимизации')

    # Добавляем легенду
    plt.legend()

    # Отображаем график
    plt.show()

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
            grad_f = grad(lambda x: problem.target.evaluate(dict(zip(names, x))))
            return grad_f(x)

        constraints = []
        for con in problem.constraints:
            expr = lambda x: con.evaluate(dict(zip(names, x)))
            grad_expr = grad(lambda x: con.evaluate(dict(zip(names, x))))
            if con.type == ConstraintType.INEQ:
                print(con.type)
                constraints.append({'type': 'ineq', 'fun': expr, 'jac': grad_expr})
            elif con.type == ConstraintType.EQ:
                constraints.append({'type': 'eq', 'fun': expr, 'jac': grad_expr})
            else :
                raise ValueError(f"Неподдерживаемый тип ограничения: {con.type}")

        bounds = [(v.lower, v.upper) for v in problem.variables]

        res = minimize(f, x0, jac=jac, constraints=constraints, bounds=bounds, method='SLSQP')
        print(problem.target.evaluate(dict(zip(names, res.x))))
        return res.x

    def test_sqp_vs_scipy(self):
        problem_path = "tests/rosenbrock_valley.json"
        task = self.load_problem(problem_path)

        optimizer = Optimizer(task)
        result_sqp, f_val_history, x_val_history = optimizer.optimize()
        result_sqp_vals = np.array([result_sqp[var.name] for var in task.variables])

        task2 = self.load_problem(problem_path)
        result_scipy = self.scipy_solve(task2)

        # plot_convergence(f_val_history)
        # plot_convergence_2(x_val_history, task.get_variable_names())

        self._print_comparison_table(task.get_variable_names(), result_sqp_vals, result_scipy)
        

