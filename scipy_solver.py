import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


from models.optimization_task import OptimizationTask
from models.design_variable import DesignVariable
from models.constraint import Constraint, ConstraintType
from models.optimization_config import OptimizationConfig
from models.ansys_target_function import AnsysMacroTargetFunction

def stress_parser(filename):
    with open(filename, 'r') as f:
        for line in f:
            print(line)
            if "Smax," in line:
                parts = line.strip().split("Smax,")
                if len(parts) > 1:
                    try:
                        return float(parts[1])
                    except ValueError:
                        raise RuntimeError(f"Не удалось преобразовать значение '{parts[1]}' в число")
    raise RuntimeError("Не удалось извлечь значение напряжения")


def scipy_solve(problem: OptimizationTask):
    """
    Решает задачу оптимизации с помощью SciPy
    """
    x0 = problem.get_variable_values()
    names = problem.get_variable_names()

    # Для отслеживания истории оптимизации
    history = {'x': [], 'fun': [], 'nit': []}

    def callback(xk):
        # Функция обратного вызова для отслеживания истории
        history['x'].append(xk.copy())
        history['fun'].append(f(xk))
        history['nit'].append(len(history['x']))

    def f(x):
        return problem.target.evaluate(dict(zip(names, x)))

    def jac(x):
        # Используем численное дифференцирование вместо autograd
        # так как autograd может не работать с ANSYS функцией
        eps = 1e-6
        grad_vals = np.zeros_like(x)
        f0 = f(x)
        for i in range(len(x)):
            x_eps = x.copy()
            x_eps[i] += eps
            f_eps = f(x_eps)
            grad_vals[i] = (f_eps - f0) / eps
        return grad_vals

    constraints = []
    for con in problem.constraints:
        expr = lambda x: con.evaluate(dict(zip(names, x)))
        if con.type == ConstraintType.INEQ:
            constraints.append({'type': 'ineq', 'fun': expr})
        elif con.type == ConstraintType.EQ:
            constraints.append({'type': 'eq', 'fun': expr})
        else:
            raise ValueError(f"Неподдерживаемый тип ограничения: {con.type}")

    bounds = [(v.lower, v.upper) for v in problem.variables]

    print("Запуск оптимизации через SciPy...")
    res = minimize(f, x0, jac=jac, constraints=constraints, bounds=bounds, method='SLSQP',
                  options={'disp': True, 'maxiter': 100, 'ftol': 1e-6}, callback=callback)

    result_dict = dict(zip(names, res.x))
    print(f"Результат SciPy: {result_dict}")
    print(f"Значение целевой функции: {problem.target.evaluate(result_dict)}")
    print(f"Норма градиента: {res.x}")
    print(f"Статус: {res.message}")
    print(f"Количество итераций: {res.nit}")

    # Построение графика сходимости SciPy
    plt.figure(figsize=(10, 6))
    plt.plot(history['nit'], history['fun'], 'g-o', label='SciPy')
    plt.title('График сходимости оптимизации SciPy')
    plt.xlabel('Итерация')
    plt.ylabel('Значение целевой функции')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('scipy_convergence_plot.png')
    plt.show()

    return result_dict, history

# Создаем копию задачи для SciPy
task_scipy = OptimizationTask(
    variables=[
        DesignVariable(name="a", value=0.5, lower=0.1, upper=5.0),
        DesignVariable(name="b", value=0.5, lower=0.1, upper=10.0)
    ],
    target=AnsysMacroTargetFunction(
        macro_template_path="kirsh.txt",
        ansys_path="C:/Program Files/ANSYS Inc/ANSYS Student/v251/ansys/bin/winx64/ANSYS251.exe",
        workdir="C:/Users/Mikhail/Desktop/NLPSQP/ansys_tmp",
        output_filename="results.txt",
        result_parser=stress_parser
    ),
    constraints=[],
    config=OptimizationConfig()
)

# Запускаем оптимизацию через SciPy
print("\n=== Оптимизация через SciPy ===")
result_scipy, scipy_history = scipy_solve(task_scipy)
print(result_scipy)
print(scipy_history)