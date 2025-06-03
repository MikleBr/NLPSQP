# from managers.problem_json_parser import ProblemJSONParser
from managers.optimizer import Optimizer
import matplotlib.pyplot as plt
import numpy as np
from typing import List
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

def mass_parser(filename):
    with open(filename, 'r') as f:
        for line in f:
            print(line)
            if "Mass," in line:
                parts = line.strip().split("Mass,")
                if len(parts) > 1:
                    try:
                        return float(parts[1])
                    except ValueError:
                        raise RuntimeError(f"Не удалось преобразовать значение '{parts[1]}' в число")
    raise RuntimeError("Не удалось извлечь значение напряжения")

def umax_parser(filename):
    with open(filename, 'r') as f:
        for line in f:
            print(line)
            if "Umax," in line:
                parts = line.strip().split("Umax,")
                if len(parts) > 1:
                    try:
                        return float(parts[1])
                    except ValueError:
                        raise RuntimeError(f"Не удалось преобразовать значение '{parts[1]}' в число")
    raise RuntimeError("Не удалось извлечь значение напряжения")


target = AnsysMacroTargetFunction(
        macro_template_path="beam.txt",
        ansys_path="C:/Program Files/ANSYS Inc/ANSYS Student/v251/ansys/bin/winx64/ANSYS251.exe",
        workdir="C:/Users/Mikhail/Desktop/NLPSQP/ansys_tmp",
        output_filename="results.txt",
        result_parser=mass_parser
)

res = target.evaluate({
    "r1": 0.029194,
    "r2": 0.022972
})

print(res)

# deformationAnsysFunction = AnsysMacroTargetFunction(
#         macro_template_path="beam.txt",
#         ansys_path="C:/Program Files/ANSYS Inc/ANSYS Student/v251/ansys/bin/winx64/ANSYS251.exe",
#         workdir="C:/Users/Mikhail/Desktop/NLPSQP/ansys_tmp",
#         output_filename="results.txt",
#         result_parser=umax_parser
# )

# stressAnsysFunction = AnsysMacroTargetFunction(
#         macro_template_path="beam.txt",
#         ansys_path="C:/Program Files/ANSYS Inc/ANSYS Student/v251/ansys/bin/winx64/ANSYS251.exe",
#         workdir="C:/Users/Mikhail/Desktop/NLPSQP/ansys_tmp",
#         output_filename="results.txt",
#         result_parser=stress_parser
# )

# deformationConstraint = Constraint(func=lambda params: deformationAnsysFunction.evaluate(params) - 0.002, type=ConstraintType.INEQ, name="Deformation < 0.002m", penalty_gain = 1000000)
# stressConstraint = Constraint(func=lambda params: stressAnsysFunction.evaluate(params) - 5000000, type=ConstraintType.INEQ, name="Stress < 5MPa")


# task = OptimizationTask(
#     variables=[
#         DesignVariable(name="r1", value=0.4, lower=0.001),
#         DesignVariable(name="r2", value=0.2, lower=0.001)
#     ],
#     target=target,
#     constraints=[deformationConstraint, stressConstraint],
#     config=OptimizationConfig(max_iter=50)
# )

# optimizer = Optimizer(task)
# results, f_val_history, x_val_history = optimizer.optimize()

# def plot_convergence(values):
#     """
#     Строит график сходимости алгоритма оптимизации.

#     :param values: Список значений для оси Y.
#     """
#     # Создаем массив индексов для оси X
#     x = list(range(len(values)))

#     # Создаем график
#     plt.plot(x, values)

#     # Добавляем подписи осей
#     plt.xlabel('Итерация')
#     plt.ylabel('Значение функции')

#     # Добавляем заголовок
#     # plt.title(title)

#     # Отображаем график
#     plt.show()

# def plot_convergence_2(arrays: List[np.ndarray], names: List[str]):
#     """
#     Строит график сходимости алгоритма оптимизации для нескольких линий.

#     :param arrays: Массив массивов NumPy, где каждый внутренний массив представляет собой значения для одной линии.
#     """
#     # Определяем количество шагов (итераций)
#     num_steps = len(arrays)

#     # Создаем массив индексов для оси X
#     x = np.arange(num_steps)

#     first_step_values = arrays[0]

#     # Создаем график для каждого массива
#     for i in range(first_step_values.shape[0]):
#         y_values = [arr[i] for arr in arrays]
#         plt.plot(x, y_values, label=f'{names[i]}')

#     # Добавляем подписи осей
#     plt.xlabel('Итерация')
#     plt.ylabel('Значение функции')

#     # Добавляем заголовок
#     # plt.title('График сходимости алгоритма оптимизации')

#     # Добавляем легенду
#     plt.legend()

#     # Отображаем график
#     plt.show()

# plot_convergence(f_val_history)
# plot_convergence(x_val_history)