import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# Пример задачи с 3 переменными и 2 ограничениями и 2 целевыми функциями
class MyMultiObjectiveProblem(Problem):

    def __init__(self):
        super().__init__(n_var=3,
                         n_obj=2,
                         n_constr=2,
                         xl=np.array([-10, -10, -10]),
                         xu=np.array([10, 10, 10]))

    def _evaluate(self, X, out, *args, **kwargs):
        # Целевые функции
        f1 = (X[:, 0] - 1)**2 + (X[:, 1] - 2)**2 + (X[:, 2] - 3)**2
        f2 = (X[:, 0])**2 + (X[:, 1])**2 + (X[:, 2])**2

        # Ограничения: x + y + z <= 5, и x^2 + y^2 <= 4
        g1 = X[:, 0] + X[:, 1] + X[:, 2] - 5
        g2 = X[:, 0]**2 + X[:, 1]**2 - 4

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])

# Алгоритм NSGA2
algorithm = NSGA2(pop_size=100)

# Оптимизация
res = minimize(MyMultiObjectiveProblem(),
               algorithm,
               termination=('n_gen', 100),
               seed=1,
               verbose=True)

# Вывод результатов
print("\n⚙️ Лучшие решения:")
for i in range(min(5, len(res.F))):
    print(f"X = {res.X[i]}, Objectives = {res.F[i]}")

# Визуализация
plot = Scatter()
plot.add(res.F, facecolor="red")
plot.show()
