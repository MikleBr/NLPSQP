# from managers.problem_json_parser import ProblemJSONParser
from managers.optimizer import Optimizer

# problem_json_parser = ProblemJSONParser()

# problem = problem_json_parser.createProblem('tests/circle.json')


# from tests.comparison import OptimizationComparisonTest

# OptimizationComparisonTest().test_sqp_vs_scipy()
from models.optimization_task import OptimizationTask
from models.design_variable import DesignVariable
from models.optimization_config import OptimizationConfig
from models.ansys_target_function import AnsysMacroTargetFunction

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
                        raise RuntimeError(f"Не удалось преобразовать значение напряжений '{parts[1]}' в число")
    raise RuntimeError("Не удалось извлечь значение перемещений")

target = AnsysMacroTargetFunction(
        macro_template_path="demo2.txt",
        ansys_path="C:/Program Files/ANSYS Inc/ANSYS Student/v251/ansys/bin/winx64/ANSYS251.exe",
        workdir="C:/Users/Mikhail/Desktop/NLPSQP/ansys_tmp",
        output_filename="results.txt",
        result_parser=umax_parser
)

# res = target.evaluate({
#     "DIAG_BEAM_POSITION": 12,
#     # "r2": 0.02726
# })

# print(res)


task = OptimizationTask(
    variables=[
        DesignVariable(name="DIAG_BEAM_POSITION", value=10, lower=5, upper=30),
    ],
    target=target,
    constraints=[],
    config=OptimizationConfig(max_iter=75, tol = 1e-6, grad_eps=1e-3)
)

optimizer = Optimizer(task)
optimizer.optimize()