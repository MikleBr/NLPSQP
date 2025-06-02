# from managers.problem_json_parser import ProblemJSONParser
from managers.optimizer import Optimizer

# problem_json_parser = ProblemJSONParser()

# problem = problem_json_parser.createProblem('tests/circle.json')


# from tests.comparison import OptimizationComparisonTest

# OptimizationComparisonTest().test_sqp_vs_scipy()
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

# target=AnsysMacroTargetFunction(
#     macro_template_path="kirsh.txt",
#     ansys_path="C:/Program Files/ANSYS Inc/ANSYS Student/v251/ansys/bin/winx64/ANSYS251.exe",
#     workdir="C:/Users/Mikhail/Desktop/NLPSQP/ansys_tmp",
#     output_filename="results.txt",
#     result_parser=stress_parser
# )

# print(target.evaluate({"a": 0.5, "b": 0.5}))
# print(target.evaluate({"a": 0.2, "b": 0.3}))

task = OptimizationTask(
    variables=[
        DesignVariable(name="a", value=0.1, lower=0.01, upper=1.0),
        DesignVariable(name="b", value=0.1, lower=0.01, upper=1.0)
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

optimizer = Optimizer(task)
optimizer.optimize()  