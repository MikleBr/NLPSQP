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

deformationAnsysFunction = AnsysMacroTargetFunction(
        macro_template_path="beam.txt",
        ansys_path="C:/Program Files/ANSYS Inc/ANSYS Student/v251/ansys/bin/winx64/ANSYS251.exe",
        workdir="C:/Users/Mikhail/Desktop/NLPSQP/ansys_tmp",
        output_filename="results.txt",
        result_parser=umax_parser
)

stressAnsysFunction = AnsysMacroTargetFunction(
        macro_template_path="beam.txt",
        ansys_path="C:/Program Files/ANSYS Inc/ANSYS Student/v251/ansys/bin/winx64/ANSYS251.exe",
        workdir="C:/Users/Mikhail/Desktop/NLPSQP/ansys_tmp",
        output_filename="results.txt",
        result_parser=stress_parser
)

deformationConstraint = Constraint(func=lambda params: deformationAnsysFunction.evaluate(params) - 0.002, type=ConstraintType.INEQ)
stressConstraint = Constraint(func=lambda params: stressAnsysFunction.evaluate(params) - 5000000, type=ConstraintType.INEQ)


task = OptimizationTask(
    variables=[
        DesignVariable(name="r1", value=0.09, upper=0.1),
        DesignVariable(name="r2", value=0.09, upper=0.1)
    ],
    target=target,
    constraints=[deformationConstraint],
    config=OptimizationConfig(max_iter=50)
)

optimizer = Optimizer(task)
optimizer.optimize()