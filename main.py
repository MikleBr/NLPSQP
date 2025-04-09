# from managers.problem_json_parser import ProblemJSONParser
# from managers.optimizer import Optimizer

# problem_json_parser = ProblemJSONParser()

# problem = problem_json_parser.createProblem('tests/circle.json')

# optimizer = Optimizer(problem)
# optimizer.optimize()  
from tests.comparison import OptimizationComparisonTest

OptimizationComparisonTest().test_sqp_vs_scipy()
