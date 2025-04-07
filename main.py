from managers.problem_json_parser import ProblemJSONParser
from managers.optimizer import Optimizer

problem_json_parser = ProblemJSONParser()

problem = problem_json_parser.createProblem('tests/test2.json')

optimizer = Optimizer(problem)
optimizer.optimize()
