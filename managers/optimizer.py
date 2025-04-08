import numpy as np
import logging
from datetime import datetime
from models.optimization_task import OptimizationTask
from models.constraint import ConstraintType
from autograd import grad, hessian


class Optimizer:
    def __init__(self, task: OptimizationTask):
        self.task = task
        self.history = []
        self._setup_logger()
        self._last_grad_norm = np.inf
        self._last_step_size = 0.0
        self._prev_alpha = 1.0

    def _setup_logger(self):
        self.logger = logging.getLogger('SQPOptimizer')
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            log_filename = f"sqp_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_filename)
            file_handler.setFormatter(formatter)

            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

        self.logger.info("Optimizer initialized")
        self.logger.info(f"Target function: {self.task.target.name}")
        self.logger.info(f"Number of variables: {len(self.task.variables)}")
        self.logger.info(f"Number of constraints: {len(self.task.constraints)}")

    def _compute_gradient(self, func, x: np.ndarray) -> np.ndarray:
        x = x.astype(float)
        grad_func = grad(lambda x: func.evaluate(dict(zip(self.task.get_variable_names(), x))))
        return grad_func(x)

    def _compute_hessian(self, func, x: np.ndarray) -> np.ndarray:
        x = x.astype(float)
        hess_func = hessian(lambda x: func.evaluate(dict(zip(self.task.get_variable_names(), x))))
        return hess_func(x)

    def _solve_qp_subproblem(self, Hessian, gradients, A, b):
        from cvxpy import Variable, Minimize, Problem, quad_form

        n = len(gradients)
        x = Variable(n)

        objective = Minimize(0.5 * quad_form(x, Hessian) + gradients.T @ x)
        constraints = [A @ x <= b] if A.shape[0] > 0 else []
        prob = Problem(objective, constraints)
        prob.solve()
        return x.value

    def optimize(self):
        x = self.task.get_variable_values()

        for iteration in range(self.task.config.max_iter):
            grad = self._compute_gradient(self.task.target, x)
            hess = self._compute_hessian(self.task.target, x)
            # TODO: Add BFGS algorithm for Hessian
            # hess = np.eye(len(x))
            A, b = self._prepare_constraints(x)

            try:
                p = self._solve_qp_subproblem(hess, grad, A, b)
            except Exception as e:
                self.logger.error(f"QP solver error: {e}")
                break

            if p is None:
                self.logger.error("❌ QP не решена. Прерывание.")
                break

            alpha = self._line_search(x, p, grad)
            extended_p = p
            x_new = x + extended_p
            self.task.set_variable_values(x_new)

            self._log_iteration(iteration, x_new, grad, extended_p)

            if self._check_convergence(grad, extended_p):
                break

            x = x_new

        return self.task.get_variable_dict()

    def _prepare_constraints(self, x):
        A, b = [], []
        for con in self.task.constraints:
            grad = self._compute_gradient(con, x)
            val = con.evaluate(self.task.get_variable_dict())
            if con.type == ConstraintType.INEQ:
                A.append(grad)
                b.append(-val)
            elif con.type == ConstraintType.EQ:
                A.append(grad)
                b.append(-val)
                A.append(-grad)
                b.append(val)
        return (np.array(A), np.array(b)) if A else (np.zeros((0, len(x))), np.zeros(0))

    def _line_search(self, x, p, grad):
        alpha = self._prev_alpha
        beta = 0.5
        c1 = 1e-4
        for _ in range(20):
            x_trial = x + alpha * p
            self.task.set_variable_values(x_trial)
            if not self._check_constraints():
                alpha *= beta
                continue
            f_new = self.task.target.evaluate(self.task.get_variable_dict())
            f_curr = self.task.target.evaluate({v.name: v.value for v in self.task.variables})
            if f_new <= f_curr + c1 * alpha * (grad @ p):
                return alpha
            alpha *= beta

        self._prev_alpha = alpha
        return alpha

    def _check_constraints(self):
        for constraint in self.task.constraints:
            val = constraint.evaluate(self.task.get_variable_dict())
            if constraint.type == ConstraintType.INEQ and val > self.task.config.tol:
                return False
            if constraint.type == ConstraintType.EQ and abs(val) > self.task.config.tol:
                return False
        return True

    def _log_iteration(self, i, x, grad, step):
        self.logger.info(
            f"Iter {i}: f(x) = {self.task.target.evaluate(self.task.get_variable_dict()):.6f} | "
            f"||∇f|| = {np.linalg.norm(grad):.6f} | Step = {np.linalg.norm(step):.6f}"
        )
        for variable in self.task.variables:
            self.logger.info(f"-- {variable.name}: {variable.value:.6f} | ")

    def _check_convergence(self, grad, step):
        grad_norm = np.linalg.norm(grad)
        step_norm = np.linalg.norm(step)

        if grad_norm < self.task.config.tol and step_norm < self.task.config.tol:
            return True

        if abs(self._last_grad_norm - grad_norm) < 1e-8 and abs(self._last_step_size - step_norm) < 1e-8:
            return True

        self._last_grad_norm = grad_norm
        self._last_step_size = step_norm
        return False
