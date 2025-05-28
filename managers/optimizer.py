import numpy as np
import logging
from datetime import datetime
from models.optimization_task import OptimizationTask
from models.constraint import ConstraintType
from cvxpy import Variable, Minimize, Problem, quad_form


class Optimizer:
    def __init__(self, task: OptimizationTask):
        self.task = task
        self.history = []
        self._setup_logger()
        self._last_grad_norm = np.inf
        self._last_step_size = 0.0
        self._prev_alpha = 1.0

        self._feasible_steps = 0
        # Для отслеживания истории оптимизации
        self.iteration_history = []
        self.f_history = []
        self.grad_norm_history = []

    def optimize(self):
        x = self.task.get_variable_values()
        hess = np.eye(len(x))  # Инициализация гессиана как единичной матрицы
        s_prev, y_prev = None, None

        for iteration in range(self.task.config.max_iter):
            self.logger.info(f"Start step {iteration}")
            
            target_function_grad = self._compute_gradient(self.task.target, x)
            if s_prev is not None and y_prev is not None:
                hess = self._update_hessian_bfgs(hess, s_prev, y_prev)

            self.logger.info(f"grad(f): {target_function_grad} | grad_norm = {np.linalg.norm(target_function_grad)}")

            A, b = self._prepare_constraints(x)

            try:
                p, lambdas = self._solve_qp_subproblem(hess, target_function_grad, A, b)
            except Exception as e:
                self.logger.error(f"QP solver error: {e}")
                break

            self.logger.info(f"QP variables: {p}")


            # Обновляем коэф
            rho_new = np.max(np.abs(lambdas)) * 1.1
            self.task.config.penalty_coeff = max(rho_new, 1.0)

            # Линейный поиск
            alpha = self._line_search(x, p)

            self.logger.info(f"Iter {iteration}: penalty_coeff = {self.task.config.penalty_coeff} | "f"||alpha|| = {alpha}")
            
            extended_p = p * alpha
            x_new = x + extended_p
        
            constraints_observed = self._check_constraints(dict(zip(self.task.get_variable_names(), x_new)))
            if not constraints_observed:
                self.logger.error("Constraints are violated")

            self.task.set_variable_values(x_new)

            self._log_iteration(iteration, x_new, target_function_grad, extended_p)

            if self._check_convergence(target_function_grad, extended_p):
                break

            x = x_new

        return self.task.get_variable_dict()


    def _solve_qp_subproblem(self, Hessian, target_function_grad, A, b):
        H = 0.5 * (Hessian + Hessian.T)

        n = H.shape[0]
        regularization_term = 1e-4 * np.eye(n)
        H += regularization_term

        eigenvalues = np.linalg.eigvals(H)
        if np.any(eigenvalues <= 0):
            self.logger.warning("Матрица Гессе не является положительно определенной. Применяется дополнительная регуляризация.")
            H += 1e-4 * np.eye(n)

        self.logger.debug(f"Hessian matrix:\n{H}")

        x = Variable(n)

        objective = Minimize(0.5 * quad_form(x, H) + target_function_grad.T @ x)
        self.logger.info(f"A.shape[0]: {A.shape[0]}")
        constraints = [A @ x <= b] if A.shape[0] > 0 else []
        prob = Problem(objective, constraints)
        prob.solve()

        if x.value is None:
            raise RuntimeError("QP subproblem failed to solve.")

        lambdas = np.array([con.dual_value for con in constraints])
        return x.value, lambdas

    def _compute_gradient(self, func, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        x = x.astype(float)
        grad = np.zeros_like(x)
        f0 = func.evaluate(dict(zip(self.task.get_variable_names(), x)))
        for i in range(len(x)):
            x_eps = x.copy()
            x_eps[i] += eps
            f_eps = func.evaluate(dict(zip(self.task.get_variable_names(), x_eps)))
            grad[i] = (f_eps - f0) / eps
        return grad

    def _update_hessian_bfgs(self, H, s, y):
        rho = 1.0 / (y @ s)

        # избегаем деления на 0 и отрицательной кривизны
        if rho <= 0 or np.isinf(rho):
            return H

        I = np.eye(len(s))
        V = I - rho * np.outer(s, y)
        H_new = V @ H @ V.T + rho * np.outer(s, s)
        return H_new

    def _prepare_constraints(self, x):
        A, b = [], []
        for constraint in self.task.constraints:
            constraint_gradient = self._compute_gradient(constraint, x)
            constraint_value = constraint.evaluate(self.task.get_variable_dict())
            if constraint.type == ConstraintType.INEQ:
                A.append(constraint_gradient)
                b.append(-constraint_value)
            elif constraint.type == ConstraintType.EQ:
                A.append(constraint_gradient)
                b.append(-constraint_value)
                A.append(-constraint_gradient)
                b.append(constraint_value)
        return (np.array(A), np.array(b)) if A else (np.zeros((0, len(x))), np.zeros(0))

    def _evaluate_merit(self, x: np.ndarray, penalty_coeff: float = None) -> float:
        if penalty_coeff is None:
            penalty_coeff = self.task.config.penalty_coeff

        var_dict = dict(zip(self.task.get_variable_names(), x))
        f_val = self.task.target.evaluate(var_dict)

        constraint_penalty = 0.0
        for constraint in self.task.constraints:
            val = constraint.evaluate(var_dict)
            if constraint.type == ConstraintType.INEQ:
                constraint_penalty += max(0, val)**2
            elif constraint.type == ConstraintType.EQ:
                constraint_penalty += val**2

        merit = f_val + penalty_coeff * constraint_penalty
        return merit

    def _line_search(self, x, p):
        alpha = self._prev_alpha
        c1 = 1e-4
        beta = 0.5
        max_trials = 20

        phi_0 = self._evaluate_merit(x)
        phi_0_grad = self._compute_merit_gradient(x)

        for _ in range(max_trials):
            x_trial = x + alpha * p
            phi_trial = self._evaluate_merit(x_trial)

            # Условие Армихо
            if phi_trial <= phi_0 + c1 * alpha * (phi_0_grad @ p):
                self._prev_alpha = alpha
                return alpha
            # на каждой итерации alpha = b^i
            alpha *= beta

        alpha = max(1e-6, alpha)

        self._prev_alpha = alpha
        return alpha

    def _check_constraints(self, variable_dict):
        for constraint in self.task.constraints:
            constraint_value = constraint.evaluate(variable_dict)
            self.logger.info(f"Constraint {constraint.name} = {constraint_value}")
            if constraint.type == ConstraintType.INEQ and constraint_value > self.task.config.tol:
                return False
            if constraint.type == ConstraintType.EQ and abs(constraint_value) > self.task.config.tol:
                return False
        return True

    def _log_iteration(self, i, x, grad, step):
        f_value = self.task.target.evaluate(self.task.get_variable_dict())
        grad_norm = np.linalg.norm(grad)
        step_norm = np.linalg.norm(step)

        # Сохраняем историю для построения графиков
        self.iteration_history.append(i)
        self.f_history.append(f_value)
        self.grad_norm_history.append(grad_norm)

        self.logger.info(
            f"f(x) = {f_value:.6f} | "
            f"Step = {step_norm:.6f}"
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

    def _compute_merit_gradient(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        x = x.astype(float)
        grad = np.zeros_like(x)
        f0 = self._evaluate_merit(x)
        for i in range(len(x)):
            x_eps = x.copy()
            x_eps[i] += eps
            f_eps = self._evaluate_merit(x_eps)
            grad[i] = (f_eps - f0) / eps
        return grad

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
