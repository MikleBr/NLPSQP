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

        self._feasible_steps = 0
        # Для отслеживания истории оптимизации
        self.iteration_history = []
        self.f_history = []
        self.grad_norm_history = []

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

    def _compute_gradient(self, func, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
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

    def _solve_qp_subproblem(self, Hessian, gradients, A, b):
        from cvxpy import Variable, Minimize, Problem, quad_form

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

        objective = Minimize(0.5 * quad_form(x, H) + gradients.T @ x)
        constraints = [A @ x <= b] if A.shape[0] > 0 else []
        prob = Problem(objective, constraints)
        prob.solve()
        return x.value

    def optimize(self):
        x = self.task.get_variable_values()
        hess = np.eye(len(x))  # Инициализация гессиана как единичной матрицы
        s_prev, y_prev = None, None

        for iteration in range(self.task.config.max_iter):
            grad = self._compute_gradient(self.task.target, x)
            if s_prev is not None and y_prev is not None:
                hess = self._update_hessian_bfgs(hess, s_prev, y_prev)

            A, b = self._prepare_constraints(x)

            try:
                p = self._solve_qp_subproblem(hess, grad, A, b)
            except Exception as e:
                self.logger.error(f"QP solver error: {e}")
                break

            if p is None:
                self.logger.error("❌ QP задача не решена.")
                break

            alpha = self._line_search(x, p, grad)
            extended_p = p * alpha
            x_new = x + extended_p
        
            # Пробуем уменьшать alpha, если x_new нарушает ограничения
            for attempt in range(2):
                x_new = x + alpha * p
                if self._check_constraints(dict(zip(self.task.get_variable_names(), x_new))):
                    self._feasible_steps += 1
                    if self._feasible_steps >= 5 and self.task.config.penalty_coeff > 10:
                        self.task.config.penalty_coeff *= 0.5
                        self.logger.info(f"^ Уменьшаем penalty_coeff до {self.task.config.penalty_coeff}")
                    break
                alpha *= 0.5  # уменьшаем шаг
                self.logger.warning(f"X restrictions with alpha = {alpha*2: .4f}, try alpha = {alpha: .4f}")
            else:
                self.logger.error("X failed to find a permissible step. We increase the fine and continue.")
                self.task.config.penalty_coeff *= 2
                continue
            


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

    def _line_search(self, x, p, grad):
        alpha = self._prev_alpha
        beta = 0.5
        c1 = 1e-4
        max_trials = 20

        phi_0 = self._evaluate_merit(x)

        for _ in range(max_trials):
            x_trial = x + alpha * p
            phi_trial = self._evaluate_merit(x_trial)

            if phi_trial <= phi_0 + c1 * alpha * (grad @ p):
                self._prev_alpha = alpha
                return alpha
            alpha *= beta

        self._prev_alpha = alpha
        return alpha

    def _check_constraints(self, variable_dict):
        for constraint in self.task.constraints:
            val = constraint.evaluate(variable_dict)
            if constraint.type == ConstraintType.INEQ and val > self.task.config.tol:
                return False
            if constraint.type == ConstraintType.EQ and abs(val) > self.task.config.tol:
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
            f"Iter {i}: f(x) = {f_value:.6f} | "
            f"||grad(f)|| = {grad_norm:.6f} | Step = {step_norm:.6f}"
        )
        self.logger.info(f"Penalty coeff: {self.task.config.penalty_coeff}")

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
