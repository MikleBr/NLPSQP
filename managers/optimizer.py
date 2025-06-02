from typing import Tuple
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
        # Нужно для увеличения штрафа при выходе за ограничения
        self.penalty_coeff_gain = 1.0

        self._feasible_steps = 0
        self.iteration_history = []
        self.f_history = []
        self.grad_norm_history = []

    def optimize(self):
        # Временно выключил, тк пока выбираю точки подходящие ограничениям
        # start_variables_dict = self.task.get_variable_dict()
        # self.logger.info("Checking start point...")
        # initial_constraints_satisfied = self._check_constraints(start_variables_dict)
        
        # if not initial_constraints_satisfied:
        #     self.logger.error("Star point ")
        #     # Выбрасываем исключение, чтобы остановить выполнение программы
        #     raise ValueError("Начальные условия не удовлетворяют ограничениям.")
        
        x = self.task.get_variable_values() 
        hess = np.eye(len(x))
        s_prev, y_prev = None, None

        for iteration in range(self.task.config.max_iter):
            self.logger.info(f"--------- Start step {iteration + 1} ---------")
            
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

            if lambdas.size == 0:
                rho_new = self.task.config.penalty_coeff
            else:
                rho_new = np.max(np.abs(lambdas)) * 1.1
            self.task.config.penalty_coeff = max(rho_new, 1.0)

            alpha = self._line_search(x, p)
            p_step = p * alpha
            x_new = x + p_step
            
            self.logger.info(
                f"penalty_coeff = {self.task.config.penalty_coeff}"
                f"| alpha = {alpha}"
                f"| p_step = {p_step}"
            )

            
            max_attempts_to_choose_alpha = 5
            choose_alpha_attempt = 0
            constraints_satisfied = False

            while choose_alpha_attempt < max_attempts_to_choose_alpha:
                if (choose_alpha_attempt > 1):
                    self.logger.info(
                        f"Attempt {choose_alpha_attempt}"
                        f"| alpha = {alpha}"
                        f"| p_step = {p_step}"
                    )
                
                attempt_constraints_satisfied = self._check_constraints(dict(zip(self.task.get_variable_names(), x_new)))

                if not attempt_constraints_satisfied:
                    alpha *= 0.5
                    p_step = p * alpha
                    x_new = x + p_step
                else:
                    constraints_satisfied = True
                    choose_alpha_attempt = max_attempts_to_choose_alpha

            if not constraints_satisfied:
                self.logger.error(f"Constraints with step {p_step} not satisfied")
                break
            
            s_prev = x_new - x
            y_prev = self._compute_gradient(self.task.target, x_new) - target_function_grad

            self.task.set_variable_values(x_new)

            self._log_iteration(iteration, x_new, target_function_grad, p_step)

            if self._check_convergence(target_function_grad, p_step):
                break

            x = x_new

        self.logger.info("---- RESULT ----")
        for variable in self.task.variables:
            self.logger.info(f"{variable.name} = {variable.value:.6f}")

        return self.task.get_variable_dict()

    def _solve_qp_subproblem(self, Hessian, target_function_grad, A, b):
        eigvals = np.linalg.eigvalsh(Hessian)
        lam_min = eigvals.min()
        print("λ_min =", lam_min)

        # проверяем компоненту g в нулевом подпространстве, если нужно
        # if lam_min < 1e-12:
        #     # собственные векторы, соответствующие ~0
        #     null_space = eigvecs[:, eigvals < 1e-12]
        #     proj = null_space.T @ g
        #     print("‖проекция g на ядро(H)‖ =", np.linalg.norm(proj))

        n = Hessian.shape[0]
        x = Variable(n)

        objective = Minimize(0.5 * quad_form(x, Hessian) + target_function_grad.T @ x)
        constraints = [A[i, :] @ x <= b[i] for i in range(A.shape[0])] if A.shape[0] > 0 else []

        prob = Problem(objective, constraints)
        prob.solve( solver="OSQP", verbose=True)

        print("Status:", prob.status)
        print("Primal objective:", prob.value)
        print("Solver stats:", prob.solver_stats)


        if x.value is None:
            raise RuntimeError("QP subproblem failed to solve.")

        lambdas = np.array([con.dual_value for con in constraints])
        return x.value, lambdas
    
    def _regularize_hessian(self, H: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, float]:
        """
        Возвращает положительно-определённую копию Гессиана и величину добавленной регуляризации (для дебага).
        """
        H_sym = 0.5 * (H + H.T)
        lam_min = np.linalg.eigvalsh(H_sym).min()

        if lam_min > eps:
            return H_sym, 0.0
        delta = eps - lam_min
        H_reg = H_sym + delta * np.eye(H_sym.shape[0])

        return H_reg, delta

    def _compute_gradient(self, func, x: np.ndarray) -> np.ndarray:
        eps = self.task.config.grad_eps
        x = x.astype(float)
        grad = np.zeros_like(x)
        f0 = func.evaluate(dict(zip(self.task.get_variable_names(), x)))
        for i in range(len(x)):
            x_eps = x.copy()
            x_eps[i] += eps
            f_eps = func.evaluate(dict(zip(self.task.get_variable_names(), x_eps)))
            grad[i] = (f_eps - f0) / eps
        return grad

    def _update_hessian_bfgs(self, B: np.ndarray, s: np.ndarray, y: np.ndarray, curvature_threshold: float = 1e-8) -> np.ndarray: 
        """
        Обновление прямой (необратной) матрицы Гессе методом BFGS.

        Параметры
        ---------
        H : (n, n) ndarray
            Текущая аппроксимация Гессе B_k.
        s : (n,) ndarray
            Шаг s_k = x_{k+1} − x_k.
        y : (n,) ndarray
            Разность градиентов y_k = g_{k+1} − g_k.
        curvature_threshold : float
            Минимальное допустимое значение yᵀs для проверки кривизны.
            При yᵀs ≤ threshold включается демпфирование Пауэлла.

        Возвращает
        ----------
        (n, n) ndarray
            Регуляризованная прямая матрица Гессе B_{k+1}.
        """

        # Приводим к вектор-столбцам
        s = s.reshape(-1, 1)
        y = y.reshape(-1, 1)

        ys = float(y.T @ s) # yᵀs — условие секущей и кривизны
        if ys <= curvature_threshold:
            # Демпфирование Пауэлла, чтобы гарантировать положит. определённость
            sBs = float(s.T @ B @ s)
            theta = 0.8 * (sBs - curvature_threshold) / (sBs - ys)
            y = theta * y + (1 - theta) * (B @ s)
            ys = float(y.T @ s)

        # BFGS-формула ранга-2 для (B_k → B_{k+1})
        Bs   = B @ s
        rho  = 1.0 / ys # = 1 / (yᵀs)
        B_new = (
            B
            - (Bs @ Bs.T) / float(s.T @ Bs) # − B_k s sᵀ B_k / (sᵀ B_k s)
            + rho * (y @ y.T)               # + y yᵀ / (yᵀ s)
        )

        B_regularized, delta = self._regularize_hessian(B_new)
        if delta > 0.0:
            self.logger.info(f"Гессиан не PD; добавлена регуляризация δ={delta:.2e}")

        return B_regularized

    def _prepare_constraints(self, x):
        A, b = [], []
        for constraint in self.task.constraints:
            constraint_gradient = self._compute_gradient(constraint, x)
            constraint_gradient = np.atleast_1d(constraint_gradient)
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

    def _evaluate_merit(self, x: np.ndarray) -> float:
        penalty_coeff = self.task.config.penalty_coeff * self.penalty_coeff_gain
        var_dict = dict(zip(self.task.get_variable_names(), x))
        f_val = self.task.target.evaluate(var_dict)
        constraint_penalty = 0.0
        for constraint in self.task.constraints:
            val = constraint.evaluate(var_dict)
            if constraint.type == ConstraintType.INEQ:
                # Для неравенств: max(0, g(x))^2
                constraint_penalty += max(0, val)**2
            elif constraint.type == ConstraintType.EQ:
                # Для равенств: h(x)^2
                constraint_penalty += val**2

        # Добавляем ограничения по переменным
        for variable in self.task.variables:
            if (variable.lower != -np.inf):
                val = variable.lower - var_dict.get(variable.name)
                # будет 0 только когда переменная БОЛЬШЕ lower
                constraint_penalty += max(0, val)**2
            if (variable.lower != np.inf):
                val = var_dict.get(variable.name) - variable.upper
                # будет 0 только когда переменная МЕНЬШЕ lower
                constraint_penalty += max(0, val)**2

        merit = f_val + penalty_coeff * constraint_penalty
        return merit

    def _line_search(self, x, p):
        alpha = self._prev_alpha
        c1 = 1e-4
        beta = 0.5
        max_trials = 20
        phi_0 = self._evaluate_merit(x)
        phi_0_grad = self._compute_merit_gradient(x)

        while max_trials > 0:
            x_trial = x + alpha * p
            phi_trial = self._evaluate_merit(x_trial)
            if phi_trial <= phi_0 + c1 * alpha * (phi_0_grad @ p):
                max_trials = 0
                self.logger.info(f"Alpha calculated with Armijo")
            alpha *= beta
            max_trials -= 1
            if (max_trials == 0):
                self.logger.info(f"It was not possible to pick up alpha by Armijo")


        alpha = max(1e-6, alpha)
        self._prev_alpha = alpha
        return alpha

    def _check_constraints(self, variable_dict):
        for variable in self.task.variables:
            current_value = variable_dict.get(variable.name)
            print(current_value)
            if (current_value < variable.lower or current_value > variable.upper):
                return False

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
        self.iteration_history.append(i)
        self.f_history.append(f_value)
        self.grad_norm_history.append(grad_norm)
        self.logger.info(
            f"f(x) = {f_value:.6f} | Step = {step_norm:.6f}"
        )
        for variable in self.task.variables:
            self.logger.info(f"-- {variable.name}: {variable.value:.6f} | ")

    def _check_convergence(self, grad, step):
        grad_norm = np.linalg.norm(grad)
        step_norm = np.linalg.norm(step)

        if grad_norm < self.task.config.tol and step_norm < self.task.config.tol:
            self.logger.info("Convergence: Gradient norm, step norm, and constraints are within tolerance.")
            return True

        # if step_norm < self.task.config.tol:
        #     self.logger.info("Convergence: Gradient norm, step norm, and constraints are within tolerance.")
        #     return True

        # Дополнительное условие, если изменения слишком малы (может указывать на застревание)
        if abs(self._last_grad_norm - grad_norm) < 1e-8 and abs(self._last_step_size - step_norm) < 1e-8:
            self.logger.info("Convergence: Little progress and constraints are satisfied.")
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
