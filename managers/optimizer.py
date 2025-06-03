from typing import Tuple
import numpy as np
import logging
from datetime import datetime
from models.optimization_task import OptimizationTask
from models.constraint import ConstraintType
from cvxpy import Variable, Minimize, Problem, quad_form
from typing import List


class Optimizer:
    def __init__(self, task: OptimizationTask):
        self.task = task
        self.history = []
        self._setup_logger()
        self._last_grad_norm = np.inf
        self._last_step_size = 0.0
        self._prev_alpha = 1.0

        # For graphs building
        self.f_val_history: List[float] = [self.task.target.evaluate(self.task.get_variable_dict())]
        self.x_val_history: List[np.ndarray] = [self.task.get_variable_values()]

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
        
        hess = self.task.config.tau * np.eye(len(x)) # Начальное приближение Гессиана
        s_prev, y_prev = None, None

        for iteration in range(self.task.config.max_iter):
            self.logger.info(f"--------- Start step {iteration + 1} ---------")
            
            target_function_grad = self._compute_gradient(self.task.target, x)
            if s_prev is not None and y_prev is not None:
                hess = self._update_hessian_bfgs(hess, s_prev, y_prev)

            self.logger.info(f"grad(f): {target_function_grad} | grad_norm = {np.linalg.norm(target_function_grad):.4e}")

            A, b = self._prepare_constraints(x)

            try:
                p, lambdas = self._solve_qp_subproblem(hess, target_function_grad, A, b)
            except Exception as e:
                self.logger.error(f"QP solver error: {e}")
                break

            self.logger.info(f"QP solution p: {p}")

            if lambdas.size == 0:
                rho_new = self.task.config.penalty_coeff
            else:
                rho_new = np.max(np.abs(lambdas)) * 1.1
            self.task.config.penalty_coeff = max(rho_new, 1.0)

            alpha = self._line_search(x, p)
            p_step = p * alpha
            x_new = x + p_step
            
            self.logger.info(
                f"penalty_coeff = {self.task.config.penalty_coeff:.4e}"
                f" | alpha = {alpha:.4e}"
                f" | p_step_norm = {np.linalg.norm(p_step):.4e}"
            )
            self.logger.info(f"p_step details: {p_step}")
            
            max_attempts_to_choose_alpha = 5
            choose_alpha_attempt = 0
            constraints_satisfied_after_step = False # Renamed for clarity

            current_alpha_for_constraint_check = alpha # Start with alpha from line search
            current_p_step_for_constraint_check = p_step
            current_x_new_for_constraint_check = x_new

            # Вообще это скорее костыль, чем хорошее решение, но всем не угодить :)
            while choose_alpha_attempt < max_attempts_to_choose_alpha:
                if (choose_alpha_attempt > 0): # Log if it's not the first attempt
                    self.logger.info(
                        f"Constraint check attempt {choose_alpha_attempt + 1} with reduced alpha."
                        f" | alpha = {current_alpha_for_constraint_check:.4e}"
                        # f" | p_step = {current_p_step_for_constraint_check}" # Can be verbose
                    )
                
                attempt_constraints_satisfied = self._check_constraints(
                    dict(zip(self.task.get_variable_names(), current_x_new_for_constraint_check))
                )

                if not attempt_constraints_satisfied:
                    current_alpha_for_constraint_check *= 0.5
                    current_p_step_for_constraint_check = p * current_alpha_for_constraint_check
                    current_x_new_for_constraint_check = x + current_p_step_for_constraint_check
                    choose_alpha_attempt += 1
                else:
                    constraints_satisfied_after_step = True
                    x_new = current_x_new_for_constraint_check # Adopt the x_new that satisfies constraints
                    p_step = current_p_step_for_constraint_check # And the corresponding step
                    choose_alpha_attempt = max_attempts_to_choose_alpha # Exit while loop

            if not constraints_satisfied_after_step:
                self.logger.error(f"Constraints not satisfied after {max_attempts_to_choose_alpha} attempts to reduce step. p_step={p_step}. Optimization terminated.")
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
        final_target_value = self.task.target.evaluate(self.task.get_variable_dict())
        self.logger.info(f"Final target function value: {final_target_value:.6f}")

        return self.task.get_variable_dict(), self.f_val_history, self.x_val_history

    def _solve_qp_subproblem(self, original_Hessian: np.ndarray, target_function_grad: np.ndarray, 
                             A: np.ndarray, b: np.ndarray, 
                             max_reg_attempts: int = 5, 
                             base_reg_eps: float = 1e-7,      # Smallest eigenvalue we'd like for QP Hessian
                             reg_eps_multiplier: float = 10.0): # How much reg_eps increases each time
        
        n = original_Hessian.shape[0]
        if n == 0: # No variables to optimize
            return np.array([]), np.array([])

        # CVXPY variable for the step p
        p_var = Variable(n) 

        # OSQP solver parameters (can be tuned)
        osqp_eps_abs = 1e-7 
        osqp_eps_rel = 1e-7
        osqp_max_iter = 10000 # Increased max iterations for OSQP

        H_for_qp = original_Hessian.copy() # Start with the passed Hessian (already BFGS-updated and regularized once)

        # Initial diagnostic log for the Hessian received by this function
        eigvals_initial = np.linalg.eigvalsh(0.5 * (H_for_qp + H_for_qp.T))
        lam_min_initial = eigvals_initial.min() if eigvals_initial.size > 0 else np.nan
        self.logger.debug(f"QP subproblem received Hessian with min_eig={lam_min_initial:.2e}")

        for attempt in range(max_reg_attempts):
            if attempt > 0:
                # For subsequent attempts, increase regularization on the *original* Hessian
                current_target_eps = base_reg_eps * (reg_eps_multiplier**(attempt)) # e.g., 1e-7, 1e-6, 1e-5...
                
                # Regularize the original Hessian passed to this function
                # _regularize_hessian returns H_sym + delta*I where H_sym is symmetrized original_Hessian
                regularized_H, delta = self._regularize_hessian(original_Hessian, eps=current_target_eps)
                
                if delta > 0:
                    self.logger.info(f"QP attempt {attempt + 1}: Regularized original Hessian for QP, targeting eps={current_target_eps:.1e}. Added delta={delta:.1e} to diagonal.")
                    H_for_qp = regularized_H
                else:
                    # If original_Hessian was already PD enough for current_target_eps (delta=0),
                    # but QP still failed in the previous attempt, we must add *some* explicit regularization.
                    # H_for_qp at this point would be original_Hessian_symmetrized.
                    explicit_add_val = current_target_eps * 0.1 # Add a fraction of the target_eps
                    H_for_qp = regularized_H + explicit_add_val * np.eye(n) 
                    self.logger.info(f"QP attempt {attempt + 1}: Original Hessian met target eps={current_target_eps:.1e}. Added explicit {explicit_add_val:.1e}*I for robustness.")
            
            # Log the minimum eigenvalue of the Hessian actually being used for this QP attempt
            eigvals_current_qp = np.linalg.eigvalsh(0.5 * (H_for_qp + H_for_qp.T))
            lam_min_current_qp = eigvals_current_qp.min() if eigvals_current_qp.size > 0 else np.nan
            self.logger.debug(f"QP attempt {attempt + 1}: Using Hessian with min_eig={lam_min_current_qp:.2e} for QP solve.")

            if lam_min_current_qp < 1e-9 and n > 0: # If still not very PD
                self.logger.warning(f"QP attempt {attempt + 1}: Hessian for QP (min_eig={lam_min_current_qp:.2e}) is very small/negative. This may cause issues.")
                # Could add more aggressive failsafe regularization here if needed
                # H_for_qp = H_for_qp + (1e-8 - lam_min_current_qp) * np.eye(n) 

            objective = Minimize(0.5 * quad_form(p_var, H_for_qp) + target_function_grad.T @ p_var)
            
            # Define constraints for the QP subproblem
            # A p <= b  (where b was -constraint_value in _prepare_constraints)
            qp_constraints = []
            if A.shape[0] > 0 : # Ensure A is not empty
                 qp_constraints = [A[i, :] @ p_var <= b[i] for i in range(A.shape[0])]
            
            prob = Problem(objective, qp_constraints)
            
            try:
                prob.solve(solver="OSQP", 
                           verbose=False, # Set to True for detailed OSQP logs
                           polish=True, 
                           eps_abs=osqp_eps_abs, 
                           eps_rel=osqp_eps_rel,
                           max_iter=osqp_max_iter,
                           check_termination=5 # Check termination criteria more often
                           ) 
            except Exception as e: # Catch errors from cvxpy/OSQP library itself
                self.logger.warning(f"CVXPY/OSQP solver raised an exception during solve on attempt {attempt + 1}: {e}")
                if attempt == max_reg_attempts - 1:
                    self.logger.error(f"QP subproblem failed after {max_reg_attempts} regularization attempts. Last exception: {e}")
                    self.logger.error(f"Final QP Hessian min_eig: {lam_min_current_qp:.2e}")
                    self.logger.error(f"Target_function_grad: {target_function_grad}")
                    # self.logger.error(f"A: {A}, b: {b}") # Can be very verbose
                    raise RuntimeError(f"QP subproblem failed due to solver exception after {max_reg_attempts} attempts: {e}")
                continue # Try next attempt with more regularization

            self.logger.info(f"QP Attempt {attempt+1}: Status: {prob.status}, Primal obj: {prob.value if prob.value is not None else 'N/A'}")

            if prob.status in ["optimal", "optimal_inaccurate"]:
                if p_var.value is not None:
                    # Check for NaN/Inf in solution, even if OSQP reports optimal
                    if np.any(np.isnan(p_var.value)) or np.any(np.isinf(p_var.value)):
                        self.logger.warning(f"QP solved (status {prob.status}) but p contains NaN/Inf on attempt {attempt+1}. p: {p_var.value}. Retrying.")
                        if attempt == max_reg_attempts - 1:
                            raise RuntimeError(f"QP subproblem resulted in NaN/Inf step p after {max_reg_attempts} attempts.")
                        continue # Failed this attempt, try more regularization

                    lambdas_val = np.array([con.dual_value for con in qp_constraints]) if qp_constraints else np.array([])
                    if prob.status == "optimal_inaccurate":
                        self.logger.warning(f"QP subproblem solved with status 'optimal_inaccurate' on attempt {attempt+1}. Solution quality might be reduced.")
                    return p_var.value, lambdas_val
                else: # Should not happen if status is optimal/optimal_inaccurate
                    self.logger.warning(f"QP solved (status {prob.status}) but p_var.value is None on attempt {attempt+1}. Retrying.")
            
            elif prob.status in ["infeasible", "infeasible_inaccurate"]:
                self.logger.error(f"QP subproblem is infeasible (status: {prob.status}) on attempt {attempt+1}. Linearized constraints may be inconsistent.")
                self.logger.debug(f"Infeasible QP details: A.shape={A.shape if A is not None else 'None'}, b.shape={b.shape if b is not None else 'None'}")
                # For infeasible QP, regularizing Hessian won't help directly. This is a critical failure.
                raise RuntimeError(f"QP subproblem is infeasible (status: {prob.status}). Check constraint formulation or problem scaling.")
            
            elif prob.status in ["unbounded", "unbounded_inaccurate"]:
                self.logger.warning(f"QP subproblem is unbounded (status: {prob.status}) on attempt {attempt+1}. Hessian may not be sufficiently PD, or constraints too loose.")
                # This should be helped by more Hessian regularization. Loop will continue.
                if attempt == max_reg_attempts - 1:
                     raise RuntimeError(f"QP subproblem remained unbounded after {max_reg_attempts} regularization attempts (status: {prob.status}).")
            
            else: # Other statuses like 'solver_error', 'user_limit', etc.
                self.logger.warning(f"QP subproblem failed with status: {prob.status} on attempt {attempt+1}. Solver stats: {prob.solver_stats}")
                if attempt == max_reg_attempts - 1:
                    raise RuntimeError(f"QP subproblem failed after {max_reg_attempts} attempts with final status: {prob.status}")
        
        # Fallback if loop finishes without returning (should ideally be caught by raises within the loop)
        raise RuntimeError(f"QP subproblem failed to solve after {max_reg_attempts} attempts. Exited loop unexpectedly.")
    
    def _regularize_hessian(self, H: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, float]:
        """
        Возвращает положительно-определённую копию Гессиана и величину добавленной регуляризации (для дебага).
        """
        if H.shape[0] == 0:
            return H.copy(), 0.0
            
        H_sym = 0.5 * (H + H.T)
        try:
            eigvals = np.linalg.eigvalsh(H_sym)
            lam_min = eigvals.min() if eigvals.size > 0 else 0
        except np.linalg.LinAlgError:
            self.logger.warning("LinAlgError during eigenvalue decomposition in _regularize_hessian. Forcing large regularization.")
            lam_min = -np.inf # Force regularization

        delta = 0.0
        if lam_min <= eps: # Check if less than or equal to ensure strictly PD if eps > 0
            delta = eps - lam_min + 1e-10 # Add a small positive to ensure > eps
            # if lam_min was very negative, delta becomes large, which is intended.
            # if lam_min was positive but just under eps, delta is small.
            # if eps is 0, delta ensures lam_min becomes 1e-10 (strictly positive)
            # if eps is, say, 1e-7, delta ensures lam_min becomes 1e-7 + 1e-10
            H_reg = H_sym + delta * np.eye(H_sym.shape[0])
            self.logger.debug(f"Regularizing Hessian: min_eig={lam_min:.2e}, eps={eps:.1e}, added_delta={delta:.1e}")
            return H_reg, delta
        
        return H_sym, 0.0 # Already sufficiently positive definite

    def _compute_gradient(self, func, x: np.ndarray) -> np.ndarray:
        """
        Вычисляет градиент функции func в точке x, используя центральные конечные разности.
        """
        eps = self.task.config.grad_eps
        x = x.astype(float) # Убедимся, что x - это float для арифметики
        grad = np.zeros_like(x, dtype=float) # Ensure grad is float
        
        # Central differences for better accuracy if affordable
        # f0 = func.evaluate(dict(zip(self.task.get_variable_names(), x))) # Not needed for central diff
        
        if len(x) == 0:
            return grad

        for i in range(len(x)):
            x_plus_eps = x.copy()
            x_plus_eps[i] += eps
            f_plus_eps = func.evaluate(dict(zip(self.task.get_variable_names(), x_plus_eps)))

            x_minus_eps = x.copy()
            x_minus_eps[i] -= eps
            f_minus_eps = func.evaluate(dict(zip(self.task.get_variable_names(), x_minus_eps)))
            grad[i] = (f_plus_eps - f_minus_eps) / (2 * eps)
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

        # Powell's Damping: Ensure yᵀs > 0 for PD update
        if ys <= curvature_threshold * np.linalg.norm(s)**2 : # Scaled threshold
            sBs = float(s.T @ B @ s)
            if sBs <= 0: # B is not PD w.r.t s, cannot reliably damp. Reset or skip update.
                self.logger.warning(f"BFGS: sBs = {sBs:.2e} <= 0. Resetting Hessian to Identity or skipping update.")
                # return np.eye(B.shape[0]) # Option: Reset Hessian
                return B # Option: Skip update if B is problematic

            # Theta calculation as in Nocedal & Wright, Eq. (6.20) or similar, adapted for direct Hessian
            # We want y_new = theta*y + (1-theta)*B*s such that y_new^T s >= threshold'
            # Standard Powell damping for direct Hessian (Nocedal & Wright, around page 143, for inverse Hessian, adapt)
            # Simpler: ensure yTs is positive before BFGS. If not, modify y.
            theta = 1.0
            if ys < 0.2 * sBs : # If curvature is not sufficiently positive compared to B's curvature along s
                theta = (0.8 * sBs) / (sBs - ys)
            else: # Curvature is OK or ys > 0.2 * sBs
                theta = 1.0 # No damping needed if ys is already good.
            
            if theta < 0 or theta > 1: # Should not happen with correct formula but as safeguard
                 self.logger.warning(f"BFGS Powell Damping: Unusual theta={theta:.2e}. ys={ys:.2e}, sBs={sBs:.2e}. Using theta=1.")
                 theta = 1.0

            y_damped = theta * y + (1 - theta) * (B @ s)
            ys_damped = float(y_damped.T @ s)

            if ys_damped <= curvature_threshold * np.linalg.norm(s)**2 : # Still bad after damping?
                self.logger.warning(f"BFGS: Curvature condition yᵀs ({ys_damped:.2e}) still too low after Powell damping. Skipping update.")
                return B # Skip update
            
            y = y_damped # Use damped y for the update
            ys = ys_damped
            self.logger.debug(f"BFGS: Powell damping applied. Original yᵀs={float(y.T @ s):.2e}, Damped yᵀs={ys:.2e}, theta={theta:.2e}")


        # Standard BFGS update formula for direct Hessian B
        Bs = B @ s
        sBs = float(s.T @ Bs) # sᵀBs

        if abs(sBs) < 1e-12 or abs(ys) < 1e-12: # Denominators too small
            self.logger.warning("BFGS: Denominator sBs or ys too small. Skipping update.")
            return B

        term1 = (Bs @ Bs.T) / sBs   # (B s sᵀ B) / (sᵀ B s)
        term2 = (y @ y.T) / ys      # (y yᵀ) / (yᵀ s)
        
        B_new = B - term1 + term2

        # Final regularization to ensure PD, _regularize_hessian's default eps is 1e-8
        B_regularized, delta_reg = self._regularize_hessian(B_new) 
        if delta_reg > 0.0:
            self.logger.info(f"Гессиан не PD; добавлена регуляризация δ={delta:.2e}")

        return B_regularized

    def _prepare_constraints(self, x: np.ndarray):
        A_list, b_list = [], [] # Use lists to append
        var_names = self.task.get_variable_names()
        
        # Variable bounds: lower <= x_i <= upper  =>  x_i - upper <= 0  AND  lower - x_i <= 0
        for i, variable in enumerate(self.task.variables):
            # Constraint: x_i <= upper_i  =>  1*x_i - upper_i <= 0
            if variable.upper != np.inf:
                grad_upper = np.zeros(len(x))
                grad_upper[i] = 1.0
                A_list.append(grad_upper)
                b_list.append(variable.upper - x[i]) # b is g(x_k) for constraint g(x_k) + nabla_g^T p <= 0
                                                     # Here, constraint is p_i <= upper_i - x_i_k
                                                     # So, A_row p <= b_val, where A_row_i = 1, b_val = upper_i - x_i_k

            # Constraint: x_i >= lower_i  =>  -1*x_i + lower_i <= 0
            if variable.lower != -np.inf:
                grad_lower = np.zeros(len(x))
                grad_lower[i] = -1.0
                A_list.append(grad_lower)
                b_list.append(x[i] - variable.lower) # Constraint -p_i <= x_i_k - lower_i
                                                     # So, A_row p <= b_val, where A_row_i = -1, b_val = x_i_k - lower_i

        # General constraints
        for constraint in self.task.constraints:
            # We need g(x_k) and grad_g(x_k)
            # The QP constraint is g(x_k) + grad_g(x_k)^T p <= 0  (for g(x) <= 0 type)
            # or h(x_k) + grad_h(x_k)^T p = 0 (for h(x) = 0 type)
            
            constraint_value_at_x = constraint.evaluate(dict(zip(var_names, x)))
            constraint_gradient_at_x = self._compute_gradient(constraint, x)
            constraint_gradient_at_x = np.atleast_1d(constraint_gradient_at_x)

            if constraint.type == ConstraintType.INEQ: # g(x) <= 0
                A_list.append(constraint_gradient_at_x) # grad_g(x_k)
                b_list.append(-constraint_value_at_x)   # -g(x_k)
            elif constraint.type == ConstraintType.EQ: # h(x) = 0
                # h(x_k) + grad_h(x_k)^T p = 0  is split into two inequalities:
                # 1)  grad_h(x_k)^T p <= -h(x_k)
                # 2) -grad_h(x_k)^T p <=  h(x_k)
                A_list.append(constraint_gradient_at_x)
                b_list.append(-constraint_value_at_x)
                
                A_list.append(-constraint_gradient_at_x)
                b_list.append(constraint_value_at_x)
        
        if not A_list: # No constraints
            return np.zeros((0, len(x))), np.zeros(0)
            
        return np.array(A_list), np.array(b_list)


    def _evaluate_merit(self, x: np.ndarray) -> float:
        # L1 exact penalty merit function: f(x) + penalty_coeff * sum(max(0, g_i(x))) + penalty_coeff * sum(|h_j(x)|)
        # Using L2 penalty for now as per original code.
        penalty_coeff = self.task.config.penalty_coeff # Use penalty from task config, updated based on lambdas
        
        var_dict = dict(zip(self.task.get_variable_names(), x))
        f_val = self.task.target.evaluate(var_dict)
        
        constraint_violation_sum_sq = 0.0
        
        # General constraints
        for constraint in self.task.constraints:
            val = constraint.evaluate(var_dict)
            if constraint.type == ConstraintType.INEQ: # g(x) <= 0
                constraint_violation_sum_sq += max(0, val)**2 # Penalty for g(x) > 0
            elif constraint.type == ConstraintType.EQ: # h(x) = 0
                constraint_violation_sum_sq += val**2      # Penalty for h(x) != 0

        # Variable bounds violation (implicitly handled by QP step if bounds are part of A,b)
        # But for merit function, it's good to include them explicitly if not using strict barrier for line search
        for variable_obj in self.task.variables:
            val_x = var_dict.get(variable_obj.name)
            if val_x < variable_obj.lower:
                constraint_violation_sum_sq += (variable_obj.lower - val_x)**2
            if val_x > variable_obj.upper:
                constraint_violation_sum_sq += (val_x - variable_obj.upper)**2
                
        merit = f_val + penalty_coeff * constraint_violation_sum_sq
        return merit

    def _line_search(self, x_k: np.ndarray, p_k: np.ndarray, c1: float = 1e-4, beta: float = 0.5, max_trials: int = 20) -> float:
        """
        Backtracking line search using the L2 merit function to find step size alpha.
        Ensures sufficient decrease condition (Armijo-like for merit function).
        Directional derivative for merit function: D_p(phi(x)) = grad_f(x)^T p - rho * (penalty term derivative)
        Approximated by D_p(phi(x)) approx grad_f(x)^T p - rho * sum(violations) (for L1 penalty)
        For simplicity with L2, we check phi(x + alpha*p) <= phi(x) + c1 * alpha * D,
        where D is an approximation of the directional derivative. A common choice for D is grad_f^T p_k.
        However, this can be problematic if constraints are violated.
        A safer bet is to ensure phi(x_k + alpha*p_k) < phi(x_k) (descent on merit function).
        Let's use a slightly more robust Armijo condition for the merit function.
        The directional derivative of the L2 merit function is complex.
        A common approach is to use phi(x_k + alpha p_k) <= phi(x_k) + c1 * alpha * D_k
        where D_k = grad_f(x_k)^T p_k - penalty_coeff * C(x_k), where C(x_k) is sum of squared violations.
        This ensures that the step decreases the merit function relative to a "linearized" decrease.
        """
        alpha = 1.0 # Start with full step
        # self._prev_alpha can be used too, but 1.0 is standard for SQP Newton-like steps.
        
        phi_k = self._evaluate_merit(x_k)
        
        # Approximate directional derivative of the merit function.
        # This is a heuristic. For L1 merit, it's grad_f^T p - penalty_coeff * L1_norm_of_violations_at_p
        # For L2, it's more complex. A simple choice often used:
        grad_f_k = self._compute_gradient(self.task.target, x_k) # Recompute if not available
        directional_derivative_approx = grad_f_k @ p_k # Approximation of target func decrease
        
        # If p_k is a good descent direction for f, directional_derivative_approx will be < 0.
        # We also want to reduce constraint violations.
        # A more conservative check: ensure phi_trial simply decreases.
        # phi_trial <= phi_k - c1 * alpha * ||p_k||^2 (forcing decrease proportional to step size squared if D is problematic)

        self.logger.debug(f"Line Search: phi(x_k)={phi_k:.4e}, approx_D={directional_derivative_approx:.4e}")

        for i in range(max_trials):
            x_trial = x_k + alpha * p_k
            phi_trial = self._evaluate_merit(x_trial)
            
            # Armijo condition for merit function.
            # Use a simplified condition: just ensure merit function decreases.
            # If directional_derivative_approx is positive (p_k is ascent for f), this condition relies on penalty term.
            # A very basic Armijo: phi_trial <= phi_k + c1 * alpha * directional_derivative_approx
            # This requires directional_derivative_approx to be negative for descent.
            # If p_k is from a QP that minimizes f subject to linearized constraints,
            # it should be a descent direction for f (or zero if at constrained optimum).
            
            # If directional_derivative_approx > 0 (ascent for f), this Armijo form is problematic.
            # Let's use a robust check: ensure phi_trial < phi_k if alpha is small enough.
            # Modified Armijo: phi_trial <= phi_k + c1 * alpha * D_phi_p
            # D_phi_p = grad_f^T p - mu * (linearized constraint violation decrease)
            # For now, let's use the simpler phi_trial <= phi_k (simple decrease)
            # with a small tolerance or ensure directional_derivative_approx is negative.
            
            # If p_k is a Newton step for the KKT system, it's a descent direction for some merit function.
            # Using the simple Armijo with D_k = grad_f^T p_k:
            # This might fail if p_k increases f but significantly reduces constraint violation.
            
            # Alternative D_k from Nocedal & Wright for L1 merit function (adapted):
            # D_k = grad_f(x_k)^T p_k - self.task.config.penalty_coeff * sum_of_abs_constraint_violations
            # For L2, it's more complex. Let's use the phi_0_grad @ p from original code.

            phi_k_grad_approx_merit = self._compute_merit_gradient(x_k) # Grad of merit function
            D_merit = phi_k_grad_approx_merit @ p_k


            if phi_trial <= phi_k + c1 * alpha * D_merit: # Armijo condition on merit function
                self.logger.info(f"Line Search: Accepted alpha={alpha:.4e} on trial {i+1}. phi_trial={phi_trial:.4e}, D_merit={D_merit:.4e}")
                self._prev_alpha = alpha # Store for potential next iteration start (adaptive)
                return alpha
            
            self.logger.debug(f"Line Search: alpha={alpha:.4e} failed. phi_trial={phi_trial:.4e}, Condition RHS={phi_k + c1 * alpha * D_merit:.4e}")
            alpha *= beta

        self.logger.warning(f"Line search failed to find suitable alpha after {max_trials} trials. Using smallest alpha={alpha:.4e}.")
        self._prev_alpha = alpha 
        return alpha # Return smallest alpha if no success


    def _check_constraints(self, variable_dict: dict, tol_scale: float = 1.0) -> bool:
        """Checks if all variable bounds and explicit constraints are satisfied."""
        # Check variable bounds
        for variable_obj in self.task.variables:
            current_value = variable_dict.get(variable_obj.name)
            # Add a small tolerance to bounds check to avoid issues with strict floating point comparison
            # This tolerance should be related to machine epsilon or problem scale.
            bound_tol = self.task.config.tol * 1e-1 # Smaller tolerance for hard bounds
            if current_value < variable_obj.lower - bound_tol or \
               current_value > variable_obj.upper + bound_tol:
                self.logger.debug(f"Constraint Check: Variable {variable_obj.name}={current_value:.4e} out of bounds [{variable_obj.lower:.4e}, {variable_obj.upper:.4e}]")
                return False

        # Check explicit constraints
        effective_tol = self.task.config.tol * tol_scale
        for constraint in self.task.constraints:
            constraint_value = constraint.evaluate(variable_dict)
            self.logger.debug(f"Constraint Check: {constraint.name} = {constraint_value:.4e}")
            if constraint.type == ConstraintType.INEQ and constraint_value > effective_tol: # g(x) <= 0
                self.logger.debug(f"Constraint Check: INEQ {constraint.name} violated ({constraint_value:.4e} > {effective_tol:.4e})")
                return False
            if constraint.type == ConstraintType.EQ and abs(constraint_value) > effective_tol: # h(x) = 0
                self.logger.debug(f"Constraint Check: EQ {constraint.name} violated (|{constraint_value:.4e}| > {effective_tol:.4e})")
                return False
        return True

    def _log_iteration(self, i, x, grad_target_f, step):
        f_value = self.task.target.evaluate(self.task.get_variable_dict())
        grad_norm_target_f = np.linalg.norm(grad_target_f)
        step_norm = np.linalg.norm(step)

        self.f_val_history.append(f_value)
        self.x_val_history.append(self.task.get_variable_values())
        
        self.logger.info(
            f"Iter {i+1}: f(x)={f_value:.6e} | ||grad(f(x))||={grad_norm_target_f:.4e} | ||step||={step_norm:.4e}"
        )

        for variable in self.task.variables:
            self.logger.info(f"-- {variable.name}: {variable.value:.6f} | ")


    def _check_convergence(self, grad_target_f, step):
        # KKT conditions are more complex to check directly without lambdas for general constraints.
        # Common convergence criteria for SQP:
        # 1. Norm of the step p_k is small.
        # 2. Norm of the gradient of the Lagrangian is small. (Approximated by grad_target_f if lambdas are unstable)
        # 3. Constraint satisfaction. (Checked before accepting step)

        grad_norm = np.linalg.norm(grad_target_f) # Norm of gradient of target function
        step_norm = np.linalg.norm(step)
        tol = self.task.config.tol

        # Basic convergence: step size and gradient norm
        # (Constraint satisfaction is handled by step acceptance logic)
        converged_step = step_norm < tol
        converged_grad = grad_norm < tol # This is for unconstrained; for constrained, Lagrangian grad needed.

        if converged_step:
            self.logger.info(f"Convergence: Step norm ({step_norm:.2e}) < tol AND Target Grad norm ({grad_norm:.2e}) < tol.")
            return True
        
        if converged_step:
             self.logger.info(f"Convergence: Step norm ({step_norm:.2e}) < tol. Checking overall progress.")
             # If step is small, check if gradient is also reasonably small or if progress has stalled.
             if abs(self._last_grad_norm - grad_norm) < tol * 1e-2 and \
                abs(self._last_step_size - step_norm) < tol * 1e-2: # Stagnation
                 self.logger.info("Convergence: Little progress in grad norm and step size. Assuming convergence.")
                 return True


        # Stagnation check (from original code)
        # This checks if successive iterations yield very small changes in grad_norm and step_norm.
        # Could be due to reaching a local minimum or getting stuck.
        if abs(self._last_grad_norm - grad_norm) < tol * 1e-3 and \
           abs(self._last_step_size - step_norm) < tol * 1e-3 and \
           iteration > 5 : # Allow a few iterations before declaring stagnation
            # And constraints must be satisfied at current point
            if self._check_constraints(self.task.get_variable_dict(), tol_scale=10.0): # Check with slightly looser tol
                self.logger.info("Convergence: Stagnation in gradient and step size changes with satisfied constraints.")
                return True

        self._last_grad_norm = grad_norm
        self._last_step_size = step_norm
        return False

    def _compute_merit_gradient(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray: # Use a slightly larger eps for merit grad
        x = x.astype(float)
        grad = np.zeros_like(x, dtype=float)
        
        if len(x) == 0:
            return grad

        # Using central differences for merit gradient
        for i in range(len(x)):
            x_plus_eps = x.copy()
            x_plus_eps[i] += eps
            f_plus_eps = self._evaluate_merit(x_plus_eps)

            x_minus_eps = x.copy()
            x_minus_eps[i] -= eps
            f_minus_eps = self._evaluate_merit(x_minus_eps)
            
            grad[i] = (f_plus_eps - f_minus_eps) / (2 * eps)
        return grad

    def _setup_logger(self):
        self.logger = logging.getLogger(f'SQPOptimizer_{id(self)}') # Unique logger name
        self.logger.setLevel(logging.INFO) # Set to DEBUG for more verbose output
        
        # Avoid adding handlers if they already exist (e.g., in Jupyter multiple runs)
        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            log_filename = f"sqp_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            try:
                file_handler = logging.FileHandler(log_filename)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.error(f"Failed to create file handler for logging: {e}")

        self.logger.info("Optimizer initialized")
        self.logger.info(f"Target function: {self.task.target.name}")
        self.logger.info(f"Number of variables: {len(self.task.variables)}")
        self.logger.info(f"Number of constraints: {len(self.task.constraints)}")
        self.logger.info(f"Config: max_iter={self.task.config.max_iter}, tol={self.task.config.tol}, grad_eps={self.task.config.grad_eps}")