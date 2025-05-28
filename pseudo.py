import numpy as np

class NLPQLPOptimizer:
    def __init__(self, objective_func, constraints_eq, constraints_ineq, x0,
                 grad_obj=None, grad_eq=None, grad_ineq=None,
                 hess_approx=None, rho0=10.0, tau=10.0, eps_f=1e-6, eps_c=1e-6, max_iter=100):
        self.f = objective_func
        self.ceq = constraints_eq
        self.cineq = constraints_ineq
        self.x = np.array(x0, dtype=float)
        self.grad_f = grad_obj
        self.grad_ceq = grad_eq
        self.grad_cineq = grad_ineq
        self.hess = hess_approx if hess_approx is not None else np.eye(len(x0))
        self.rho = rho0
        self.tau = tau
        self.eps_f = eps_f
        self.eps_c = eps_c
        self.max_iter = max_iter

    def penalty(self, x):
        penalty_val = 0.0
        if self.cineq:
            penalty_val += sum(max(0, ci(x)) ** 2 for ci in self.cineq)
        if self.ceq:
            penalty_val += sum(ce(x) ** 2 for ce in self.ceq)
        return penalty_val

    def psi(self, x):
        return self.f(x) + self.rho * self.penalty(x)

    def gradient(self, x):
        if self.grad_f is None:
            eps = 1e-8
            return np.array([(self.f(x + eps * np.eye(len(x))[i]) - self.f(x)) / eps for i in range(len(x))])
        return self.grad_f(x)

    def optimize(self):
        for k in range(self.max_iter):
            grad = self.gradient(self.x)
            d = -np.linalg.solve(self.hess, grad)

            # Line search with penalty
            alpha = 1.0
            while self.psi(self.x + alpha * d) > self.psi(self.x) + 1e-4 * alpha * np.dot(grad, d):
                alpha *= 0.5
                if alpha < 1e-8:
                    break

            x_new = self.x + alpha * d

            # Constraint violation check
            max_viol = 0.0
            if self.cineq:
                max_viol = max(max_viol, max(max(0, ci(x_new)) for ci in self.cineq))
            if self.ceq:
                max_viol = max(max_viol, max(abs(ce(x_new)) for ce in self.ceq))

            if max_viol > self.eps_c:
                self.rho *= self.tau

            # Update
            s = x_new - self.x
            y = self.gradient(x_new) - grad
            ys = np.dot(y, s)
            if ys > 1e-10:
                self.hess += np.outer(y, y) / ys - np.dot(self.hess @ np.outer(s, s) @ self.hess.T, 1.0 / np.dot(s, self.hess @ s))

            self.x = x_new

            # Check convergence
            if np.linalg.norm(grad) < self.eps_f and self.penalty(self.x) < self.eps_c:
                break

        return self.x, self.f(self.x), self.penalty(self.x)
