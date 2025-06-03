class OptimizationConfig:
    def __init__(self, max_iter=100, tol=1e-5, eps=1, grad_eps=1e-6, penalty_coeff=0.0, delta_tol=1e-6, tau: float = 1.0):
        self.max_iter = max_iter
        self.tol = tol
        self.penalty_coeff = penalty_coeff
        self.delta_tol = delta_tol
        self.eps = eps
        self.grad_eps = grad_eps
        # параметр, на который умножается единичная диагональная матрица B на первом шаге
        # очень полезно, если на первом шаге подзадача QP плохо решается
        # и нужно поменять матрицу Гессе для улучшение глобальной сходимости
        self.tau = tau