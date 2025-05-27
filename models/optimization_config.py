class OptimizationConfig:
    def __init__(self, max_iter=100, tol=1e-5, eps=1, penalty_coeff=50.0, delta_tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.penalty_coeff = penalty_coeff
        self.delta_tol = delta_tol
        self.eps = eps
