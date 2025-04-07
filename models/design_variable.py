import numpy as np

class DesignVariable:
    def __init__(self, name: str, value: float, lower: float = -np.inf, upper: float = np.inf):
        self.name = name
        self.value = value
        self.lower = lower
        self.upper = upper

    def clip(self):
        self.value = max(min(self.value, self.upper), self.lower)
