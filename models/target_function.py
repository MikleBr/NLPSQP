class TargetFunction:
    def __init__(self, func, name: str = ""):
        self.func = func
        self.name = name

    def evaluate(self, variables: dict) -> float:
        return self.func(variables)
