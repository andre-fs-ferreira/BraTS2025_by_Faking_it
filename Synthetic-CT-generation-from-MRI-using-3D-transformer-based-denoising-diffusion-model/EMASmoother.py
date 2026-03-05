class EMASmoother:
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * self.value + (1.0 - self.alpha) * x
        return self.value