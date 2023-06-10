class Module:
    def __init__(self):
        pass
    def __len__(self):
        return 0
    def __str__(self):
        s = f"{self.__class__.__name__}"
        if self.parameters() is not None:
            s += f"\t{self.parameters()}"
        s += "\n"
        return s
    def __call__(self, x):
        self.input = x
        self.output = self.forward(x)
        return self.output
    def forward(self, x):
        pass
    def backward(self, x):
        pass
    def grad(self):
        return None
    def parameters(self):
        return None
    def reset(self):
        pass
    def set_param(self, x):
        pass