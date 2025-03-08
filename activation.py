import numpy as np

class ReLU():
    def __init__(self):
        self.Z = None

    def forward(self, X):
        self.Z = np.maximum(0, X)
        return self.Z

    def backward(self, dA):
        return dA * (self.Z > 0).astype(float)
    
class Sigmoid():
    def __init__(self):
        self.A = None
        
    def forward(self, X):
        self.A = 1 / (1 + np.exp(-X))
        return self.A
    
    def backward(self, dA):
        return dA * self.A * (1 - self.A)
    
class Tanh():
    def __init__(self):
        self.A = None
        
    def forward(self, X):
        self.A = np.tanh(X)
        return self.A
    
    def backward(self, dA):
        return dA * (1 - self.A ** 2)
    
class LeakyReLU():
    def __init__(self):
        self.Z = None

    def forward(self, X):
        self.Z = np.maximum(0.01 * X, X)
        return self.Z

    def backward(self, dA):
        return dA * (self.Z > 0).astype(float) + 0.01 * dA * (self.Z <= 0).astype(float)
    
class Softmax():
    def __init__(self):
        self.Z = None
        
    def forward(self, X):
        exps = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.Z = exps / np.sum(exps, axis=1, keepdims=True)
        return self.Z
    
    def backward(self, dA):
        #NOTE 这里的梯度直接反传，在交叉熵损失函数中一并简化计算，只能在输出层使用
        return dA
    
    