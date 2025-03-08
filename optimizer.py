import numpy as np


class MiniBatchGradientDescent:
    def __init__(self, learning_rate=0.01, batch_size=32, shuffle=True):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.shuffle = shuffle

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.learning_rate * grads[key]

    def get_minibatches(self, X, y):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, n_samples, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            yield X[batch_idx], y[batch_idx]
            

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, batch_size=32, shuffle=True):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.m = None
        self.v = None
        self.t = 0
        
    def update(self, params, grads):
        if self.m is None:
            self.m = {}
            self.v = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
                
        self.t += 1
        lr_t = self.learning_rate * np.sqrt(1.0 - self.beta2**self.t) / (1.0 - self.beta1**self.t)
        
        for key in params.keys():
            # 更新一阶矩估计
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            # 更新二阶矩估计
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key]**2
            # 参数更新
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + self.epsilon)
            
    def get_minibatches(self, X, y):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, n_samples, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            yield X[batch_idx], y[batch_idx]