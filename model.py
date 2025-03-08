import numpy as np
from utils import L2Regularization
from activation import Softmax


class FullyConnectedLayer:
    """全连接层
    参数:
        input_dim: 输入维度
        output_dim: 输出维度
        activation: 激活函数
    """

    def __init__(self, input_dim, output_dim, activation=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._activate = activation

        self.weights = np.random.randn(self.input_dim, self.output_dim) * 0.01
        self.biases = np.zeros((1, self.output_dim))

    def forward(self, X):
        self.X = X
        self.Z = np.dot(X, self.weights) + self.biases
        if self._activate:
            self.Z = self._activate.forward(self.Z)
        return self.Z

    def backward(self, dZ):
        """反向传播计算梯度
        参数:
            dZ: 损失函数L对本层输出Z的梯度（来自下一层的梯度）

        返回:
            self.dA: 损失函数对上一层输出的梯度（传递给上一层）
        """
        if self._activate:
            # ∂L/∂Z_linear = ∂L/∂Z · ∂Z/∂Z_linear = dZ · activate'(Z_linear)
            dZ = self._activate.backward(dZ)

        # 权重梯度: ∂L/∂W = ∂L/∂Z_linear · ∂Z_linear/∂W = dZ · X,这里考虑到矩阵运算的特性，所以代码是X^T · dZ
        self.dW = np.dot(self.X.T, dZ)

        # 计算偏置梯度: ∂L/∂b = ∂L/∂Z_linear · ∂Z_linear/∂b = dZ · 1，由于每个样本都有梯度贡献，需要对所有样本求和
        self.db = np.sum(dZ, axis=0, keepdims=True)

        # 计算向上一层传递的梯度: 上一层梯度：∂L/∂X = ∂L/∂Z_linear · ∂Z_linear/∂X = dZ · W^T
        self.dA = np.dot(dZ, self.weights.T)

        return self.dA


class NeuralNetwork:
    """
    神经网络模型
    参数:
        hidden_dims: 隐藏层维度
        activation: 激活函数
        lammbda: 正则化系数
    """

    def __init__(self, input_dim, output_dim, hidden_dims, activation, lammbda):
        self.layers = []
        # 输入层
        self.layers.append(FullyConnectedLayer(input_dim, hidden_dims[0], None))
        # 隐藏层
        for i in range(len(hidden_dims) - 1):
            self.layers.append(
                FullyConnectedLayer(hidden_dims[i], hidden_dims[i + 1], activation)
            )
        # 输出层
        self.layers.append(FullyConnectedLayer(hidden_dims[-1], output_dim, Softmax()))
        self.l2 = L2Regularization(lammbda)

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def _forward(self, X):
        """前向传播通过所有层"""
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def _backward(self, dZ):
        """反向传播通过所有层"""
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)
        return dZ

    def predict(self, X):
        """预测新数据"""
        return self._forward(X)

    def evaluate(self, X, y):
        """评估模型性能"""
        y_pred = self.predict(X)
        # 对于分类问题，取最大概率的类别
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y, axis=1)
        accuracy = np.mean(y_pred_classes == y_true_classes)

        # 计算损失
        loss = self.loss.forward(y_pred, y)

        return loss, accuracy

    def fit(self, X, y, X_val=None, y_val=None, epochs=10, verbose=True):
        """训练神经网络模型

        参数:
            X: 训练数据
            y: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            epochs: 训练轮数
        """
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        for epoch in range(epochs):
            epoch_loss = 0
            batches = 0

            for X_batch, y_batch in self.optimizer.get_minibatches(X, y):
                # 前向传播
                y_pred = self._forward(X_batch)

                # 计算原始损失
                base_loss = self.loss.forward(y_pred, y_batch)

                # 计算L2正则化损失并添加到总损失
                reg_loss = 0
                for layer in self.layers:
                    reg_loss += self.l2.forward(layer.weights)

                total_loss = base_loss + reg_loss
                epoch_loss += total_loss
                batches += 1

                # 反向传播
                dZ = self.loss.backward()
                gradients = self._backward(dZ)

                # 构建参数和梯度字典
                self.params = {}
                self.grads = {}
                for i, layer in enumerate(self.layers):
                    # 添加每一层的参数
                    self.params[f"W{i}"] = layer.weights
                    self.params[f"b{i}"] = layer.biases

                    # 添加每一层的梯度，并加上L2正则化梯度
                    self.l2.forward(layer.weights)
                    reg_grad = self.l2.backward()

                    self.grads[f"W{i}"] = layer.dW + reg_grad
                    self.grads[f"b{i}"] = layer.db

                # 更新参数
                self.optimizer.update(self.params, self.grads)

            # 计算训练集性能
            train_loss, train_acc = self.evaluate(X, y)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # 如果提供了验证集，则计算验证集性能
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                if verbose:
                    print(
                        f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
                    )
            else:
                if verbose:
                    print(
                        f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f}"
                    )

        return self.history
