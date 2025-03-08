import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os

class CrossEntropyLoss:
    def __init__(self):
        self.y_hat = None
        self.y = None
    
    def forward(self, y_hat, y):
        # 添加数值稳定性
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1.0 - epsilon)
        self.y_hat = y_hat
        self.y = y
        return -np.sum(y * np.log(y_hat)) / y.shape[0]

    def backward(self):
        # 叉熵梯度，由于y_hat是softmax输出
        return (self.y_hat - self.y) / self.y.shape[0]
    
    
class L2Regularization:
    def __init__(self, lambda_):
        self.lambda_ = lambda_
        self.weights = None
    
    def forward(self, weights):
        self.weights = weights
        return 0.5 * self.lambda_ * np.sum(weights * weights)
    
    def backward(self):
        return self.lambda_ * self.weights
    
            
            
def load_mnist_data(cache_path='mnist_cache.npz', val_size=0.1):
    # 如果缓存文件存在则直接加载
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']
        y_train_encoded = data['y_train_encoded']
        y_val_encoded = data['y_val_encoded']
        y_test_encoded = data['y_test_encoded']
        print("从缓存加载 MNIST 数据。")
        return X_train, X_val, X_test, y_train_encoded, y_val_encoded, y_test_encoded

    # 通过 fetch_openml 加载 MNIST 数据集
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist.data.to_numpy(), mnist.target.astype(int).to_numpy()
    
    # 将像素值归一化到 0~1
    X = X.astype(np.float32) / 255.0

    # 先划分出测试集
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 然后从剩余数据中划分训练集和验证集
    test_val_ratio = val_size / (1 - 0.2)  # 调整验证集比例
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=test_val_ratio, random_state=42)
    

    # 确保标签是数字类型
    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)
    y_test = y_test.astype(np.int32)
    
    # 获取类别数量
    classes = np.unique(np.concatenate([y_train, y_val, y_test]))
    n_classes = len(classes)
    
    # 独热编码
    y_train_encoded = np.eye(n_classes)[y_train.astype(int)]
    y_val_encoded = np.eye(n_classes)[y_val.astype(int)]
    y_test_encoded = np.eye(n_classes)[y_test.astype(int)]

    # 保存到本地缓存
    np.savez(cache_path,
             X_train=X_train, X_val=X_val, X_test=X_test,
             y_train_encoded=y_train_encoded, y_val_encoded=y_val_encoded, y_test_encoded=y_test_encoded)
    print("MNIST数据已缓存到本地。")

    return X_train, X_val, X_test, y_train_encoded, y_val_encoded, y_test_encoded