import os
import numpy as np
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

from model import NeuralNetwork
from utils import load_mnist_data, CrossEntropyLoss
from activation import ReLU, LeakyReLU, Tanh, Sigmoid
from optimizer import MiniBatchGradientDescent, Adam
from visualize import plot_training_history, plot_confusion_matrix, plot_misclassified_examples

def parse_args():
    parser = argparse.ArgumentParser(description='MNIST 手写数字分类')
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'model1', 'model2', 'custom'],
                        help='要使用的模型 (默认: baseline)')
    parser.add_argument('--hidden', type=str, default='256,128',
                        help='隐藏层维度，用逗号分隔 (默认: 256,128)')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'leaky_relu', 'tanh', 'sigmoid'],
                        help='激活函数 (默认: relu)')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'momentum', 'adam'],
                        help='优化器 (默认: sgd)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学习率 (默认: 0.01)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批量大小 (默认: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数 (默认: 100)')
    parser.add_argument('--lambda', type=float, dest='lambda_val', default=1e-4,
                        help='L2正则化系数 (默认: 1e-4)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout比例 (默认: 0.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    return parser.parse_args()

def get_activation(name):
    """根据名称获取激活函数实例"""
    if name == 'relu':
        return ReLU()
    elif name == 'leaky_relu':
        return LeakyReLU()
    elif name == 'tanh':
        return Tanh()
    elif name == 'sigmoid':
        return Sigmoid()
    else:
        raise ValueError(f"不支持的激活函数: {name}")

def get_optimizer(name, lr, batch_size):
    """根据名称获取优化器实例"""
    if name == 'sgd':
        return MiniBatchGradientDescent(learning_rate=lr, batch_size=batch_size)
    elif name == 'adam':
        return Adam(learning_rate=lr, batch_size=batch_size)
    else:
        raise ValueError(f"不支持的优化器: {name}")

def main():
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 加载数据
    print("正在加载 MNIST 数据集...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_mnist_data()
    print(f"数据集加载完成: 训练集 {X_train.shape[0]} 样本, 验证集 {X_val.shape[0]} 样本, 测试集 {X_test.shape[0]} 样本")
    
    input_dim = 784  # 28x28 像素
    output_dim = 10  # 0-9 数字
    
    # 解析隐藏层维度
    hidden_dims = [int(dim) for dim in args.hidden.split(',')]
    
    # 创建模型
    if args.model == 'baseline':
        model = NeuralNetwork(input_dim, output_dim, hidden_dims=[256, 128], 
                             activation=ReLU(), lammbda=1e-4)
        print("使用 baseline 模型: [784, 256, 128, 10], ReLU, lambda=1e-4")
    elif args.model == 'model1':
        model = NeuralNetwork(input_dim, output_dim, hidden_dims=[512, 256], 
                             activation=LeakyReLU(), lammbda=1e-3)
        print("使用 baseline1 模型: [784, 512, 256, 10], LeakyReLU, lambda=1e-3")
    elif args.model == 'model2':
        model = NeuralNetwork(input_dim, output_dim, hidden_dims=[128], 
                             activation=Tanh(), lammbda=0)
        print("使用 model2 模型: [784, 128, 10], Tanh, lambda=0")
    else:  # 自定义模型
        activation = get_activation(args.activation)
        
        model = NeuralNetwork(input_dim, output_dim, hidden_dims=hidden_dims,
                             activation=activation, lammbda=args.lambda_val)
        print(f"使用自定义模型:")
        print(f"  - 结构: [784, {', '.join(map(str, hidden_dims))}, 10]")
        print(f"  - 激活函数: {args.activation}")
        print(f"  - L2系数: {args.lambda_val}")
    
    # 创建优化器和损失函数
    optimizer = get_optimizer(args.optimizer, args.lr, args.batch_size)
    loss = CrossEntropyLoss()
    print(f"使用优化器: {args.optimizer}, 学习率: {args.lr}, 批大小: {args.batch_size}")
    
    # 编译模型
    model.compile(loss, optimizer)
    
    # 创建结果目录
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join('runs', f'run_{run_time}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"结果将保存到: {run_dir}")
    
    # 保存配置信息
    with open(os.path.join(run_dir, 'config.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # 训练模型
    print("\n开始训练...\n")
    history = model.fit(X_train, y_train, epochs=args.epochs, X_val=X_val, y_val=y_val, verbose=1)
    
    # 可视化训练历史
    history_path = os.path.join(run_dir, 'training_history.png')
    plot_training_history(model.history, save_path=history_path)
    print(f"\n训练历史图已保存到: {history_path}")
    
    # 在测试集上评估模型
    print("\n在测试集上评估模型...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    
    # 保存训练指标
    metrics_path = os.path.join(run_dir, 'training_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"最终训练准确率: {model.history['train_acc'][-1]:.4f}\n")
        f.write(f"最终验证准确率: {model.history['val_acc'][-1]:.4f}\n")
        f.write(f"最终训练损失: {model.history['train_loss'][-1]:.4f}\n")
        f.write(f"最终验证损失: {model.history['val_loss'][-1]:.4f}\n")
        f.write(f"测试集损失: {test_loss:.4f}\n")
        f.write(f"测试集准确率: {test_acc:.4f}\n")
    print(f"训练指标已保存到: {metrics_path}")
    
    # 预测并可视化结果
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # 保存混淆矩阵
    cm_path = os.path.join(run_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, class_names=list(range(10)), normalize=False, save_path=cm_path)
    
    cm_norm_path = os.path.join(run_dir, 'confusion_matrix_normalized.png')
    plot_confusion_matrix(y_true, y_pred, class_names=list(range(10)), normalize=True, save_path=cm_norm_path)
    print(f"混淆矩阵已保存到: {cm_path} 和 {cm_norm_path}")
    
    # 保存错误分类样本
    misclass_path = os.path.join(run_dir, 'misclassified_examples.png')
    plot_misclassified_examples(X_test, y_true, y_pred, class_names=list(range(10)), save_path=misclass_path)
    print(f"错误分类样本已保存到: {misclass_path}")
    
    print("\n模型训练和评估完成！")

if __name__ == "__main__":
    main()