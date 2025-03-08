import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

def plot_training_history(history, figsize=(14, 5), save_path=None):
    """
    可视化训练过程
    """
    plt.figure(figsize=figsize)
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    if len(history['val_loss']) > 0:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    if len(history['val_acc']) > 0:
        plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    
def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8), cmap='Blues', 
                         normalize=False, save_path=None):
    """
    绘制画图矩阵
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Create figure
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    # Print classification report
    print("Classification Accuracy: {:.2f}%".format(100 * np.sum(np.diag(cm)) / np.sum(cm)))
    if not normalize:
        for i in range(len(cm)):
            precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
            recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print(f"Class {i if class_names is None else class_names[i]}:")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1-score: {f1:.3f}")
            
            
def plot_misclassified_examples(X, y_true, y_pred, class_names=None, n_examples=25, 
                               figsize=(12, 10), save_path=None):
    """
    可视化绘图样例
    """
    # Find indices of misclassified samples
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    if len(misclassified_idx) == 0:
        print("No misclassified samples found!")
        return
    
    # Limit the number of examples to display
    n_examples = min(n_examples, len(misclassified_idx))
    
    # Randomly select samples (if there are too many)
    if len(misclassified_idx) > n_examples:
        selected_idx = np.random.choice(misclassified_idx, n_examples, replace=False)
    else:
        selected_idx = misclassified_idx
    
    # Calculate row and column numbers for the grid
    n_cols = 5
    n_rows = (n_examples + n_cols - 1) // n_cols
    
    plt.figure(figsize=figsize)
    for i, idx in enumerate(selected_idx):
        plt.subplot(n_rows, n_cols, i+1)
        
        # For MNIST dataset, reshape samples to 28x28 images
        if X[idx].shape[0] == 784:  # MNIST original features are 784-dimensional
            img = X[idx].reshape(28, 28)
        else:
            img = X[idx]
        
        plt.imshow(img, cmap='gray')
        
        # Add title showing true and predicted labels
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        title = f"True: {true_label if class_names is None else class_names[true_label]}\nPred: {pred_label if class_names is None else class_names[pred_label]}"
        plt.title(title, color='red')
        plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle("Misclassified Examples", fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show()