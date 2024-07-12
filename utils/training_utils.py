import torch
import os
import matplotlib.pyplot as plt

def save_model(model, path):
    """Save the trained model to the specified path."""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Load the model from the specified path."""
    model.load_state_dict(torch.load(path))
    return model

def plot_metrics(metrics, path):
    """Plot and save training metrics."""
    plt.figure()
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(path, 'loss.png'))

    plt.figure()
    plt.plot(metrics['accuracy'], label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(path, 'accuracy.png'))

def calculate_accuracy(outputs, labels):
    """Calculate accuracy of model predictions."""
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total

def save_logs(logs, path):
    """Save training logs to the specified path."""
    with open(os.path.join(path, 'logs.txt'), 'w') as f:
        for log in logs:
            f.write(log + '\n')
