import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import torch
import torch.nn.functional as F
from tqdm import tqdm

def evaluate_validation_metrics(model, val_loader, device, model_name, num_classes=3):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Evaluating {model_name}"):
            images, labels = images.to(device), labels.to(device)
            if isinstance(model, ViTForClassification):
                outputs, _ = model(images)
            else:  # SwinTransformer
                outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    val_acc = 100 * (all_preds == all_labels).sum() / len(all_labels)
    print(f'{model_name} Validation Accuracy: {val_acc:.2f}%')
    
    cm = confusion_matrix(all_labels, all_preds)
    print(f'{model_name} Confusion Matrix:\n{cm}')
    
    sensitivity = []
    specificity = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity.append(sens)
        specificity.append(spec)
        print(f'{model_name} Class {i} - Sensitivity: {sens:.4f}, Specificity: {spec:.4f}')
    
    macro_sensitivity = np.mean(sensitivity)
    macro_specificity = np.mean(specificity)
    print(f'{model_name} Macro Sensitivity: {macro_sensitivity:.4f}, Macro Specificity: {macro_specificity:.4f}')
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'{model_name} Macro F1 Score: {f1:.4f}')
    
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure()
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{model_name} ROC curve (class {i}, AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    return val_acc, macro_sensitivity, macro_specificity, f1, roc_auc

def evaluate_accuracy(model, data_loader, device, dataset_type):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            if isinstance(model, ViTForClassification):
                outputs, _ = model(images)
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'{model.__class__.__name__} {dataset_type} Accuracy: {accuracy:.2f}%')
    return accuracy

def plot_accuracy_graph(train_acc_history, val_acc_history, model_name="Model"):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_acc_history) + 1)
    plt.plot(epochs, train_acc_history, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc_history, 'r-', label='Validation Accuracy')
    plt.title(f'{model_name} Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy_comparison(models, train_accuracies, val_accuracies):
    x = np.arange(len(models))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, train_accuracies, width, label='Training Accuracy', color='blue')
    plt.bar(x + width/2, val_accuracies, width, label='Validation Accuracy', color='orange')
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.title('Training vs Validation Accuracy Comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.ylim(0, 100)
    plt.grid(True)
    plt.show()