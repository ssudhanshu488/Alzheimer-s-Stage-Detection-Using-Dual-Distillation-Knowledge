import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .visualization_utils import plot_accuracy_graph

def train_teacher_model(model, train_loader, val_loader, test_loader, num_epochs, device, model_name, output_dir):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_acc = 0.0
    train_acc_history = []
    val_acc_history = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if isinstance(model, ViTForClassification):
                outputs, _ = model(images)
            else:  # SwinTransformer
                outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        train_acc_history.append(train_acc)
        
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                if isinstance(model, ViTForClassification):
                    outputs, _ = model(images)
                else:
                    outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_acc_history.append(val_acc)
        
        print(f'{model_name} Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'{output_dir}/best_{model_name}_model.pth')
    
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            if isinstance(model, ViTForClassification):
                outputs, _ = model(images)
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = 100 * test_correct / test_total
    print(f'{model_name} Test Accuracy: {test_acc:.2f}%')
    
    plot_accuracy_graph(train_acc_history, val_acc_history, model_name)
    return test_acc

def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    distill_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * distill_loss + (1 - alpha) * hard_loss

def train_student_with_distillation(student_model, vit_teacher, swin_teacher, train_loader, val_loader, test_loader, num_epochs, device, output_dir):
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)
    best_acc = 0.0
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        student_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                vit_logits, _ = vit_teacher(images)
                swin_logits = swin_teacher(images)
                teacher_logits = (vit_logits + swin_logits) / 2

            student_logits, _ = student_model(images)
            loss = distillation_loss(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(student_logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        train_acc_history.append(train_acc)

        student_model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                student_logits, _ = student_model(images)
                _, predicted = torch.max(student_logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_acc_history.append(val_acc)

        print(f'Student Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Acc: {val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student_model.state_dict(), f'{output_dir}/best_student_model.pth')

    student_model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            student_logits, _ = student_model(images)
            _, predicted = torch.max(student_logits.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = 100 * test_correct / test_total
    print(f'Student Test Accuracy: {test_acc:.2f}%')
    
    plot_accuracy_graph(train_acc_history, val_acc_history, "Student")
    return test_acc