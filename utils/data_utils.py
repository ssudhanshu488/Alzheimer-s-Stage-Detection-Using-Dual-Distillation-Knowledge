import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
import os
import numpy as np
import torchvision.transforms as transforms

def load_and_split_folder(folder_path, transform, train_ratio=0.7, val_ratio=0.15):
    images = []
    labels = []
    label_map = {'AD': 0, 'CN': 1, 'MCI': 2}
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            label = filename.split('_')[0]
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            images.append(img.numpy())
            labels.append(label_map[label])
    
    images = np.array(images)
    labels = np.array(labels)
    
    indices = np.random.permutation(len(images))
    n_train = int(train_ratio * len(images))
    n_val = int(val_ratio * len(images))
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    return (images[train_idx], labels[train_idx],
            images[val_idx], labels[val_idx],
            images[test_idx], labels[test_idx])

def create_data_loaders(ax_folder, cr_folder, batch_size, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    ax_train_x, ax_train_y, ax_val_x, ax_val_y, ax_test_x, ax_test_y = load_and_split_folder(ax_folder, transform)
    cr_train_x, cr_train_y, cr_val_x, cr_val_y, cr_test_x, cr_test_y = load_and_split_folder(cr_folder, transform)
    
    x_train_concatenated = np.concatenate((ax_train_x, cr_train_x), axis=0)
    y_train_concatenated = np.concatenate((ax_train_y, cr_train_y), axis=0)
    x_val_concatenated = np.concatenate((ax_val_x, cr_val_x), axis=0)
    y_val_concatenated = np.concatenate((ax_val_y, cr_val_y), axis=0)
    x_test_concatenated = np.concatenate((ax_test_x, cr_test_x), axis=0)
    y_test_concatenated = np.concatenate((ax_test_y, cr_test_y), axis=0)
    
    x_train = torch.tensor(x_train_concatenated, dtype=torch.float32)
    y_train = torch.tensor(y_train_concatenated, dtype=torch.long)
    x_val = torch.tensor(x_val_concatenated, dtype=torch.float32)
    y_val = torch.tensor(y_val_concatenated, dtype=torch.long)
    x_test = torch.tensor(x_test_concatenated, dtype=torch.float32)
    y_test = torch.tensor(y_test_concatenated, dtype=torch.long)
    
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader