#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import torch
from models.vit import ViTForClassification
from models.swin_transformer import SwinTransformer
from models.configurations import vit_teacher_config, swin_teacher_config, student_config
from utils.data_utils import create_data_loaders
from utils.training_utils import train_teacher_model, train_student_with_distillation, evaluate_validation_metrics, evaluate_accuracy, plot_accuracy_comparison

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu_id != '-1' else "cpu")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        args.ax_folder, args.cr_folder, args.batch_size, args.input_size
    )
    
    if args.mode == 'train':
        vit_teacher = ViTForClassification(vit_teacher_config).to(device)
        swin_teacher = SwinTransformer(**swin_teacher_config).to(device)
        student_model = ViTForClassification(student_config).to(device)
        
        if args.model_type in ['vit', 'both']:
            print("Training ViT Teacher...")
            train_teacher_model(vit_teacher, train_loader, val_loader, test_loader, 
                             args.num_epochs, device, 'vit', args.output_dir)
        
        if args.model_type in ['swin', 'both']:
            print("Training Swin Teacher...")
            train_teacher_model(swin_teacher, train_loader, val_loader, test_loader, 
                             args.num_epochs, device, 'swin', args.output_dir)
        
        if args.model_type == 'student':
            print("Training Student with Distillation...")
            vit_teacher.load_state_dict(torch.load(f'{args.output_dir}/best_vit_model.pth'))
            vit_teacher.eval()
            swin_teacher.load_state_dict(torch.load(f'{args.output_dir}/best_swin_model.pth'))
            swin_teacher.eval()
            train_student_with_distillation(student_model, vit_teacher, swin_teacher,
                                         train_loader, val_loader, test_loader,
                                         args.num_epochs, device, args.output_dir)
    
    elif args.mode == 'evaluate':
        vit_teacher = ViTForClassification(vit_teacher_config).to(device)
        swin_teacher = SwinTransformer(**swin_teacher_config).to(device)
        student_model = ViTForClassification(student_config).to(device)
        
        vit_teacher.load_state_dict(torch.load(f'{args.output_dir}/best_vit_model.pth'))
        swin_teacher.load_state_dict(torch.load(f'{args.output_dir}/best_swin_model.pth'))
        student_model.load_state_dict(torch.load(f'{args.output_dir}/best_student_model.pth'))
        
        vit_metrics = evaluate_validation_metrics(vit_teacher, val_loader, device, "ViT Teacher")
        swin_metrics = evaluate_validation_metrics(swin_teacher, val_loader, device, "Swin Teacher")
        student_metrics = evaluate_validation_metrics(student_model, val_loader, device, "Student")
        
        vit_train_acc = evaluate_accuracy(vit_teacher, train_loader, device, "Training")
        vit_val_acc = evaluate_accuracy(vit_teacher, val_loader, device, "Validation")
        swin_train_acc = evaluate_accuracy(swin_teacher, train_loader, device, "Training")
        swin_val_acc = evaluate_accuracy(swin_teacher, val_loader, device, "Validation")
        student_train_acc = evaluate_accuracy(student_model, train_loader, device, "Training")
        student_val_acc = evaluate_accuracy(student_model, val_loader, device, "Validation")
        
        plot_accuracy_comparison(
            ['ViT Teacher', 'Swin Teacher', 'Student'],
            [vit_train_acc, swin_train_acc, student_train_acc],
            [vit_val_acc, swin_val_acc, student_val_acc]
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Alzheimer Disease Classification using Vision Transformers')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'], help='Mode: train or evaluate')
    parser.add_argument('--model_type', type=str, default='both', choices=['vit', 'swin', 'student', 'both'], help='Model to train: vit, swin, student, or both')
    parser.add_argument('--gpu_id', type=str, default='0', help='Device id to run (-1 for CPU)')
    parser.add_argument('--ax_folder', type=str, default='/fab3/btech/2022/sudhanshu.singh/AlziehmerOnDiffFolders/Slices_Separate_Folders_T1_weighted/ax_AD_CN_MCI', 
                        help='Path to axial view dataset')
    parser.add_argument('--cr_folder', type=str, default='/fab3/btech/2022/sudhanshu.singh/AlziehmerOnDiffFolders/Slices_Separate_Folders_T1_weighted/cr_AD_CN_MCI',
                        help='Path to coronal view dataset')
    parser.add_argument('--output_dir', type=str, default='./snapshots', help='Output directory for model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.gpu_id != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    main(args)