# Attention-Based Dual Knowledge Distillation for Alzheimer’s Disease Stage Detection

This repository contains the implementation of the research paper titled **“Attention-Based Dual Knowledge Distillation for Alzheimer’s Disease Stage Detection Using MRI Scans”**, accepted for publication in the *IEEE Open Journal of Instrumentation and Measurement*. The project focuses on classifying Alzheimer’s disease stages (AD, CN, MCI) using MRI scans through a dual knowledge distillation approach, leveraging Vision Transformer (ViT) and Swin Transformer as teacher models to train a lightweight student model.

## Project Overview

This project implements a deep learning framework for Alzheimer’s disease stage detection using MRI scans. It employs:
- **Vision Transformer (ViT)** and **Swin Transformer** as teacher models.
- A **lightweight ViT-based student model** trained using knowledge distillation.
- A dataset of MRI scans in axial (ax) and coronal (cr) views, categorized into Alzheimer’s Disease (AD), Cognitively Normal (CN), and Mild Cognitive Impairment (MCI).

The framework processes MRI scans, trains teacher models independently, and then uses their knowledge to train a compact student model via distillation. The codebase includes data loading, model training, evaluation metrics (accuracy, sensitivity, specificity, F1 score, ROC curves), and visualization utilities.

## Project Structure

```
alzheimer_classification/
├── models/
│   ├── __init__.py
│   ├── activations.py       # Custom GELU activation function
│   ├── vit.py               # Vision Transformer model components
│   ├── swin_transformer.py  # Swin Transformer model components
│   └── configurations.py    # Model configurations
├── utils/
│   ├── __init__.py
│   ├── data_utils.py        # Data loading and preprocessing
│   ├── training_utils.py    # Training and distillation functions
│   └── visualization_utils.py # Evaluation and visualization functions
├── main.py                  # Main script with command-line arguments
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/alzheimer_classification.git
   cd alzheimer_classification
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Ensure you have Python 3.8+ installed, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes:
   ```
   torch>=1.8.0
   torchvision>=0.9.0
   numpy>=1.19.0
   Pillow>=8.0.0
   tqdm>=4.0.0
   matplotlib>=3.3.0
   scikit-learn>=0.24.0
   ```

4. **Set Up Data**:
   - The project expects MRI scan data in two folders: one for axial (ax) and one for coronal (cr) views.
   - Default paths in `main.py`:
     - Axial: `/fab3/btech/2022/sudhanshu.singh/AlziehmerOnDiffFolders/Slices_Separate_Folders_T1_weighted/ax_AD_CN_MCI`
     - Coronal: `/fab3/btech/2022/sudhanshu.singh/AlziehmerOnDiffFolders/Slices_Separate_Folders_T1_weighted/cr_AD_CN_MCI`
   - Update these paths in the command-line arguments or `main.py` to match your dataset location.
   - The dataset should contain images labeled with prefixes `AD_`, `CN_`, or `MCI_` (e.g., `AD_image1.png`).

## Usage

The project supports two modes: **training** and **evaluation**. You can specify the model to train/evaluate (ViT teacher, Swin teacher, student, or both teachers) and other parameters via command-line arguments.

### Training
To train the models (ViT teacher, Swin teacher, or student with distillation):
```bash
python main.py --mode train --model_type both --ax_folder /path/to/ax_folder --cr_folder /path/to/cr_folder --output_dir ./snapshots --batch_size 32 --num_epochs 200 --input_size 224 --lr 0.0001 --gpu_id 0
```

- `--mode`: `train` to train models.
- `--model_type`: Choose `vit`, `swin`, `student`, or `both` (for both teachers).
- `--ax_folder` and `--cr_folder`: Paths to the axial and coronal MRI datasets.
- `--output_dir`: Directory to save model checkpoints (default: `./snapshots`).
- `--batch_size`: Batch size for training (default: 32).
- `--num_epochs`: Number of training epochs (default: 200).
- `--input_size`: Input image size (default: 224).
- `--lr`: Learning rate (default: 0.0001).
- `--gpu_id`: GPU ID to use (`-1` for CPU, default: `0`).

The trained models are saved as `best_vit_model.pth`, `best_swin_model.pth`, and `best_student_model.pth` in the specified `output_dir`.

### Evaluation
To evaluate the trained models and generate metrics (accuracy, confusion matrix, sensitivity, specificity, F1 score, ROC curves):
```bash
python main.py --mode evaluate --output_dir ./snapshots --ax_folder /path/to/ax_folder --cr_folder /path/to/cr_folder --batch_size 32 --input_size 224 --gpu_id 0
```

- Ensure the model checkpoints (`best_vit_model.pth`, `best_swin_model.pth`, `best_student_model.pth`) exist in the `output_dir`.
- The script outputs validation accuracy, confusion matrix, sensitivity, specificity, F1 score, and ROC curves for each model, along with a bar plot comparing training and validation accuracies.

## Research Paper

This project is based on the research paper:

**Title**: Attention-Based Dual Knowledge Distillation for Alzheimer’s Disease Stage Detection Using MRI Scans  
**Authors**: Sudhanshu Singh, Chandita Barman, Shovan Barma
**Journal**: IEEE Open Journal of Instrumentation and Measurement  
**Status**: Accepted for publication  

The paper proposes a novel approach for Alzheimer’s disease stage detection using a dual knowledge distillation framework, where a lightweight ViT-based student model learns from both ViT and Swin Transformer teacher models to achieve high accuracy with reduced computational complexity.

For citation details, please refer to the published article in the *IEEE Open Journal of Instrumentation and Measurement* (link to be updated upon publication).

## Requirements

- Python 3.8+
- PyTorch 1.8.0+
- torchvision 0.9.0+
- NumPy 1.19.0+
- Pillow 8.0.0+
- tqdm 4.0.0+
- Matplotlib 3.3.0+
- scikit-learn 0.24.0+

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes. Ensure that any changes align with the project's coding style and include appropriate tests.

## Acknowledgment 
We would like to thank to various open source codes present in Pytorch for Vision based Transformers.
