# Install rich if not already installed
# !pip install rich

from rich.console import Console
from rich.syntax import Syntax

# ==========================================================
# Define pipeline tree as a multi-line string
# ==========================================================
pipeline_text = """
Pipeline: EfficientNet-B3 COCO Multi-label Classification
│
├─ Step 1: Imports & Environment Setup
│   ├─ Import packages (torch, torchvision, PIL, sklearn, pandas)
│   └─ Set random seeds, configure device (GPU/CPU)
│
├─ Step 2: Hyperparameters & Globals
│   ├─ Batch size, learning rate, epochs, threshold, image size
│   ├─ Number of classes (NUM_CLASSES = 80)
│   └─ Paths: dataset root, checkpoint, experiment reports
│
├─ Step 3: Data Transforms
│   ├─ train_transform: Resize, RandomFlip, Rotation, ColorJitter, Normalize
│   └─ val_transform: Resize, Normalize only
│
├─ Step 4: Dataset Preparation & DataLoaders
│   ├─ Unzip and flatten train/test images and annotations
│   ├─ Define COCOTrainImageDataset and COCOTestImageDataset
│   ├─ Safe train/validation split (80/20)
│   ├─ Create Subsets with independent transforms
│   └─ Create DataLoaders (train_loader, val_loader, test_loader)
│
├─ Step 5: Model Setup
│   ├─ Load pretrained EfficientNet-B3
│   ├─ Optional backbone freezing
│   ├─ Replace classifier head with NUM_CLASSES output
│   └─ Move model to device
│
├─ Step 6: Loss Function
│   ├─ Compute per-class frequencies on training set
│   ├─ Compute pos_weight = num_negatives / num_positives
│   └─ Define BCEWithLogitsLoss with pos_weight
│
├─ Step 7: Optimizer & Scheduler
│   ├─ Optimizer: AdamW
│   ├─ Scheduler: ReduceLROnPlateau on validation loss
│   └─ AMP setup: GradScaler
│
├─ Step 8: Training + Validation
│   ├─ Loop over epochs
│   │   ├─ Training Phase
│   │   │   ├─ Forward pass
│   │   │   ├─ Compute loss
│   │   │   ├─ Backward pass with AMP
│   │   │   └─ Update optimizer
│   │   ├─ Validation Phase
│   │   │   ├─ Forward pass
│   │   │   ├─ Compute loss and metrics (accuracy, micro/macro F1)
│   │   │   └─ Track class imbalance performance
│   │   └─ Scheduler step & best model saving
│   └─ Track training log per epoch
│
├─ Step 9: Final Validation Metrics
│   ├─ Load best model checkpoint
│   ├─ Compute validation metrics using validation_loop
│   ├─ Extract avg_val_loss, micro_f1, macro_f1, accuracy
│   ├─ Compute per-class mAP
│   └─ Print metrics & class-wise F1/Precision/Recall
│
├─ Step 10: Final Test Evaluation
│   ├─ Forward pass on test_loader
│   ├─ Compute predictions per image
│   ├─ Generate submission JSON
│   └─ Optionally compute test metrics if labels available
│
├─ Step 11: Save Best Model
│   └─ Save model.state_dict() to DRIVE_ROOT
│
├─ Step 12: Inference Function
│   └─ predict_image(model, img_path, transform, device, threshold)
│
├─ Step 13: Pipeline Completion
│   └─ Print pipeline completed message
│
└─ Step 14: JSON Experiment Report
    ├─ Store hyperparameters, model summary, dataset info
    ├─ Store metrics: val_loss, micro/macro F1, mAP, class-wise metrics
    ├─ Store training log
    ├─ Store best_model_path and submission JSON path
    └─ Save experiment_report.json
"""

# ==========================================================
# Display pipeline with rich colors
# ==========================================================
console = Console()
syntax = Syntax(pipeline_text, "python", theme="monokai", line_numbers=False)
console.print(syntax)
