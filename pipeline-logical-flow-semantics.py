from rich.console import Console
from rich.text import Text

# ==========================================================
# Color-coded pipeline tree
# ==========================================================
pipeline_text = Text()

# Function to add a line with optional color
def add_line(line, color=None):
    if color:
        pipeline_text.append(line + "\n", style=color)
    else:
        pipeline_text.append(line + "\n")

# Pipeline header
add_line("Pipeline: EfficientNet-B3 COCO Multi-label Classification", "bold magenta")
add_line("│")

# Step 1
add_line("├─ Step 1: Imports & Environment Setup", "cyan bold")
add_line("│   ├─ WHAT: Import packages (torch, torchvision, PIL, sklearn, pandas)", "yellow")
add_line("│   ├─ WHY: Ensure reproducibility and GPU usage", "green")
add_line("│   └─ HOW: Set random seeds, configure device (CPU/GPU)", "blue")

# Step 2
add_line("├─ Step 2: Hyperparameters & Globals", "cyan bold")
add_line("│   ├─ WHAT: Define batch size, learning rate, epochs, threshold, image size", "yellow")
add_line("│   ├─ WHY: Centralized config for consistency", "green")
add_line("│   └─ HOW: Set NUM_CLASSES, paths, and other global vars", "blue")

# Step 3
add_line("├─ Step 3: Data Transforms", "cyan bold")
add_line("│   ├─ WHAT: Train & validation transforms", "yellow")
add_line("│   ├─ WHY: Augmentation improves generalization", "green")
add_line("│   └─ HOW: Compose Resize, Flip, Rotation, ColorJitter, Normalize", "blue")

# Step 4
add_line("├─ Step 4: Dataset Preparation & DataLoaders", "cyan bold")
add_line("│   ├─ WHAT: Load images/labels, split train/val, build DataLoaders", "yellow")
add_line("│   ├─ WHY: Efficient loading and no transform leakage", "green")
add_line("│   └─ HOW: Use COCOTrainImageDataset, COCOTestImageDataset, Subsets, DataLoader", "blue")

# Step 5
add_line("├─ Step 5: Model Setup", "cyan bold")
add_line("│   ├─ WHAT: Load pretrained EfficientNet-B3 & modify classifier", "yellow")
add_line("│   ├─ WHY: Transfer learning accelerates convergence", "green")
add_line("│   └─ HOW: Optionally freeze backbone, replace classifier, move to device", "blue")

# Step 6
add_line("├─ Step 6: Loss Function", "cyan bold")
add_line("│   ├─ WHAT: BCEWithLogitsLoss for multi-label classification", "yellow")
add_line("│   ├─ WHY: Each class is an independent binary problem", "green")
add_line("│   └─ HOW: Compute pos_weight = num_negatives / num_positives per class", "blue")

# Step 7
add_line("├─ Step 7: Optimizer & Scheduler", "cyan bold")
add_line("│   ├─ WHAT: Configure AdamW optimizer & ReduceLROnPlateau scheduler", "yellow")
add_line("│   ├─ WHY: Stabilize convergence and improve generalization", "green")
add_line("│   └─ HOW: Setup optimizer, scheduler, AMP scaler", "blue")

# Step 8
add_line("├─ Step 8: Training + Validation", "cyan bold")
add_line("│   ├─ WHAT: Train model and validate per epoch", "yellow")
add_line("│   ├─ WHY: Track loss, accuracy, F1, and save best model", "green")
add_line("│   └─ HOW: Forward → Loss → Backward → Optimizer → AMP → Validation → Scheduler → Save Best", "blue")

# Step 9
add_line("├─ Step 9: Final Validation Metrics", "cyan bold")
add_line("│   ├─ WHAT: Compute final validation metrics including class-wise F1/Precision/Recall and mAP", "yellow")
add_line("│   ├─ WHY: Evaluate best model performance and class imbalance", "green")
add_line("│   └─ HOW: Load best checkpoint → validation_loop → compute per-class metrics → print", "blue")

# Step 10
add_line("├─ Step 10: Final Test Evaluation", "cyan bold")
add_line("│   ├─ WHAT: Generate predictions for test set and save submission JSON", "yellow")
add_line("│   ├─ WHY: Evaluate generalization and prepare leaderboard submission", "green")
add_line("│   └─ HOW: Forward pass → Sigmoid → Threshold → Build JSON → Save", "blue")

# Step 11
add_line("├─ Step 11: Save Best Model", "cyan bold")
add_line("│   └─ Save model.state_dict() to DRIVE_ROOT", "yellow")

# Step 12
add_line("├─ Step 12: Inference Function", "cyan bold")
add_line("│   └─ predict_image(model, img_path, transform, device, threshold)", "yellow")

# Step 13
add_line("├─ Step 13: Pipeline Completion", "cyan bold")
add_line("│   └─ Print pipeline completion message", "yellow")

# Step 14
add_line("└─ Step 14: JSON Experiment Report", "cyan bold")
add_line("    ├─ Store hyperparameters, model summary, dataset info", "yellow")
add_line("    ├─ Store metrics: val_loss, micro/macro F1, mAP, class-wise metrics", "yellow")
add_line("    ├─ Store training log, best_model_path, submission JSON path", "yellow")
add_line("    └─ Save experiment_report.json", "yellow")

# ==========================================================
# Print with rich console
# ==========================================================
console = Console()
console.print(pipeline_text)
