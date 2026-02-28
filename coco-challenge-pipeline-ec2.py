# ==========================================================
# STEP 1/14: IMPORTS & ENVIRONMENT SETUP (Dependency-Safe)
# ==========================================================

print("Step 1/14: Imports & Environment Setup")
print("--------------------------------------")
print("WHAT:\n  Import required Python packages, define global dependencies, and configure device.")
print("WHY:\n  Ensure reproducibility, avoid undefined variables in later steps, and enable GPU usage.")
print("HOW:\n  Import torch, torchvision, PIL, pandas, sklearn, json, datetime,\n"
      "  define global constants, and initialize device + random seeds.\n")

# ----------------------------------------------------------
# Core Python & Data Libraries
# ----------------------------------------------------------
import os
import json
import random
import shutil
import zipfile
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from datetime import datetime
from PIL import Image

# ----------------------------------------------------------
# PyTorch & Vision
# ----------------------------------------------------------
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models

# Modern AMP (Torch 2.x compatible)
from torch.amp import autocast, GradScaler

# ----------------------------------------------------------
# Metrics
# ----------------------------------------------------------
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# ----------------------------------------------------------
# Global Constants (Used Across Steps 2–14)
# ----------------------------------------------------------
NUM_CLASSES = 80
THRESHOLD = 0.5
SEED = 42

# Model Saving Path (Used in Step 8/14 & 14/14)
best_model_path = "best_model.pth"

# Class Names Placeholder (Used in Step 9/14 reporting)
classes = [f"class_{i}" for i in range(NUM_CLASSES)]

# ----------------------------------------------------------
# Device Configuration
# ----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✔ Using device: {device}")

if torch.cuda.is_available():
    print(f"✔ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ Running on CPU — training will be slower.")

# ----------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print("✔ Random seeds initialized.")
print("✔ Global dependencies defined successfully.\n")

# ==========================================================
# STEP 2/14: HYPERPARAMETERS & GLOBAL VARIABLES
# ==========================================================

print("Step 2/14: Hyperparameters & Globals")
print("------------------------------------")
print("WHAT:\n  Define core training hyperparameters and model configuration.")
print("WHY:\n  Centralized configuration ensures consistency across all steps\n"
      "  and simplifies experimentation and tuning.")
print("HOW:\n  Set batch size, learning rate, epochs, image size,\n"
      "  classification threshold, and backbone training policy.\n")

# ----------------------------------------------------------
# Training Hyperparameters
# ----------------------------------------------------------
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
EPOCHS = 25

# ----------------------------------------------------------
# Model / Input Configuration
# ----------------------------------------------------------
IMG_SIZE = 300            # Must match EfficientNet-B3 input size
THRESHOLD = 0.5           # Multi-label sigmoid threshold
FREEZE_BACKBONE = False   # If True → only classifier head trains

# ----------------------------------------------------------
# Training Behavior Flags
# ----------------------------------------------------------
USE_AMP = torch.cuda.is_available()  # Enable mixed precision only if GPU
MULTI_LABEL = True                   # MS COCO is multi-label

print(f"✔ Batch size: {BATCH_SIZE}")
print(f"✔ Learning rate: {LEARNING_RATE}")
print(f"✔ Epochs: {EPOCHS}")
print(f"✔ Image size: {IMG_SIZE}")
print(f"✔ Mixed Precision Enabled: {USE_AMP}")
print("✔ Hyperparameters initialized successfully.\n")

# ==========================================================
# STEP 3/14: TRANSFORMS
# ==========================================================
print("Step 3/14: Data Augmentation & Preprocessing")
print("---------------------------------------------")
print("WHAT:\n  Define image transforms for training and validation/test sets.")
print("WHY:\n  Augmentation improves generalization; validation uses only normalization to reflect real data.")
print("HOW:\n  Use torchvision transforms with Resize, Normalize, RandomFlip, Rotation, ColorJitter.\n")

# ---------------------------
# Centralized normalization (reusable)
# ---------------------------
imagenet_norm = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# ---------------------------
# Training transforms
# ---------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    imagenet_norm
])

# ---------------------------
# Validation / Test transforms
# ---------------------------
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    imagenet_norm
])

print("✔ Transforms defined for train and validation/test.\n")

# ==========================================================
# STEP 4A/14: Dataset Extraction (EC2)
# ==========================================================
print("\nStep 4A/14: Dataset Extraction (EC2)")
print("--------------------------------------")
print("WHAT:\n  Ensure MS COCO dataset files are extracted and properly organized.")
print("WHY:\n  1) Flatten nested folders for deterministic training.\n"
      "  2) Prepare target directories compatible with Step 4B.\n"
      "  3) Avoid runtime errors due to missing images or labels.\n")
print("HOW:\n  1) Check target directories.\n"
      "  2) Unzip datasets only if needed.\n"
      "  3) Flatten any number of nested subdirectories.\n"
      "  4) Prepare train, validation, and test folders.\n")

import os
import zipfile
import shutil

# ---------------------------
# Dataset paths
# ---------------------------
ROOT_DIR = "/home/ubuntu/data"
TRAIN_IMG_ZIP   = os.path.join(ROOT_DIR, "train-resized.zip")
TEST_IMG_ZIP    = os.path.join(ROOT_DIR, "test-resized.zip")
TRAIN_LABEL_ZIP = os.path.join(ROOT_DIR, "train.zip")

TRAIN_IMG_DIR   = os.path.join(ROOT_DIR, "images/train")
TRAIN_LABEL_DIR = os.path.join(ROOT_DIR, "labels/train")
TEST_IMG_DIR    = os.path.join(ROOT_DIR, "images/test")

# ---------------------------
# Utility Functions
# ---------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def dir_ready(path):
    """Check if directory exists and contains files"""
    return os.path.exists(path) and len(os.listdir(path)) > 0

def unzip_and_flatten_ec2(zip_path, target_dir, allowed_exts=None):
    """
    Extract zip into target_dir and flatten all nested folders.
    - allowed_exts: tuple of allowed file extensions (e.g., ('.jpg', '.cls'))
    """
    if dir_ready(target_dir):
        print(f"✔ {target_dir} already prepared → skipping extraction.")
        return

    temp_dir = target_dir + "_tmp"
    ensure_dir(temp_dir)
    print(f"⚡ Extracting {zip_path} → {temp_dir}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    ensure_dir(target_dir)
    moved_files = 0

    # Walk through all nested directories
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if allowed_exts is None or file.lower().endswith(allowed_exts):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(target_dir, file)
                if not os.path.exists(dst_path):
                    shutil.move(src_path, dst_path)
                    moved_files += 1

    # Clean up temp extraction folder
    shutil.rmtree(temp_dir)
    print(f"✔ Consolidated {moved_files} files into {target_dir}\n")

# ---------------------------
# Ensure target directories exist
# ---------------------------
ensure_dir(TRAIN_IMG_DIR)
ensure_dir(TRAIN_LABEL_DIR)
ensure_dir(TEST_IMG_DIR)

# ---------------------------
# Extract & flatten datasets
# ---------------------------
unzip_and_flatten_ec2(TRAIN_IMG_ZIP, TRAIN_IMG_DIR, allowed_exts=(".jpg", ".jpeg", ".png"))
unzip_and_flatten_ec2(TEST_IMG_ZIP, TEST_IMG_DIR, allowed_exts=(".jpg", ".jpeg", ".png"))
unzip_and_flatten_ec2(TRAIN_LABEL_ZIP, TRAIN_LABEL_DIR, allowed_exts=(".cls", ".txt"))

print("✔ Dataset extraction and flattening complete for EC2.")
print(f"✔ Train images: {len(os.listdir(TRAIN_IMG_DIR))}")
print(f"✔ Test images : {len(os.listdir(TEST_IMG_DIR))}")
print(f"✔ Train labels: {len(os.listdir(TRAIN_LABEL_DIR))}\n")

# ==========================================================
# STEP 4B/14: Dataset Objects, Train/Validation Split & DataLoaders
# ==========================================================

print("Step 4B/14: Dataset Objects, Safe Split & DataLoaders")
print("-----------------------------------------------------")
print("WHAT:\n  Extract zip files, flatten nested folders, create PyTorch datasets and DataLoaders.")
print("WHY:\n  Deterministic extraction ensures reproducibility, train/validation split prevents data leakage,"
      "\n  and GPU-optimized DataLoaders maximize throughput.")
print("HOW:\n  1. Unzip and consolidate images/labels\n"
      "  2. Define dataset classes\n"
      "  3. Apply train/validation transforms\n"
      "  4. Perform 80/20 train/val split\n"
      "  5. Build DataLoaders with pin_memory and persistent_workers.\n")

import zipfile
from glob import glob
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# ---------------------------
# Helper: unzip and flatten
# ---------------------------
def unzip_and_flatten(zip_path, target_dir, extension):
    """Extract zip to target_dir and consolidate files with given extension."""
    if os.path.exists(target_dir) and len(glob(f"{target_dir}/*{extension}")) > 0:
        print(f"✔ {target_dir} already prepared → skipping extraction.")
        return

    temp_dir = target_dir + "_tmp"
    os.makedirs(temp_dir, exist_ok=True)

    print(f"⚡ Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    os.makedirs(target_dir, exist_ok=True)
    moved = 0
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.lower().endswith(extension):
                dst = os.path.join(target_dir, file)
                if not os.path.exists(dst):
                    shutil.move(os.path.join(root, file), dst)
                    moved += 1
    shutil.rmtree(temp_dir)
    print(f"✔ Consolidated {moved} {extension} files into {target_dir}\n")

# ---------------------------
# Extract all zips
# ---------------------------
unzip_and_flatten(TRAIN_IMG_ZIP, TRAIN_IMG_DIR, ".jpg")
unzip_and_flatten(TEST_IMG_ZIP, TEST_IMG_DIR, ".jpg")
unzip_and_flatten(TRAIN_LABEL_ZIP, TRAIN_LABEL_DIR, ".cls")

# ---------------------------
# Dataset Classes
# ---------------------------
class COCOTrainImageDataset(Dataset):
    """MS COCO Training Dataset for multi-label classification"""
    def __init__(self, img_dir, annotations_dir, transform=None):
        self.img_labels = sorted(glob(os.path.join(annotations_dir, "*.cls")))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label_path = self.img_labels[idx]
        img_name = Path(label_path).stem + ".jpg"
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        with open(label_path, 'r') as f:
            labels = [int(label.strip()) for label in f.readlines()]

        label_tensor = torch.zeros(NUM_CLASSES)
        label_tensor[labels] = 1

        if self.transform:
            image = self.transform(image)

        return image, label_tensor

class COCOTestImageDataset(Dataset):
    """MS COCO Test Dataset (no labels)"""
    def __init__(self, img_dir, transform=None):
        self.img_list = sorted(glob(os.path.join(img_dir, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, Path(img_path).stem

# ---------------------------
# Safe Train/Validation Split
# ---------------------------
print("Applying safe train/validation split with independent transforms...")

base_dataset = COCOTrainImageDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, transform=None)
total_size = len(base_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

generator = torch.Generator().manual_seed(SEED)
indices = torch.randperm(total_size, generator=generator)

train_indices = indices[:train_size]
val_indices   = indices[train_size:]

train_dataset_full = COCOTrainImageDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, transform=train_transform)
val_dataset_full   = COCOTrainImageDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, transform=val_transform)

train_dataset = Subset(train_dataset_full, train_indices)
val_dataset   = Subset(val_dataset_full, val_indices)

# ---------------------------
# GPU-Optimized DataLoaders
# ---------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

test_loader = DataLoader(
    COCOTestImageDataset(TEST_IMG_DIR, transform=val_transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

# ---------------------------
# Dataset Summary
# ---------------------------
dataset_summary = pd.DataFrame({
    "Component": [
        "Total Training Samples",
        "Training Subset (80%)",
        "Validation Subset (20%)",
        "Test Images",
        "Batch Size",
        "Image Resolution",
        "Train Augmentation",
        "Validation Augmentation",
        "Storage Mode"
    ],
    "Value": [
        total_size,
        train_size,
        val_size,
        len(test_loader.dataset),
        BATCH_SIZE,
        f"{IMG_SIZE}x{IMG_SIZE}",
        "Flip + Rotation + ColorJitter",
        "Resize + Normalize Only",
        "Local EC2 Disk (Deterministic Consolidation)"
    ]
})

print("\nDataset Summary:")
print(dataset_summary)

print(f"\n✔ DataLoaders created successfully.")
print(f"✔ Train batches: {len(train_loader)}")
print(f"✔ Validation batches: {len(val_loader)}")
print(f"✔ Test batches: {len(test_loader)}\n")

# ==========================================================
# STEP 5/14: MODEL SETUP
# ==========================================================

print("Step 5/14: Model Architecture Setup")
print("------------------------------------")
print("WHAT:\n  Load pretrained EfficientNet-B3 and adapt classifier head")
print("  for 80-class multi-label classification.\n")
print("WHY:\n  Transfer learning leverages ImageNet features,")
print("  accelerating convergence and improving generalization.\n")
print("HOW:\n  Load official pretrained weights → optionally freeze backbone →"
      " replace classifier layer → move model to GPU.\n")

# ----------------------------------------------------------
# Load EfficientNet-B3 with pretrained ImageNet weights
# ----------------------------------------------------------
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

weights = EfficientNet_B3_Weights.DEFAULT
model = efficientnet_b3(weights=weights)

# ----------------------------------------------------------
# Optional Backbone Freezing
# ----------------------------------------------------------
if FREEZE_BACKBONE:
    for param in model.features.parameters():
        param.requires_grad = False
    print("✔ Backbone frozen (training classifier only).")
else:
    print("✔ Full fine-tuning enabled (backbone + classifier).")

# ----------------------------------------------------------
# Replace Classification Head for NUM_CLASSES output
# ----------------------------------------------------------
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

# ----------------------------------------------------------
# Move model to device
# ----------------------------------------------------------
model = model.to(device)

# ----------------------------------------------------------
# Model Summary
# ----------------------------------------------------------
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\nModel Summary:")
print(f"  Architecture        : EfficientNet-B3")
print(f"  Input Resolution    : {IMG_SIZE}x{IMG_SIZE}")
print(f"  Output Classes      : {NUM_CLASSES}")
print(f"  Trainable Parameters: {trainable_params:,}")
print("✔ Model initialized successfully.\n")

# ==========================================================
# STEP 6/14: LOSS FUNCTION (CLASS IMBALANCE HANDLING)
# ==========================================================
print("Step 6/14: Loss Function Definition")
print("------------------------------------")
print("WHAT:\n  Define objective function for multi-label classification with class imbalance handling.\n")
print("WHY:\n  1) Each class is treated as an independent binary problem.\n"
      "  2) MS COCO exhibits significant label imbalance.\n"
      "  3) Per-class positive weighting stabilizes training and improves rare-class learning.\n")
print("HOW:\n  1) Compute per-class positive frequencies from training set.\n"
      "  2) Derive pos_weight = num_negatives / (num_positives + epsilon) → optionally apply smoothing.\n"
      "  3) Apply nn.BCEWithLogitsLoss with pos_weight on GPU.\n")

# -----------------------------
# Compute Class Frequencies
# -----------------------------
print("Computing class frequency statistics...")

num_positives = torch.zeros(NUM_CLASSES, device=device, dtype=torch.float32)
num_samples = 0

for _, labels in train_loader:
    labels = labels.float().to(device)
    num_positives += labels.sum(dim=0)
    num_samples += labels.size(0)

num_negatives = num_samples - num_positives

# Optional smoothing to avoid extreme weights
epsilon = 1e-6
smooth_factor = 0.05
pos_weight = (num_negatives + smooth_factor) / (num_positives + smooth_factor + epsilon)
pos_weight = torch.clamp(pos_weight, max=10.0).to(device)  # optional clipping

# -----------------------------
# Define Loss
# -----------------------------
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print("Loss Configuration Summary:")
print(f"  Loss Function : BCEWithLogitsLoss")
print(f"  Multi-label   : Yes")
print(f"  Class Weight  : Inverse frequency weighting with smoothing ({smooth_factor})")
print(f"  Threshold     : {THRESHOLD}")
print("✔ Loss function initialized successfully.\n")

# ==========================================================
# STEP 7/14: OPTIMIZER & LR STRATEGY
# ==========================================================
print("\nStep 7/14: Optimizer & Learning Rate Strategy")
print("----------------------------------------------")
print("WHAT:\n  Configure optimizer and adaptive learning rate scheduler.")
print("WHY:\n  AdamW improves generalization via decoupled weight decay.\n"
      "  Scheduler reduces learning rate when validation loss plateaus, stabilizing convergence.")
print("HOW:\n  Use AdamW optimizer with ReduceLROnPlateau scheduler.\n")

# ---------------- Optimizer ----------------
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),  # respects FREEZE_BACKBONE
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

# ---------------- Scheduler ----------------
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2,
    verbose=True
)

# ---------------- AMP Setup ----------------
scaler = GradScaler(enabled=USE_AMP)  # Automatic Mixed Precision if GPU available

# ---------------- Tracking ----------------
training_log = []          # per-epoch metrics
best_val_loss = float("inf")  # track best validation loss

print("Optimizer & Scheduler Summary:")
print(f"  Optimizer       : AdamW")
print(f"  Learning Rate   : {LEARNING_RATE}")
print(f"  Weight Decay    : 1e-4")
print(f"  Scheduler       : ReduceLROnPlateau")
print(f"  LR Reduction    : factor=0.5 after 2 stagnant epochs")
print(f"  AMP Enabled     : {USE_AMP}")
print("✔ Optimizer, scheduler, and AMP initialized.\n")

# ==========================================================
# STEP 8/14: TRAINING + VALIDATION (BEST MODEL SAVING)
# ==========================================================
print("\nStep 8/14: Training + Validation (Best Model Saving)")
print("------------------------------------------------------")
print("WHAT:\n  Train and validate the model while saving only the best model.")
print("WHY:\n  Preserve the model with the lowest validation loss while monitoring F1 performance.")
print("HOW:\n  1) Define unified validation loop.\n"
      "  2) Train model using AMP.\n"
      "  3) Validate using micro/macro F1.\n"
      "  4) Update scheduler.\n"
      "  5) Save best model.\n")

# ==========================================================
# Unified Validation Loop
# ==========================================================
def validation_loop(
    loader,
    model,
    criterion,
    num_classes,
    device,
    multi_label=True,
    th_multi_label=0.5,
    class_metrics=False
):
    model.eval()
    val_loss = 0.0

    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > th_multi_label).float()

            all_targets.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    avg_val_loss = val_loss / len(loader)

    all_targets = torch.cat(all_targets)
    all_preds = torch.cat(all_preds)
    all_probs = torch.cat(all_probs)

    correct = (all_preds == all_targets).sum().item()
    total = all_targets.numel()
    accuracy = correct / total

    y_true = all_targets.numpy()
    y_pred = all_preds.numpy()

    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    val_results = {
        "loss": avg_val_loss,
        "accuracy": accuracy,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1
    }

    val_class_results = []
    if class_metrics:
        for c in range(num_classes):
            cls_true = y_true[:, c]
            cls_pred = y_pred[:, c]

            val_class_results.append({
                "precision": precision_score(cls_true, cls_pred, zero_division=0),
                "recall": recall_score(cls_true, cls_pred, zero_division=0),
                "f1": f1_score(cls_true, cls_pred, zero_division=0)
            })

    return val_results, val_class_results


# -----------------------------
# Training Loop
# -----------------------------
best_model_path = "best_model.pth"
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = datetime.now()

    print(f"\n================ Epoch {epoch}/{EPOCHS} ================")
    print("Step 8/14: Training Phase")
    print("---------------------------")
    print("WHAT: Update model weights.")
    print("WHY: Minimize BCE multi-label loss.")
    print("HOW: Forward → loss → backward → optimizer → AMP.\n")

    model.train()
    running_train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    print(f"✔ Avg Training Loss: {avg_train_loss:.4f}")

    # ---------------- Validation ----------------
    print("\nStep 8/14: Validation Phase")
    print("-----------------------------")
    print("WHAT: Evaluate model on validation set.")
    print("WHY: Monitor overfitting and guide scheduler.")
    print("HOW: Forward pass → compute loss → compute micro/macro F1.\n")

    val_results, _ = validation_loop(
        val_loader,
        model,
        criterion,
        NUM_CLASSES,
        device,
        multi_label=MULTI_LABEL,
        th_multi_label=THRESHOLD,
        class_metrics=False
    )

    avg_val_loss = val_results["loss"]
    accuracy = val_results["accuracy"]
    micro_f1 = val_results["micro_f1"]
    macro_f1 = val_results["macro_f1"]

    print(f"✔ Validation Loss: {avg_val_loss:.4f}")
    print(f"✔ Validation Accuracy: {accuracy:.4f}")
    print(f"✔ Micro F1: {micro_f1:.4f}")
    print(f"✔ Macro F1: {macro_f1:.4f}")

    scheduler.step(avg_val_loss)
    print("✔ Scheduler updated based on validation loss.")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print("✔ New best model saved!")

    training_log.append({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_accuracy": accuracy,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1
    })

print(f"\n✔ Training completed. Best model stored as '{best_model_path}'.")


# ==========================================================
# STEP 9/14: FINAL VALIDATION METRICS
# ==========================================================
print("\nStep 9/14: Final Validation Metrics (Class-wise + mAP)")
print("--------------------------------------------------------")
print("WHAT:\n  Evaluate best model with detailed class-wise metrics.")
print("WHY:\n  Detect class imbalance and measure true generalization.")
print("HOW:\n  Load best model → run validation_loop(class_metrics=True) → compute mAP.\n")

model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

val_results, val_class_results = validation_loop(
    val_loader,
    model,
    criterion,
    NUM_CLASSES,
    device,
    multi_label=MULTI_LABEL,
    th_multi_label=THRESHOLD,
    class_metrics=True
)

avg_val_loss = val_results["loss"]
accuracy = val_results["accuracy"]
micro_f1 = val_results["micro_f1"]
macro_f1 = val_results["macro_f1"]

# Compute mAP
all_labels, all_probs = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)

        all_labels.append(labels)
        all_probs.append(probs.cpu())

all_labels = torch.cat(all_labels).numpy()
all_probs = torch.cat(all_probs).numpy()

mAP_val = average_precision_score(all_labels, all_probs, average="macro")

print(f"\n✔ Validation Loss  : {avg_val_loss:.4f}")
print(f"✔ Validation Acc   : {accuracy:.4f}")
print(f"✔ Micro F1         : {micro_f1:.4f}")
print(f"✔ Macro F1         : {macro_f1:.4f}")
print(f"✔ mAP              : {mAP_val:.4f}\n")

print("✔ Class-wise Validation Metrics:")
for i, cls_metrics in enumerate(val_class_results):
    print(f"{classes[i]:20s} | "
          f"F1: {cls_metrics['f1']:.4f} | "
          f"Precision: {cls_metrics['precision']:.4f} | "
          f"Recall: {cls_metrics['recall']:.4f}")
      
# ==========================================================
# STEP 10/14: Final Test Evaluation + Submission JSON
# ==========================================================
print("\nStep 10/14: Test Evaluation")
print("---------------------------")
print("WHAT:\n  Generate predictions for the test dataset and prepare submission JSON.")
print("WHY:\n  1) Final assessment on unseen data.\n"
      "  2) Create leaderboard-ready submission.\n"
      "  3) Optionally compute metrics if labels available.")
print("HOW:\n  1) Iterate over test_loader.\n"
      "  2) Apply sigmoid + threshold → predicted classes.\n"
      "  3) Store results in submission dictionary.\n"
      "  4) Save as JSON to ROOT_DIR.\n")

all_probs_test, submission_dict = [], {}
with torch.no_grad():
    for images, img_ids in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        all_probs_test.append(probs.cpu())

        # Build submission dict
        for idx, img_id in enumerate(img_ids):
            pred_classes = [i for i, v in enumerate(probs[idx].cpu().numpy()) if v > THRESHOLD]
            submission_dict[img_id] = pred_classes

# Save submission JSON to ROOT_DIR
submission_path = os.path.join(ROOT_DIR, "coco_test_submission.json")
with open(submission_path, 'w') as f:
    json.dump(submission_dict, f, indent=4)

print(f"✔ Test submission JSON saved to {submission_path}")

# ==========================================================
# STEP 11/14: POST-TRAINING & EXPERIMENT REPORT
# ==========================================================
print("\nStep 11/14: Post-Training Tasks & Experiment Report")
print("----------------------------------------------------")
print("WHAT:\n  1) Save the best model.\n"
      "  2) Prepare an inference function for single images.\n"
      "  3) Mark pipeline completion.\n"
      "  4) Generate a comprehensive JSON experiment report.\n")
print("WHY:\n  1) Preserve trained model for future inference.\n"
      "  2) Enable reproducible predictions.\n"
      "  3) Track all training, validation, and dataset info for traceability.\n")
print("HOW:\n  1) Save model state_dict.\n"
      "  2) Define predict_image() function.\n"
      "  3) Print pipeline completion messages.\n"
      "  4) Aggregate hyperparameters, metrics, dataset info, augmentation, class imbalance, and submission paths into JSON.\n")

# ----------------------------- 
# Save Best Model 
# -----------------------------
MODEL_PATH = os.path.join(ROOT_DIR, "efficientnet_b3_coco_best.pth")
torch.save(model.state_dict(), MODEL_PATH)
print(f"✔ Best model saved at {MODEL_PATH}")

# -----------------------------
# Inference Function
# -----------------------------
def predict_image(model, img_path, transform, device, threshold=THRESHOLD):
    """
    Predicts multi-label classes for a single image.
    Returns a list of class indices above the threshold.
    """
    model.eval()
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(image)).cpu().numpy().flatten()
    return [i for i, v in enumerate(probs) if v > threshold]

print("✔ Inference function ready.")

# -----------------------------
# Pipeline Completion Notice
# -----------------------------
print("\n✔ Pipeline Completed")
print("✔ Model trained, validated, test JSON ready, inference ready.")

# -----------------------------
# JSON Experiment Report
# -----------------------------
pos_weight_list = pos_weight.cpu().tolist() if 'pos_weight' in globals() else None

experiment_report = {
    "experiment_name": "EfficientNet-B3 COCO Multi-label Classification",
    "experiment_start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Ideally captured at pipeline start
    "experiment_end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "hyperparameters": {
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "num_classes": NUM_CLASSES,
        "threshold": THRESHOLD,
        "image_size": IMG_SIZE
    },
    "model": {
        "architecture": "EfficientNet-B3",
        "input_resolution": f"{IMG_SIZE}x{IMG_SIZE}",
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "freezed_backbone": FREEZE_BACKBONE
    },
    "metrics": {
        "val_loss": avg_val_loss,
        "micro_f1_val": micro_f1,
        "macro_f1_val": macro_f1,
        "accuracy_val": accuracy,
        "mAP_val": mAP_val,
        "classwise_val": val_class_results
    },
    "training_log": training_log,
    "dataset_info": {
        "train_images": len(train_loader.dataset),
        "val_images": len(val_loader.dataset),
        "test_images": len(test_loader.dataset),
        "train_augmentation": "Flip + Rotation + ColorJitter",
        "validation_augmentation": "Resize + Normalize Only"
    },
    "class_imbalance": {
        "pos_weight": pos_weight_list,
        "description": "Inverse frequency weighting used in BCEWithLogitsLoss per class."
    },
    "best_model_path": MODEL_PATH,
    "submission_json_path": submission_path
}

json_output_path = os.path.join(ROOT_DIR, "experiment_report.json")
with open(json_output_path, 'w') as json_file:
    json.dump(experiment_report, json_file, indent=4)

print(f"✔ JSON experiment report successfully saved to {json_output_path}")
print("✔ Report includes hyperparameters, model info, training log, validation metrics (overall + class-wise + mAP), dataset info, augmentation strategies, class imbalance weights, and test submission path.\n")

