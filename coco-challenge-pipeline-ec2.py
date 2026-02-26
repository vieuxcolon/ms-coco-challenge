# ==========================================================
# STEP 1/14: IMPORTS & ENVIRONMENT SETUP
# ==========================================================

print("Step 1/14: Imports & Environment Setup")
print("--------------------------------------")
print("WHAT:\n  Import required Python packages and configure device.")
print("WHY:\n  Ensure reproducibility and GPU usage for training.")
print("HOW:\n  Use torch, torchvision, PIL, pandas, sklearn, and set torch device.\n")

import os
import random
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from PIL import Image
import zipfile
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, average_precision_score

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✔ Using device: {device}\n")

# ==========================================================
# STEP 2/14: HYPERPARAMETERS & GLOBAL VARIABLES
# ==========================================================
print("Step 2/14: Hyperparameters & Globals")
print("------------------------------------")
print("WHAT:\n  Define training hyperparameters, thresholds, paths, and number of classes.")
print("WHY:\n  Centralized configuration simplifies adjustments and ensures consistency.")
print("HOW:\n  Set batch size, learning rate, epochs, number of classes, threshold, image size.\n")

BATCH_SIZE = 16
LEARNING_RATE = 2e-4
EPOCHS = 25
NUM_CLASSES = 80  # MS COCO classes
THRESHOLD = 0.5
IMG_SIZE = 300
FREEZE_BACKBONE = False
DRIVE_ROOT = "/content/drive/MyDrive/ML_Experiments"  # EC2 or Google Drive path


# ==========================================================
# STEP 3/14: TRANSFORMS
# ==========================================================
print("Step 3/14: Data Augmentation & Preprocessing")
print("---------------------------------------------")
print("WHAT:\n  Define image transforms for training and validation/test sets.")
print("WHY:\n  Augmentation improves generalization; validation uses only normalization to reflect real data.")
print("HOW:\n  Use torchvision transforms with Resize, Normalize, RandomFlip, Rotation, ColorJitter.\n")

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
print("✔ Transforms defined for train and validation/test.\n")

# ==========================================================
# STEP 4/14: Dataset Preparation & DataLoaders
# (EC2 + Google Drive + Deterministic Consolidation + Augmentation)
# ==========================================================

print("\nStep 4/14: Dataset & DataLoaders Setup (Production EC2 Version)")
print("-----------------------------------------------------------------")
print("WHAT:\n  Mount Google Drive, copy MS COCO zip files locally,"
      "\n  safely unzip with deterministic subdirectory handling,"
      "\n  split into train/validation subsets without leakage,"
      "\n  apply train/val augmentations, and prepare GPU-optimized DataLoaders.\n")

print("WHY:\n  1. Training directly from Google Drive is network-bound and slow.\n"
      "  2. Copying to local EC2 NVMe/EBS maximizes GPU utilization.\n"
      "  3. Zip files may contain arbitrary nested sub-subdirectories.\n"
      "  4. Deterministic consolidation prevents dataset corruption.\n"
      "  5. Train and validation must use independent dataset instances.\n"
      "  6. Preserve full compatibility with Steps 5–14 pipeline.\n")

print("HOW:\n  1. Install & verify rclone.\n"
      "  2. Mount Google Drive automatically.\n"
      "  3. Copy zip files Drive → Local EC2 disk.\n"
      "  4. Recursively extract and consolidate files by extension.\n"
      "  5. Create independent dataset objects for train/val with transforms.\n"
      "  6. Build GPU-optimized DataLoaders.\n")

# ==========================================================
# 1️⃣ GOOGLE DRIVE AUTO-MOUNT (EC2)
# ==========================================================
import os, shutil, subprocess, zipfile
from glob import glob
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torch
import pandas as pd

def command_exists(cmd):
    return shutil.which(cmd) is not None

def run_command(cmd):
    return subprocess.run(cmd, shell=True).returncode == 0

print("Checking rclone installation...")

if not command_exists("rclone"):
    print("⚡ Installing rclone...")
    run_command("curl https://rclone.org/install.sh | sudo bash")
else:
    print("✔ rclone already installed.")

RCLONE_REMOTE = "gdrive"
GDRIVE_MOUNT = "/mnt/gdrive"
os.makedirs(GDRIVE_MOUNT, exist_ok=True)

print("Attempting Google Drive mount...")
mount_cmd = f"rclone mount {RCLONE_REMOTE}: {GDRIVE_MOUNT} --daemon --vfs-cache-mode writes"
if run_command(mount_cmd):
    print("✔ Google Drive mounted successfully.")
else:
    print("⚠ Mount skipped or already active. Ensure 'rclone config' was completed once.")

# ==========================================================
# 2️⃣ DATASET PATHS (LOCAL EC2 DISK)
# ==========================================================
ROOT_DIR = "/home/ubuntu/data"
DRIVE_DATA_DIR = os.path.join(GDRIVE_MOUNT, "ms-coco")

os.makedirs(ROOT_DIR, exist_ok=True)

TRAIN_IMG_ZIP   = os.path.join(ROOT_DIR, "train-resized.zip")
TEST_IMG_ZIP    = os.path.join(ROOT_DIR, "test-resized.zip")
TRAIN_LABEL_ZIP = os.path.join(ROOT_DIR, "train.zip")

TRAIN_IMG_DIR   = os.path.join(ROOT_DIR, "images/train")
TRAIN_LABEL_DIR = os.path.join(ROOT_DIR, "labels/train")
TEST_IMG_DIR    = os.path.join(ROOT_DIR, "images/test")

# ==========================================================
# 3️⃣ COPY ZIP FILES FROM DRIVE → LOCAL EC2
# ==========================================================
def copy_if_needed(src, dst):
    if os.path.exists(dst):
        print(f"✔ {os.path.basename(dst)} already exists locally.")
        return
    if not os.path.exists(src):
        raise FileNotFoundError(f"❌ Missing file in Google Drive: {src}")
    print(f"⚡ Copying {src} → {dst}")
    shutil.copy(src, dst)

print("\nSyncing MS COCO zip files from Google Drive...")
copy_if_needed(os.path.join(DRIVE_DATA_DIR, "train-resized.zip"), TRAIN_IMG_ZIP)
copy_if_needed(os.path.join(DRIVE_DATA_DIR, "test-resized.zip"), TEST_IMG_ZIP)
copy_if_needed(os.path.join(DRIVE_DATA_DIR, "train.zip"), TRAIN_LABEL_ZIP)

# ==========================================================
# 4️⃣ DETERMINISTIC UNZIP + RECURSIVE CONSOLIDATION
# ==========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def dir_ready(path, ext):
    return os.path.exists(path) and len(glob(os.path.join(path, f"*{ext}"))) > 0

def unzip_and_consolidate(zip_path, target_dir, extension):
    if dir_ready(target_dir, extension):
        print(f"✔ {target_dir} already prepared → skipping extraction.")
        return

    print(f"⚡ Extracting {zip_path}")
    temp_dir = target_dir + "_tmp"
    ensure_dir(temp_dir)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    print("⚡ Recursively consolidating nested directories...")
    ensure_dir(target_dir)
    moved = 0
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.lower().endswith(extension):
                src = os.path.join(root, file)
                dst = os.path.join(target_dir, file)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
                    moved += 1
    shutil.rmtree(temp_dir)
    print(f"✔ Consolidated {moved} {extension} files into {target_dir}\n")

# Execute deterministic extraction
unzip_and_consolidate(TRAIN_IMG_ZIP, TRAIN_IMG_DIR, ".jpg")
unzip_and_consolidate(TEST_IMG_ZIP, TEST_IMG_DIR, ".jpg")
unzip_and_consolidate(TRAIN_LABEL_ZIP, TRAIN_LABEL_DIR, ".cls")

# ==========================================================
# 5️⃣ DATASET CLASSES
# ==========================================================
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

# ==========================================================
# 6️⃣ SAFE TRAIN / VALIDATION SPLIT + AUGMENTATIONS
# ==========================================================
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

# ==========================================================
# 7️⃣ GPU-OPTIMIZED DATALOADERS
# ==========================================================
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True,
                          persistent_workers=True)

val_loader = DataLoader(val_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True,
                        persistent_workers=True)

test_loader = DataLoader(
    COCOTestImageDataset(TEST_IMG_DIR, transform=val_transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

# ==========================================================
# 8️⃣ DATASET SUMMARY + AUGMENTATION DOCUMENTATION
# ==========================================================
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
        "Google Drive → Local EC2 Disk (Deterministic Consolidation)"
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

# -----------------------------
# Load EfficientNet-B3
# -----------------------------
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

weights = EfficientNet_B3_Weights.DEFAULT
model = efficientnet_b3(weights=weights)

# -----------------------------
# Optional Backbone Freezing
# -----------------------------
if FREEZE_BACKBONE:
    for param in model.features.parameters():
        param.requires_grad = False
    print("✔ Backbone frozen (training classifier only).")
else:
    print("✔ Full fine-tuning enabled (backbone + classifier).")

# -----------------------------
# Replace Classification Head
# -----------------------------
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

# Move to device
model = model.to(device)

# -----------------------------
# Model Summary
# -----------------------------
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\nModel Summary:")
print(f"  Architecture        : EfficientNet-B3")
print(f"  Input Resolution    : {IMG_SIZE}x{IMG_SIZE}")
print(f"  Output Classes      : {NUM_CLASSES}")
print(f"  Trainable Parameters: {trainable_params:,}")
print("✔ Model initialized successfully.\n")

# ==========================================================
# STEP 6/14: LOSS FUNCTION
# ==========================================================
print("Step 6/14: Loss Function Definition")
print("------------------------------------")
print("WHAT:\n  Define objective function for multi-label classification.\n")
print("WHY:\n  Each class is treated as an independent binary problem.\n"
      "  Class imbalance is addressed using per-class positive weights.\n")
print("HOW:\n  Compute per-class positive frequencies → derive pos_weight →"
      " apply BCEWithLogitsLoss.\n")

# -----------------------------
# Compute Class Frequencies
# -----------------------------
print("Computing class frequency statistics...")

num_positives = torch.zeros(NUM_CLASSES)
num_samples = 0

for _, labels in train_loader:
    num_positives += labels.sum(dim=0)
    num_samples += labels.size(0)

num_negatives = num_samples - num_positives

# Prevent division by zero
epsilon = 1e-6
pos_weight = num_negatives / (num_positives + epsilon)
pos_weight = pos_weight.to(device)

# -----------------------------
# Define Loss
# -----------------------------
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print("Loss Configuration Summary:")
print("  Loss Function : BCEWithLogitsLoss")
print("  Multi-label   : Yes")
print("  Class Weight  : Inverse frequency weighting")
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
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

# ---------------- Scheduler ----------------
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2
)

print("Optimizer & Scheduler Summary:")
print(f"  Optimizer       : AdamW")
print(f"  Learning Rate   : {LEARNING_RATE}")
print(f"  Weight Decay    : 1e-4")
print(f"  Scheduler       : ReduceLROnPlateau")
print(f"  LR Reduction    : factor=0.5 after 2 stagnant epochs")
print("✔ Optimizer and scheduler initialized.\n")

# ---------------- AMP Setup ----------------
scaler = GradScaler()  # Automatic Mixed Precision
training_log = []      # Track per-epoch metrics
best_val_loss = float("inf")  # Track best validation loss


# ==========================================================
# STEP 8/14: TRAINING + VALIDATION (BEST MODEL SAVING)
# ==========================================================
print("\nStep 8/14: Training + Validation (Best Model Saving)")
print("------------------------------------------------------")
print("WHAT:\n  Train and validate the model while saving only the best model.")
print("WHY:\n  Preserve the model with the lowest validation loss, avoiding unnecessary checkpoints.")
print("HOW:\n  Forward → compute loss → backward → optimizer → AMP → validate → update scheduler → save best model.\n")

for epoch in range(1, EPOCHS + 1):
    print(f"\n================ Epoch {epoch}/{EPOCHS} ================")

    # ---------------- Training Phase ----------------
    print("Step 8/14: Training")
    print("--------------------")
    print("WHAT: Update model weights.")
    print("WHY: Minimize BCE multi-label loss.")
    print("HOW: Forward → loss → backward → optimizer → AMP.\n")

    model.train()
    running_train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    print(f"✔ Avg Training Loss: {avg_train_loss:.4f}")

    # ---------------- Validation Phase ----------------
    print("\nStep 8/14: Validation")
    print("-----------------------")
    print("WHAT: Evaluate model on validation set.")
    print("WHY: Monitor overfitting and guide scheduler adjustments.")
    print("HOW: Forward pass → compute loss → compute metrics.\n")

    model.eval()
    val_loss = 0.0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            predicted = (outputs.sigmoid() > THRESHOLD).float()
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.numel()

    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct_preds / total_samples

    print(f"✔ Validation Loss: {avg_val_loss:.4f}")
    print(f"✔ Validation Accuracy: {accuracy:.4f}")

    # ---------------- Scheduler Step ----------------
    scheduler.step(avg_val_loss)
    print("✔ Scheduler updated based on validation loss.\n")

    # ---------------- Best Model Saving ----------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("✔ New best model saved!\n")

    # Track metrics
    training_log.append({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_accuracy": accuracy
    })

print("✔ Training completed. Best model stored as 'best_model.pth'.")

# ==========================================================
# STEP 9/14: FINAL VALIDATION METRICS
# ==========================================================
print("\nStep 9/14: Final Validation Metrics (Class-wise + mAP)")
print("--------------------------------------------------------")
print("WHAT:\n  Evaluate the best model on the validation set including per-class metrics and mean Average Precision (mAP).")
print("WHY:\n  1) Verify overall generalization.\n"
      "  2) Detect class-specific performance issues.\n"
      "  3) Prepare comprehensive metrics for experiment logging.")
print("HOW:\n  1) Load best model checkpoint.\n"
      "  2) Run validation_loop with class_metrics=True.\n"
      "  3) Collect all labels and probabilities to compute mAP.\n"
      "  4) Print overall and per-class metrics.\n")

# Load best model checkpoint
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

# -----------------------------
# Compute validation metrics using validation_loop
# -----------------------------
val_results, val_class_results = validation_loop(
    val_loader,
    model,
    criterion,
    NUM_CLASSES,
    device,
    multi_label=True,
    th_multi_label=THRESHOLD,
    class_metrics=True
)

# Extract metrics
avg_val_loss = val_results["loss"]
micro_f1 = val_results["f1"]
macro_f1 = val_results["f1"]  # weighted by class frequency
accuracy = val_results["accuracy"]

# -----------------------------
# Compute per-class mAP
# -----------------------------
all_labels, all_probs = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

all_labels = torch.cat(all_labels).numpy()
all_probs = torch.cat(all_probs).numpy()
mAP_val = average_precision_score(all_labels, all_probs, average='macro')

# -----------------------------
# Print metrics
# -----------------------------
print(f"✔ Validation Loss  : {avg_val_loss:.4f}")
print(f"✔ Validation Acc   : {accuracy:.4f}")
print(f"✔ Micro F1         : {micro_f1:.4f}")
print(f"✔ Macro F1         : {macro_f1:.4f}")
print(f"✔ mAP              : {mAP_val:.4f}\n")

print("✔ Class-wise Validation Metrics:")
for i, cls_metrics in enumerate(val_class_results):
    print(f"{classes[i]:20s} | F1: {cls_metrics['f1']:.4f} | "
          f"Precision: {cls_metrics['precision']:.4f} | Recall: {cls_metrics['recall']:.4f}")

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
      "  4) Save as JSON to DRIVE_ROOT.\n")

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

# Save submission JSON
submission_path = os.path.join(DRIVE_ROOT, "coco_test_submission.json")
with open(submission_path, 'w') as f:
    json.dump(submission_dict, f, indent=4)

print(f"✔ Test submission JSON saved to {submission_path}")

# ==========================================================
# STEP 11/14: SAVE BEST MODEL
# ==========================================================
MODEL_PATH = os.path.join(DRIVE_ROOT, "efficientnet_b3_coco_best.pth")
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n✔ Best model saved at {MODEL_PATH}")

# ==========================================================
# STEP 12/14: INFERENCE FUNCTION
# ==========================================================
def predict_image(model, img_path, transform, device, threshold=THRESHOLD):
    model.eval()
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(image)).cpu().numpy().flatten()
    return [i for i, v in enumerate(probs) if v > threshold]

print("✔ Inference function ready.")

# ==========================================================
# STEP 13/14: PIPELINE COMPLETION
# ==========================================================
print("\nStep 13/14: Pipeline Completed")
print("✔ Model trained, validated, test JSON ready, inference ready.")

# ==========================================================
# STEP 14/14: JSON EXPERIMENT REPORT
# ==========================================================
print("\nStep 14/14: JSON Experiment Report Generation")
print("------------------------------------------------")
print("WHAT:\n  Generate a structured report summarizing hyperparameters, model, training log, metrics, dataset info, and submission path.")
print("WHY:\n  1) Reproducibility and traceability.\n"
      "  2) Keeps all experiment information consolidated.\n"
      "  3) Ready for sharing or automated logging.")
print("HOW:\n  1) Collect hyperparameters and model details.\n"
      "  2) Include training log.\n"
      "  3) Add validation metrics (overall + class-wise + mAP).\n"
      "  4) Include best model path and submission JSON path.\n"
      "  5) Include dataset info and save JSON.\n")

experiment_report = {
    "experiment_name": "EfficientNet-B3 COCO Multi-label Classification",
    "experiment_start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # capture start ideally
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
    "best_model_path": best_model_path,
    "submission_json_path": submission_path,
    "training_log": training_log,
    "dataset_info": {
        "train_images": len(train_loader.dataset),
        "val_images": len(val_loader.dataset),
        "test_images": len(test_loader.dataset)
    }
}

# Save JSON report
json_output_path = os.path.join(DRIVE_ROOT, "experiment_report.json")
with open(json_output_path, 'w') as json_file:
    json.dump(experiment_report, json_file, indent=4)

print(f"✔ JSON experiment report successfully saved to {json_output_path}")
print("✔ Report includes hyperparameters, model info, validation metrics (overall + class-wise + mAP), training log, dataset info, and test submission path.\n")

