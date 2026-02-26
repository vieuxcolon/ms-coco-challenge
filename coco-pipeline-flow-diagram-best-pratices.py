ms_coco_pipeline_tree = """
MS-COCO Multi-Label Classification Pipeline
└─ Step 1: Setup & Environment
   ├─ Install libraries, set seeds, configure device
   └─ Mount Google Drive (rclone) → local EC2 storage
└─ Step 2-3: Data Preparation
   ├─ Copy zip files from Drive → EC2
   └─ Deterministic unzip & recursive consolidation
└─ Step 4: Dataset & DataLoaders
   ├─ Define COCOTrainImageDataset & COCOTestImageDataset
   ├─ Train/Validation split
   │  └─ Safe split: independent dataset instances (no transform leakage)
   ├─ Data Augmentation (train_transform)
   │  └─ Flip, Rotation, ColorJitter
   ├─ Standardization / Normalization
   │  └─ Resize + Normalize (ImageNet mean/std)
   ├─ GPU-optimized DataLoaders
   │  └─ pin_memory=True, persistent_workers=True
   └─ Dataset Summary: train/val/test counts, augmentation types
└─ Step 5: Model Initialization
   └─ EfficientNet-B3 (pretrained backbone)
└─ Step 6: Loss Function (Class Imbalance Handling)
   ├─ Compute per-class positive/negative counts
   ├─ Compute pos_weight = num_negatives / num_positives
   └─ BCEWithLogitsLoss(pos_weight=pos_weight)
└─ Step 7: Optimizer & LR Scheduler
   ├─ AdamW optimizer
   └─ ReduceLROnPlateau scheduler (validation loss-driven)
└─ Step 8: Training + Validation
   ├─ For each epoch:
   │  ├─ Timestamp start/end (rec #5)
   │  ├─ Training Phase
   │  │  └─ Forward → loss → backward → optimizer → AMP
   │  ├─ Validation Phase
   │  │  └─ Forward → compute loss & metrics
   │  └─ Update scheduler
   └─ Save best model based on validation loss
└─ Step 9: Final Validation Metrics
   ├─ Load best model checkpoint
   ├─ Compute metrics
   │  ├─ Micro F1 (per-sample)
   │  ├─ Macro F1 (per-class)
   │  ├─ Accuracy
   │  └─ Per-class metrics
   └─ Compute mAP across all classes
└─ Step 10: Test Evaluation + Submission JSON
   └─ Forward pass on test set → sigmoid → threshold → predicted classes
└─ Step 11-13: Save best model, inference function, pipeline completion
└─ Step 14: JSON Experiment Report
   ├─ Hyperparameters
   ├─ Model architecture
   ├─ Training log
   ├─ Validation metrics (micro/macro F1 + mAP + class-wise)
   ├─ Dataset info
   └─ Submission path
"""

# Example print
print(ms_coco_pipeline_tree)
