ms_coco_pipeline_tree_full = """
STEP 1–3: Environment & Setup
└─ Prepare environment, install dependencies, set seeds, define device
   ├─ Step 1: Install packages, verify versions
   ├─ Step 2: Set random seeds for reproducibility
   └─ Step 3: Detect GPU, configure device

STEP 4: Dataset Preparation & DataLoaders
   ├─ Mount Google Drive & copy zip files to local EC2 disk
   ├─ Flatten nested/sub-subdirectories → deterministic TRAIN_IMG_DIR, TRAIN_LABEL_DIR, TEST_IMG_DIR
   ├─ Apply Data Augmentation
   │   ├─ Training set: Flip, Rotation, ColorJitter
   │   └─ Validation/Test set: Resize + Normalize
   └─ Safe Train/Validation split → separate dataset instances (no transform leakage)

STEP 5: Model Setup
   ├─ Load EfficientNet-B3 pretrained
   ├─ Replace classifier head for 80 classes
   └─ Optional freeze backbone

STEP 6: Loss Function (Class Imbalance Handling)
   ├─ Compute pos_weight from training labels
   ├─ Clip extreme weights (0.1–10)
   └─ BCEWithLogitsLoss → multi-label

STEP 7: Optimizer & LR Strategy
   ├─ AdamW optimizer
   ├─ ReduceLROnPlateau scheduler
   └─ Automatic Mixed Precision (AMP)

STEP 8: Training + Validation
   ├─ Epoch loop with start/end timestamps
   ├─ Training: Forward → Loss → Backward → Optimizer → AMP
   ├─ Validation: Forward → Compute loss & metrics
   └─ Save Best Model (lowest val loss)

STEP 9: Final Validation Metrics
   ├─ Load best model checkpoint
   ├─ Compute class-wise metrics (precision, recall, F1)
   └─ Compute overall mAP (micro & macro F1 distinguished)

STEP 10: Test Evaluation
   ├─ Forward pass on test set
   ├─ Sigmoid + threshold → predicted classes
   └─ Save submission JSON

STEP 11: Save Best Model
   └─ Store model checkpoint for reproducibility & inference

STEP 12: Inference Function (predict_image)
   └─ Load image, apply transform, forward pass → predicted classes

STEP 13: Pipeline Completion
   └─ All components ready: training, validation, test JSON, inference

STEP 14: JSON Experiment Report
   ├─ Hyperparameters
   ├─ Model info (architecture, trainable params, backbone freeze)
   ├─ Dataset info + augmentations
   ├─ Validation metrics (overall + class-wise + mAP)
   ├─ Best model path
   ├─ Test submission path
   ├─ Training log + epoch durations
   └─ pos_weight vector (class imbalance)
"""

# Print the pipeline tree
print(ms_coco_pipeline_tree_full)
