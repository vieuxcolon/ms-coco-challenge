# ==========================================================
# COCO Backbone Comparison Experiment Tree
# ==========================================================

experiment_tree = """
COCO Multi-label Backbone Comparison
├─ Dataset Preparation
│  ├─ Train/Validation split (80/20)
│  ├─ Data Augmentation (train)
│  └─ Normalization (val/test)
├─ Candidate Backbones
│  ├─ ResNet-50
│  │  ├─ Pros: strong baseline, widely used
│  │  ├─ Cons: larger memory, slower than EfficientNet-B3
│  │  └─ Metrics to track:
│  │     ├─ Micro F1
│  │     ├─ Macro F1
│  │     ├─ mAP
│  │     ├─ Class-wise F1/Precision/Recall
│  │     └─ Training time & GPU usage
│  ├─ EfficientNet-B3
│  │  ├─ Pros: high accuracy, efficient scaling, pretrained weights
│  │  ├─ Cons: medium GPU memory requirement
│  │  └─ Metrics to track: same as ResNet-50
│  └─ MobileNetV3-Large
│     ├─ Pros: fast, low memory, suitable for edge deployment
│     ├─ Cons: slightly lower accuracy on complex datasets
│     └─ Metrics to track: same as others
├─ Training Pipeline
│  ├─ BCEWithLogitsLoss with class weights
│  ├─ AdamW optimizer + ReduceLROnPlateau scheduler
│  ├─ AMP (mixed precision)
│  └─ Early stopping: keep best validation loss
├─ Evaluation
│  ├─ Validation Metrics
│  │  ├─ Overall: loss, micro F1, macro F1, mAP
│  │  └─ Class-wise: F1, precision, recall
│  ├─ Test Metrics / JSON Submission
│  └─ Compare speed/memory usage
└─ Decision Criteria
   ├─ Accuracy: micro/macro F1, mAP
   ├─ Per-class performance: rare class detection
   ├─ Training/inference efficiency
   └─ Deployment suitability (if mobile/edge required)
"""

# ------------------------------
# Optional: color printing for Python terminal
# ------------------------------
def print_tree(tree_text):
    from colorama import Fore, Style, init
    init(autoreset=True)
    
    for line in tree_text.split("\n"):
        if line.startswith("├─") or line.startswith("└─"):
            print(Fore.CYAN + line)       # branch nodes
        elif line.strip().startswith("├─") or line.strip().startswith("└─"):
            print(Fore.MAGENTA + line)    # sub-branches
        else:
            print(Fore.GREEN + line)      # leaf nodes

# ------------------------------
# Display experiment tree
# ------------------------------
print_tree(experiment_tree)
