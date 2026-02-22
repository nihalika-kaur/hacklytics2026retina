"""
RetinaScope AI — Baseline Evaluation + Head Fine-Tuning
========================================================
Loads the ODIR-5K dataset, establishes baselines, fine-tunes only the
classification head of RetinaVisionModel, and generates publication-quality
evaluation plots.

Dataset:
  Kaggle — andrewmvd/ocular-disease-recognition-odir5k
  Expected layout:
    <data_dir>/
      full_df.csv  (or ODIR-5K_Training_Annotations.xlsx)
      ODIR-5K_Training_Dataset/<image_files>
      ODIR-5K_Testing_Images/<image_files>   (optional)

Usage:
  python finetune_evaluate.py --data-dir /path/to/odir5k
  python finetune_evaluate.py --data-dir /path/to/odir5k --eval-only --checkpoint retinascope_finetuned.pth
  python finetune_evaluate.py --data-dir /path/to/odir5k --epochs 20 --batch-size 16
"""

import argparse
import json
import os
import sys
import warnings

import matplotlib
if os.environ.get("DISPLAY") is None and sys.platform != "darwin" and sys.platform != "win32":
    matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

from retina import (  # noqa: E402
    RetinaVisionModel,
    _generate_synthetic_fundus,
    apply_clahe,
    extract_vessel_features,
    segment_vessels,
)

# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("tab10")


# ===========================================================================
# ODIR-5K Dataset Helpers
# ===========================================================================

def _find_annotation_file(data_dir: str) -> str:
    """Search for the ODIR-5K annotation CSV/Excel file."""
    candidates = [
        "full_df.csv",
        "ODIR-5K_Training_Annotations.xlsx",
        "ODIR-5K_Training_Annotations.csv",
        "odir_annotations.csv",
        "annotation.xlsx",
        "annotation.csv",
    ]
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            return path
    # Recursive search one level down
    for entry in os.scandir(data_dir):
        if entry.is_dir():
            for name in candidates:
                path = os.path.join(entry.path, name)
                if os.path.isfile(path):
                    return path
    raise FileNotFoundError(
        f"Could not find ODIR-5K annotation file in '{data_dir}'. "
        "Expected one of: " + ", ".join(candidates)
    )


def _find_image_dirs(data_dir: str) -> list:
    """Return candidate image directories in order of preference."""
    candidates = [
        os.path.join(data_dir, "ODIR-5K_Training_Dataset"),
        os.path.join(data_dir, "ODIR-5K_Testing_Images"),
        os.path.join(data_dir, "images"),
        data_dir,
    ]
    return [d for d in candidates if os.path.isdir(d)]


def _is_hypertension_label(label_str: str) -> bool:
    """Return True if the diagnostic label indicates hypertension."""
    if not isinstance(label_str, str):
        return False
    label_lower = label_str.lower()
    return (
        label_lower == "h"
        or "hypertension" in label_lower
        or "hypertensive retinopathy" in label_lower
    )


def load_odir5k(data_dir: str):
    """
    Parse the ODIR-5K annotation file and return a list of
    (image_path, label) tuples where label ∈ {0, 1}.

    Searches image sub-directories and handles both left-eye and right-eye
    columns.  Images whose files cannot be located on disk are skipped.
    """
    import pandas as pd

    ann_path = _find_annotation_file(data_dir)
    print(f"[INFO] Annotation file: {ann_path}")

    if ann_path.endswith(".xlsx"):
        df = pd.read_excel(ann_path)
    else:
        df = pd.read_csv(ann_path)

    print(f"[INFO] Loaded {len(df)} annotation rows. Columns: {list(df.columns)}")

    # Normalise column names to lower-case with underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    image_dirs = _find_image_dirs(data_dir)
    print(f"[INFO] Image search paths: {image_dirs}")

    def resolve_image(filename):
        if not isinstance(filename, str):
            return None
        for d in image_dirs:
            p = os.path.join(d, filename)
            if os.path.isfile(p):
                return p
        return None

    samples = []

    # Try to detect the filename and label columns automatically
    # Common column patterns in ODIR-5K datasets
    left_img_col = next(
        (c for c in df.columns if "left" in c and ("file" in c or "fundus" in c or "image" in c)),
        None,
    )
    right_img_col = next(
        (c for c in df.columns if "right" in c and ("file" in c or "fundus" in c or "image" in c)),
        None,
    )
    # Diagnostic label column — may appear as "labels", "diagnosis", "left_label", etc.
    label_col = next(
        (c for c in df.columns if c in ("labels", "label", "diagnosis", "diagnostic_keywords")),
        None,
    )
    left_label_col = next(
        (c for c in df.columns if "left" in c and "label" in c), None
    )
    right_label_col = next(
        (c for c in df.columns if "right" in c and "label" in c), None
    )

    for _, row in df.iterrows():
        # Determine hypertension label
        is_positive = False
        if label_col and pd.notna(row.get(label_col)):
            is_positive = _is_hypertension_label(str(row[label_col]))
        elif left_label_col and pd.notna(row.get(left_label_col)):
            is_positive = _is_hypertension_label(str(row[left_label_col]))
        elif right_label_col and pd.notna(row.get(right_label_col)):
            is_positive = _is_hypertension_label(str(row[right_label_col]))
        else:
            # Fall back: scan all string columns for 'H' or 'hypertension'
            for col in df.columns:
                val = row.get(col)
                if _is_hypertension_label(str(val) if pd.notna(val) else ""):
                    is_positive = True
                    break

        label = int(is_positive)

        # Add left eye
        if left_img_col:
            path = resolve_image(str(row.get(left_img_col, "")))
            if path:
                samples.append((path, label))
        # Add right eye
        if right_img_col:
            path = resolve_image(str(row.get(right_img_col, "")))
            if path:
                samples.append((path, label))

        # If no dedicated image columns found, try generic filename column
        if not left_img_col and not right_img_col:
            for col in ("filename", "file", "image", "id"):
                if col in df.columns:
                    path = resolve_image(str(row.get(col, "")))
                    if path:
                        samples.append((path, label))
                    break

    if not samples:
        raise RuntimeError(
            "No image/label pairs could be loaded from the annotation file. "
            "Check that image files are present in the expected sub-directories."
        )

    pos = sum(1 for _, l in samples if l == 1)
    neg = len(samples) - pos
    print(f"[INFO] Loaded {len(samples)} samples: {pos} positive (hypertension), {neg} negative")
    return samples


def stratified_split(samples, train_frac=0.70, val_frac=0.15, seed=42):
    """Stratified 70/15/15 split."""
    from sklearn.model_selection import train_test_split

    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    train_p, temp_p, train_l, temp_l = train_test_split(
        paths, labels, test_size=(1 - train_frac), stratify=labels, random_state=seed
    )
    # Split the remaining ~30% evenly: val ~= 15%, test ~= 15%
    val_ratio = val_frac / (1 - train_frac)
    val_p, test_p, val_l, test_l = train_test_split(
        temp_p, temp_l, test_size=(1 - val_ratio), stratify=temp_l, random_state=seed
    )
    return (
        list(zip(train_p, train_l)),
        list(zip(val_p, val_l)),
        list(zip(test_p, test_l)),
    )


# ===========================================================================
# PyTorch Dataset
# ===========================================================================

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

IMG_TRANSFORM_TRAIN = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ODIRDataset(Dataset):
    """PyTorch Dataset for ODIR-5K fundus images."""

    def __init__(self, samples, transform=None, use_vessel_features=True):
        self.samples = samples
        self.transform = transform or IMG_TRANSFORM
        self.use_vessel_features = use_vessel_features

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                raise ValueError("cv2 returned None")
        except Exception:
            image_bgr = _generate_synthetic_fundus()

        if self.use_vessel_features:
            enhanced = apply_clahe(image_bgr)
            vessel_mask = segment_vessels(enhanced)
            feats = extract_vessel_features(vessel_mask)
            vessel_vec = torch.tensor(
                [feats["vessel_density"], feats["mean_width"], feats["tortuosity"]],
                dtype=torch.float32,
            )
        else:
            vessel_vec = torch.zeros(3, dtype=torch.float32)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        img_tensor = self.transform(pil_img)

        return img_tensor, vessel_vec, torch.tensor(label, dtype=torch.float32)


def make_weighted_sampler(samples):
    """Create a WeightedRandomSampler that up-samples the minority class."""
    labels = [s[1] for s in samples]
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    sample_weights = torch.tensor([weights[l] for l in labels], dtype=torch.float32)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# ===========================================================================
# Vessel Feature Extraction (for LR baseline)
# ===========================================================================

def extract_vessel_feature_matrix(samples, desc="Extracting vessel features"):
    """Extract (N, 3) numpy array of vessel features from a list of (path, label) samples."""
    features = []
    labels = []
    for img_path, label in tqdm(samples, desc=desc):
        try:
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                raise ValueError()
        except Exception:
            image_bgr = _generate_synthetic_fundus()
        enhanced = apply_clahe(image_bgr)
        vessel_mask = segment_vessels(enhanced)
        feats = extract_vessel_features(vessel_mask)
        features.append([feats["vessel_density"], feats["mean_width"], feats["tortuosity"]])
        labels.append(label)
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


# ===========================================================================
# Metrics Helpers
# ===========================================================================

def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute a full set of binary classification metrics."""
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true = np.array(y_true)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn + 1e-9)  # recall
    specificity = tn / (tn + fp + 1e-9)
    ppv = tp / (tp + fp + 1e-9)          # precision
    npv = tn / (tn + fn + 1e-9)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")

    return {
        "auc_roc": roc_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "ppv": ppv,
        "npv": npv,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# ===========================================================================
# Baselines
# ===========================================================================

def random_baseline(n_samples, seed=42):
    """Random classifier: predict uniform random probabilities."""
    rng = np.random.default_rng(seed)
    return rng.random(n_samples).tolist()


def majority_class_baseline(n_samples, majority_label=0):
    """Majority class classifier: always predict the majority class probability."""
    if majority_label == 0:
        return [0.0] * n_samples
    else:
        return [1.0] * n_samples


def vessel_lr_baseline(train_samples, test_samples):
    """
    Train a Logistic Regression on 3 vessel features extracted from each image.
    Returns predicted probabilities on the test set.
    """
    print("[INFO] Extracting vessel features for LR baseline (train)…")
    X_train, y_train = extract_vessel_feature_matrix(train_samples, "  Train features")
    print("[INFO] Extracting vessel features for LR baseline (test)…")
    X_test, y_test = extract_vessel_feature_matrix(test_samples, "  Test features")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train_s, y_train)

    probs = lr.predict_proba(X_test_s)[:, 1].tolist()
    return probs, y_test.tolist()


# ===========================================================================
# Training Loop
# ===========================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    for imgs, vessel_feats, labels in tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", leave=False):
        imgs = imgs.to(device)
        vessel_feats = vessel_feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        ht_prob, _, _ = model(imgs, vessel_feats)
        loss = criterion(ht_prob, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        all_probs.extend(ht_prob.detach().cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    try:
        ep_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        ep_auc = float("nan")
    return avg_loss, ep_auc


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    for imgs, vessel_feats, labels in tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [val]  ", leave=False):
        imgs = imgs.to(device)
        vessel_feats = vessel_feats.to(device)
        labels = labels.to(device)

        ht_prob, _, _ = model(imgs, vessel_feats)
        loss = criterion(ht_prob, labels)

        total_loss += loss.item() * len(labels)
        all_probs.extend(ht_prob.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    try:
        ep_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        ep_auc = float("nan")
    return avg_loss, ep_auc, all_probs, all_labels


# ===========================================================================
# Plot Helpers
# ===========================================================================

def plot_training_curves(train_losses, val_losses, train_aucs, val_aucs, out_path="training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, label="Train loss", color=PALETTE[0])
    axes[0].plot(epochs, val_losses, label="Val loss", color=PALETTE[1], linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_aucs, label="Train AUC", color=PALETTE[0])
    axes[1].plot(epochs, val_aucs, label="Val AUC", color=PALETTE[1], linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("Training & Validation AUC")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def plot_roc_curves(roc_data: dict, out_path="roc_curves.png"):
    """
    roc_data: dict mapping model_name -> (y_true, y_prob)
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")

    for i, (name, (y_true, y_prob)) in enumerate(roc_data.items()):
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=PALETTE[i % len(PALETTE)])
        except ValueError:
            pass

    ax.set_xlabel("False Positive Rate (1 – Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def plot_metrics_comparison(metrics_dict: dict, out_path="metrics_comparison.png"):
    """
    metrics_dict: dict mapping model_name -> metrics dict
    """
    metric_keys = ["auc_roc", "sensitivity", "specificity", "f1"]
    metric_labels = ["AUC-ROC", "Sensitivity", "Specificity", "F1 Score"]
    model_names = list(metrics_dict.keys())
    n_models = len(model_names)
    n_metrics = len(metric_keys)

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(model_names):
        vals = [metrics_dict[name].get(k, 0.0) for k in metric_keys]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=PALETTE[i % len(PALETTE)])

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Key Metrics")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def plot_confusion_matrix(y_true, y_prob, threshold=0.5, model_name="Fine-tuned ViT", out_path="confusion_matrix.png"):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def plot_risk_distribution(y_true, y_prob, model_name="Fine-tuned ViT", out_path="risk_distribution.png"):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        y_prob[y_true == 0], bins=30, alpha=0.6,
        label="Negative (no hypertension)", color=PALETTE[0], density=True,
    )
    ax.hist(
        y_prob[y_true == 1], bins=30, alpha=0.6,
        label="Positive (hypertension)", color=PALETTE[1], density=True,
    )
    ax.set_xlabel("Predicted Hypertension Probability")
    ax.set_ylabel("Density")
    ax.set_title(f"Risk Score Distribution — {model_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


# ===========================================================================
# Single Image Prediction
# ===========================================================================

def predict_single(image_path: str, checkpoint: str, device: str = None) -> dict:
    """
    Predict hypertension probability for a single fundus image.

    Parameters
    ----------
    image_path : str  Path to the fundus image.
    checkpoint : str  Path to the saved model checkpoint (.pth).
    device     : str  Optional device override.

    Returns
    -------
    dict with 'hypertension_prob' and 'vessel_features'.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RetinaVisionModel().to(device)
    state = torch.load(checkpoint, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    try:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError()
    except Exception:
        print("[WARN] Could not read image; using synthetic demo image.")
        image_bgr = _generate_synthetic_fundus()

    enhanced = apply_clahe(image_bgr)
    vessel_mask = segment_vessels(enhanced)
    feats = extract_vessel_features(vessel_mask)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    img_tensor = IMG_TRANSFORM(pil_img).unsqueeze(0).to(device)

    vessel_tensor = torch.tensor(
        [[feats["vessel_density"], feats["mean_width"], feats["tortuosity"]]],
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        ht_prob, _, _ = model(img_tensor, vessel_tensor)

    return {
        "hypertension_prob": float(ht_prob.item()),
        "vessel_features": feats,
    }


# ===========================================================================
# Main Training/Evaluation Pipeline
# ===========================================================================

def run(args):
    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[INFO] Device: {device}")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    samples = load_odir5k(args.data_dir)
    train_samples, val_samples, test_samples = stratified_split(samples)
    print(
        f"[INFO] Split — Train: {len(train_samples)}, "
        f"Val: {len(val_samples)}, Test: {len(test_samples)}"
    )

    # ------------------------------------------------------------------
    # Majority class for baseline
    # ------------------------------------------------------------------
    test_labels = [s[1] for s in test_samples]
    majority_label = int(np.bincount(test_labels).argmax())

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    num_workers = args.num_workers
    sampler = make_weighted_sampler(train_samples)
    train_dataset = ODIRDataset(train_samples, transform=IMG_TRANSFORM_TRAIN)
    val_dataset = ODIRDataset(val_samples)
    test_dataset = ODIRDataset(test_samples)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=sampler, num_workers=num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"),
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = RetinaVisionModel().to(device)

    if args.eval_only and args.checkpoint:
        print(f"[INFO] Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
    elif not args.eval_only:
        # Fine-tune only the classification head
        for param in model.backbone.parameters():
            param.requires_grad = False
        head_params = [p for p in model.head.parameters() if p.requires_grad]
        criterion = nn.BCELoss()  # WeightedRandomSampler handles class imbalance

        optimizer = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        train_losses, val_losses, train_aucs, val_aucs = [], [], [], []
        best_val_auc = 0.0

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_auc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, args.epochs)
            vl_loss, vl_auc, _, _ = evaluate_epoch(model, val_loader, criterion, device, epoch, args.epochs)
            scheduler.step()

            train_losses.append(tr_loss)
            val_losses.append(vl_loss)
            train_aucs.append(tr_auc)
            val_aucs.append(vl_auc)

            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"Train loss={tr_loss:.4f} AUC={tr_auc:.4f} | "
                f"Val   loss={vl_loss:.4f} AUC={vl_auc:.4f}"
            )

            if vl_auc > best_val_auc:
                best_val_auc = vl_auc
                torch.save(
                    {"model_state_dict": model.state_dict(), "epoch": epoch, "val_auc": vl_auc},
                    "retinascope_finetuned.pth",
                )
                print(f"  ✓ Saved best checkpoint (val AUC={vl_auc:.4f})")

        plot_training_curves(train_losses, val_losses, train_aucs, val_aucs)

        # Load best checkpoint for final evaluation
        state = torch.load("retinascope_finetuned.pth", map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(f"[INFO] Loaded best checkpoint (epoch {state['epoch']}, val AUC={state['val_auc']:.4f})")

    # ------------------------------------------------------------------
    # Get model predictions on the test set
    # ------------------------------------------------------------------
    model.eval()
    criterion_eval = nn.BCELoss()
    _, _, finetuned_probs, test_labels_list = evaluate_epoch(
        model, test_loader, criterion_eval, device, 0, 0
    )

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    rand_probs = random_baseline(len(test_labels_list))
    maj_probs = majority_class_baseline(len(test_labels_list), majority_label)

    print("[INFO] Computing vessel+LR baseline…")
    vessel_probs, vessel_labels = vessel_lr_baseline(train_samples, test_samples)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    all_metrics = {
        "Random Baseline": compute_metrics(test_labels_list, rand_probs),
        "Majority Class": compute_metrics(test_labels_list, maj_probs),
        "Vessel LR": compute_metrics(vessel_labels, vessel_probs),
        "Fine-tuned ViT (head)": compute_metrics(test_labels_list, finetuned_probs),
    }

    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)
    header = f"{'Model':<28} {'AUC-ROC':>8} {'Sens.':>8} {'Spec.':>8} {'F1':>8} {'PPV':>8} {'NPV':>8}"
    print(header)
    print("-" * 70)
    for name, m in all_metrics.items():
        print(
            f"{name:<28} "
            f"{m['auc_roc']:>8.4f} "
            f"{m['sensitivity']:>8.4f} "
            f"{m['specificity']:>8.4f} "
            f"{m['f1']:>8.4f} "
            f"{m['ppv']:>8.4f} "
            f"{m['npv']:>8.4f}"
        )
    print("=" * 70)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    roc_data = {
        "Random Baseline": (test_labels_list, rand_probs),
        "Majority Class": (test_labels_list, maj_probs),
        "Vessel LR": (vessel_labels, vessel_probs),
        "Fine-tuned ViT (head)": (test_labels_list, finetuned_probs),
    }
    plot_roc_curves(roc_data)
    plot_metrics_comparison(all_metrics)
    plot_confusion_matrix(test_labels_list, finetuned_probs)
    plot_risk_distribution(test_labels_list, finetuned_probs)

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    results = {
        "split_sizes": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "metrics": all_metrics,
    }
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[INFO] Saved evaluation_results.json")

    return results


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RetinaScope AI — Baseline Evaluation + Head Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Root directory of the ODIR-5K dataset.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of fine-tuning epochs (default: 10).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="DataLoader batch size (default: 16). Use 4–8 on CPU.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate for the classification head (default: 1e-4).",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; load checkpoint and evaluate only.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (.pth) for --eval-only mode.",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda", "mps"],
        help="PyTorch device (default: auto-detect).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader worker processes (default: 4). Set to 0 if issues arise.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
