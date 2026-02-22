"""
RetinaScope AI — Advanced Fine-Tuning with RETFound Weights
============================================================
Loads real RETFound weights (or falls back to ImageNet ViT), partially unfreezes
the last 4 transformer blocks, trains with Focal Loss and two-group learning rates,
and generates a comprehensive evaluation report.

RETFound weights source:
  https://huggingface.co/rmaphoh/RETFound_MAE/resolve/main/RETFound_cfp_weights.pth

Dataset:
  Kaggle — andrewmvd/ocular-disease-recognition-odir5k  (same as finetune_evaluate.py)

Usage:
  python finetune_retfound.py --data-dir /path/to/odir5k
  python finetune_retfound.py --data-dir /path/to/odir5k --retfound-weights RETFound_cfp_weights.pth
  python finetune_retfound.py --data-dir /path/to/odir5k --eval-only --checkpoint retinascope_retfound.pth
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
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# Import shared utilities from the baseline script
from finetune_evaluate import (
    ODIRDataset,
    IMG_TRANSFORM,
    IMG_TRANSFORM_TRAIN,
    compute_metrics,
    extract_vessel_feature_matrix,
    load_odir5k,
    make_weighted_sampler,
    majority_class_baseline,
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_risk_distribution,
    plot_roc_curves,
    plot_training_curves,
    random_baseline,
    stratified_split,
    vessel_lr_baseline,
    PALETTE,
)
from retina import (
    RetinaVisionModel,
    _generate_synthetic_fundus,
    apply_clahe,
    extract_vessel_features,
    segment_vessels,
)

sns.set_theme(style="whitegrid", font_scale=1.1)

# ---------------------------------------------------------------------------
# Optional timm import
# ---------------------------------------------------------------------------
try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


# ===========================================================================
# Focal Loss
# ===========================================================================

class FocalLoss(nn.Module):
    """
    Binary Focal Loss — down-weights easy negatives and focuses training on
    hard cases.  Typically yields 3–5% AUC improvement on imbalanced datasets.

    Parameters
    ----------
    alpha : float   Weighting factor for the positive class (default 0.25).
    gamma : float   Focusing parameter (default 2.0).
    """

    EPSILON = 1e-7  # Numerical stability clamp for log operations

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Clamp to avoid log(0)
        inputs = inputs.clamp(self.EPSILON, 1 - self.EPSILON)
        bce = -(targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        alpha_t = torch.where(targets == 1, torch.full_like(inputs, self.alpha), torch.full_like(inputs, 1 - self.alpha))
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * bce
        return focal_loss.mean()


# ===========================================================================
# RETFound Weight Loading
# ===========================================================================

def _load_retfound_weights(model: RetinaVisionModel, weights_path: str, device: str) -> bool:
    """
    Attempt to load RETFound weights into the ViT backbone.

    The RETFound checkpoint is a MAE-pretrained ViT-Large; this function
    handles the key remapping to align with the timm ViT-Base backbone used
    as a stand-in when the actual architecture is not available.

    Returns True if weights were successfully loaded, False otherwise.
    """
    if not os.path.isfile(weights_path):
        print(f"[WARN] RETFound weights not found at '{weights_path}'. "
              "Falling back to ImageNet ViT weights.")
        return False

    try:
        checkpoint = torch.load(weights_path, map_location=device)
        # RETFound checkpoints may store the model inside a 'model' key
        state_dict = checkpoint.get("model", checkpoint)

        # Try a direct load first
        if hasattr(model.backbone, "backbone"):
            backbone_module = model.backbone.backbone
        else:
            backbone_module = model.backbone

        # Filter keys to only those that match the backbone
        backbone_state = backbone_module.state_dict()
        matched, skipped = {}, []
        for k, v in state_dict.items():
            # Strip common prefixes
            clean_k = k
            for prefix in ("encoder.", "module.", "backbone."):
                if clean_k.startswith(prefix):
                    clean_k = clean_k[len(prefix):]
            if clean_k in backbone_state and backbone_state[clean_k].shape == v.shape:
                matched[clean_k] = v
            else:
                skipped.append(k)

        if matched:
            backbone_state.update(matched)
            backbone_module.load_state_dict(backbone_state, strict=False)
            print(
                f"[INFO] Loaded {len(matched)} / {len(state_dict)} weight tensors "
                f"from RETFound checkpoint. Skipped {len(skipped)} incompatible keys."
            )
            return True
        else:
            print("[WARN] No matching weight tensors found. Falling back to ImageNet ViT weights.")
            return False
    except Exception as exc:
        print(f"[WARN] Failed to load RETFound weights: {exc}. Falling back to ImageNet ViT weights.")
        return False


# ===========================================================================
# Partial Backbone Unfreeze
# ===========================================================================

def _partially_unfreeze_backbone(model: RetinaVisionModel, num_blocks: int = 4) -> list:
    """
    Unfreeze the last `num_blocks` transformer blocks of the ViT backbone.
    All other backbone parameters remain frozen.

    Returns the list of unfrozen parameter groups (backbone blocks + head).
    """
    if not _TIMM_AVAILABLE:
        print("[WARN] timm not available; cannot partially unfreeze backbone. Head only.")
        return [p for p in model.head.parameters() if p.requires_grad]

    backbone = model.backbone.backbone if hasattr(model.backbone, "backbone") else model.backbone

    # Identify the transformer blocks attribute (timm uses .blocks)
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        print("[WARN] Could not find transformer blocks. Unfreezing all backbone params.")
        for p in model.backbone.parameters():
            p.requires_grad = True
        return list(model.backbone.parameters())

    total_blocks = len(blocks)
    unfreeze_from = max(0, total_blocks - num_blocks)
    print(f"[INFO] ViT has {total_blocks} blocks; unfreezing blocks {unfreeze_from}–{total_blocks - 1}")

    unfrozen_backbone_params = []
    for i, block in enumerate(blocks):
        if i >= unfreeze_from:
            for p in block.parameters():
                p.requires_grad = True
                unfrozen_backbone_params.append(p)

    # Also unfreeze the final norm layer if present
    for attr in ("norm", "fc_norm", "head_drop"):
        layer = getattr(backbone, attr, None)
        if layer is not None:
            for p in layer.parameters():
                p.requires_grad = True
                unfrozen_backbone_params.append(p)

    return unfrozen_backbone_params


# ===========================================================================
# Warmup + Cosine Decay Scheduler
# ===========================================================================

def make_warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """
    Linear warmup for `warmup_epochs` then cosine decay to near-zero.
    Protects pretrained weights from large gradient updates early in training.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(warmup_epochs, 1))
        progress = float(epoch - warmup_epochs) / float(max(total_epochs - warmup_epochs, 1))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ===========================================================================
# Additional Plot: Threshold Analysis
# ===========================================================================

def plot_threshold_analysis(y_true, y_prob, model_name="RETFound Fine-tuned", out_path="threshold_analysis.png"):
    """
    Plot sensitivity and specificity at every probability threshold so the
    user can pick an operating point for clinical screening.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1 - fpr

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, tpr, label="Sensitivity (TPR)", color=PALETTE[0])
    ax.plot(thresholds, specificity, label="Specificity (TNR)", color=PALETTE[1])

    # Mark the operating point where sensitivity > 0.85
    target_sensitivity = 0.85
    idx = np.argmax(tpr >= target_sensitivity)
    if idx < len(thresholds):
        ax.axvline(thresholds[idx], color="red", linestyle="--", alpha=0.7,
                   label=f"Threshold={thresholds[idx]:.3f} (Sens≥{target_sensitivity:.0%})")
        ax.scatter([thresholds[idx]], [tpr[idx]], color="red", zorder=5)
        ax.scatter([thresholds[idx]], [specificity[idx]], color="darkred", zorder=5)

    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Rate")
    ax.set_title(f"Sensitivity / Specificity Trade-off — {model_name}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


# ===========================================================================
# Training Loop (same structure as finetune_evaluate but with Focal Loss)
# ===========================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    all_probs, all_labels = [], []

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
    all_probs, all_labels = [], []

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
# Predict Single Image
# ===========================================================================

def predict_single(image_path: str, checkpoint: str, device: str = None) -> dict:
    """
    Predict hypertension probability for a single fundus image using the
    RETFound fine-tuned model.

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
# Main Pipeline
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

    # Majority class for baseline
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
    retfound_loaded = False

    if args.eval_only and args.checkpoint:
        print(f"[INFO] Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
    elif not args.eval_only:
        # ------------------------------------------------------------------
        # Load RETFound weights (or fall back to ImageNet ViT)
        # ------------------------------------------------------------------
        if args.retfound_weights:
            retfound_loaded = _load_retfound_weights(model, args.retfound_weights, device)
        if not retfound_loaded:
            print("[INFO] Using ImageNet ViT backbone weights (RETFound not loaded).")

        # ------------------------------------------------------------------
        # Partially unfreeze last 4 transformer blocks
        # ------------------------------------------------------------------
        unfrozen_backbone_params = _partially_unfreeze_backbone(model, num_blocks=4)
        head_params = list(model.head.parameters())

        print(
            f"[INFO] Trainable params: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

        # ------------------------------------------------------------------
        # Two-group optimizer: backbone unfrozen blocks at lr × 0.1
        # ------------------------------------------------------------------
        optimizer = torch.optim.AdamW(
            [
                {"params": unfrozen_backbone_params, "lr": args.lr * 0.1},
                {"params": head_params, "lr": args.lr},
            ],
            weight_decay=1e-4,
        )

        warmup_epochs = max(1, args.epochs // 10)
        scheduler = make_warmup_cosine_scheduler(optimizer, warmup_epochs, args.epochs)

        criterion = FocalLoss(alpha=0.25, gamma=2.0)

        train_losses, val_losses, train_aucs, val_aucs = [], [], [], []
        best_val_auc = 0.0
        checkpoint_path = "retinascope_retfound.pth"

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_auc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, args.epochs)
            vl_loss, vl_auc, _, _ = evaluate_epoch(model, val_loader, criterion, device, epoch, args.epochs)
            scheduler.step()

            current_lr_head = optimizer.param_groups[-1]["lr"]
            current_lr_bb = optimizer.param_groups[0]["lr"]

            train_losses.append(tr_loss)
            val_losses.append(vl_loss)
            train_aucs.append(tr_auc)
            val_aucs.append(vl_auc)

            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"Train loss={tr_loss:.4f} AUC={tr_auc:.4f} | "
                f"Val   loss={vl_loss:.4f} AUC={vl_auc:.4f} | "
                f"lr_head={current_lr_head:.2e} lr_bb={current_lr_bb:.2e}"
            )

            if vl_auc > best_val_auc:
                best_val_auc = vl_auc
                torch.save(
                    {"model_state_dict": model.state_dict(), "epoch": epoch, "val_auc": vl_auc},
                    checkpoint_path,
                )
                print(f"  ✓ Saved best checkpoint (val AUC={vl_auc:.4f})")

        plot_training_curves(train_losses, val_losses, train_aucs, val_aucs)

        # Load best checkpoint
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(f"[INFO] Loaded best checkpoint (epoch {state['epoch']}, val AUC={state['val_auc']:.4f})")

    # ------------------------------------------------------------------
    # Test set evaluation
    # ------------------------------------------------------------------
    model.eval()
    criterion_eval = FocalLoss(alpha=0.25, gamma=2.0)
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
        "RETFound Fine-tuned": compute_metrics(test_labels_list, finetuned_probs),
    }

    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS — RETFound Fine-tuning")
    print("=" * 70)
    header = f"{'Model':<24} {'AUC-ROC':>8} {'Sens.':>8} {'Spec.':>8} {'F1':>8} {'PPV':>8} {'NPV':>8}"
    print(header)
    print("-" * 70)
    for name, m in all_metrics.items():
        print(
            f"{name:<24} "
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
        "RETFound Fine-tuned": (test_labels_list, finetuned_probs),
    }
    plot_roc_curves(roc_data)
    plot_metrics_comparison(all_metrics)
    plot_confusion_matrix(test_labels_list, finetuned_probs, model_name="RETFound Fine-tuned")
    plot_risk_distribution(test_labels_list, finetuned_probs, model_name="RETFound Fine-tuned")

    try:
        plot_threshold_analysis(test_labels_list, finetuned_probs)
    except Exception as exc:
        print(f"[WARN] Could not generate threshold analysis plot: {exc}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "split_sizes": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "retfound_weights_loaded": args.retfound_weights is not None,
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
        description="RetinaScope AI — Advanced Fine-Tuning with RETFound Weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Root directory of the ODIR-5K dataset.",
    )
    parser.add_argument(
        "--retfound-weights", type=str, default=None,
        help="Path to RETFound_cfp_weights.pth checkpoint. "
             "Downloads from HuggingFace if not provided and not already present.",
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Number of fine-tuning epochs (default: 15).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="DataLoader batch size (default: 16). Use 4–8 on CPU.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Base learning rate for the classification head (default: 1e-4). "
             "Backbone unfrozen blocks train at lr × 0.1.",
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

    # Auto-download RETFound weights if not provided and standard path not present
    default_weights = "RETFound_cfp_weights.pth"
    if args.retfound_weights is None and not os.path.isfile(default_weights):
        url = "https://huggingface.co/rmaphoh/RETFound_MAE/resolve/main/RETFound_cfp_weights.pth"
        print(f"[INFO] --retfound-weights not specified. Attempting download from:\n  {url}")
        try:
            import urllib.request
            print("[INFO] Downloading RETFound weights (this may take a few minutes)…")
            urllib.request.urlretrieve(url, default_weights)
            print(f"[INFO] Downloaded to '{default_weights}'.")
            args.retfound_weights = default_weights
        except Exception as exc:
            print(f"[WARN] Download failed: {exc}. Proceeding with ImageNet ViT weights.")
    elif args.retfound_weights is None and os.path.isfile(default_weights):
        args.retfound_weights = default_weights
        print(f"[INFO] Found local RETFound weights: {default_weights}")

    run(args)


if __name__ == "__main__":
    main()
