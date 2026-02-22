import os
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import Tuple, List

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader, random_split
import timm

# ------------------------
# Config
# ------------------------
@dataclass
class TrainConfig:
    data_dir: str
    labels_csv: str
    retfound_weights: str
    output_dir: str = "outputs/retfound_odir/"
    image_size: int = 384
    batch_size: int = 8
    num_workers: int = 4
    max_epochs: int = 25
    warmup_epochs: int = 3
    lr_head: float = 5e-4
    lr_backbone: float = 5e-5
    weight_decay: float = 0.05
    drop_path_rate: float = 0.2
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 0.2
    label_smoothing: float = 0.0
    val_split: float = 0.15
    seed: int = 42
    num_classes: int = 8

# ------------------------
# Utils
# ------------------------
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Dataset
# ------------------------
class ODIRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_root: str, image_size: int, augment: bool):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.image_size = image_size
        self.augment = augment
        self.transforms = self.build_transforms()

    def build_transforms(self):
        train_tf = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.2),
            A.CLAHE(p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05, p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
        ])
        val_tf = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
        ])
        return train_tf if self.augment else val_tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row["image"])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = self.transforms(image=img)["image"]
        img_t = torch.from_numpy(np.transpose(aug, (2, 0, 1))).float()
        labels = torch.tensor(row[["N","D","G","C","A","H","M","O"]].values.astype(np.float32))
        return img_t, labels

# ------------------------
# Model
# ------------------------
class RetFoundClassifier(nn.Module):
    def __init__(self, num_classes: int, weights_path: str, drop_path_rate: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        self.load_retfound_weights(weights_path)
        self.head = nn.Sequential(
            nn.LayerNorm(self.backbone.num_features),
            nn.Linear(self.backbone.num_features, num_classes)
        )

    def load_retfound_weights(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"RETFound weights not found at {path}")
        state = torch.load(path, map_location="cpu")
        if "model" in state:
            state = state["model"]
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        print(f"[load weights] missing: {len(missing)}, unexpected: {len(unexpected)}")

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits

# ------------------------
# Mixup/Cutmix helper
# ------------------------
def apply_mixup_cutmix(x, y, alpha_mix=0.2, alpha_cut=0.2):
    if alpha_mix <= 0 and alpha_cut <= 0:
        return x, y, 1.0
    use_cutmix = random.random() < 0.5
    alpha = alpha_cut if use_cutmix else alpha_mix
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    if use_cutmix:
        h, w = x.size(2), x.size(3)
        cx, cy = np.random.randint(w), np.random.randint(h)
        cut_w, cut_h = int(w * math.sqrt(1 - lam)), int(h * math.sqrt(1 - lam))
        x1, y1 = np.clip(cx - cut_w // 2, 0, w), np.clip(cy - cut_h // 2, 0, h)
        x2, y2 = np.clip(cx + cut_w // 2, 0, w), np.clip(cy + cut_h // 2, 0, h)
        x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    else:
        x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return x, (y_a, y_b), lam

def mixup_criterion(criterion, preds, targets, lam):
    y_a, y_b = targets
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)

# ------------------------
# Train / Eval
# ------------------------
def get_loaders(cfg: TrainConfig):
    df = pd.read_csv(cfg.labels_csv)
    assert all(c in df.columns for c in ["image","N","D","G","C","A","H","M","O"])
    val_len = int(len(df) * cfg.val_split)
    train_len = len(df) - val_len
    train_df, val_df = random_split(df, [train_len, val_len], generator=torch.Generator().manual_seed(cfg.seed))
    train_ds = ODIRDataset(train_df.dataset.iloc[train_df.indices], cfg.data_dir, cfg.image_size, augment=True)
    val_ds = ODIRDataset(val_df.dataset.iloc[val_df.indices], cfg.data_dir, cfg.image_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader

def evaluate(model, loader, device):
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    all_logits, all_labels, val_loss = [], [], 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = crit(logits, labels)
            val_loss += loss.item() * imgs.size(0)
            all_logits.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())
    probs = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy()
    aucs = []
    for i in range(labels_np.shape[1]):
        try:
            aucs.append(roc_auc_score(labels_np[:, i], probs[:, i]))
        except ValueError:
            aucs.append(float("nan"))
    mean_auc = float(np.nanmean(aucs))
    return val_loss / len(loader.dataset), mean_auc, aucs

def run_training(cfg: TrainConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    seed_all(cfg.seed)
    device = get_device()
    print(f"Device: {device}")

    model = RetFoundClassifier(cfg.num_classes, cfg.retfound_weights, drop_path_rate=cfg.drop_path_rate).to(device)
    criterion = nn.BCEWithLogitsLoss()

    head_params = list(model.head.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("head")]

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": cfg.lr_backbone},
        {"params": head_params, "lr": cfg.lr_head},
    ], weight_decay=cfg.weight_decay)

    train_loader, val_loader = get_loaders(cfg)

    best_auc = -1
    metrics_path = os.path.join(cfg.output_dir, "metrics.json")
    for epoch in range(cfg.max_epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            use_aug = cfg.mixup_alpha > 0 or cfg.cutmix_alpha > 0
            if use_aug:
                imgs, (y_a, y_b), lam = apply_mixup_cutmix(imgs, labels, cfg.mixup_alpha, cfg.cutmix_alpha)
            optimizer.zero_grad()
            logits = model(imgs)
            if use_aug:
                loss = mixup_criterion(criterion, logits, (y_a, y_b), lam)
            else:
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, mean_auc, per_class_auc = evaluate(model, val_loader, device)

        log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mean_auc": mean_auc,
            "per_class_auc": per_class_auc,
        }
        print(json.dumps(log, indent=2))

        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best_model.pt"))
            with open(metrics_path, "w") as f:
                json.dump({"best_mean_auc": mean_auc, "per_class_auc": per_class_auc, "epoch": epoch + 1}, f, indent=2)

    return metrics_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory with images")
    parser.add_argument("--labels_csv", required=True, help="CSV with columns image,N,D,G,C,A,H,M,O")
    parser.add_argument("--retfound_weights", required=True, help="Path to RETFound ViT-Large weights")
    parser.add_argument("--output_dir", default="outputs/retfound_odir/")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        labels_csv=args.labels_csv,
        retfound_weights=args.retfound_weights,
        output_dir=args.output_dir,
    )
    run_training(cfg)