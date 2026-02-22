"""
RetinaScope AI Prototype (with multi-disease RETFound head + evaluation)
------------------------------------------------------------------------
- Single-image inference: vessel features + 8-class fundus probabilities + Gemini summary
- Dataset evaluation: mean AUC and per-class AUC on ODIR-style labels.csv

Classes: ["N","D","G","C","A","H","M","O"]
"""

import argparse
import os
import warnings
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage import morphology
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings("ignore", category=UserWarning)


# add near the imports
from safetensors.torch import load_file
from safetensors.torch import load_file  # add near other imports

# add this helper somewhere above run_eval / run_single_image
def load_checkpoint_into_model(model: nn.Module, path: str, device: str):
    if path.endswith(".safetensors"):
        state = load_file(path, device=device)
    else:
        state = torch.load(path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[INFO] Loaded weights: {path} (missing={len(missing)}, unexpected={len(unexpected)})")

# in run_single_image, replace the weight-loading block with:
    if args.model_weights:
        if not os.path.isfile(args.model_weights):
            raise FileNotFoundError(args.model_weights)
        load_checkpoint_into_model(model, args.model_weights, device)

# in run_eval, replace the weight-loading block with:
    if args.model_weights:
        if not os.path.isfile(args.model_weights):
            raise FileNotFoundError(args.model_weights)
        load_checkpoint_into_model(model, args.model_weights, device)

# add a small helper somewhere above run_eval / run_single_image
def load_checkpoint_into_model(model: nn.Module, path: str, device: str):
    if path.endswith(".safetensors"):
        state = load_file(path, device=device)
    else:
        state = torch.load(path, map_location=device, weights_only=False)  # explicit False
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[INFO] Loaded weights: {path} (missing={len(missing)}, unexpected={len(unexpected)})")

# replace the inline torch.load(...) in both run_eval and run_single_image with:
# ...
    if args.model_weights:
        if not os.path.isfile(args.model_weights):
            raise FileNotFoundError(args.model_weights)
        load_checkpoint_into_model(model, args.model_weights, device)
# ...

# Optional imports
try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False

# Gemini (new or legacy)
try:
    from google import genai as genai_new  # google-genai >= 0.8
    _GENAI_AVAILABLE = True
    _GENAI_LEGACY = False
except ImportError:
    try:
        import google.generativeai as genai  # legacy
        _GENAI_AVAILABLE = True
        _GENAI_LEGACY = True
    except ImportError:
        _GENAI_AVAILABLE = False
        _GENAI_LEGACY = False

CLASSES = ["N","D","G","C","A","H","M","O"]

# -----------------------------------------------------------------------------
# Vision backbone + head
# -----------------------------------------------------------------------------
class RETFoundBackbone(nn.Module):
    """ViT backbone stand-in. Choose base or large via arch."""
    def __init__(self, arch: str = "vit_base_patch16_224"):
        super().__init__()
        if _TIMM_AVAILABLE:
            try:
                self.backbone = timm.create_model(
                    arch, pretrained=True, num_classes=0
                )
            except Exception:
                self.backbone = timm.create_model(
                    arch, pretrained=False, num_classes=0
                )
            self.embed_dim = self.backbone.num_features
        else:
            self.embed_dim = 768
            self.backbone = _MockBackbone(self.embed_dim)

        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class _MockBackbone(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(3 * 224 * 224, embed_dim)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class MultiDiseaseHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = len(CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RetinaVisionModel(nn.Module):
    """Backbone + vessel features + multi-disease head."""
    VESSEL_FEATURE_DIM = 3
    def __init__(self, arch: str = "vit_base_patch16_224"):
        super().__init__()
        self.backbone = RETFoundBackbone(arch=arch)
        total_dim = self.backbone.embed_dim + self.VESSEL_FEATURE_DIM
        self.head = MultiDiseaseHead(total_dim, num_classes=len(CLASSES))

    def forward(self, image: torch.Tensor, vessel_features: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(image)
        combined = torch.cat([emb, vessel_features], dim=-1)
        return self.head(combined)


# -----------------------------------------------------------------------------
# Vessel preprocessing
# -----------------------------------------------------------------------------
def apply_clahe(image_bgr: np.ndarray) -> np.ndarray:
    green = image_bgr[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(green)

def segment_vessels(enhanced_gray: np.ndarray) -> np.ndarray:
    thresh = cv2.adaptiveThreshold(
        enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=15, C=3,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    vessel_mask = morphology.remove_small_objects(opened.astype(bool), max_size=50)
    return vessel_mask

def extract_vessel_features(vessel_mask: np.ndarray) -> Dict[str, float]:
    total = vessel_mask.size
    vessel_pixels = vessel_mask.sum()
    vessel_density = float(vessel_pixels) / float(total)

    mask_uint8 = vessel_mask.astype(np.uint8) * 255
    dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    nz = dist[dist > 0]
    mean_width = float(2.0 * nz.mean()) if nz.size > 0 else 0.0

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours and vessel_pixels > 0:
        total_perim = sum(cv2.arcLength(c, closed=False) for c in contours)
        area = float(vessel_pixels)
        tortuosity = float(total_perim / (np.sqrt(4.0 * np.pi * area) + 1e-6))
    else:
        tortuosity = 0.0

    return {
        "vessel_density": vessel_density,
        "mean_width": mean_width,
        "tortuosity": tortuosity,
    }

def preprocess_image(image_path: Optional[str]) -> Tuple[torch.Tensor, Dict[str, float], np.ndarray]:
    if image_path and os.path.isfile(image_path):
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"cv2 could not read image at '{image_path}'")
    else:
        if image_path:
            print(f"[WARN] Image not found at '{image_path}'. Using synthetic demo.")
        image_bgr = _generate_synthetic_fundus()

    enhanced = apply_clahe(image_bgr)
    vessel_mask = segment_vessels(enhanced)
    vessel_feats = extract_vessel_features(vessel_mask)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    img_tensor = tfm(pil_image).unsqueeze(0)  # (1,3,224,224)
    return img_tensor, vessel_feats, image_bgr

def _generate_synthetic_fundus() -> np.ndarray:
    h, w = 512, 512
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(img, (w//2, h//2), 230, (30, 60, 120), -1)
    cv2.circle(img, (w//2 + 60, h//2), 30, (200, 210, 240), -1)
    rng = np.random.default_rng(seed=42)
    for _ in range(8):
        angle = rng.uniform(0, 2*np.pi)
        r = rng.uniform(50, 200)
        x1, y1 = int(w//2 + 60), int(h//2)
        x2, y2 = int(x1 + r*np.cos(angle)), int(y1 + r*np.sin(angle))
        thickness = rng.integers(1, 4)
        cv2.line(img, (x1, y1), (x2, y2), (20, 80, 160), int(thickness))
    noise = rng.integers(0, 15, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img

# -----------------------------------------------------------------------------
# Gemini + citations (same as before, minimal)
# -----------------------------------------------------------------------------
def _build_clinical_prompt(prob_dict: Dict[str, float], vessel_feats: Dict[str, float]) -> str:
    lines = [f"  • {k}: {v:.2%}" for k, v in prob_dict.items()]
    return (
        "You are an expert ophthalmologist reviewing multi-disease retinal AI outputs.\n"
        "Provide: (1) Clinician summary (2–3 sentences), "
        "(2) Patient-friendly summary (2–3 sentences), "
        "(3) Three follow-up actions.\n\n"
        "Probabilities:\n" + "\n".join(lines) + "\n"
        f"Vessel density: {vessel_feats['vessel_density']:.4f}, "
        f"Mean width: {vessel_feats['mean_width']:.2f}, "
        f"Tortuosity: {vessel_feats['tortuosity']:.4f}"
    )

def call_gemini(prob_dict: Dict[str, float], vessel_feats: Dict[str, float], api_key: Optional[str] = None) -> str:
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not resolved_key or not _GENAI_AVAILABLE:
        reason = (
            "google-genai not installed"
            if not _GENAI_AVAILABLE
            else "GEMINI_API_KEY not set"
        )
        print(f"[INFO] Gemini unavailable ({reason}). Using mock text.")
        return _mock_gemini(prob_dict, vessel_feats)
    prompt = _build_clinical_prompt(prob_dict, vessel_feats)
    try:
        if not _GENAI_LEGACY:
            client = genai_new.Client(api_key=resolved_key)  # type: ignore[name-defined]
            model_candidates = [
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
                "gemini-1.5-flash",
            ]
            last_error: Optional[Exception] = None
            for model_name in model_candidates:
                try:
                    resp = client.models.generate_content(model=model_name, contents=prompt)
                    return resp.text
                except Exception as exc:
                    last_error = exc
                    continue
            if last_error is not None:
                raise last_error
            raise RuntimeError("Gemini call failed for all model candidates.")
        else:
            genai.configure(api_key=resolved_key)  # type: ignore[name-defined]
            model = genai.GenerativeModel("gemini-1.5-flash")  # type: ignore[name-defined]
            resp = model.generate_content(prompt)
            return resp.text
    except Exception as exc:
        print(f"[WARN] Gemini call failed: {exc}. Using mock text.")
        return _mock_gemini(prob_dict, vessel_feats)

def _mock_gemini(prob_dict: Dict[str, float], vessel_feats: Dict[str, float]) -> str:
    return (
        "Clinician Summary: Model indicates retinal findings consistent with "
        f"risks: {prob_dict}. Vessel density {vessel_feats['vessel_density']:.4f}, "
        f"tortuosity {vessel_feats['tortuosity']:.4f}. Correlate clinically.\n"
        "Patient Summary: The AI saw patterns that might relate to eye or systemic "
        "conditions. Please discuss with your doctor.\n"
        "Suggested Follow-ups: 1) Full eye exam, 2) Blood pressure & labs, "
        "3) Cardiovascular risk assessment."
    )

def call_citations() -> str:
    return (
        "[1] Zhou Y. et al. (2023). A foundation model for generalizable disease detection from retinal images. Nature.\n"
        "[2] Poplin R. et al. (2018). Prediction of cardiovascular risk factors from retinal fundus photographs. Nat Biomed Eng.\n"
        "[3] Cheung C.Y. et al. (2021). Retinal-vessel calibre and CVD risk. Nat Biomed Eng."
    )

# -----------------------------------------------------------------------------
# Dataset for evaluation
# -----------------------------------------------------------------------------
class ODIRDataset(Dataset):
    def __init__(self, csv_path: str, img_root: str, image_size: int = 224):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        assert all(c in self.df.columns for c in ["image", *CLASSES])
        self.img_root = img_root
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row["image"])
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        img_t = self.transform(pil)
        label = torch.tensor(row[CLASSES].values.astype(np.float32))
        return img_t, label

# -----------------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            vessel_feats = torch.zeros((imgs.size(0), 3), device=device)  # no vessel feats in eval
            logits = model(imgs, vessel_feats)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    probs = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    aucs = []
    for i in range(labels.shape[1]):
        try:
            aucs.append(roc_auc_score(labels[:, i], probs[:, i]))
        except ValueError:
            aucs.append(float("nan"))
    mean_auc = float(np.nanmean(aucs))
    return mean_auc, aucs

# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def run_single_image(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== RetinaScope AI — Single Image ===\nDevice: {device}\nImage: {args.image or '[synthetic demo]'}\n")

    img_tensor, vessel_feats, _ = preprocess_image(args.image)
    img_tensor = img_tensor.to(device)
    vessel_tensor = torch.tensor([[
        vessel_feats["vessel_density"],
        vessel_feats["mean_width"],
        vessel_feats["tortuosity"],
    ]], dtype=torch.float32, device=device)

    model = RetinaVisionModel(arch=args.arch).to(device)
    if args.model_weights:
        if not os.path.isfile(args.model_weights):
            raise FileNotFoundError(args.model_weights)
        state = torch.load(args.model_weights, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"[INFO] Loaded weights: {args.model_weights}")

    model.eval()
    with torch.no_grad():
        probs = model(img_tensor, vessel_tensor).squeeze(0).cpu().numpy()

    prob_dict = {cls: float(p) for cls, p in zip(CLASSES, probs)}
    explanation = call_gemini(prob_dict, vessel_feats, api_key=args.gemini_api_key)
    citations = call_citations()

    print("\nProbabilities:")
    for k, v in prob_dict.items():
        print(f"  {k}: {v:.1%}")
    print("\nVessel features:")
    print(f"  density={vessel_feats['vessel_density']:.6f}, "
          f"mean_width={vessel_feats['mean_width']:.2f}, "
          f"tortuosity={vessel_feats['tortuosity']:.4f}")
    print("\nGemini explanation:\n", explanation)
    print("\nCitations:\n", citations)
    return prob_dict

# Add near the imports
from safetensors.torch import load_file

# add near the imports
from safetensors.torch import load_file

# add this helper (place above run_eval / run_single_image)
def load_checkpoint_into_model(model: nn.Module, path: str, device: str):
    if path.endswith(".safetensors"):
        state = load_file(path, device=device)
    else:
        state = torch.load(path, map_location=device, weights_only=False)  # explicit False
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[INFO] Loaded weights: {path} (missing={len(missing)}, unexpected={len(unexpected)})")

# in run_single_image, replace the weight-loading block with:
    if args.model_weights:
        if not os.path.isfile(args.model_weights):
            raise FileNotFoundError(args.model_weights)
        load_checkpoint_into_model(model, args.model_weights, device)

# in run_eval, replace the weight-loading block with:
    if args.model_weights:
        if not os.path.isfile(args.model_weights):
            raise FileNotFoundError(args.model_weights)
        load_checkpoint_into_model(model, args.model_weights, device)

def run_eval(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== RetinaScope AI — Evaluation ===\nDevice: {device}\nData: {args.data_dir}\nCSV: {args.labels_csv}\n")
    ds = ODIRDataset(args.labels_csv, args.data_dir, image_size=224)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = RetinaVisionModel(arch=args.arch).to(device)
    if args.model_weights:
        if not os.path.isfile(args.model_weights):
            raise FileNotFoundError(args.model_weights)
        state = torch.load(args.model_weights, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"[INFO] Loaded weights: {args.model_weights}")
    model.eval()

    mean_auc, aucs = evaluate(model, loader, device)
    print(f"\nMean AUC: {mean_auc:.4f}")
    for cls, auc in zip(CLASSES, aucs):
        print(f"  {cls}: {auc:.4f}")
    return mean_auc, aucs

def parse_args():
    p = argparse.ArgumentParser(description="RetinaScope AI — fundus inference + evaluation")
    p.add_argument("--image", type=str, default=None, help="Path to fundus image for single inference. Omit for synthetic demo.")
    p.add_argument("--gemini-api-key", type=str, default=None, help="Gemini API key (overrides env).")
    p.add_argument("--model-weights", type=str, default=None, help="Path to trained checkpoint (.pth).")
    p.add_argument("--arch", type=str, default="vit_base_patch16_224",
                   choices=["vit_base_patch16_224","vit_large_patch16_224"])
    p.add_argument("--device", type=str, default=None, choices=["cpu","cuda","mps"])
    # Eval options
    p.add_argument("--evaluate", action="store_true", help="Run dataset evaluation instead of single-image inference.")
    p.add_argument("--data-dir", type=str, default=None, help="Image root for evaluation.")
    p.add_argument("--labels-csv", type=str, default=None, help="CSV with columns: image,N,D,G,C,A,H,M,O")
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.evaluate:
        if not args.data_dir or not args.labels_csv:
            raise ValueError("Evaluation requires --data-dir and --labels-csv")
        run_eval(args)
    else:
        run_single_image(args)