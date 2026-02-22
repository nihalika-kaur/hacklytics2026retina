"""
test_visual.py — RetinaScope AI Pipeline Visualiser
====================================================
Displays every stage of the retina.py processing pipeline in a single
matplotlib figure so you can SEE what the model is doing.
Usage:
    python test_visual.py                      # synthetic demo image
    python test_visual.py --image path/to/fundus.jpg
"""
import argparse
import sys
import cv2
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit(
        "[ERROR] matplotlib is not installed.\n"
        "Run: pip install matplotlib>=3.7.0"
    )
from retina import (
    _generate_synthetic_fundus,
    apply_clahe,
    extract_vessel_features,
    preprocess_image,
    segment_vessels,
    RetinaVisionModel,
)
import torch
def run_visual_test(image_path: str | None = None) -> None:
    """Run the pipeline and display each stage in a matplotlib figure."""
    # ------------------------------------------------------------------
    # 1. Load / generate the fundus image
    # ------------------------------------------------------------------
    if image_path and __import__("os").path.isfile(image_path):
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"[ERROR] cv2 could not read '{image_path}'. Using synthetic image.")
            image_bgr = _generate_synthetic_fundus()
    else:
        if image_path:
            print(f"[WARN] '{image_path}' not found. Using synthetic demo image.")
        else:
            print("[INFO] No image supplied — generating synthetic fundus demo.")
        image_bgr = _generate_synthetic_fundus()
    # ------------------------------------------------------------------
    # 2. Run pipeline stages
    # ------------------------------------------------------------------
    enhanced_gray = apply_clahe(image_bgr)
    vessel_mask = segment_vessels(enhanced_gray)
    vessel_feats = extract_vessel_features(vessel_mask)
    # Distance transform (used internally for mean width)
    mask_uint8 = vessel_mask.astype(np.uint8) * 255
    dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    # ------------------------------------------------------------------
    # 3. Run the vision model to get risk scores
    # ------------------------------------------------------------------
    img_tensor, _, _ = preprocess_image(image_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RetinaVisionModel().to(device)
    model.eval()
    vessel_tensor = torch.tensor(
        [[
            vessel_feats["vessel_density"],
            vessel_feats["mean_width"],
            vessel_feats["tortuosity"],
        ]],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        ht_prob, cvd_prob, bio_age = model(img_tensor.to(device), vessel_tensor)
    hypertension_risk = float(ht_prob.item())
    cvd_risk = float(cvd_prob.item())
    biological_age = max(20.0, min(90.0, abs(float(bio_age.item()))))
    # ------------------------------------------------------------------
    # 4. Build subplot 4 — vessel overlay
    # ------------------------------------------------------------------
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay = image_rgb.copy()
    # Paint detected vessels green
    overlay[vessel_mask] = [0, 220, 0]
    # ------------------------------------------------------------------
    # 5. Plot
    # ------------------------------------------------------------------
    print("\n[INFO] Opening figure: 'RetinaScope AI: Pipeline Visualization'")
    print("       Close the window to exit.\n")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("RetinaScope AI: Pipeline Visualization", fontsize=16, fontweight="bold")
    # (1,1) Original fundus image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original Fundus Image", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")
    # (1,2) CLAHE enhanced green channel
    axes[0, 1].imshow(enhanced_gray, cmap="gray")
    axes[0, 1].set_title("CLAHE Enhanced (Green Channel)", fontsize=13, fontweight="bold")
    axes[0, 1].axis("off")
    # (1,3) Vessel segmentation mask
    axes[0, 2].imshow(vessel_mask, cmap="hot")
    axes[0, 2].set_title("Vessel Segmentation Mask", fontsize=13, fontweight="bold")
    axes[0, 2].axis("off")
    # (2,1) Vessel overlay on original
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Vessel Overlay on Original", fontsize=13, fontweight="bold")
    axes[1, 0].axis("off")
    # (2,2) Distance transform
    axes[1, 1].imshow(dist_transform, cmap="magma")
    axes[1, 1].set_title("Distance Transform (Vessel Width)", fontsize=13, fontweight="bold")
    axes[1, 1].axis("off")
    # (2,3) Results summary — text only
    axes[1, 2].axis("off")
    axes[1, 2].set_title("Results Summary", fontsize=13, fontweight="bold")
    summary = (
        f"Hypertension Risk : {hypertension_risk:.1%}\n"
        f"CVD Risk          : {cvd_risk:.1%}\n"
        f"Biological Age    : {biological_age:.1f} yrs\n\n"
        f"Vessel Density    : {vessel_feats['vessel_density']:.6f}\n"
        f"Mean Width (px)   : {vessel_feats['mean_width']:.2f}\n"
        f"Tortuosity Index  : {vessel_feats['tortuosity']:.4f}"
    )
    axes[1, 2].text(
        0.05, 0.5,
        summary,
        transform=axes[1, 2].transAxes,
        fontsize=12,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )
    plt.tight_layout()
    plt.show()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="RetinaScope AI — Visual pipeline test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a retinal fundus image (JPEG/PNG). "
             "Omit to use a synthetic demo image.",
    )
    args = parser.parse_args()
    run_visual_test(args.image)
if __name__ == "__main__":
    main()