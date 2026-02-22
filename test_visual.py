"""
RetinaScope AI — Visual Test Script
====================================
Demonstrates each stage of the retinal fundus analysis pipeline using
matplotlib subplots so the user can visually inspect intermediate results.

Usage:
  python test_visual.py                    # uses synthetic demo image
  python test_visual.py --image fundus.jpg # uses a real fundus image
"""

import argparse
import os
import sys

import cv2
import matplotlib
import numpy as np

# Use non-interactive backend when no display is available (e.g., CI servers).
if os.environ.get("DISPLAY") is None and sys.platform != "darwin" and sys.platform != "win32":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from retina import (  # noqa: E402
    RetinaVisionModel,
    _generate_synthetic_fundus,
    apply_clahe,
    extract_vessel_features,
    segment_vessels,
)


def run_visual_test(image_path=None):
    """Run the visual pipeline test and display a 2×3 subplot figure."""

    # ------------------------------------------------------------------
    # Load or generate the fundus image
    # ------------------------------------------------------------------
    if image_path and os.path.isfile(image_path):
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"[ERROR] Could not read image at '{image_path}'.")
            sys.exit(1)
        source = image_path
    else:
        if image_path:
            print(f"[WARN] Image not found at '{image_path}'. Using synthetic demo image.")
        image_bgr = _generate_synthetic_fundus()
        source = "synthetic demo"

    print(f"\n{'='*60}")
    print("  RetinaScope AI — Visual Pipeline Test")
    print(f"{'='*60}")
    print(f"  Image source: {source}")
    print(f"  Image size  : {image_bgr.shape[1]}×{image_bgr.shape[0]}\n")

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------
    print("[1/5] Converting BGR → RGB for display…")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print("[2/5] Applying CLAHE enhancement…")
    enhanced = apply_clahe(image_bgr)

    print("[3/5] Segmenting vessels…")
    vessel_mask = segment_vessels(enhanced)

    print("[4/5] Extracting vessel features…")
    vessel_feats = extract_vessel_features(vessel_mask)

    print("[5/5] Running vision model inference…")
    # Prepare tensor for the model
    from torchvision import transforms
    from PIL import Image

    pil_image = Image.fromarray(image_rgb)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(pil_image).unsqueeze(0)

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
        ht_prob, cvd_prob, bio_age = model(img_tensor, vessel_tensor)

    hypertension_prob = float(ht_prob.item())
    cvd_probability = float(cvd_prob.item())
    biological_age = float(bio_age.item())
    biological_age = max(20.0, min(90.0, abs(biological_age)))

    # ------------------------------------------------------------------
    # Compute distance transform for visualization
    # ------------------------------------------------------------------
    mask_uint8 = vessel_mask.astype(np.uint8) * 255
    dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)

    # ------------------------------------------------------------------
    # Create vessel overlay on original image
    # ------------------------------------------------------------------
    overlay = image_rgb.copy()
    overlay[vessel_mask, 1] = np.clip(
        overlay[vessel_mask, 1].astype(np.int16) + 120, 0, 255
    ).astype(np.uint8)

    # ------------------------------------------------------------------
    # Build 2×3 subplot figure
    # ------------------------------------------------------------------
    print("\nGenerating visualization figure…")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("RetinaScope AI: Pipeline Visualization", fontsize=18, fontweight="bold")

    # (1,1) Original Fundus Image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original Fundus Image", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")

    # (1,2) CLAHE Enhanced (Green Channel)
    axes[0, 1].imshow(enhanced, cmap="gray")
    axes[0, 1].set_title("CLAHE Enhanced (Green Channel)", fontsize=13, fontweight="bold")
    axes[0, 1].axis("off")

    # (1,3) Vessel Segmentation Mask
    axes[0, 2].imshow(vessel_mask, cmap="hot")
    axes[0, 2].set_title("Vessel Segmentation Mask", fontsize=13, fontweight="bold")
    axes[0, 2].axis("off")

    # (2,1) Vessel Overlay on Original
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Vessel Overlay on Original", fontsize=13, fontweight="bold")
    axes[1, 0].axis("off")

    # (2,2) Distance Transform (Vessel Width)
    axes[1, 1].imshow(dist_transform, cmap="magma")
    axes[1, 1].set_title("Distance Transform (Vessel Width)", fontsize=13, fontweight="bold")
    axes[1, 1].axis("off")

    # (2,3) Results Summary (text-only)
    axes[1, 2].axis("off")
    axes[1, 2].set_title("Results Summary", fontsize=13, fontweight="bold")
    summary_text = (
        f"Hypertension risk : {hypertension_prob:.1%}\n"
        f"CVD risk          : {cvd_probability:.1%}\n"
        f"Biological age    : {biological_age:.1f} years\n"
        f"\n"
        f"Vessel density    : {vessel_feats['vessel_density']:.6f}\n"
        f"Mean width (px)   : {vessel_feats['mean_width']:.2f}\n"
        f"Tortuosity index  : {vessel_feats['tortuosity']:.4f}"
    )
    axes[1, 2].text(
        0.1, 0.5, summary_text,
        transform=axes[1, 2].transAxes,
        fontsize=12,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gray"),
    )

    plt.tight_layout()

    # Show or save depending on environment
    backend = matplotlib.get_backend().lower()
    if backend == "agg":
        out_path = "retinascope_visual_test.png"
        fig.savefig(out_path, dpi=150)
        print(f"\n[INFO] No display detected. Figure saved to '{out_path}'.")
    else:
        print("\n[INFO] Displaying figure — close the window to exit.")
        plt.show()

    print("\n✅ Visual test completed successfully!\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RetinaScope AI — Visual pipeline test",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a retinal fundus image (JPEG/PNG). "
             "If omitted, a synthetic demo image is used.",
    )
    args = parser.parse_args()
    run_visual_test(image_path=args.image)


if __name__ == "__main__":
    main()
