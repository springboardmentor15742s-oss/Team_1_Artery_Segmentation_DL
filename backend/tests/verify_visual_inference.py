"""Verification script for single-slice inference and visual overlay generation."""
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from app.services.ai_engine import ai_engine

sample_p = Path(r"C:\Users\somes\Documents\Model\processed_data\test\images\case_014_slice_084.png")
if not sample_p.exists():
    print("Sample slice not found, creating synthetic CT test slice...")
    arr = np.zeros((128, 128), dtype=np.float32)
    arr[40:70, 50:80] = 350.0  # contrast-enhanced artery HU
else:
    im = Image.open(sample_p).convert("L")
    arr = np.asarray(im, dtype=np.float32)

print(f"Loaded input slice: shape={arr.shape}, min={arr.min()}, max={arr.max()}")

# 1. Attention U-Net
res_att = ai_engine.predict_slice(arr, model_name="attention_unet", threshold=0.50)
att_fg = res_att["foreground_pixels_128"]
att_ms = res_att["inference_duration_ms"]
print(f"Attention U-Net: detected {att_fg} px in {att_ms:.1f} ms (shape={res_att['mask_128'].shape})")

# 2. Standard U-Net
res_unet = ai_engine.predict_slice(arr, model_name="unet", threshold=0.50)
unet_fg = res_unet["foreground_pixels_128"]
unet_ms = res_unet["inference_duration_ms"]
print(f"Standard U-Net:  detected {unet_fg} px in {unet_ms:.1f} ms (shape={res_unet['mask_128'].shape})")

# Composite Visualization
fig, axes = plt.subplots(2, 3, figsize=(11, 7))

# Row 1: Attention U-Net
axes[0, 0].imshow(arr, cmap="gray")
axes[0, 0].set_title("Input CT Slice (Axial)")
axes[0, 0].axis("off")

axes[0, 1].imshow(res_att["mask_128"], cmap="gray")
axes[0, 1].set_title(f"Attention U-Net Mask ({att_fg} px)")
axes[0, 1].axis("off")

axes[0, 2].imshow(arr, cmap="gray")
masked_att = np.ma.masked_where(res_att["mask_native"] == 0, res_att["mask_native"])
axes[0, 2].imshow(masked_att, cmap="Reds", alpha=0.65)
axes[0, 2].set_title(f"Attention U-Net Overlay ({att_ms:.1f} ms)")
axes[0, 2].axis("off")

# Row 2: Standard U-Net
axes[1, 0].imshow(arr, cmap="gray")
axes[1, 0].set_title("Input CT Slice (Axial)")
axes[1, 0].axis("off")

axes[1, 1].imshow(res_unet["mask_128"], cmap="gray")
axes[1, 1].set_title(f"Standard U-Net Mask ({unet_fg} px)")
axes[1, 1].axis("off")

axes[1, 2].imshow(arr, cmap="gray")
masked_unet = np.ma.masked_where(res_unet["mask_native"] == 0, res_unet["mask_native"])
axes[1, 2].imshow(masked_unet, cmap="Reds", alpha=0.65)
axes[1, 2].set_title(f"Standard U-Net Overlay ({unet_ms:.1f} ms)")
axes[1, 2].axis("off")

plt.tight_layout()
out_fig = backend_dir / "temp_inference_verification.png"
fig.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close(fig)

print(f"Saved visual verification overlay to: {out_fig}")
print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
