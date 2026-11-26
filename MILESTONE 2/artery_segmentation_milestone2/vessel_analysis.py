import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mask", required=True, help="Path to binary mask PNG")
    p.add_argument("--min-size", type=int, default=10)
    args = p.parse_args()
    arr = np.asarray(Image.open(args.mask).convert("L")) > 0
    labels, n = ndi.label(arr)
    counts = np.bincount(labels.ravel())
    keep = counts >= args.min_size
    keep[0] = False
    cleaned = keep[labels]
    skeleton = skeletonize(cleaned)
    distance = ndi.distance_transform_edt(cleaned)
    diameter_px = 2.0 * distance[skeleton]
    out = Path("outputs/analysis"); out.mkdir(parents=True, exist_ok=True)
    Image.fromarray((cleaned * 255).astype(np.uint8)).save(out / "cleaned_mask.png")
    Image.fromarray((skeleton * 255).astype(np.uint8)).save(out / "centerline.png")
    values = {
        "mask": args.mask,
        "foreground_pixels": int(cleaned.sum()),
        "connected_components_kept": int(keep.sum() - 1 if len(keep) else 0),
        "skeleton_pixels": int(skeleton.sum()),
        "mean_diameter_estimate_pixels": float(np.mean(diameter_px)) if diameter_px.size else 0.0,
        "median_diameter_estimate_pixels": float(np.median(diameter_px)) if diameter_px.size else 0.0,
        "diameter_note": "Rough 2D pixel-width proxy only; not a physical or clinical diameter."
    }
    pd.DataFrame([values]).to_csv(out / "vessel_measurements.csv", index=False)
    print(values)
    print(f"Saved cleaned mask, centerline and CSV under {out}")


if __name__ == "__main__":
    main()
