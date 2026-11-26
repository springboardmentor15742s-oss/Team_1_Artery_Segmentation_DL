import argparse, csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from models import build_model
from data_utils import paired_files, split_pairs


def metrics(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    dice = 1.0 if (pred.sum() + gt.sum()) == 0 else 2 * tp / (2 * tp + fp + fn)
    iou = 1.0 if (pred.sum() + gt.sum() - tp) == 0 else tp / (tp + fp + fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"dice": dice, "iou": iou, "precision": precision,
            "sensitivity": sensitivity, "specificity": specificity}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["unet", "attention_unet"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="data")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    size = int(ckpt.get("size", 256))
    model = build_model(args.model); model.load_state_dict(ckpt["model"]); model.eval()
    pairs = paired_files(args.data)
    _, _, test_pairs = split_pairs(pairs, 42)
    rows = []
    with torch.no_grad():
        for image_path, mask_path in test_pairs:
            im = Image.open(image_path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
            gt_im = Image.open(mask_path).convert("L").resize((size, size), Image.Resampling.NEAREST)
            x = torch.from_numpy(np.asarray(im, dtype=np.float32)[None, None] / 255.0)
            gt = np.asarray(gt_im) > 0
            pred = (torch.sigmoid(model(x))[0, 0].numpy() >= args.threshold)
            row = {"case": image_path.stem, **metrics(pred, gt)}
            rows.append(row)
    out = Path("outputs/metrics"); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.model}_per_case.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    metric_names = ["dice", "iou", "precision", "sensitivity", "specificity"]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in metric_names}
    summary["model"] = args.model
    with open(out / f"{args.model}_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model"] + metric_names); writer.writeheader(); writer.writerow(summary)
    print("Mean test metrics (per-case averages):")
    for k in metric_names: print(f"  {k}: {summary[k]:.4f}")
    print(f"Saved {path} and summary CSV. No boundary metrics are implemented in this starter.")


if __name__ == "__main__":
    main()
