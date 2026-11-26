import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from models import build_model
from data_utils import paired_files, split_pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["unet", "attention_unet"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="data")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    size = int(ckpt.get("size", 256))
    model = build_model(args.model)
    model.load_state_dict(ckpt["model"])
    model.eval()
    pairs = paired_files(args.data)
    _, _, test_pairs = split_pairs(pairs, 42)
    pred_dir, overlay_dir = Path("outputs/predictions"), Path("outputs/overlays")
    pred_dir.mkdir(parents=True, exist_ok=True); overlay_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for image_path, _ in test_pairs:
            im = Image.open(image_path).convert("L")
            original = np.asarray(im, dtype=np.uint8)
            resized = im.resize((size, size), Image.Resampling.BILINEAR)
            x = torch.from_numpy(np.asarray(resized, dtype=np.float32)[None, None] / 255.0)
            prob = torch.sigmoid(model(x))[0, 0].numpy()
            mask = (prob >= args.threshold).astype(np.uint8) * 255
            Image.fromarray(mask).save(pred_dir / f"{image_path.stem}_mask.png")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(np.asarray(resized), cmap="gray")
            ax.imshow(mask, cmap="Reds", alpha=0.35)
            ax.axis("off"); fig.tight_layout()
            fig.savefig(overlay_dir / f"{image_path.stem}_overlay.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved {len(test_pairs)} test masks to {pred_dir} and overlays to {overlay_dir}")


if __name__ == "__main__":
    main()
