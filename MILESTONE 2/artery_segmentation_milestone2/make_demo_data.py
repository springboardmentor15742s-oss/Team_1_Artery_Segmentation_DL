"""Create synthetic tube-like images/masks for a software smoke test only."""
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw


def main():
    random.seed(9)
    root = Path("data")
    im_dir, mask_dir = root / "images", root / "masks"
    im_dir.mkdir(parents=True, exist_ok=True); mask_dir.mkdir(parents=True, exist_ok=True)
    for i in range(20):
        size = 128
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        # Draw a main tube and a small branch.
        x0, y0 = random.randint(20, 35), random.randint(10, 30)
        x1, y1 = random.randint(85, 110), random.randint(90, 115)
        draw.line((x0, y0, x1, y1), fill=255, width=random.randint(5, 10))
        bx, by = random.randint(45, 75), random.randint(45, 80)
        draw.line((bx, by, min(size-5, bx+random.randint(20, 35)), max(5, by-random.randint(15, 30))),
                  fill=255, width=random.randint(3, 6))
        m = np.asarray(mask, dtype=np.uint8)
        noise = np.random.default_rng(i).normal(15, 12, (size, size))
        img = np.clip(noise + (m > 0) * 110, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(im_dir / f"case_{i+1:03d}.png")
        mask.save(mask_dir / f"case_{i+1:03d}.png")
    print("Created 20 synthetic demo pairs in data/. For software testing only, not medical data.")


if __name__ == "__main__":
    main()
