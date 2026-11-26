import argparse, csv, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import build_model
from data_utils import paired_files, split_pairs, VesselDataset, get_device


def dice_score(prob, target, threshold=0.5):
    pred = prob >= threshold
    target = target >= 0.5
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return float(2 * (pred & target).sum() / denom)


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_dice, batches = 0.0, 0.0, 0
    bce = torch.nn.BCEWithLogitsLoss()
    for x, y, _ in loader:
        x, y = x.to("cpu"), y.to("cpu")
        with torch.set_grad_enabled(training):
            logits = model(x)
            probs = torch.sigmoid(logits)
            dims = (1, 2, 3)
            intersection = (probs * y).sum(dims)
            dice_loss = 1 - ((2 * intersection + 1e-6) /
                             (probs.sum(dims) + y.sum(dims) + 1e-6)).mean()
            loss = bce(logits, y) + dice_loss
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item()
        total_dice += dice_score(probs.detach().numpy() >= 0.5, y.numpy())
        batches += 1
    return total_loss / max(batches, 1), total_dice / max(batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unet", "attention_unet"], default="unet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--data", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    pairs = paired_files(args.data)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, args.seed)
    print(f"Cases: total={len(pairs)}, train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
    train_ds = VesselDataset(train_pairs, args.size, augment=True)
    val_ds = VesselDataset(val_pairs, args.size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(args.model).to(get_device())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    out = Path("checkpoints"); out.mkdir(exist_ok=True)
    checkpoint = out / f"{args.model}_best.pth"
    best_dice = -1
    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_dice = run_epoch(model, train_loader, optimizer)
        va_loss, va_dice = run_epoch(model, val_loader)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_dice": tr_dice,
                        "val_loss": va_loss, "val_dice": va_dice})
        print(f"Epoch {epoch:03d}/{args.epochs} | train loss {tr_loss:.4f} dice {tr_dice:.4f} | val loss {va_loss:.4f} dice {va_dice:.4f}")
        if va_dice > best_dice:
            best_dice = va_dice
            torch.save({"model": model.state_dict(), "model_name": args.model,
                        "size": args.size, "best_val_dice": best_dice}, checkpoint)
    with open(out / f"{args.model}_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    print(f"Best checkpoint saved to {checkpoint}; best validation Dice={best_dice:.4f}")
    print(f"Note: this script uses a deterministic image-level split. For multi-slice scans, split by patient before slice extraction.")


if __name__ == "__main__":
    main()
