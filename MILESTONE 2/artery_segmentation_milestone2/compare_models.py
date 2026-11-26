from pathlib import Path
import pandas as pd

root = Path("outputs/metrics")
paths = [root / "unet_summary.csv", root / "attention_unet_summary.csv"]
available = [p for p in paths if p.exists()]
if not available:
    raise SystemExit("No summary files found. Run evaluate.py for both models first.")
df = pd.concat([pd.read_csv(p) for p in available], ignore_index=True)
df = df.sort_values("dice", ascending=False)
root.mkdir(parents=True, exist_ok=True)
df.to_csv(root / "model_comparison.csv", index=False)
print(df.to_string(index=False))
print(f"\nSaved {root / 'model_comparison.csv'}")
print("Choose a model using validation performance first; treat test comparison as final reporting, not repeated tuning.")
