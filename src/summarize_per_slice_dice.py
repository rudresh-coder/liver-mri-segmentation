import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
csv_path = PROJECT_ROOT / "checkpoints" / "per_slice_dice.csv"

df = pd.read_csv(csv_path)

summary = df.groupby("model")["dice"].agg(
    mean_dice="mean",
    std_dev="std",
)

summary["mean_dice"] = summary["mean_dice"].round(4)
summary["std_dev"] = summary["std_dev"].round(3)

summary_csv = PROJECT_ROOT / "checkpoints" / "per_slice_dice_summary.csv"
summary.to_csv(summary_csv)
print(f"Saved summary CSV to: {summary_csv}")

print(summary)