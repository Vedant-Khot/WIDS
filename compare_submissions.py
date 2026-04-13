import pandas as pd

print("=== Current submission.csv (v3) ===")
v3 = pd.read_csv(r"d:\wids\submission.csv")
print(v3["prob_72h"].describe())
print(f"Unique values: {v3['prob_72h'].nunique()}")

print()
print("=== submission1.csv (v2) ===")
v2 = pd.read_csv(r"d:\wids\submission1.csv")
print(v2["prob_72h"].describe())
print(f"Unique values: {v2['prob_72h'].nunique()}")

print()
print("=== Full column comparison (all horizons) ===")
for col in ["prob_12h", "prob_24h", "prob_48h", "prob_72h"]:
    print(f"\n--- {col} ---")
    print(f"  v2: mean={v2[col].mean():.4f}, std={v2[col].std():.4f}, min={v2[col].min():.4f}, max={v2[col].max():.4f}, unique={v2[col].nunique()}")
    print(f"  v3: mean={v3[col].mean():.4f}, std={v3[col].std():.4f}, min={v3[col].min():.4f}, max={v3[col].max():.4f}, unique={v3[col].nunique()}")

print()
print("=== Side-by-side prob_72h (first 15 rows) ===")
comp = pd.DataFrame({
    "event_id": v2["event_id"],
    "v2_72h": v2["prob_72h"],
    "v3_72h": v3["prob_72h"]
})
print(comp.head(15).to_string())
