import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT

print("="*60)
print("AMAZON ML CHALLENGE 2026 — EDA REPORT")
print("="*60)

s1 = pd.read_csv(TRAIN_S1, sep="\t")
s2 = pd.read_csv(TRAIN_S2, sep="\t")
s3 = pd.read_csv(TRAIN_S3, sep="\t")
gt = pd.read_csv(TRAIN_GT, sep="\t")
gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

print(f"\n── Dataset Sizes ──────────────────────────")
print(f"  S1 (reference):  {len(s1):,} entities")
print(f"  S2 (noisy copy): {len(s2):,} entities")
print(f"  S3 (noisy copy): {len(s3):,} entities")
print(f"  Ground truth:    {len(gt):,} rows")

print(f"\n── Singleton Analysis ─────────────────────")
singletons = gt[gt["matched_entity_ids"] == ""]
non_singletons = gt[gt["matched_entity_ids"] != ""]
print(f"  Singletons (no match): {len(singletons):,} ({100*len(singletons)/len(gt):.1f}%)")
print(f"  With matches:          {len(non_singletons):,} ({100*len(non_singletons)/len(gt):.1f}%)")

print(f"\n── Match Cardinality ──────────────────────")
gt["match_count"] = gt["matched_entity_ids"].apply(lambda x: len(x.split(",")) if x else 0)
for cnt in sorted(gt["match_count"].unique())[:10]:
    n = (gt["match_count"] == cnt).sum()
    print(f"  {cnt} matches: {n:,} entities ({100*n/len(gt):.1f}%)")

print(f"\n── Country Distribution ───────────────────")
for src, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    print(f"  {src}: {dict(df['country'].value_counts().head(10))}")

print(f"\n── Missing Values ─────────────────────────")
for src, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    mn = df["business_name"].isna().sum()
    ma = df["business_address"].isna().sum()
    mc = df["country"].isna().sum()
    print(f"  {src}: name={mn} ({100*mn/len(df):.1f}%)  addr={ma} ({100*ma/len(df):.1f}%)  country={mc} ({100*mc/len(df):.1f}%)")

print(f"\n── Name / Address Lengths ─────────────────")
for src, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    nl = df["business_name"].dropna().str.len()
    al = df["business_address"].dropna().str.len()
    print(f"  {src} name len: mean={nl.mean():.0f} median={nl.median():.0f}")
    print(f"  {src} addr len: mean={al.mean():.0f} median={al.median():.0f}")

print(f"\n{'='*60}")
print("EDA COMPLETE")
print("="*60)
