# predict.py
import pandas as pd
import numpy as np
import pickle, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from normalize import apply_normalization
from blocking import build_candidate_pairs
from features import compute_features, FEATURE_NAMES
from train import get_entity_probs, apply_decision

def main():
    print("Loading test data...")
    s1 = apply_normalization(pd.read_csv(TEST_S1, sep="\t"))
    s2 = apply_normalization(pd.read_csv(TEST_S2, sep="\t"))
    s3 = apply_normalization(pd.read_csv(TEST_S3, sep="\t"))

    countries = set(s1["country"].dropna().str.lower().unique())
    print(f"Countries in test S1: {countries}")

    print("Loading model...")
    with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "idf_name.pkl"), "rb") as f:
        idf_name = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "idf_addr.pkl"), "rb") as f:
        idf_addr = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        cfg = json.load(f)

    st = cfg["singleton_threshold"]
    at = cfg["accept_threshold"]
    print(f"Thresholds: singleton={st}, accept={at}")

    print("Building test candidates...")
    candidates = build_candidate_pairs(s1, s2, s3)

    target_df     = pd.concat([s2, s3], ignore_index=True)
    s1_lookup     = {r["entity_id"]: r for _, r in s1.iterrows()}
    target_lookup = {r["entity_id"]: r for _, r in target_df.iterrows()}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    matching_rows  = []
    candidate_rows = []

    print("Running inference...")
    for _, s1_row in s1.iterrows():
        s1_id    = s1_row["entity_id"]
        cand_ids = list(candidates.get(s1_id, set()))
        ep = get_entity_probs(model, s1_id, cand_ids, s1_lookup, target_lookup, idf_name, idf_addr)
        matched = apply_decision(ep, st, at)
        matching_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": ",".join(matched)})
        candidate_rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": ",".join(cand_ids)})

    mr_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    cp_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    pd.DataFrame(matching_rows).to_csv(mr_path, sep="\t", index=False)
    pd.DataFrame(candidate_rows).to_csv(cp_path, sep="\t", index=False)

    total      = len(matching_rows)
    with_match = sum(1 for r in matching_rows if r["matched_entity_ids"])
    singletons = total - with_match
    print(f"\nOutput written:")
    print(f"  {mr_path}")
    print(f"  {cp_path}")
    print(f"  Total S1: {total}  |  With matches: {with_match}  |  Singletons: {singletons}")
    print(f"\nNext step — run validator:")
    print(f"  cd .. && python student_resource/utils/validate_submission.py --matching business_entity_resolution/output/matching_results.tsv --candidate business_entity_resolution/output/candidate_pairs.tsv --test-dir student_resource/dataset/test")

if __name__ == "__main__":
    main()