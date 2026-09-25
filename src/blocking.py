# blocking.py — HIGH RECALL VERSION
import pandas as pd
import numpy as np
from collections import defaultdict
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from normalize import apply_normalization, get_tokens, get_numbers

MAX_POSTINGS     = 2000  # raised from 500 — catches more true matches
MIN_TOKEN_OVERLAP_BLOCKING = 1   # lower than config — blocking needs high recall
MIN_BIGRAM_LEN   = 4     # minimum bigram character length to index

def get_char_bigrams(text, min_len=4):
    """Character bigrams of length 4 from normalized name."""
    if not text or len(text) < min_len:
        return []
    return [text[i:i+min_len] for i in range(len(text) - min_len + 1)]

def build_candidate_pairs(s1_df, s2_df, s3_df):
    target_df = pd.concat([s2_df, s3_df], ignore_index=True)
    candidates = {eid: set() for eid in s1_df["entity_id"]}

    print("  Building inverted indexes...", flush=True)
    t0 = time.time()
    exact_idx    = defaultdict(list)
    name_tok_idx = defaultdict(list)
    addr_tok_idx = defaultdict(list)
    num_idx      = defaultdict(list)
    bigram_idx   = defaultdict(list)  # NEW: char-4gram index for typo matching

    for _, row in target_df.iterrows():
        eid = row["entity_id"]
        nm  = row.get("name_norm", "") or ""

        # Rule 1: exact
        if nm:
            exact_idx[nm].append(eid)

        # Rule 2: name tokens
        for tok in get_tokens(row.get("name_tokens_str", "")):
            if len(tok) >= MIN_TOKEN_LEN:
                name_tok_idx[tok].append(eid)

        # Rule 3: address tokens
        for tok in get_tokens(row.get("addr_tokens_str", "")):
            if len(tok) >= MIN_ADDR_TOKEN_LEN:
                addr_tok_idx[tok].append(eid)

        # Rule 4: numeric
        for num in get_numbers(row.get("addr_numeric_str", "")):
            if len(num) >= MIN_NUM_LEN:
                num_idx[num].append(eid)

        # Rule 5: char-4gram on name (catches typos like "Ponr" vs "Power")
        for bg in get_char_bigrams(nm, MIN_BIGRAM_LEN):
            bigram_idx[bg].append(eid)

    # Prune high-frequency entries
    name_tok_idx = {k: v for k, v in name_tok_idx.items() if len(v) <= MAX_POSTINGS}
    addr_tok_idx = {k: v for k, v in addr_tok_idx.items() if len(v) <= MAX_POSTINGS}
    num_idx      = {k: v for k, v in num_idx.items()      if len(v) <= MAX_POSTINGS}
    bigram_idx   = {k: v for k, v in bigram_idx.items()   if len(v) <= MAX_POSTINGS}

    print(f"    Done in {time.time()-t0:.1f}s", flush=True)
    print(f"    name_tok={len(name_tok_idx):,}  addr_tok={len(addr_tok_idx):,}  bigram={len(bigram_idx):,}", flush=True)
    gc.collect()

    print("  Scanning S1 entities...", flush=True)
    t0 = time.time()
    s1_list = list(s1_df.iterrows())

    for i, (_, s1_row) in enumerate(s1_list):
        if i % 50000 == 0:
            print(f"    {i:,}/{len(s1_list):,}  ({time.time()-t0:.0f}s)", flush=True)

        s1_id = s1_row["entity_id"]
        nm    = s1_row.get("name_norm", "") or ""
        cands = candidates[s1_id]

        # Rule 1: Exact name
        for eid in exact_idx.get(nm, []):
            cands.add(eid)

        # Rule 2: Name token overlap >= 1 (lowered for recall)
        tok_hits = {}
        for tok in get_tokens(s1_row.get("name_tokens_str", "")):
            if len(tok) >= MIN_TOKEN_LEN:
                posting = name_tok_idx.get(tok)
                if posting:
                    for eid in posting:
                        tok_hits[eid] = tok_hits.get(eid, 0) + 1
        for eid, cnt in tok_hits.items():
            if cnt >= MIN_TOKEN_OVERLAP_BLOCKING:
                cands.add(eid)

        # Rule 3: Address token
        for tok in get_tokens(s1_row.get("addr_tokens_str", "")):
            if len(tok) >= MIN_ADDR_TOKEN_LEN:
                posting = addr_tok_idx.get(tok)
                if posting:
                    for eid in posting:
                        cands.add(eid)

        # Rule 4: Numeric
        for num in get_numbers(s1_row.get("addr_numeric_str", "")):
            if len(num) >= MIN_NUM_LEN:
                posting = num_idx.get(num)
                if posting:
                    for eid in posting:
                        cands.add(eid)

        # Rule 5: Char-4gram (typo tolerance — catches "Ponr" vs "Power")
        bg_hits = {}
        for bg in get_char_bigrams(nm, MIN_BIGRAM_LEN):
            posting = bigram_idx.get(bg)
            if posting:
                for eid in posting:
                    bg_hits[eid] = bg_hits.get(eid, 0) + 1
        # Need >= 3 matching 4-grams to add as candidate
        for eid, cnt in bg_hits.items():
            if cnt >= 3:
                cands.add(eid)

    total = sum(len(v) for v in candidates.values())
    print(f"\n  TOTAL candidate pairs: {total:,}")
    print(f"  S1 with >= 1 candidate: {sum(1 for v in candidates.values() if v):,}")
    print(f"  Predicted singletons:   {sum(1 for v in candidates.values() if not v):,}")
    return candidates


def measure_candidate_recall(candidates, gt_df):
    found = total = 0
    missed_pairs = []
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        if pd.isna(row["matched_entity_ids"]) or row["matched_entity_ids"] == "":
            continue
        true_matches = set(row["matched_entity_ids"].split(","))
        total += len(true_matches)
        cands  = candidates.get(s1_id, set())
        missed = true_matches - cands
        found += len(true_matches) - len(missed)
        for m in missed:
            missed_pairs.append((s1_id, m))
    recall = found / total if total > 0 else 0.0
    print(f"\n{'='*55}")
    print(f"  CANDIDATE RECALL: {recall:.4f}  ({found}/{total})")
    if recall < 0.90:
        print("  WARNING: Below 90% — fix blocking before training.")
    elif recall < 0.95:
        print("  CAUTION: Below 95%.")
    else:
        print("  Strong candidate recall.")
    print("="*55)
    return recall, missed_pairs


if __name__ == "__main__":
    t_start = time.time()

    print("Loading data...", flush=True)
    s1 = pd.read_csv(TRAIN_S1, sep="\t")
    s2 = pd.read_csv(TRAIN_S2, sep="\t")
    s3 = pd.read_csv(TRAIN_S3, sep="\t")
    gt = pd.read_csv(TRAIN_GT, sep="\t")
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

    sample_ids = set(s1["entity_id"].sample(frac=0.20, random_state=42).tolist())
    s1_sample  = s1[s1["entity_id"].isin(sample_ids)].reset_index(drop=True)
    gt_sample  = gt[gt["source1_entity_id"].isin(sample_ids)]

    print(f"\nNormalizing {len(s1_sample):,} S1 + S2 + S3...", flush=True)
    s1_s = apply_normalization(s1_sample)
    s2_n = apply_normalization(s2)
    s3_n = apply_normalization(s3)
    del s1, s2, s3
    gc.collect()

    print(f"\nBuilding candidates for {len(s1_s):,} S1 entities...", flush=True)
    candidates = build_candidate_pairs(s1_s, s2_n, s3_n)
    recall, missed = measure_candidate_recall(candidates, gt_sample)

    if missed:
        print(f"\nFirst 5 missed pairs:")
        target_df = pd.concat([s2_n, s3_n]).set_index("entity_id")
        s1_idx    = s1_s.set_index("entity_id")
        for s1_id, tgt_id in missed[:5]:
            if s1_id in s1_idx.index and tgt_id in target_df.index:
                print(f"  S1:  {s1_idx.loc[s1_id]['business_name']!r}")
                print(f"  TGT: {target_df.loc[tgt_id]['business_name']!r}")

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")