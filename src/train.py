# train.py — updated for fast normalize (token strings)
import numpy as np
import pandas as pd
import pickle, json, os, sys, time
from collections import defaultdict
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from normalize import apply_normalization, get_tokens, get_numbers
from blocking import build_candidate_pairs, measure_candidate_recall
from features import compute_features, FEATURE_NAMES, build_idf
from evaluate import evaluate_predictions


def build_pairwise_dataset(candidates, gt_dict, s1_lookup, target_lookup,
                           idf_name, idf_addr, hard_negatives=False):
    X, y = [], []
    pos_count = neg_count = 0
    for s1_id, cand_ids in candidates.items():
        s1_row = s1_lookup.get(s1_id)
        if s1_row is None:
            continue
        true_ids = gt_dict.get(s1_id, set())
        for cid in cand_ids:
            tgt = target_lookup.get(cid)
            if tgt is None:
                continue
            feats = compute_features(s1_row, tgt, idf_name, idf_addr)
            label = int(cid in true_ids)
            X.append(feats)
            y.append(label)
            if label == 1:
                pos_count += 1
            else:
                neg_count += 1
                if hard_negatives:
                    name_jac = feats[FEATURE_NAMES.index("name_token_jac")]
                    if name_jac > HARD_NEG_THRESHOLD:
                        X.append(feats)
                        y.append(0)
                        neg_count += 1
    print(f"  Dataset: {len(y):,} pairs  pos={pos_count:,}  neg={neg_count:,}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train_xgboost(X, y):
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    spw = neg / pos if pos > 0 else 1.0
    print(f"  Training XGBoost: {len(y):,} pairs, SPW={spw:.2f}")
    base = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LR,
        scale_pos_weight=spw,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
        tree_method="hist",   # faster on large datasets
    )
    model = CalibratedClassifierCV(base, cv=3, method="isotonic")
    t0 = time.time()
    model.fit(X, y)
    print(f"  Training done in {time.time()-t0:.0f}s")
    return model


def get_entity_probs(model, s1_id, cand_ids, s1_lookup, target_lookup,
                     idf_name, idf_addr):
    s1_row = s1_lookup.get(s1_id)
    if s1_row is None or not cand_ids:
        return {}
    feats, valid = [], []
    for cid in cand_ids:
        tgt = target_lookup.get(cid)
        if tgt is None:
            continue
        feats.append(compute_features(s1_row, tgt, idf_name, idf_addr))
        valid.append(cid)
    if not feats:
        return {}
    probs = model.predict_proba(np.array(feats, dtype=np.float32))[:, 1]
    return dict(zip(valid, probs))


def apply_decision(entity_probs, singleton_threshold, accept_threshold):
    if not entity_probs:
        return []
    max_score = max(entity_probs.values())
    if max_score < singleton_threshold:
        return []
    return [cid for cid, p in entity_probs.items() if p >= accept_threshold]


def threshold_sweep(model, val_ids, val_candidates, s1_lookup, target_lookup,
                    idf_name, idf_addr, gt_val_df):
    print("\nPre-computing validation probabilities...", flush=True)
    all_probs = {}
    for i, s1_id in enumerate(val_ids):
        if i % 10000 == 0:
            print(f"  {i:,}/{len(val_ids):,}", flush=True)
        cands = list(val_candidates.get(s1_id, set()))
        all_probs[s1_id] = get_entity_probs(
            model, s1_id, cands, s1_lookup, target_lookup, idf_name, idf_addr)

    best_f05, best_st, best_at = 0.0, 0.35, 0.55
    print(f"\n{'ST':>5} {'AT':>5} {'F0.5':>8} {'Prec':>8} {'Rec':>8} {'SingAcc':>8}")
    print("-" * 52)

    for st in np.arange(0.20, 0.65, 0.05):
        for at in np.arange(0.35, 0.85, 0.05):
            if st >= at:
                continue
            preds = {s1_id: apply_decision(all_probs[s1_id], st, at)
                     for s1_id in val_ids}
            m = evaluate_predictions(preds, gt_val_df, verbose=False)
            f05 = m["macro_f05"]
            if f05 > best_f05:
                best_f05, best_st, best_at = f05, float(st), float(at)
            if f05 >= best_f05 - 0.005:
                sing = m.get("singleton_acc") or 0
                print(f"{st:5.2f} {at:5.2f} {f05:8.4f} "
                      f"{m['mean_precision']:8.4f} {m['mean_recall']:8.4f} {sing:8.4f}")

    print(f"\n  BEST: ST={best_st:.2f}  AT={best_at:.2f}  F0.5={best_f05:.6f}")
    return best_st, best_at, best_f05


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EXP_DIR, exist_ok=True)
    t_total = time.time()

    print("Loading training data...", flush=True)
    s1 = pd.read_csv(TRAIN_S1, sep="\t")
    s2 = pd.read_csv(TRAIN_S2, sep="\t")
    s3 = pd.read_csv(TRAIN_S3, sep="\t")
    gt = pd.read_csv(TRAIN_GT, sep="\t")
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

    # Entity-level train/val split
    all_ids = s1["entity_id"].tolist()
    train_ids, val_ids = train_test_split(
        all_ids, test_size=VAL_SIZE, random_state=RANDOM_SEED)
    train_set = set(train_ids)
    val_set   = set(val_ids)

    s1_train = s1[s1["entity_id"].isin(train_set)].reset_index(drop=True)
    s1_val   = s1[s1["entity_id"].isin(val_set)].reset_index(drop=True)
    gt_train = gt[gt["source1_entity_id"].isin(train_set)].reset_index(drop=True)
    gt_val   = gt[gt["source1_entity_id"].isin(val_set)].reset_index(drop=True)

    print(f"Split: {len(s1_train):,} train  {len(s1_val):,} val", flush=True)

    print("\nNormalizing data...", flush=True)
    s1_train_n = apply_normalization(s1_train)
    s1_val_n   = apply_normalization(s1_val)
    s2_n       = apply_normalization(s2)
    s3_n       = apply_normalization(s3)
    del s1, s2, s3
    gc.collect()

    target_df = pd.concat([s2_n, s3_n], ignore_index=True)

    print("\nBuilding row lookups...", flush=True)
    s1_all_n  = apply_normalization(pd.read_csv(TRAIN_S1, sep="\t"))
    s1_lookup     = {r["entity_id"]: r.to_dict() for _, r in s1_all_n.iterrows()}
    target_lookup = {r["entity_id"]: r.to_dict() for _, r in target_df.iterrows()}
    print(f"  s1_lookup={len(s1_lookup):,}  target_lookup={len(target_lookup):,}")
    del s1_all_n
    gc.collect()

    print("\nBuilding IDF...", flush=True)
    idf_name = build_idf(target_df["name_tokens_str"].fillna("").tolist())
    idf_addr = build_idf(target_df["addr_tokens_str"].fillna("").tolist())
    print(f"  idf_name vocab={len(idf_name):,}  idf_addr vocab={len(idf_addr):,}")

    print("\nBuilding TRAINING candidates...", flush=True)
    train_cands = build_candidate_pairs(s1_train_n, s2_n, s3_n)
    train_recall, _ = measure_candidate_recall(train_cands, gt_train)

    print("\nBuilding VALIDATION candidates...", flush=True)
    val_cands = build_candidate_pairs(s1_val_n, s2_n, s3_n)
    val_recall, _ = measure_candidate_recall(val_cands, gt_val)

    gt_train_dict = {}
    for _, row in gt_train.iterrows():
        gt_train_dict[row["source1_entity_id"]] = (
            set(row["matched_entity_ids"].split(","))
            if row["matched_entity_ids"] else set())

    # ── EXP-008: XGBoost baseline ────────────────────────────────
    print("\n" + "="*55)
    print("EXP-008: XGBoost (no hard negatives, threshold=0.50)")
    X1, y1 = build_pairwise_dataset(
        train_cands, gt_train_dict, s1_lookup, target_lookup,
        idf_name, idf_addr, hard_negatives=False)
    model_v1 = train_xgboost(X1, y1)
    del X1, y1
    gc.collect()

    preds_v1 = {s1_id: apply_decision(
        get_entity_probs(model_v1, s1_id,
                         list(val_cands.get(s1_id, set())),
                         s1_lookup, target_lookup, idf_name, idf_addr),
        0.35, 0.50)
        for s1_id in val_ids}
    m_v1 = evaluate_predictions(preds_v1, gt_val)
    print(f"EXP-008 RESULT: F0.5={m_v1['macro_f05']:.6f}")

    # ── EXP-009: Threshold sweep ─────────────────────────────────
    print("\n" + "="*55)
    print("EXP-009: Threshold sweep on v1")
    best_st, best_at, f09 = threshold_sweep(
        model_v1, val_ids, val_cands, s1_lookup, target_lookup,
        idf_name, idf_addr, gt_val)

    # ── EXP-010/011: Hard negatives ──────────────────────────────
    print("\n" + "="*55)
    print("EXP-010/011: XGBoost with hard negatives + sweep")
    X2, y2 = build_pairwise_dataset(
        train_cands, gt_train_dict, s1_lookup, target_lookup,
        idf_name, idf_addr, hard_negatives=True)
    model_v2 = train_xgboost(X2, y2)
    del X2, y2
    gc.collect()

    best_st2, best_at2, f11 = threshold_sweep(
        model_v2, val_ids, val_cands, s1_lookup, target_lookup,
        idf_name, idf_addr, gt_val)

    # ── Select best ───────────────────────────────────────────────
    if f11 >= f09:
        best_model, final_st, final_at, final_f = model_v2, best_st2, best_at2, f11
        print(f"\nChose EXP-011 (hard negatives): F0.5={f11:.6f}")
    else:
        best_model, final_st, final_at, final_f = model_v1, best_st, best_at, f09
        print(f"\nChose EXP-009 (no hard negatives): F0.5={f09:.6f}")

    # ── Save ──────────────────────────────────────────────────────
    print("\nSaving model...", flush=True)
    with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    with open(os.path.join(MODEL_DIR, "idf_name.pkl"), "wb") as f:
        pickle.dump(idf_name, f)
    with open(os.path.join(MODEL_DIR, "idf_addr.pkl"), "wb") as f:
        pickle.dump(idf_addr, f)

    cfg = {
        "singleton_threshold": float(final_st),
        "accept_threshold":    float(final_at),
        "val_f05":             float(final_f),
        "train_cand_recall":   float(train_recall),
        "val_cand_recall":     float(val_recall),
    }
    with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  DONE. Val F0.5 = {final_f:.6f}")
    print(f"  ST={final_st:.2f}  AT={final_at:.2f}")
    print(f"  Total time: {(time.time()-t_total)/60:.1f} min")
    print(f"  Run next: python src/predict.py")
    print("="*55)


if __name__ == "__main__":
    main()