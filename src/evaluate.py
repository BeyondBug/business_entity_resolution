import numpy as np
import pandas as pd

def f05_entity(pred_set, true_set):
    if not true_set:
        if not pred_set:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 1.0
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0:
        return 0.0, prec, rec
    f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
    return f05, prec, rec

def evaluate_predictions(predictions, ground_truth_df, verbose=True):
    gt_dict = {}
    for _, row in ground_truth_df.iterrows():
        s1_id = row["source1_entity_id"]
        if pd.isna(row["matched_entity_ids"]) or row["matched_entity_ids"] == "":
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(str(row["matched_entity_ids"]).split(","))

    all_f05, all_prec, all_rec = [], [], []
    singleton_correct = singleton_total = 0
    non_singleton_f05 = []
    tp_total = fp_total = fn_total = 0

    for s1_id, true_set in gt_dict.items():
        pred_set = set(predictions.get(s1_id, []))
        f05, prec, rec = f05_entity(pred_set, true_set)
        all_f05.append(f05)
        all_prec.append(prec)
        all_rec.append(rec)
        if not true_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1
        else:
            non_singleton_f05.append(f05)
            tp_total += len(pred_set & true_set)
            fp_total += len(pred_set - true_set)
            fn_total += len(true_set - pred_set)

    results = {
        "macro_f05":         round(float(np.mean(all_f05)), 6),
        "mean_precision":    round(float(np.mean(all_prec)), 6),
        "mean_recall":       round(float(np.mean(all_rec)), 6),
        "singleton_acc":     round(singleton_correct / singleton_total, 6) if singleton_total > 0 else None,
        "non_singleton_f05": round(float(np.mean(non_singleton_f05)), 6) if non_singleton_f05 else None,
        "n_entities":        len(all_f05),
        "n_singletons":      singleton_total,
        "singleton_correct": singleton_correct,
        "tp": tp_total, "fp": fp_total, "fn": fn_total,
    }

    if verbose:
        print("\n" + "="*60)
        print(f"  MACRO F0.5:          {results['macro_f05']:.6f}")
        print(f"  Mean Precision:      {results['mean_precision']:.6f}")
        print(f"  Mean Recall:         {results['mean_recall']:.6f}")
        if results["singleton_acc"] is not None:
            print(f"  Singleton Accuracy:  {results['singleton_acc']:.6f}  ({singleton_correct}/{singleton_total})")
        if results["non_singleton_f05"] is not None:
            print(f"  Non-Singleton F0.5: {results['non_singleton_f05']:.6f}")
        print(f"  TP={tp_total}  FP={fp_total}  FN={fn_total}")
        print(f"  Total Entities: {results['n_entities']} ({singleton_total} singletons)")
        print("="*60)

    return results

if __name__ == "__main__":
    print("Testing evaluate.py...")
    pred = {"S1-001": ["S2-001", "S3-001"]}
    gt   = pd.DataFrame([{"source1_entity_id": "S1-001", "matched_entity_ids": "S2-001,S3-001"}])
    r = evaluate_predictions(pred, gt, verbose=False)
    assert r["macro_f05"] == 1.0
    print("  Test 1 PASS: perfect match")

    pred = {"S1-002": []}
    gt   = pd.DataFrame([{"source1_entity_id": "S1-002", "matched_entity_ids": ""}])
    r = evaluate_predictions(pred, gt, verbose=False)
    assert r["macro_f05"] == 1.0
    print("  Test 2 PASS: singleton correct")

    pred = {"S1-002": ["S2-999"]}
    gt   = pd.DataFrame([{"source1_entity_id": "S1-002", "matched_entity_ids": ""}])
    r = evaluate_predictions(pred, gt, verbose=False)
    assert r["macro_f05"] == 0.0
    print("  Test 3 PASS: singleton false merge = 0.0")

    print("\nAll tests passed. evaluate.py is correct.")
