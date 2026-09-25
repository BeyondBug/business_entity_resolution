# features.py — UPDATED to use token strings (compatible with fast normalize.py)
import numpy as np
import math
from rapidfuzz import fuzz
from collections import defaultdict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import get_tokens, get_numbers


def token_jaccard(a, b):
    a, b = set(a or []), set(b or [])
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return len(a & b) / len(a | b)

def ngram_jaccard(a, b):
    # ngrams no longer computed — returns 0.0 gracefully
    a, b = set(a or []), set(b or [])
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return len(a & b) / len(a | b)

def len_ratio(s1, s2):
    l1, l2 = len(s1 or ""), len(s2 or "")
    if max(l1, l2) == 0: return 1.0
    return min(l1, l2) / max(l1, l2)

def idf_weighted_overlap(tokens_a, tokens_b, idf_dict):
    a, b = set(tokens_a or []), set(tokens_b or [])
    common = a & b
    if not common: return 0.0
    return sum(idf_dict.get(t, 1.0) for t in common)

def build_idf(token_str_list):
    """
    Build IDF from a list of pipe-separated token strings
    e.g. ["hotel|india|ltd", "raj|traders", ...]
    """
    doc_count = defaultdict(int)
    total = len(token_str_list)
    for tok_str in token_str_list:
        if not tok_str:
            continue
        for t in set(tok_str.split("|")):
            doc_count[t] += 1
    return {t: math.log(total / (1 + c)) for t, c in doc_count.items()}


FEATURE_NAMES = [
    "name_exact", "name_lev", "name_jaro", "name_token_jac",
    "name_tok_overlap", "name_ngram3_jac", "name_token_sort",
    "name_len_ratio", "name_prefix_match", "name_digit_overlap",
    "name_rare_overlap", "name_tfidf_cos",
    "addr_lev", "addr_token_jac", "addr_tok_overlap", "addr_ngram3_jac",
    "addr_len_ratio", "addr_num_match", "addr_rare_overlap",
    "addr_tfidf_cos", "addr_empty_flag", "addr_prefix_match",
    "country_exact", "country_both_known",
    "both_high_sim", "name_high_addr_low", "max_field_sim",
    "source_pair", "field_agree_score"
]
N_FEATURES = len(FEATURE_NAMES)


def compute_features(s1_row, tgt_row, idf_name, idf_addr,
                     name_tfidf_sim=0.0, addr_tfidf_sim=0.0):
    """
    Compute all 29 pairwise features for one S1 <-> candidate pair.
    Works with the fast normalize.py output (token strings, no ngram columns).
    """
    # ── Pull raw strings ─────────────────────────────────────────
    n1 = s1_row.get("name_norm", "") or ""
    n2 = tgt_row.get("name_norm", "") or ""
    a1 = s1_row.get("addr_norm", "") or ""
    a2 = tgt_row.get("addr_norm", "") or ""
    c1 = s1_row.get("country_norm", "") or ""
    c2 = tgt_row.get("country_norm", "") or ""

    # ── Pull token lists from pipe-strings ───────────────────────
    nt1 = get_tokens(s1_row.get("name_tokens_str", ""))
    nt2 = get_tokens(tgt_row.get("name_tokens_str", ""))
    at1 = get_tokens(s1_row.get("addr_tokens_str", ""))
    at2 = get_tokens(tgt_row.get("addr_tokens_str", ""))
    num1 = set(get_numbers(s1_row.get("addr_numeric_str", "")))
    num2 = set(get_numbers(tgt_row.get("addr_numeric_str", "")))

    # ngrams not computed in fast mode — empty sets → features return 0
    nn1, nn2 = set(), set()
    an1, an2 = set(), set()

    # ── NAME features ────────────────────────────────────────────
    name_exact      = float(n1 == n2 and n1 != "")
    name_lev        = fuzz.ratio(n1, n2) / 100.0
    name_jaro       = fuzz.WRatio(n1, n2) / 100.0
    name_token_jac  = token_jaccard(nt1, nt2)
    name_tok_ovlp   = float(len(set(nt1) & set(nt2)))
    name_ngram3_jac = ngram_jaccard(nn1, nn2)   # 0.0 in fast mode (no ngrams)
    name_token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_len_r      = len_ratio(n1, n2)
    name_prefix     = float(len(n1) >= 3 and len(n2) >= 3 and n1[:3] == n2[:3])
    d1 = set(c for c in n1 if c.isdigit())
    d2 = set(c for c in n2 if c.isdigit())
    name_digit_ovlp = float(bool(d1 & d2)) if (d1 or d2) else 1.0
    name_rare       = idf_weighted_overlap(nt1, nt2, idf_name)

    # ── ADDRESS features ─────────────────────────────────────────
    addr_lev        = fuzz.ratio(a1, a2) / 100.0
    addr_token_jac  = token_jaccard(at1, at2)
    addr_tok_ovlp   = float(len(set(at1) & set(at2)))
    addr_ngram3_jac = ngram_jaccard(an1, an2)   # 0.0 in fast mode
    addr_len_r      = len_ratio(a1, a2)
    ln1 = {n for n in num1 if len(n) >= 3}
    ln2 = {n for n in num2 if len(n) >= 3}
    addr_num_match  = float(bool(ln1 & ln2)) if (ln1 or ln2) else 1.0
    addr_rare       = idf_weighted_overlap(at1, at2, idf_addr)
    addr_empty      = float((not a1) or (not a2))
    addr_prefix     = float(bool(at1 and at2 and at1[0] == at2[0]))

    # ── COUNTRY features ─────────────────────────────────────────
    country_exact      = float(c1 == c2 and c1 != "")
    country_both_known = float(c1 != "" and c2 != "")

    # ── CROSS-FIELD features ─────────────────────────────────────
    name_sim        = max(name_lev, name_token_jac, name_token_sort)
    addr_sim        = max(addr_lev, addr_token_jac) if not addr_empty else 0.0
    both_high       = float(name_sim > 0.80 and addr_sim > 0.65)
    name_hi_addr_lo = float(name_sim > 0.80 and addr_sim < 0.25)
    max_field_sim   = max(name_sim, addr_sim)
    source_pair     = float(str(tgt_row.get("entity_id", "")).startswith("S3"))
    field_agree     = 0.5 * name_sim + 0.4 * addr_sim + 0.1 * country_exact

    return [
        name_exact, name_lev, name_jaro, name_token_jac,
        name_tok_ovlp, name_ngram3_jac, name_token_sort,
        name_len_r, name_prefix, name_digit_ovlp,
        name_rare, name_tfidf_sim,
        addr_lev, addr_token_jac, addr_tok_ovlp, addr_ngram3_jac,
        addr_len_r, addr_num_match, addr_rare,
        addr_tfidf_sim, addr_empty, addr_prefix,
        country_exact, country_both_known,
        both_high, name_hi_addr_lo, max_field_sim,
        source_pair, field_agree
    ]


if __name__ == "__main__":
    # Quick sanity check — run: python features.py
    print("Testing features.py...")

    idf_name = build_idf(["hotel|india|ltd", "raj|traders", "hotel|california"])
    idf_addr = build_idf(["mg|road|bangalore", "5th|avenue|new|york"])

    row1 = {
        "entity_id": "S1-001",
        "name_norm": "hotel california limited",
        "name_tokens_str": "california|hotel|limited",
        "addr_norm": "5th avenue new york",
        "addr_tokens_str": "5th|avenue|new|york",
        "addr_numeric_str": "",
        "country_norm": "us",
    }
    row2 = {
        "entity_id": "S2-001",
        "name_norm": "hotel california ltd",
        "name_tokens_str": "california|hotel|limited",
        "addr_norm": "5th ave new york",
        "addr_tokens_str": "5th|avenue|new|york",
        "addr_numeric_str": "",
        "country_norm": "us",
    }
    feats = compute_features(row1, row2, idf_name, idf_addr)
    assert len(feats) == N_FEATURES, f"Expected {N_FEATURES} features, got {len(feats)}"
    assert feats[0] == 0.0   # name_exact: different (ltd vs limited)
    assert feats[1] > 0.8    # name_lev: very similar
    assert feats[3] > 0.8    # name_token_jac: same tokens
    print(f"  Feature count: {len(feats)} — PASS")
    print(f"  name_exact={feats[0]}, name_lev={feats[1]:.2f}, name_token_jac={feats[3]:.2f}")
    print(f"  country_exact={feats[22]}, both_high={feats[24]}")
    print("\nfeatures.py OK — all checks passed")