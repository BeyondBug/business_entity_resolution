# normalize.py — VECTORIZED VERSION (handles 10M rows fast)
import re
import unicodedata
import pandas as pd
import numpy as np

NAME_ABBREVS = {
    "pvt": "private", "ltd": "limited", "corp": "corporation",
    "inc": "incorporated", "co": "company", "&": "and",
    "mfg": "manufacturing", "intl": "international", "mgmt": "management",
    "assoc": "associates", "dept": "department", "univ": "university",
    "hosp": "hospital", "natl": "national", "inst": "institute",
    "svcs": "services", "svc": "service", "grp": "group",
    "ent": "enterprises", "enterp": "enterprises",
}

ADDR_ABBREVS = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "ln": "lane", "dr": "drive", "ct": "court", "pl": "place",
    "sq": "square", "hwy": "highway", "pkwy": "parkway",
    "apt": "apartment", "ste": "suite", "bldg": "building",
}

# Build regex patterns once (much faster than token-by-token replace)
def _build_abbrev_pattern(abbrevs):
    escaped = {re.escape(k): v for k, v in abbrevs.items()}
    pattern = r'\b(' + '|'.join(escaped.keys()) + r')\b'
    return re.compile(pattern)

_NAME_PATTERN = _build_abbrev_pattern(NAME_ABBREVS)
_ADDR_PATTERN = _build_abbrev_pattern(ADDR_ABBREVS)

def _norm_series(series, pattern):
    """Vectorized normalization of an entire pandas Series."""
    s = series.fillna("")
    # Unicode normalize + ascii encode (vectorized via str methods where possible)
    s = s.str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("ascii")
    s = s.str.lower()
    s = s.str.replace(r"[^\w\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    # Abbreviation expansion — must be per-row but fast with compiled regex
    s = s.apply(lambda x: pattern.sub(lambda m: {re.escape(k): v for k, v in 
                (NAME_ABBREVS if pattern is _NAME_PATTERN else ADDR_ABBREVS).items()}
                .get(re.escape(m.group(0)), m.group(0)), x) if x else x)
    return s

def _fast_norm(series, abbrevs, pattern):
    """Faster: vectorize everything except abbrev substitution."""
    s = series.fillna("").astype(str)
    s = s.str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("ascii")
    s = s.str.lower()
    s = s.str.replace(r"[^\w\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    # Abbrev: apply only if there's something to expand
    if abbrevs:
        keys = list(abbrevs.keys())
        vals = list(abbrevs.values())
        for k, v in zip(keys, vals):
            s = s.str.replace(r'\b' + re.escape(k) + r'\b', v, regex=True)
    return s

def apply_normalization(df):
    """
    Vectorized normalization — no row-by-row apply for the heavy ops.
    Skips ngram columns (too slow at 10M rows, not needed for blocking).
    Uses token lists as pipe-separated strings for memory efficiency.
    """
    df = df.copy()
    print(f"    Normalizing names ({len(df):,} rows)...", flush=True)
    df["name_norm"] = _fast_norm(df["business_name"], NAME_ABBREVS, _NAME_PATTERN)
    
    print(f"    Normalizing addresses...", flush=True)
    df["addr_norm"] = _fast_norm(df["business_address"], ADDR_ABBREVS, _ADDR_PATTERN)
    
    print(f"    Building token sets...", flush=True)
    # Store tokens as sorted pipe-separated string — fast, memory efficient
    df["name_tokens_str"] = df["name_norm"].str.split().apply(
        lambda x: "|".join(sorted(set(x))) if x else "")
    df["addr_tokens_str"] = df["addr_norm"].str.split().apply(
        lambda x: "|".join(sorted(set(x))) if x else "")
    
    # Extract numbers from address
    df["addr_numeric_str"] = df["business_address"].fillna("").str.findall(r'\d{4,}').apply(
        lambda x: "|".join(x) if x else "")
    
    df["country_norm"] = df["country"].fillna("").str.lower().str.strip()
    return df

def get_tokens(token_str):
    """Convert stored pipe-string back to list."""
    if not token_str:
        return []
    return token_str.split("|")

def get_numbers(num_str):
    if not num_str:
        return []
    return num_str.split("|")

if __name__ == "__main__":
    import sys, os, time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import TRAIN_S1
    t0 = time.time()
    df = pd.read_csv(TRAIN_S1, sep="\t")
    print(f"Loaded {len(df):,} rows")
    df = apply_normalization(df)
    print(f"Done in {time.time()-t0:.1f}s")
    print("Columns:", [c for c in df.columns if c not in ["entity_id","business_name","business_address","country"]])
    print("\nSample:")
    print("  name_raw: ", df.iloc[0]["business_name"])
    print("  name_norm:", df.iloc[0]["name_norm"])
    print("  name_tok: ", df.iloc[0]["name_tokens_str"])