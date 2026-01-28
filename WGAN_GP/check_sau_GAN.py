import pandas as pd
import re

REG_FILE = "regularpep_AIPs_generated.csv"
WGAN_FILE = "AIP_merged_all_WGAN_GP_pos.csv"
OUT_FILE = "regularpep_AIPs_generated_clean.csv"

SEQ_COL_CANDIDATES = ["sequence", "seq", "Sequence", "SEQ", "peptide", "Peptide", "aa_seq", "AA_SEQ"]

def detect_seq_col(df, candidates=SEQ_COL_CANDIDATES):
    for c in candidates:
        if c in df.columns:
            return c
    # fallback heuristic: choose object column most like AA strings
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not obj_cols:
        raise ValueError("Không tìm thấy cột dạng text để làm sequence.")
    best, best_score = None, -1
    for c in obj_cols:
        s = df[c].astype(str).head(200)
        score = s.str.match(r"^[A-Za-z\s\-]+$").mean()
        if score > best_score:
            best_score = score
            best = c
    return best

def normalize_seq(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    s = re.sub(r"\s+", "", s)      # remove whitespace
    s = s.replace("-", "")         # remove gaps
    s = re.sub(r"[^A-Z]", "", s)   # keep only A-Z
    return s

# =========================
# LOAD
# =========================
reg = pd.read_csv(REG_FILE)
wgan = pd.read_csv(WGAN_FILE)

reg_seq_col = detect_seq_col(reg)
wgan_seq_col = detect_seq_col(wgan)

# =========================
# NORMALIZE
# =========================
reg = reg.copy()
wgan = wgan.copy()

reg["seq_norm"] = reg[reg_seq_col].apply(normalize_seq)
wgan["seq_norm"] = wgan[wgan_seq_col].apply(normalize_seq)

# drop empty sequences (after normalize)
reg_nonempty = reg[reg["seq_norm"] != ""].copy()
wgan_nonempty = wgan[wgan["seq_norm"] != ""].copy()

# =========================
# 1) DEDUP WITHIN regularpep
# =========================
dup_within_mask = reg_nonempty.duplicated(subset=["seq_norm"], keep="first")
n_within_dup_rows_removed = int(dup_within_mask.sum())
reg_unique = reg_nonempty[~dup_within_mask].copy()  # keep first occurrence

# =========================
# 2) REMOVE OVERLAP WITH WGAN
# =========================
wgan_set = set(wgan_nonempty["seq_norm"].unique())
overlap_mask = reg_unique["seq_norm"].isin(wgan_set)
n_overlap_removed = int(overlap_mask.sum())

reg_clean = reg_unique[~overlap_mask].copy()

# =========================
# SAVE (ONLY ONE OUTPUT)
# =========================
reg_clean_out = reg_clean.drop(columns=["seq_norm"], errors="ignore")
reg_clean_out.to_csv(OUT_FILE, index=False)

# =========================
# PRINT SUMMARY
# =========================
print("============= CLEANING SUMMARY =============")
print(f"REG_FILE : {REG_FILE}")
print(f"WGAN_FILE: {WGAN_FILE}  (read-only; unchanged)")
print(f"Detected seq column (REG) : {reg_seq_col}")
print(f"Detected seq column (WGAN): {wgan_seq_col}")
print("-------------------------------------------")
print(f"[REG] total rows                         : {len(reg)}")
print(f"[REG] empty/invalid seq after normalize  : {int((reg['seq_norm']=='').sum())}")
print(f"[REG] non-empty rows used                : {len(reg_nonempty)}")
print("-------------------------------------------")
print(f"[STEP 1] removed within-file duplicates : {n_within_dup_rows_removed}")
print(f"[STEP 1] remaining unique sequences     : {len(reg_unique)}")
print("-------------------------------------------")
print(f"[STEP 2] removed overlap vs WGAN         : {n_overlap_removed}")
print(f"[OUTPUT] clean rows saved                : {len(reg_clean)}")
print("-------------------------------------------")
print(f"Saved: {OUT_FILE}")
print("===========================================")
