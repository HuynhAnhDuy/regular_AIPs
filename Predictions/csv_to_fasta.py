import pandas as pd

# ====== cấu hình ======
CSV_PATH = "regularpep_AIPs_generated.csv"
FASTA_PATH = "regularpep_AIPs_generated.fasta"

ID_COL = "ID"
SEQ_COL = "Sequence"

# ====== load csv ======
df = pd.read_csv(CSV_PATH)

# ====== kiểm tra cột ======
missing = [c for c in [ID_COL, SEQ_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Thiếu cột trong CSV: {missing}. Cột hiện có: {list(df.columns)}")

# ====== làm sạch & lọc ======
df = df[[ID_COL, SEQ_COL]].copy()
df[ID_COL] = df[ID_COL].astype(str).str.strip()
df[SEQ_COL] = df[SEQ_COL].astype(str).str.strip().str.upper()

# bỏ dòng rỗng
df = df[(df[ID_COL] != "") & (df[SEQ_COL] != "")]

# (tuỳ chọn) chỉ giữ amino acid chuẩn + X (nếu bạn muốn)
# allowed = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")  # tùy bạn
# df = df[df[SEQ_COL].apply(lambda s: set(s).issubset(allowed))]

# ====== ghi fasta ======
with open(FASTA_PATH, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        seq_id = row[ID_COL]
        seq = row[SEQ_COL]
        f.write(f">{seq_id}\n{seq}\n")

print(f"Done. Wrote {len(df)} sequences to {FASTA_PATH}")
