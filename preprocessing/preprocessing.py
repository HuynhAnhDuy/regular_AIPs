import pandas as pd
import numpy as np
from collections import Counter

# ========= 0. Cấu hình cơ bản =========
INPUT_CSV  = "cyclicpep_all_core.csv"
OUTPUT_CSV = "cyclicpep_all_core_clean.csv"
OUTPUT_FASTA = "cyclicpep_all_core_clean.fasta"
OUTPUT_NUMPY = "cyclicpep_all_core_X.npy"
OUTPUT_AA_MAP = "aa_mapping.npy"

# 20 aa chuẩn (nếu bạn muốn cho phép thêm, chỉ cần bổ sung thêm ký tự)
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# ========= 1. Đọc & chuẩn hoá chuỗi =========
df = pd.read_csv(INPUT_CSV)

# Giữ đúng 3 cột cần thiết, đổi tên cho nhất quán
df = df[["ID", "Name", "Sequence"]].copy()
df["Sequence"] = df["Sequence"].astype(str).str.strip().str.upper()

# Bỏ dòng không có sequence
df = df[df["Sequence"] != ""].reset_index(drop=True)

print("Số dòng ban đầu:", len(df))


# ========= 2. Kiểm tra amino acid lạ (non-canonical) =========
def noncanonical_chars(seq):
    return "".join(sorted(set([aa for aa in seq if aa not in VALID_AA])))

df["noncanonical"] = df["Sequence"].apply(noncanonical_chars)

print("\nThống kê ký tự lạ:")
print(df["noncanonical"].value_counts(dropna=False))

# Tuỳ chiến lược:
#  - Nếu bạn CHƯA muốn xử lý aa lạ (X, U, Z...), có thể loại bỏ cho giai đoạn đầu:
clean_df = df[df["noncanonical"] == ""].copy()
print("\nSau khi loại chuỗi có aa lạ, còn:", len(clean_df), "dòng")


# ========= 3. Gộp các mẫu trùng Sequence (giữ peptide core) =========
# Nhiều tên khác nhau nhưng cùng core -> gộp lại 1 dòng
agg_dict = {
    "ID":   lambda x: ";".join(sorted(set(map(str, x)))),
    "Name": lambda x: ";".join(sorted(set(map(str, x))))
}

core_df = clean_df.groupby("Sequence", as_index=False).agg(agg_dict)
core_df["n_variants"] = core_df["ID"].apply(lambda s: len(s.split(";")))
core_df["length"] = core_df["Sequence"].str.len()

print("\nSau khi gộp trùng sequence (giữ peptide core duy nhất):", len(core_df), "dòng")
print(core_df[["Sequence", "length", "n_variants"]])

# Lưu CSV clean cho step sau
core_df.to_csv(OUTPUT_CSV, index=False)
print("\nĐã lưu CSV clean ->", OUTPUT_CSV)


# ========= 4. Xuất FASTA cho phân tích khác / docking =========
with open(OUTPUT_FASTA, "w") as f:
    for _, row in core_df.iterrows():
        header = f">{row['ID']}|{row['Name']}|nvar={row['n_variants']}"
        f.write(header + "\n")
        f.write(row["Sequence"] + "\n")

print("Đã lưu FASTA ->", OUTPUT_FASTA)


# ========= 5. Mã hoá sequence thành số cho WGAN =========
# Alphabet & mapping
AA_ALPHABET = sorted(list(VALID_AA))  # cố định thứ tự
aa_to_idx = {aa: i+1 for i, aa in enumerate(AA_ALPHABET)}  # 0 dành cho PAD
idx_to_aa = {i+1: aa for i, aa in enumerate(AA_ALPHABET)}

print("\nBảng mã amino acid:")
print(aa_to_idx)

# Độ dài tối đa (bạn có thể set cứng nếu muốn)
MAX_LEN = core_df["length"].max()
print("Độ dài max:", MAX_LEN)

PAD_IDX = 0

def encode_seq(seq, max_len=MAX_LEN):
    """Trả về vector độ dài max_len, padding = 0."""
    arr = np.full(max_len, PAD_IDX, dtype=np.int64)
    for i, aa in enumerate(seq[:max_len]):
        arr[i] = aa_to_idx.get(aa, PAD_IDX)
    return arr

X = np.stack(core_df["Sequence"].apply(encode_seq).values)
print("Shape ma trận X:", X.shape)  # (n_seq, MAX_LEN)

# Lưu lại để dùng cho training WGAN
np.save(OUTPUT_NUMPY, X)
np.save(OUTPUT_AA_MAP, np.array([AA_ALPHABET], dtype=object))

print("Đã lưu ma trận X ->", OUTPUT_NUMPY)
print("Đã lưu alphabet ->", OUTPUT_AA_MAP)
