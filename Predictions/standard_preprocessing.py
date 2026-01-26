"""
Pipeline chuẩn bị dữ liệu AIP cho mô hình (dùng CD-HIT 0.9):

Bước 1: Đọc AIP_primary_data.csv
        - Chuẩn hóa sequence: upper, bỏ khoảng trắng, chỉ giữ amino acid chuẩn.
        - Xóa các dòng có sequence trùng hoàn toàn nhưng label khác nhau (conflict).

Bước 2: Ghi tất cả (active + inactive) ra 1 FASTA chung.

Bước 3: Chạy CD-HIT với identity = 0.9 trên toàn bộ dataset
        - Gom các chuỗi tương đồng >= 90% vào cùng cluster.
        - Giữ lại 1 representative cho mỗi cluster (non-redundant across cả 2 class).

Bước 4: Tạo file CSV cuối:
        AIP_primary_data_CDhit_0p9.csv
        (dùng để train model và/hoặc chạy tiếp CD-HIT 40% giữa actives/inactives để kiểm tra).
"""

import os
import subprocess
import pandas as pd

# ========= CẤU HÌNH =========

INPUT_CSV = "AIP_merged_all_labeled_cleaned.csv"               # File gốc
OUTPUT_CSV = "AIP_CDhit_0p9.csv"    # File CSV sau CD-HIT 0.9

SEQ_COL = "Sequence"
LABEL_COL = "Label"
ID_COL = "ID"

# File FASTA chung và prefix output CD-HIT
COMBINED_FASTA = "AIP_all.fasta"
CDHIT_ALL_PREFIX = "AIP_all_cdhit_0.9"

# Đường dẫn binary CD-HIT (nếu đã có trong PATH thì chỉ cần tên)
CD_HIT_BIN = "cd-hit"

# Tham số CD-HIT
ALL_IDENTITY = 0.9   # 90% identity
ALL_N = 5            # với c >= 0.7, CD-HIT khuyến nghị n = 5

# Amino acid chuẩn (20 aa thường gặp)
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ========= HÀM TIỆN ÍCH =========

def run_command(cmd):
    """
    Chạy lệnh hệ thống bằng subprocess.
    Nếu lệnh fail, raise RuntimeError với stdout/stderr.
    """
    print(">>> Running command:")
    print("    " + " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        raise RuntimeError(f"Command failed with return code {result.returncode}: {' '.join(cmd)}")
    else:
        print("    Done.\n")


def write_fasta_from_df(df, fasta_path):
    """
    Ghi DataFrame (có cột ID_COL, SEQ_COL) ra file FASTA:
      >id
      SEQUENCE
    """
    with open(fasta_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            seq_id = str(row[ID_COL])
            seq = str(row[SEQ_COL])
            f.write(f">{seq_id}\n{seq}\n")
    print(f"[STEP 2] Saved FASTA: {fasta_path} (n={len(df)})")


def parse_cd_hit_clstr(clstr_path):
    """
    Parse file .clstr của CD-HIT.
    Trả về:
      - clusters: list các cluster, mỗi cluster là list các id (chuỗi).
      - representatives: set id là representative (dòng có dấu '*').
    """
    clusters = []
    current_cluster = []
    representatives = set()

    with open(clstr_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">Cluster"):
                # bắt đầu cluster mới
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = []
            else:
                # dòng sequence trong cluster, ví dụ:
                # "0   25aa, >ID123... *"
                try:
                    start = line.index(">") + 1
                    end = line.index("...", start)
                    seq_id = line[start:end]
                    current_cluster.append(seq_id)

                    if line.endswith("*"):
                        representatives.add(seq_id)
                except ValueError:
                    continue

        # thêm cluster cuối cùng
        if current_cluster:
            clusters.append(current_cluster)

    print(f"[STEP 4] Parsed {len(clusters)} clusters from {clstr_path}")
    print(f"[STEP 4] Found {len(representatives)} representative sequences.")

    # In thêm thống kê kích thước cluster
    cluster_sizes = [len(c) for c in clusters]
    if cluster_sizes:
        print(f"[STEP 4] Cluster size stats: min={min(cluster_sizes)}, "
              f"max={max(cluster_sizes)}, avg={sum(cluster_sizes)/len(cluster_sizes):.2f}")
    else:
        print("[STEP 4] No clusters found (unexpected).")

    return clusters, representatives


def is_valid_aa_sequence(seq: str) -> bool:
    """
    Trả về True nếu sequence chỉ chứa amino acid chuẩn trong VALID_AA.
    """
    return all(aa in VALID_AA for aa in seq)


# ========= PIPELINE CHÍNH =========

def main():
    # 1. Đọc file CSV gốc
    print("=== STEP 1: Reading and cleaning input CSV ===")
    print(f"[STEP 1] Reading input CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # Kiểm tra cột bắt buộc
    required_cols = {ID_COL, SEQ_COL, LABEL_COL}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Input CSV must contain columns: {required_cols}. "
            f"Found: {df.columns.tolist()}"
        )

    print(f"[STEP 1] Original rows: {len(df)}")
    print("[STEP 1] Original label distribution:")
    print(df[LABEL_COL].value_counts(dropna=False))

    # Bỏ các dòng thiếu sequence hoặc label
    before_drop_na = len(df)
    df = df.dropna(subset=[SEQ_COL, LABEL_COL])
    print(f"[STEP 1] Dropped {before_drop_na - len(df)} rows with NaN in sequence/label.")
    print(f"[STEP 1] Rows after dropping NaN: {len(df)}")

    # Chuẩn hóa sequence: upper, bỏ khoảng trắng, tab
    df[SEQ_COL] = (
        df[SEQ_COL]
        .astype(str)
        .str.replace(" ", "")
        .str.replace("\t", "")
        .str.strip()
        .str.upper()
    )

    print("[STEP 1] Label distribution after basic cleaning:")
    print(df[LABEL_COL].value_counts(dropna=False))

    # 1b. Giữ lại chỉ các sequence có amino acid chuẩn
    mask_valid = df[SEQ_COL].apply(is_valid_aa_sequence)
    invalid_count = (~mask_valid).sum()
    if invalid_count > 0:
        print(f"[STEP 1b] Dropping {invalid_count} rows with non-standard amino acids.")
        # In thử vài sequence không chuẩn
        print("[STEP 1b] Example invalid sequences:")
        print(df.loc[~mask_valid, SEQ_COL].head().to_string(index=False))

    df = df[mask_valid].copy()
    print(f"[STEP 1b] Rows after filtering non-standard AA: {len(df)}")
    print("[STEP 1b] Label distribution after filtering AA:")
    print(df[LABEL_COL].value_counts(dropna=False))

    # 1c. Xóa sequence xuất hiện ở cả 2 class (conflict về nhãn)
    print("\n=== STEP 1c: Removing sequences that appear in both classes (label conflict) ===")
    seq_label_counts = df.groupby(SEQ_COL)[LABEL_COL].nunique()
    conflict_seqs = seq_label_counts[seq_label_counts > 1].index.tolist()

    print(f"[STEP 1c] Number of sequences with conflicting labels (0 & 1): {len(conflict_seqs)}")

    if len(conflict_seqs) > 0:
        before_conflict = len(df)
        df = df[~df[SEQ_COL].isin(conflict_seqs)].copy()
        print(f"[STEP 1c] Dropped {before_conflict - len(df)} rows due to label conflicts.")
    else:
        print("[STEP 1c] No conflicting sequences found.")

    print(f"[STEP 1c] Rows after dropping conflicting sequences: {len(df)}")
    print("[STEP 1c] Label distribution after removing conflicts:")
    print(df[LABEL_COL].value_counts(dropna=False))

    # 2. Ghi tất cả (active + inactive) ra 1 FASTA chung
    print("\n=== STEP 2: Writing combined FASTA (all sequences) ===")
    write_fasta_from_df(df, COMBINED_FASTA)

    # 3. Chạy cd-hit trên toàn bộ dataset với c = 0.9
    print("\n=== STEP 3: Running cd-hit on all sequences (90% identity) ===")
    print(f"[STEP 3] Input FASTA: {COMBINED_FASTA}")
    print(f"[STEP 3] Identity threshold: {ALL_IDENTITY}, word length: {ALL_N}")
    cmd_all = [
        CD_HIT_BIN,
        "-i", COMBINED_FASTA,
        "-o", CDHIT_ALL_PREFIX,
        "-c", str(ALL_IDENTITY),
        "-n", str(ALL_N),
        "-d", "0",  # giữ header đầy đủ
    ]
    run_command(cmd_all)
    print(f"[STEP 3] CD-HIT output prefix: {CDHIT_ALL_PREFIX}")

    # 4. Phân tích file .clstr để lấy representative
    print("\n=== STEP 4: Parsing clusters and representatives from CD-HIT output ===")
    clstr_all_path = CDHIT_ALL_PREFIX + ".clstr"
    if not os.path.exists(clstr_all_path):
        raise FileNotFoundError(f"Cannot find {clstr_all_path}. Check cd-hit output.")

    clusters, representatives = parse_cd_hit_clstr(clstr_all_path)

    # 5. Xây DataFrame cuối: chỉ giữ representative (non-redundant)
    print("\n=== STEP 5: Building final non-redundant DataFrame ===")
    before_final = len(df)
    df_final = df[df[ID_COL].astype(str).isin(representatives)].copy()
    after_final = len(df_final)

    print(f"[STEP 5] Rows before CD-HIT filtering: {before_final}")
    print(f"[STEP 5] Rows after keeping representatives only: {after_final}")
    print(f"[STEP 5] Number of sequences removed by CD-HIT redundancy filtering: {before_final - after_final}")
    print("[STEP 5] Final label distribution:")
    print(df_final[LABEL_COL].value_counts(dropna=False))

    # 6. Ghi ra CSV cuối cùng dùng để train / chạy kiểm tra CD-HIT 40% sau này
    df_final.to_csv(OUTPUT_CSV, index=False)

    # In tổng số mẫu ở file cuối cùng
    total_samples = len(df_final)
    print("\n=== DONE ===")
    print(f"[FINAL] Final cleaned dataset saved to: {OUTPUT_CSV}")
    print(f"[FINAL] Total samples in final CSV ({OUTPUT_CSV}): {total_samples}")
    print("[FINAL] You can now run your CD-HIT 40% inter-class check on this CSV.")


if __name__ == "__main__":
    main()
