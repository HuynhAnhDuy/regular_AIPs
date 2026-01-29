#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline inter-class cho AIP dataset:

Input:
    - AIP_merged_all_labeled_cdhit90.csv  (có cột 'Sequence', 'Label')

Bước:
    1) Gắn orig_index cho từng hàng (ổn định index gốc).
    2) Xuất 2 FASTA cho CD-HIT-2D:
         - class1_for_cd-hit-2d.fasta (Label 1)
         - class0_for_cd-hit-2d.fasta (Label 0)
    3) Chạy cd-hit-2d (bên trong script).
    4) Đọc .clstr từ cd-hit-2d, loại bỏ các peptide Label 0
       trong cluster có cả 0 và 1.
       -> AIP_interclass_filtered.csv

Chạy:
    python interclass_check.py
"""

import os
import subprocess
import sys
from typing import List

import pandas as pd

# =========================
# 1. CONFIG
# =========================

# File input gốc
INPUT_CSV = "AIP_merged_all_labeled_cdhit90.csv"

# Tên cột
SEQ_COL = "Sequence"   # cột chứa chuỗi peptide
LABEL_COL = "Label"    # cột chứa nhãn 0/1

# Binary cd-hit-2d
CDHIT2D_BIN = "cd-hit-2d"

# File trung gian / output cho inter-class filter
FASTA_LABEL1 = "class1_for_cd-hit-2d.fasta"
FASTA_LABEL0 = "class0_for_cd-hit-2d.fasta"
CDHIT2D_PREFIX = "pos_vs_neg_cdhit80"
CDHIT2D_CLSTR = CDHIT2D_PREFIX + ".clstr"
FILTERED_CSV = "AIP_interclass_filtered.csv"

# Tham số logic
INTERCLASS_IDENTITY = 0.8      # cd-hit-2d -c
DROP_LABEL_WHEN_CONFLICT = 0   # nếu cluster có cả 0 và 1 -> bỏ label này (bỏ Label 0)


# =========================
# 2. Helper functions
# =========================

def ensure_orig_index(df: pd.DataFrame) -> pd.DataFrame:
    """Đảm bảo có cột 'orig_index' lưu index gốc."""
    if "orig_index" not in df.columns:
        df = df.copy()
        df["orig_index"] = df.index
    return df


def make_fasta_id(label: int, orig_idx: int) -> str:
    """Tạo ID cho FASTA, encode luôn Label và orig_index."""
    return f"L{label}_{orig_idx}"


def parse_fasta_id(fasta_id: str):
    """Giải mã ID FASTA dạng 'L0_123' -> (label, orig_index)."""
    try:
        left, right = fasta_id.split("_")
        label = int(left[1:])  # bỏ 'L'
        orig_idx = int(right)
        return label, orig_idx
    except Exception:
        return None, None


def parse_cdhit_clstr(path: str) -> List[list]:
    """
    Đọc file .clstr của CD-HIT / CD-HIT-2D.
    Trả về list các cluster, mỗi cluster là list các fasta_id (không có '>').
    """
    clusters = []
    current = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                if current:
                    clusters.append(current)
                    current = []
            else:
                # Ví dụ: "0   94aa, >L1_5... *"
                if ">" in line:
                    after = line.split(">", 1)[1]
                    fasta_id = after.split("...", 1)[0].strip()
                    current.append(fasta_id)

    if current:
        clusters.append(current)

    return clusters


def run_cmd(cmd: list, desc: str):
    """Chạy lệnh external và in log."""
    print(f"\n[RUN] {desc}")
    print("      " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Lệnh thất bại: {e}", file=sys.stderr)
        sys.exit(1)


# =========================
# 3. Pipeline inter-class
# =========================

def export_fastas_for_interclass(df: pd.DataFrame):
    """Xuất 2 FASTA cho cd-hit-2d (Label 1 và Label 0)."""
    df1 = df[df[LABEL_COL] == 1]
    df0 = df[df[LABEL_COL] == 0]

    print(f"  - Số mẫu Label 1: {len(df1)}")
    print(f"  - Số mẫu Label 0: {len(df0)}")

    # Ghi FASTA cho Label 1
    with open(FASTA_LABEL1, "w") as f1:
        for _, row in df1.iterrows():
            fasta_id = make_fasta_id(int(row[LABEL_COL]), int(row["orig_index"]))
            seq = str(row[SEQ_COL]).strip()
            if not seq:
                continue
            f1.write(f">{fasta_id}\n{seq}\n")

    # Ghi FASTA cho Label 0
    with open(FASTA_LABEL0, "w") as f0:
        for _, row in df0.iterrows():
            fasta_id = make_fasta_id(int(row[LABEL_COL]), int(row["orig_index"]))
            seq = str(row[SEQ_COL]).strip()
            if not seq:
                continue
            f0.write(f">{fasta_id}\n{seq}\n")

    print(f"  -> Đã ghi FASTA: {FASTA_LABEL1}, {FASTA_LABEL0}")


def run_cdhit2d():
    """Chạy cd-hit-2d để so khớp Label 0 vs 1."""
    cmd = [
        CDHIT2D_BIN,
        "-i", FASTA_LABEL1,       # database: Label 1
        "-i2", FASTA_LABEL0,      # query:   Label 0
        "-o", CDHIT2D_PREFIX,
        "-c", str(INTERCLASS_IDENTITY),
        "-n", "5",                # word size cho c >= 0.7
        "-d", "0",
        "-T", "4",
        "-M", "16000",
    ]
    run_cmd(cmd, "cd-hit-2d (inter-class, 0.8 identity)")


def apply_interclass_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Đọc .clstr từ cd-hit-2d, loại các peptide Label == DROP_LABEL_WHEN_CONFLICT
    trong các cluster chứa cả 0 & 1.
    """
    if not os.path.exists(CDHIT2D_CLSTR):
        print(f"[ERROR] Không tìm thấy file {CDHIT2D_CLSTR}. Hãy chắc chắn cd-hit-2d đã chạy xong.", file=sys.stderr)
        sys.exit(1)

    clusters = parse_cdhit_clstr(CDHIT2D_CLSTR)
    print(f"  - Số cluster trong CD-HIT-2D: {len(clusters)}")

    to_drop_orig_index = set()
    num_conflict_clusters = 0

    for cl in clusters:
        labels_in_cluster = set()
        ids_in_cluster = []

        for fasta_id in cl:
            label, orig_idx = parse_fasta_id(fasta_id)
            if label is None:
                continue
            labels_in_cluster.add(label)
            ids_in_cluster.append((label, orig_idx))

        # Cluster có cả 0 và 1 => conflict inter-class
        if 0 in labels_in_cluster and 1 in labels_in_cluster:
            num_conflict_clusters += 1
            for label, orig_idx in ids_in_cluster:
                if label == DROP_LABEL_WHEN_CONFLICT:
                    to_drop_orig_index.add(orig_idx)

    print(f"  - Số cluster có cả Label 0 & 1: {num_conflict_clusters}")
    print(f"  - Số peptide sẽ bị loại (Label = {DROP_LABEL_WHEN_CONFLICT} trong cluster conflict): {len(to_drop_orig_index)}")

    df_filtered = df[~df["orig_index"].isin(to_drop_orig_index)].copy()
    df_filtered.to_csv(FILTERED_CSV, index=False)

    print(f"  -> Ghi file sau lọc inter-class: {FILTERED_CSV}")
    print("  -> Phân bố Label sau lọc:")
    print(df_filtered[LABEL_COL].value_counts())

    return df_filtered


def main():
    print("==== AIP inter-class CD-HIT-2D pipeline ====")

    # 1) Đọc input
    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] Không tìm thấy file {INPUT_CSV}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    df = ensure_orig_index(df)
    df.to_csv(INPUT_CSV, index=False)  # lưu lại với orig_index (ổn định)

    print(f"[INFO] Đọc {INPUT_CSV}: {len(df)} dòng")
    print("[INFO] Phân bố Label ban đầu:")
    print(df[LABEL_COL].value_counts())

    # 2) Xuất FASTA cho inter-class
    print("\n[STEP 1] Xuất FASTA cho cd-hit-2d (inter-class)...")
    export_fastas_for_interclass(df)

    # 3) Chạy cd-hit-2d
    print("\n[STEP 2] Chạy cd-hit-2d...")
    run_cdhit2d()

    # 4) Lọc inter-class
    print("\n[STEP 3] Lọc inter-class theo .clstr...")
    apply_interclass_filter(df)

    print("\n==== DONE ====")
    print(f"File sử dụng để tính feature & train model: {FILTERED_CSV}")


if __name__ == "__main__":
    main()
