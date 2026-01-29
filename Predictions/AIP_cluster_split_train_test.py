#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cluster-based train/test split cho AIP dataset.

Input:
    - AIP_interclass_filtered.csv  (có cột 'Sequence', 'Label')

Bước:
    1) Gắn 'orig_index' cho từng dòng.
    2) Xuất FASTA cho toàn bộ dataset, ID dạng 'L{Label}_{orig_index}'.
    3) Chạy cd-hit để cluster hoá theo sequence identity (ví dụ 40%).
    4) Đọc .clstr, gán cluster_id cho từng peptide.
    5) Chia cluster vào train/test (~80/20) -> không xé cluster.
    6) Ghi:
        - AIP_train.csv
        - AIP_test.csv

Chạy:
    python cluster_split_train_test.py
"""

import os
import sys
import random
import subprocess
from typing import List

import pandas as pd

# =========================
# 1. CONFIG
# =========================

INPUT_CSV = "AIP_interclass_filtered_for_predictive.csv"

SEQ_COL = "Sequence"
LABEL_COL = "Label"

CDHIT_BIN = "cd-hit"

FASTA_ALL = "aip_for_split_cdhit40.fasta"
CDHIT_PREFIX = "aip_cdhit40"
CDHIT_CLSTR = CDHIT_PREFIX + ".clstr"

HOMOLOGY_IDENTITY = 0.4   # cd-hit -c
TRAIN_RATIO = 0.8         # phần train, test = 1 - TRAIN_RATIO
RANDOM_SEED = 42


# =========================
# 2. Helper
# =========================

def ensure_orig_index(df: pd.DataFrame) -> pd.DataFrame:
    """Đảm bảo có cột 'orig_index' lưu index gốc."""
    if "orig_index" not in df.columns:
        df = df.copy()
        df["orig_index"] = df.index
    return df


def make_fasta_id(label: int, orig_idx: int) -> str:
    """Tạo ID cho FASTA, encode Label + orig_index, ví dụ: L1_123."""
    return f"L{label}_{orig_idx}"


def parse_fasta_id(fasta_id: str):
    """Giải mã 'L0_123' -> (label, orig_index)."""
    try:
        left, right = fasta_id.split("_")
        label = int(left[1:])   # bỏ 'L'
        orig_idx = int(right)
        return label, orig_idx
    except Exception:
        return None, None


def parse_cdhit_clstr(path: str) -> List[list]:
    """
    Đọc file .clstr của CD-HIT, trả về list cluster:
        [ [fasta_id1, fasta_id2, ...], [ ... ], ... ]
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
# 3. Các bước pipeline
# =========================

def export_fasta_for_cdhit(df: pd.DataFrame):
    """Xuất toàn bộ dataset thành FASTA cho cd-hit."""
    with open(FASTA_ALL, "w") as f:
        for _, row in df.iterrows():
            label = int(row[LABEL_COL])
            orig_idx = int(row["orig_index"])
            fasta_id = make_fasta_id(label, orig_idx)
            seq = str(row[SEQ_COL]).strip()
            if not seq:
                continue
            f.write(f">{fasta_id}\n{seq}\n")

    print(f"[INFO] Đã ghi FASTA cho cd-hit: {FASTA_ALL}")


def run_cdhit():
    """Chạy cd-hit để cluster hoá theo HOMOLOGY_IDENTITY."""
    cmd = [
        CDHIT_BIN,
        "-i", FASTA_ALL,
        "-o", CDHIT_PREFIX,
        "-c", str(HOMOLOGY_IDENTITY),
        "-n", "2",      # word size phù hợp cho c ~ 0.4
        "-d", "0",
        "-T", "4",
        "-M", "16000",
    ]
    run_cmd(cmd, f"cd-hit (homology clustering, c={HOMOLOGY_IDENTITY})")


def assign_cluster_and_split(df: pd.DataFrame):
    """Đọc .clstr, gán cluster_id, rồi chia cluster vào train/test."""
    if not os.path.exists(CDHIT_CLSTR):
        print(f"[ERROR] Không tìm thấy file {CDHIT_CLSTR}. Hãy chắc chắn cd-hit đã chạy xong.", file=sys.stderr)
        sys.exit(1)

    clusters = parse_cdhit_clstr(CDHIT_CLSTR)
    print(f"[INFO] Số cluster đọc từ cd-hit: {len(clusters)}")

    # map fasta_id -> cluster_id
    fasta_to_cluster = {}
    for cid, cl in enumerate(clusters):
        for fasta_id in cl:
            fasta_to_cluster[fasta_id] = cid

    # gán cluster_id cho từng dòng
    cluster_ids = []
    missing = 0
    for _, row in df.iterrows():
        label = int(row[LABEL_COL])
        orig_idx = int(row["orig_index"])
        fasta_id = make_fasta_id(label, orig_idx)
        cid = fasta_to_cluster.get(fasta_id, -1)
        if cid == -1:
            missing += 1
        cluster_ids.append(cid)

    df = df.copy()
    df["cluster_id"] = cluster_ids

    if missing > 0:
        print(f"[WARN] Có {missing} peptide không tìm thấy trong .clstr (cluster_id = -1)")

    # thống kê size cluster
    cluster_sizes = df.groupby("cluster_id").size().to_dict()
    print(f"[INFO] Số cluster khác nhau: {len(cluster_sizes)}")

    # chia cluster -> train/test
    total_n = len(df)
    target_train = total_n * TRAIN_RATIO

    cluster_items = list(cluster_sizes.items())  # (cid, size)
    random.seed(RANDOM_SEED)
    random.shuffle(cluster_items)  # random thứ tự cụm

    cluster_to_split = {}
    current_train = 0

    for cid, size in cluster_items:
        # nếu train còn thiếu nhiều so với target -> cho vào train
        if current_train + size <= target_train:
            split = "train"
            current_train += size
        else:
            split = "test"
        cluster_to_split[cid] = split

    df["split"] = df["cluster_id"].map(cluster_to_split)

    print("[INFO] Số mẫu theo split:")
    print(df["split"].value_counts())
    print("\n[INFO] Bảng Label x split:")
    print(df.groupby(["split", LABEL_COL]).size().unstack(fill_value=0))

    # ghi file train/test (bỏ cột kỹ thuật)
    train_df = df[df["split"] == "train"].drop(columns=["split", "cluster_id"])
    test_df  = df[df["split"] == "test"].drop(columns=["split", "cluster_id"])

    train_df.to_csv("AIP_x_train.csv", index=False)
    test_df.to_csv("AIP_x_test.csv", index=False)

    print("\n[INFO] Đã ghi AIP_train.csv và AIP_test.csv")
    print("       Train size:", len(train_df))
    print("       Test size :", len(test_df))


def main():
    print("==== Cluster-based train/test split cho AIP ====")

    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] Không tìm thấy {INPUT_CSV}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    if SEQ_COL not in df.columns or LABEL_COL not in df.columns:
        print(f"[ERROR] File phải có cột '{SEQ_COL}' và '{LABEL_COL}'", file=sys.stderr)
        sys.exit(1)

    df = ensure_orig_index(df)

    print(f"[INFO] Đọc {INPUT_CSV}: {len(df)} dòng")
    print("[INFO] Phân bố Label:")
    print(df[LABEL_COL].value_counts())

    print("\n[STEP 1] Xuất FASTA cho cd-hit...")
    export_fasta_for_cdhit(df)

    print("\n[STEP 2] Chạy cd-hit để cluster hoá...")
    run_cdhit()

    print("\n[STEP 3] Gán cluster_id và chia train/test theo cluster...")
    assign_cluster_and_split(df)

    print("\n==== DONE ====")


if __name__ == "__main__":
    main()
