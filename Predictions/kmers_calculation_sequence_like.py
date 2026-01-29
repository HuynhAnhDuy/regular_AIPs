#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
K-mer encoding dạng CHUỖI cho BiGRU / LSTM / Transformer.

Ý tưởng:
    - Đọc AIP_x_train.csv và AIP_x_test.csv (có cột 'Sequence' và 'Label').
    - Với mỗi sequence, sinh list k-mer chồng lấp: [s[0:k], s[1:k+1], ...].
    - Xây vocab k-mer từ TRAIN (chỉ train).
        + id 0: PAD
        + id 1: UNK (k-mer lạ chỉ xuất hiện ở test)
        + id >=2: các k-mer thực tế
    - Mã hoá mỗi peptide thành list id k-mer.
    - Pad/truncate về cùng độ dài max_len_kmer (tính trên train/test).
    - Lưu:
        - AIP_x_train_kmer_ids.csv: Sequence + pos_0 ... pos_{max_len-1}
        - AIP_x_test_kmer_ids.csv
        - AIP_kmer_vocab.csv: kmer, idx
        - AIP_kmer_meta.txt: k, max_len, vocab_size

Chạy:
    python make_kmer_for_bigru.py
"""

import pandas as pd
from collections import Counter
import numpy as np

# =========================
# 1. K-mer utils
# =========================

def generate_kmers(sequence, k=3):
    """
    Sinh list k-mer chồng lấp từ một peptide.
    Ví dụ: "ACDEFG", k=3 -> ["ACD", "CDE", "DEF", "EFG"]
    """
    seq = str(sequence).strip().upper()
    if len(seq) < k:
        return []
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


def build_kmer_vocab(kmer_lists, min_freq=1):
    """
    Xây vocab k-mer từ list các list k-mer (TRAIN).

    Trả về:
        - kmer2idx: dict {kmer: idx}
          với:
            0 -> <PAD>
            1 -> <UNK>
        - idx2kmer: list, sao cho idx2kmer[idx] = kmer
    """
    counter = Counter()
    for kmers in kmer_lists:
        counter.update(kmers)

    # bắt đầu với token đặc biệt
    kmer2idx = {"<PAD>": 0, "<UNK>": 1}
    idx2kmer = ["<PAD>", "<UNK>"]
    next_idx = 2

    for kmer, freq in counter.items():
        if freq >= min_freq:
            kmer2idx[kmer] = next_idx
            idx2kmer.append(kmer)
            next_idx += 1

    print(f"[INFO] Vocab size (k-mer): {len(kmer2idx)} (bao gồm PAD/UNK)")
    return kmer2idx, idx2kmer


def encode_kmer_sequences(kmer_lists, kmer2idx, max_len=None):
    """
    Mã hoá list các list k-mer -> numpy array int32, pad/truncate về max_len.

    Nếu max_len=None, dùng max length trong kmer_lists.
    """
    unk_idx = kmer2idx["<UNK>"]

    lengths = [len(kmers) for kmers in kmer_lists]
    if max_len is None:
        max_len = max(lengths) if lengths else 0

    print(f"[INFO] max_len k-mer: {max_len}")

    N = len(kmer_lists)
    X = np.zeros((N, max_len), dtype="int32")  # PAD = 0

    for i, kmers in enumerate(kmer_lists):
        # chuyển k-mer -> id
        ids = [kmer2idx.get(k, unk_idx) for k in kmers]
        # truncate nếu dài hơn max_len
        ids = ids[:max_len]
        # pad/truncate vào X[i]
        X[i, :len(ids)] = ids

    return X, max_len


# =========================
# 2. Pipeline
# =========================

def process_csv_to_kmer_ids(
    train_csv="AIP_x_train.csv",
    test_csv="AIP_x_test.csv",
    seq_col="Sequence",
    label_col="Label",
    k=3,
    min_freq=1,
    out_train_ids="AIP_x_train_kmer_ids.csv",
    out_test_ids="AIP_x_test_kmer_ids.csv",
    out_vocab="AIP_kmer_vocab.csv",
    out_meta="AIP_kmer_meta.txt",
):
    # ---- đọc train/test ----
    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    assert seq_col in train_df.columns, f"'{seq_col}' not in train CSV"
    assert label_col in train_df.columns, f"'{label_col}' not in train CSV"
    assert seq_col in test_df.columns, f"'{seq_col}' not in test CSV"
    assert label_col in test_df.columns, f"'{label_col}' not in test CSV"

    train_seqs = train_df[seq_col].astype(str).str.strip().str.upper().tolist()
    test_seqs  = test_df[seq_col].astype(str).str.strip().str.upper().tolist()

    y_train = train_df[label_col].values
    y_test  = test_df[label_col].values  # giữ lại nếu bạn muốn lưu y riêng

    print(f"[INFO] Train size: {len(train_seqs)}, Test size: {len(test_seqs)}")

    # ---- sinh k-mer list cho từng sequence ----
    train_kmer_lists = [generate_kmers(s, k=k) for s in train_seqs]
    test_kmer_lists  = [generate_kmers(s, k=k) for s in test_seqs]

    print(f"[INFO] Ví dụ k-mer (train[0]): {train_kmer_lists[0][:10] if train_kmer_lists else []}")

    # ---- build vocab từ train ----
    kmer2idx, idx2kmer = build_kmer_vocab(train_kmer_lists, min_freq=min_freq)

    # ---- encode train/test -> id + pad ----
    X_train_ids, max_len_train = encode_kmer_sequences(train_kmer_lists, kmer2idx, max_len=None)
    X_test_ids, max_len_test   = encode_kmer_sequences(test_kmer_lists,  kmer2idx, max_len=max_len_train)

    print(f"[INFO] X_train_ids shape: {X_train_ids.shape}")
    print(f"[INFO] X_test_ids  shape: {X_test_ids.shape}")

    # ---- lưu train/test id ra CSV ----
    # cột đầu: Sequence, sau đó là pos_0 ... pos_{max_len-1}
    train_ids_df = pd.DataFrame(
        X_train_ids,
        columns=[f"pos_{i}" for i in range(X_train_ids.shape[1])]
    )
    train_ids_df.insert(0, seq_col, train_seqs)
    train_ids_df.to_csv(out_train_ids, index=False)

    test_ids_df = pd.DataFrame(
        X_test_ids,
        columns=[f"pos_{i}" for i in range(X_test_ids.shape[1])]
    )
    test_ids_df.insert(0, seq_col, test_seqs)
    test_ids_df.to_csv(out_test_ids, index=False)

    print(f"[INFO] Saved train k-mer ids to: {out_train_ids} (shape: {train_ids_df.shape})")
    print(f"[INFO] Saved test  k-mer ids to: {out_test_ids} (shape: {test_ids_df.shape})")

    # ---- lưu vocab ----
    vocab_df = pd.DataFrame(
        [{"kmer": kmer, "idx": idx} for kmer, idx in kmer2idx.items()]
    ).sort_values("idx")
    vocab_df.to_csv(out_vocab, index=False)
    print(f"[INFO] Saved k-mer vocab to: {out_vocab} (size: {len(vocab_df)})")

    # ---- lưu meta ----
    with open(out_meta, "w") as f:
        f.write(f"k = {k}\n")
        f.write(f"max_len_kmer = {X_train_ids.shape[1]}\n")
        f.write(f"vocab_size = {len(kmer2idx)}\n")
    print(f"[INFO] Saved meta to: {out_meta}")


# =========================
# 3. Main
# =========================

if __name__ == "__main__":
    process_csv_to_kmer_ids(
        train_csv="AIP_x_train.csv",
        test_csv="AIP_x_test.csv",
        seq_col="Sequence",
        label_col="Label",
        k=3,
        min_freq=1,
        out_train_ids="AIP_x_train_kmer_ids.csv",
        out_test_ids="AIP_x_test_kmer_ids.csv",
        out_vocab="AIP_kmer_vocab.csv",
        out_meta="AIP_kmer_meta.txt",
    )
