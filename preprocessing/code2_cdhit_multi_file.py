import pandas as pd
import subprocess
import os
import time

INPUT_CSV   = "AIP_merged_all_labeled_cleaned.csv"
OUTPUT_CSV  = "AIP_merged_all_labeled_cdhit90.csv"

# Các job CD-HIT: (mô tả, df_filter_label, fasta_tmp, fasta_cdhit)
JOBS = [
    (
        "AIP_positive",
        1,  # Label = 1
        "AIP_pos_clean.fasta",
        "AIP_pos_cdhit90.fasta",
    ),
    (
        "AIP_negative",
        0,  # Label = 0
        "AIP_neg_clean.fasta",
        "AIP_neg_cdhit90.fasta",
    ),
]

CDHIT_THRESHOLD = 0.9   # 0.9 hoặc 0.95 tuỳ bạn


# ===== STEP 1: subset class + DF -> FASTA =====
def df_to_fasta(df, fasta_file):
    """
    Ghi DataFrame (đã lọc theo Label) ra FASTA để chạy CD-HIT.
    """
    if df.empty:
        raise ValueError(f"DataFrame rỗng, không thể tạo FASTA '{fasta_file}'.")

    # Chuẩn hoá Sequence
    df = df.copy()
    df["Sequence"] = (
        df["Sequence"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    with open(fasta_file, "w") as f:
        for i, row in df.iterrows():
            # header chỉ cần unique, ta sẽ join lại theo Sequence
            header = f">seq_{i+1}"
            f.write(header + "\n")
            f.write(row["Sequence"] + "\n")

    n_seq = len(df)
    print(f"[✓] FASTA file '{fasta_file}' created successfully. N = {n_seq}")
    return n_seq


# ===== STEP 2: CHECK CD-HIT INSTALLATION =====
def check_cdhit():
    try:
        subprocess.run(
            ["cd-hit", "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("[✓] CD-HIT is installed and ready to use.")
    except FileNotFoundError:
        raise EnvironmentError(
            "CD-HIT is not installed or not found in your system PATH."
        )


# ===== STEP 3: RUN CD-HIT =====
def run_cdhit(input_fasta, output_fasta, threshold=0.8):
    print(f"[...] Running CD-HIT on '{input_fasta}' (c = {threshold:.2f}, {threshold*100:.1f}% identity) ...")
    start_time = time.time()

    cmd = [
        "cd-hit",
        "-i", input_fasta,
        "-o", output_fasta,
        "-c", str(threshold),
        "-n", "5",
        "-d", "0",
    ]
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    print(f"[✓] CD-HIT finished. Output: '{output_fasta}'")
    print(f"[⏱] Time taken: {time.time() - start_time:.2f} seconds")

    if result.stdout:
        print("[CD-HIT stdout] (first 500 chars):")
        print(result.stdout[:500])
    if result.stderr:
        print("[CD-HIT stderr] (first 500 chars):")
        print(result.stderr[:500])


# ===== STEP 4: FASTA (CD-HIT) -> DataFrame (giữ lại ID + Label) =====
def fasta_to_df(fasta_file, df_source, label_value):
    """
    df_source: DataFrame nguồn đã được lọc theo Label = label_value
    Join theo Sequence để giữ lại ID + Label.
    Trả về DataFrame đã khử trùng lặp cho class đó.
    """
    sequences = []
    with open(fasta_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    for i in range(0, len(lines), 2):
        if lines[i].startswith(">") and i + 1 < len(lines):
            sequence = lines[i + 1].strip().upper()
            sequences.append(sequence)

    df_fasta = pd.DataFrame({"Sequence": sequences})
    print(f"[INFO] Số sequence sau CD-HIT (đọc từ FASTA): {len(df_fasta)}")

    df_src = df_source.copy()
    df_src["Sequence"] = (
        df_src["Sequence"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # Join theo Sequence, trong 1 class nên mỗi Sequence chỉ có 1 Label
    df_merged = pd.merge(df_src, df_fasta, on="Sequence", how="inner")
    n_seq = len(df_merged)
    print(f"[INFO] Số dòng join được (ID, Sequence, Label={label_value}) sau CD-HIT: {n_seq}")

    df_out = df_merged[["ID", "Sequence", "Label"]].copy()
    return df_out


# ===== MAIN WORKFLOW =====
if __name__ == "__main__":
    try:
        check_cdhit()

        # Đọc toàn bộ dataset sạch
        df_all = pd.read_csv(INPUT_CSV)
        print(f"[INFO] Tổng số dòng trong {INPUT_CSV}: {len(df_all)}")

        all_clean_dfs = []

        for desc, label_value, fasta_file, cdhit_output in JOBS:
            print("\n======================================")
            print(f"Processing subset: {desc} (Label = {label_value})")
            print("======================================")

            df_subset = df_all[df_all["Label"] == label_value].copy()
            print(f"[INFO] Số dòng ban đầu của subset {desc}: {len(df_subset)}")

            if df_subset.empty:
                print(f"[WARN] Subset {desc} rỗng, bỏ qua.")
                continue

            # 1) DF subset -> FASTA
            n_before = df_to_fasta(df_subset, fasta_file)

            # 2) Run CD-HIT
            run_cdhit(fasta_file, cdhit_output, threshold=CDHIT_THRESHOLD)

            # 3) FASTA -> DF giữ lại ID + Label
            df_after = fasta_to_df(cdhit_output, df_subset, label_value)
            n_after = len(df_after)

            # Stats riêng cho subset
            removed = n_before - n_after
            pct_kept = (n_after / n_before * 100) if n_before else 0.0
            pct_removed = 100.0 - pct_kept

            print(f"\n[STATS] {desc} (Label = {label_value})")
            print(f"        CD-HIT threshold: c = {CDHIT_THRESHOLD:.2f} ({CDHIT_THRESHOLD*100:.1f}% identity)")
            print(f"        Sequences before CD-HIT: {n_before}")
            print(f"        Sequences after  CD-HIT: {n_after}")
            print(f"        Removed by CD-HIT:       {removed} ({pct_removed:.1f}% )")
            print(f"        Retained after CD-HIT:   {pct_kept:.1f}%")

            all_clean_dfs.append(df_after)

        # ===== GỘP 2 CLASS LẠI THÀNH 1 FILE CHUNG =====
        if all_clean_dfs:
            df_final = pd.concat(all_clean_dfs, ignore_index=True)

            print("\n======================================")
            print(f"[INFO] TỔNG KẾT SAU CD-HIT THEO TỪNG CLASS")
            print(f"      Tổng số dòng cuối cùng: {len(df_final)}")
            print("\n[INFO] Phân bố Label trong file cuối cùng:")
            print(df_final["Label"].value_counts().sort_index())

            df_final.to_csv(OUTPUT_CSV, index=False)
            print(f"\n[✓] Đã lưu file chung cuối cùng: {OUTPUT_CSV}")
        else:
            print("[WARN] Không có subset nào được xử lý, không tạo file chung.")

    except Exception as e:
        print(f"[❌] Error: {e}")
