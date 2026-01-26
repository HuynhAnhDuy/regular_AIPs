import pandas as pd
import subprocess
import os
import time

# ===== CẤU HÌNH CHO cyclicpep_all_core_clean =====

INPUT_CSV      = "cyclicpep_all_core_clean.csv"
TMP_FASTA      = "cyclicpep_all_core_clean.fasta"
CDHIT_FASTA    = "cyclicpep_all_core_cdhit90.fasta"
OUTPUT_CSV     = "cyclicpep_all_core_cdhit90.csv"

CDHIT_THRESHOLD = 0.90   # 90% identity


# ===== STEP 1: CSV -> FASTA =====
def csv_to_fasta(csv_file, fasta_file):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Input file '{csv_file}' does not exist.")
    
    df = pd.read_csv(csv_file)

    # Tìm cột ID (chấp nhận 'ID' hoặc 'id')
    id_col = None
    for cand in ["ID", "id"]:
        if cand in df.columns:
            id_col = cand
            break
    if id_col is None:
        raise ValueError("CSV must contain an 'ID' or 'id' column.")

    # Tìm cột Sequence
    if "Sequence" not in df.columns:
        raise ValueError("CSV must contain a 'Sequence' column.")

    # Kiểm tra trùng ID
    if df[id_col].duplicated().any():
        raise ValueError("Duplicate sequence IDs detected in input CSV.")

    # Ghi file FASTA
    with open(fasta_file, "w") as f:
        for _, row in df.iterrows():
            seq_id = str(row[id_col]).strip()
            seq = str(row["Sequence"]).strip().upper()
            f.write(f">{seq_id}\n{seq}\n")
    
    n_seq = len(df)
    print(f"[✓] FASTA file '{fasta_file}' created successfully. N = {n_seq}")
    return n_seq  # số sequence ban đầu


# ===== STEP 2: CHECK CD-HIT INSTALLATION =====
def check_cdhit():
    try:
        result = subprocess.run(
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
def run_cdhit(input_fasta, output_fasta, threshold=0.9):
    print(f"[...] Running CD-HIT on '{input_fasta}' (c = {threshold:.2f}, {threshold*100:.1f}% identity) ...")
    start_time = time.time()

    cmd = [
        "cd-hit",
        "-i", input_fasta,
        "-o", output_fasta,
        "-c", str(threshold),
        "-n", "5",   # phù hợp cho c>=0.9 với aa
        "-d", "0",   # keep full header
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


# ===== STEP 4: FASTA (CD-HIT) -> CSV (ID, Sequence) =====
def fasta_to_csv(fasta_file, output_csv):
    sequences = []
    with open(fasta_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Giả sử FASTA sau CD-HIT vẫn dạng 2 dòng: >ID, SEQ
    for i in range(0, len(lines), 2):
        if lines[i].startswith(">") and i + 1 < len(lines):
            seq_id = lines[i][1:].strip()
            sequence = lines[i + 1].strip().upper()
            sequences.append({"ID": seq_id, "Sequence": sequence})

    df_out = pd.DataFrame(sequences)
    df_out.to_csv(output_csv, index=False)
    n_seq = len(df_out)
    print(f"[✓] Filtered CSV saved to '{output_csv}'. N = {n_seq}")
    return n_seq  # số sequence sau CD-HIT


# ===== MAIN WORKFLOW =====
if __name__ == "__main__":
    try:
        print("=== CD-HIT for cyclicpep_all_core_clean ===")
        check_cdhit()

        print("\n[1] Converting CSV -> FASTA ...")
        n_before = csv_to_fasta(INPUT_CSV, TMP_FASTA)

        print("\n[2] Running CD-HIT ...")
        run_cdhit(TMP_FASTA, CDHIT_FASTA, threshold=CDHIT_THRESHOLD)

        print("\n[3] Converting CD-HIT FASTA -> CSV ...")
        n_after = fasta_to_csv(CDHIT_FASTA, OUTPUT_CSV)

        removed = n_before - n_after
        pct_kept = (n_after / n_before * 100) if n_before else 0.0
        pct_removed = 100.0 - pct_kept

        print("\n[STATS] cyclicpep_all_core_clean")
        print(f"        CD-HIT threshold: c = {CDHIT_THRESHOLD:.2f} ({CDHIT_THRESHOLD*100:.1f}% identity)")
        print(f"        Sequences before CD-HIT: {n_before}")
        print(f"        Sequences after  CD-HIT: {n_after}")
        print(f"        Removed by CD-HIT:       {removed} ({pct_removed:.1f}% )")
        print(f"        Retained after CD-HIT:   {pct_kept:.1f}%")

        print("\n[✓] Done. You can now use 'cyclicpep_all_core_cdhit90.csv' for WGAN-GP.")

    except Exception as e:
        print(f"[❌] Error: {e}")
