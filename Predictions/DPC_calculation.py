import pandas as pd
import itertools

# ===== 1. Generate all possible dipeptides once (20 x 20 = 400) =====
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # Standard 20 amino acids
DIPEPTIDES = [a1 + a2 for a1, a2 in itertools.product(AMINO_ACIDS, repeat=2)]


# ===== 2. Hàm tính dipeptide composition cho 1 sequence =====
def calculate_dipeptide_composition(sequence):
    """
    Trả về dict {dipeptide: frequency} với 400 khóa (AA, AC, AD, ..., YY).
    Tần suất được chuẩn hóa theo (len(seq) - 1).
    """
    seq = str(sequence).strip().upper()
    seq_length = len(seq) - 1  # số dipeptide trong chuỗi

    # Khởi tạo tất cả dipeptide = 0
    dipeptide_counts = {dp: 0.0 for dp in DIPEPTIDES}

    if seq_length <= 0:
        # Sequence quá ngắn (len < 2) → tất cả = 0
        return dipeptide_counts

    for i in range(seq_length):
        dipeptide = seq[i:i+2]
        if dipeptide in dipeptide_counts:
            dipeptide_counts[dipeptide] += 1.0

    # Chuẩn hóa theo số dipeptide
    for dp in dipeptide_counts:
        dipeptide_counts[dp] /= seq_length

    return dipeptide_counts


# ===== 3. Hàm xử lý 1 file CSV (CHỈ giữ cột Sequence trong output) =====
def process_dpc_for_file(input_csv, output_csv, seq_col="Sequence"):
    """
    Đọc CSV input, tính DPC cho từng sequence trong cột seq_col,
    rồi ghi ra CSV mới với:
    [Sequence] + [400 cột DPC].
    Các cột khác (ID, Label, ...) sẽ KHÔNG được giữ lại.
    """
    df = pd.read_csv(input_csv)

    if seq_col not in df.columns:
        raise ValueError(f"The CSV file must have a column named '{seq_col}'.")

    compositions = []
    for seq in df[seq_col]:
        compositions.append(calculate_dipeptide_composition(seq))

    df_dpc = pd.DataFrame(compositions)  # 400 cột, tên = dipeptide (AA, AC, ...)

    # CHỈ giữ cột Sequence, bỏ ID và các cột khác
    df_base = df[[seq_col]].copy()

    # Sequence ở cột đầu, sau đó đến 400 cột DPC
    df_result = pd.concat([df_base, df_dpc], axis=1)

    df_result.to_csv(output_csv, index=False)
    print(f"✅ DPC saved to {output_csv} (shape: {df_result.shape})")


# ===== 4. Main: chạy cho cả x_train và x_test =====
if __name__ == "__main__":
    # Đặt tên file ở đây
    train_input_csv = "AIP_x_train.csv"
    test_input_csv = "AIP_x_test.csv"

    train_output_csv = "AIP_x_train_DPC.csv"
    test_output_csv = "AIP_x_test_DPC.csv"

    print("Processing train DPC...")
    process_dpc_for_file(train_input_csv, train_output_csv, seq_col="Sequence")

    print("Processing test DPC...")
    process_dpc_for_file(test_input_csv, test_output_csv, seq_col="Sequence")

    print("Dipeptide composition calculation completed for train and test!")
