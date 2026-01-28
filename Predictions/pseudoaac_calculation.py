import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# ===== 1. Các thang thuộc tính cho 20 amino acid =====

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"  # thứ tự cố định

# 1) Hydrophobicity (Kyte–Doolittle) – bạn đang dùng
hydro_kd = {
    'A': 1.8,  'C': 2.5,  'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5,  'K': -3.9, 'L': 3.8,
    'M': 1.9,  'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2,  'W': -0.9, 'Y': -1.3
}

# 2) Hydrophilicity (Hopp–Woods) – một thang hydrophilicity cổ điển
hydro_hw = {
    'A': -0.5, 'C':  2.5, 'D':  3.0, 'E':  3.0, 'F': -2.5,
    'G':  0.0, 'H': -0.5, 'I': -1.8, 'K':  3.0, 'L': -1.8,
    'M': -1.3, 'N':  0.2, 'P':  0.2, 'Q':  0.8, 'R':  3.0,
    'S':  0.3, 'T':  0.4, 'V': -1.5, 'W': -3.4, 'Y': -2.3
}

# 3) Charge scale (custom) – side-chain charge ở pH sinh lý
charge_scale = {
    'A': 0.0, 'C': 0.0, 'D': -1.0, 'E': -1.0, 'F': 0.0,
    'G': 0.0, 'H':  0.5, 'I': 0.0,  'K':  1.0, 'L': 0.0,
    'M': 0.0, 'N': 0.0, 'P': 0.0,   'Q': 0.0, 'R':  1.0,
    'S': 0.0, 'T': 0.0, 'V': 0.0,   'W': 0.0, 'Y': 0.0
}


def _standardize_property_dict(prop_dict):
    """
    Chuẩn hóa 1 thang thuộc tính: z-score trên 20 aa.
    Trả về dict mới (AA -> giá trị chuẩn hóa).
    """
    vals = np.array([prop_dict[aa] for aa in AA_ORDER], dtype=float)
    mean = vals.mean()
    std = vals.std()
    if std == 0:
        std = 1.0
    zvals = (vals - mean) / std
    return {aa: float(z) for aa, z in zip(AA_ORDER, zvals)}


# Chuẩn hóa trước 3 thang
hydro_kd_std = _standardize_property_dict(hydro_kd)
hydro_hw_std = _standardize_property_dict(hydro_hw)
charge_std   = _standardize_property_dict(charge_scale)

PROPERTY_DICTS = [hydro_kd_std, hydro_hw_std, charge_std]  # M = 3 thuộc tính


# ===== 2. Hàm tính PseAAC đa-thuộc tính cho 1 sequence =====

def compute_pseAAC_multi(sequence, lambda_value=2, weight=0.05, property_dicts=None):
    """
    PseAAC multi-property kiểu Chou:
      - 20 thành phần đầu: composition (AA_ORDER)
      - M * lambda thành phần sau: sequence-order correlation cho M thuộc tính

    M = len(property_dicts). Ở đây M = 3 (KD, Hopp-Woods, Charge).
    Output length = 20 + M * lambda_value.
    """
    if property_dicts is None:
        property_dicts = PROPERTY_DICTS

    seq = str(sequence).strip().upper()
    L = len(seq)

    # Chuỗi quá ngắn -> vector 0
    if L < 2 or L <= lambda_value:
        return [0.0] * (20 + len(property_dicts) * lambda_value)

    # Composition
    try:
        seq_analysis = ProteinAnalysis(seq)
        aa_freq = seq_analysis.get_amino_acids_percent()
    except Exception:
        aa_freq = {aa: 0.0 for aa in AA_ORDER}

    # Tính các theta_k^(m) cho từng thuộc tính m, lag k
    all_thetas = []   # list độ dài M, mỗi phần tử là list length lambda_value
    sum_theta_total = 0.0

    for prop in property_dicts:
        prop_thetas = []
        for k in range(1, lambda_value + 1):
            sum_corr = 0.0
            count_pairs = 0
            for i in range(L - k):
                aa1, aa2 = seq[i], seq[i + k]
                if (aa1 in prop) and (aa2 in prop):
                    diff = prop[aa1] - prop[aa2]
                    sum_corr += diff * diff
                    count_pairs += 1
            if count_pairs == 0:
                theta_k = 0.0
            else:
                theta_k = sum_corr / count_pairs
            prop_thetas.append(theta_k)
            sum_theta_total += theta_k
        all_thetas.append(prop_thetas)

    if sum_theta_total == 0.0:
        sum_theta_total = 1e-10

    denom = 1.0 + weight * sum_theta_total

    # 20 thành phần composition
    pse_vec = []
    for aa in AA_ORDER:
        freq = aa_freq.get(aa, 0.0)
        pse_vec.append(freq / denom)

    # M * lambda thành phần sequence-order (theo thứ tự: thuộc tính 1..M, k=1..lambda)
    for prop_thetas in all_thetas:
        for theta_k in prop_thetas:
            pse_vec.append((weight * theta_k) / denom)

    return pse_vec


# ===== 3. Hàm xử lý 1 file CSV → PseAAC (CHỈ giữ cột Sequence) =====

def process_pseAAC_for_file(input_csv, output_csv,
                            seq_col="Sequence",
                            lambda_value=2,
                            weight=0.05):
    """
    Đọc CSV input, tính PseAAC đa-thuộc tính cho từng sequence trong cột seq_col,
    ghi ra CSV mới với:
       Sequence, PseAAC_1, ..., PseAAC_n
    Các cột khác (ID, Label, ...) KHÔNG giữ lại.
    """
    df = pd.read_csv(input_csv)

    if seq_col not in df.columns:
        raise ValueError(f"CSV file must contain a '{seq_col}' column.")

    pse_list = []
    for seq in df[seq_col]:
        pse_list.append(
            compute_pseAAC_multi(
                seq,
                lambda_value=lambda_value,
                weight=weight,
                property_dicts=PROPERTY_DICTS
            )
        )

    pse_df = pd.DataFrame(pse_list)
    num_features = pse_df.shape[1]
    pse_df.columns = [f"PseAAC_{i+1}" for i in range(num_features)]

    # chỉ giữ cột Sequence
    df_base = df[[seq_col]].copy()
    final_df = pd.concat([df_base, pse_df], axis=1)

    final_df.to_csv(output_csv, index=False)
    print(f"✅ PseAAC saved to {output_csv} (shape: {final_df.shape})")


# ===== 4. Main: chạy cho cả x_train và x_test =====

if __name__ == "__main__":
    # Đổi tên theo dataset của bạn
    train_input_csv = "AIP_x_train.csv"
    test_input_csv  = "AIP_x_test.csv"

    train_output_csv = "AIP_x_train_PAAC.csv"
    test_output_csv  = "AIP_x_test_PAAC.csv"

    lambda_value = 2   # có thể tăng lên 3 nếu muốn thêm thông tin order
    weight = 0.05      # hyper-parameter PseAAC, thường 0.05–0.1

    print("Processing train PseAAC (multi-property)...")
    process_pseAAC_for_file(
        train_input_csv,
        train_output_csv,
        seq_col="Sequence",
        lambda_value=lambda_value,
        weight=weight
    )

    print("Processing test PseAAC (multi-property)...")
    process_pseAAC_for_file(
        test_input_csv,
        test_output_csv,
        seq_col="Sequence",
        lambda_value=lambda_value,
        weight=weight
    )

    print("PseAAC multi-property calculation completed for train and test!")
