import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# =========================
# 1. Amino acid property dicts
# =========================
hydrophobicity = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

molecular_weight = {
    'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2, 'Q': 146.2, 'E': 147.1,
    'G': 75.1, 'H': 155.2, 'I': 131.2, 'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2,
    'P': 115.1, 'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
}


# =========================
# 2. Hàm tính feature cho từng loại
# =========================
def calculate_property_sum(sequence, property_dict):
    """Tính tổng giá trị của một property (hydrophobicity, MW, ...) cho sequence."""
    values = [property_dict.get(aa, 0) for aa in sequence if aa in property_dict]
    return {'sum': np.sum(values)} if values else {'sum': 0.0}


def calculate_biopython_features(sequence):
    """Tính các feature từ Biopython ProteinAnalysis."""
    try:
        analyzed_seq = ProteinAnalysis(sequence)
        return {
            'IsoelectricPoint': analyzed_seq.isoelectric_point(),
            'Aromaticity': analyzed_seq.aromaticity(),
            'InstabilityIndex': analyzed_seq.instability_index(),
            'Flexibility': float(np.mean(analyzed_seq.flexibility())) if analyzed_seq.flexibility() else 0.0,
            'Gravy': analyzed_seq.gravy(),
            'NetChargeAt7.4': analyzed_seq.charge_at_pH(7.4)
        }
    except Exception:
        return {
            'IsoelectricPoint': 0.0,
            'Aromaticity': 0.0,
            'InstabilityIndex': 0.0,
            'Flexibility': 0.0,
            'Gravy': 0.0,
            'NetChargeAt7.4': 0.0
        }


def calculate_secondary_structure(sequence):
    """Tính fraction helix/turn/sheet."""
    try:
        analyzed_seq = ProteinAnalysis(sequence)
        helix, turn, sheet = analyzed_seq.secondary_structure_fraction()
        return {
            'HelixFraction': helix,
            'TurnFraction': turn,
            'SheetFraction': sheet
        }
    except Exception:
        return {
            'HelixFraction': 0.0,
            'TurnFraction': 0.0,
            'SheetFraction': 0.0
        }


def calculate_amphiphilic_character(sequence):
    """Độ lệch chuẩn hydrophobicity → đặc trưng tính amphiphilic."""
    hydro_values = [hydrophobicity.get(aa, 0) for aa in sequence if aa in hydrophobicity]
    return {'AmphiphilicCharacter': float(np.std(hydro_values)) if hydro_values else 0.0}


def detect_cysteine_features(sequence):
    """Cysteine có mặt? Có khả năng tạo cầu disulfide?"""
    reduced_cysteines = sequence.count('C')
    return {
        'WithReducedCysteines': int(reduced_cysteines > 0),
        'WithDisulfidBridges': int(reduced_cysteines >= 2)
    }


# =========================
# 3. Hàm trích feature cho 1 sequence
# =========================
def extract_features_for_sequence(sequence):
    """Trả về dict feature cho 1 sequence (giữ nguyên logic cũ)."""
    seq = str(sequence).strip().upper()

    feature_vector = {
        'Sequence': seq,
        'SequenceLength': len(seq)
    }

    # Physicochemical features (sum)
    hydro_feat = calculate_property_sum(seq, hydrophobicity)
    mw_feat = calculate_property_sum(seq, molecular_weight)

    # Đổi tên key cho rõ
    feature_vector.update({
        f"Hydrophobicity_{k}": v for k, v in hydro_feat.items()
    })
    feature_vector.update({
        f"MolecularWeight_{k}": v for k, v in mw_feat.items()
    })

    # Biopython, secondary structure, amphiphilic, cysteine
    feature_vector.update(calculate_biopython_features(seq))
    feature_vector.update(calculate_secondary_structure(seq))
    feature_vector.update(calculate_amphiphilic_character(seq))
    feature_vector.update(detect_cysteine_features(seq))

    return feature_vector


# =========================
# 4. Hàm xử lý DataFrame (dùng chung train / test)
# =========================
def extract_features_from_df(df, seq_col='Sequence'):
    """
    Nhận DataFrame có cột sequence, trả về DataFrame features.
    Giữ nguyên thứ tự mẫu như df ban đầu.
    """
    if seq_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{seq_col}' trong DataFrame.")

    features = []
    for seq in df[seq_col]:
        features.append(extract_features_for_sequence(seq))

    feat_df = pd.DataFrame(features)
    return feat_df


# =========================
# 5. Main: chạy cho cả x_train và x_test
# =========================
def main():
    # ---- ĐẶT TÊN FILE Ở ĐÂY ----
    train_input_csv = "AIP_x_train.csv"
    test_input_csv = "AIP_x_test.csv"

    train_output_csv = "AIP_x_train_phychem.csv"
    test_output_csv = "AIP_x_test_phychem.csv"
    # -----------------------------

    # Đọc dữ liệu
    train_df = pd.read_csv(train_input_csv)
    test_df = pd.read_csv(test_input_csv)

    # Tính features
    train_feat_df = extract_features_from_df(train_df, seq_col='Sequence')
    test_feat_df = extract_features_from_df(test_df, seq_col='Sequence')

    # Lưu
    train_feat_df.to_csv(train_output_csv, index=False)
    test_feat_df.to_csv(test_output_csv, index=False)

    print(f"✅ Train features saved to: {train_output_csv} (shape: {train_feat_df.shape})")
    print(f"✅ Test features saved to:  {test_output_csv} (shape: {test_feat_df.shape})")


if __name__ == "__main__":
    main()
