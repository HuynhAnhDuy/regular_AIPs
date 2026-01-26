import pandas as pd
import numpy as np

INPUT_CSV   = "AIP_merged_all_labeled.csv"
OUTPUT_CSV  = "AIP_merged_all_labeled_cleaned.csv"

SEQ_COL = "Sequence"   # cột chứa peptide sequence trong file AIP
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")  # 20 amino acid chuẩn


def has_invalid_aa(seq: str) -> bool:
    return any(ch not in VALID_AA for ch in seq)


def process_aip_file(infile: str, outfile: str, seq_col: str = SEQ_COL):
    print(f"\n===== XỬ LÝ FILE: {infile} =====")
    df = pd.read_csv(infile)
    print(f"[INFO] Tổng số dòng ban đầu: {len(df)}")

    # Kiểm tra cột bắt buộc
    required_cols = {"ID", seq_col, "Label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"File {infile} phải có đủ các cột: {required_cols}. "
            f"Hiện tại có: {list(df.columns)}"
        )

    # 1. Chuẩn hoá chuỗi: UPPERCASE + strip
    df["seq_raw"] = (
        df[seq_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # 2. Chỉ giữ ký tự A–Z
    df["seq_clean"] = df["seq_raw"].str.replace(r"[^A-Z]", "", regex=True)

    # 3. Bỏ chuỗi rỗng sau khi làm sạch
    before_empty_filter = len(df)
    df = df[df["seq_clean"].str.len() > 0].copy()
    print(f"[INFO] Số dòng bị loại vì seq rỗng sau làm sạch: {before_empty_filter - len(df)}")
    print(f"[INFO] Số dòng còn lại sau bước làm sạch ký tự: {len(df)}")

    # 4. Lọc chỉ giữ 20 amino acid chuẩn
    df["has_invalid_aa"] = df["seq_clean"].apply(has_invalid_aa)
    invalid_count = df["has_invalid_aa"].sum()
    print(f"[INFO] Số dòng có amino acid lạ (không thuộc 20 aa chuẩn): {invalid_count}")

    df = df[~df["has_invalid_aa"]].copy()
    df.drop(columns=["has_invalid_aa"], inplace=True)
    print(f"[INFO] Số dòng còn lại sau khi bỏ amino acid lạ: {len(df)}")

    # 4b. KIỂM TRA SEQUENCE GIỐNG NHAU NHƯNG KHÁC LABEL
    grp = df.groupby("seq_clean")["Label"].nunique()
    n_conflict_seq = (grp > 1).sum()
    print(f"\n[CHECK] Số sequence xuất hiện với >1 Label khác nhau: {n_conflict_seq}")

    if n_conflict_seq > 0:
        conflict_seqs = grp[grp > 1].index
        print(f"[INFO] Số sequence xung đột label sẽ bị loại hoàn toàn: {len(conflict_seqs)}")

        # LOẠI HẲN các sequence conflict khỏi dataset sạch
        before_conflict_filter = len(df)
        df = df[~df["seq_clean"].isin(conflict_seqs)].copy()
        print(f"[INFO] Số dòng bị loại vì xung đột label: {before_conflict_filter - len(df)}")
        print(f"[INFO] Số dòng còn lại sau khi bỏ các sequence xung đột: {len(df)}")

    # 5. Loại trùng lặp theo (seq_clean, Label) trên PHẦN CÒN LẠI (không conflict)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["seq_clean", "Label"]).reset_index(drop=True)
    print(f"[INFO] Số dòng trùng lặp bị loại (trùng seq_clean + Label): {before_dedup - len(df)}")
    print(f"[INFO] Số dòng còn lại sau khi bỏ trùng: {len(df)}")

    # 6. Cập nhật lại cột Sequence bằng bản sạch
    df[seq_col] = df["seq_clean"]

    # 7. Chỉ lưu lại 3 cột ID, Sequence, Label
    df_out = df[["ID", seq_col, "Label"]].copy()

    # 8. Thống kê lại Label
    print("\n[INFO] Phân bố Label sau khi làm sạch (ĐÃ BỎ conflict):")
    print(df_out["Label"].value_counts().sort_index())

    df_out.to_csv(outfile, index=False)
    print(f"\n[✓] Đã lưu file AIP sạch: {outfile}, số chuỗi cuối cùng = {len(df_out)}")


if __name__ == "__main__":
    process_aip_file(INPUT_CSV, OUTPUT_CSV)
