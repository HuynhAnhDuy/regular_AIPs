import pandas as pd
from sklearn.model_selection import train_test_split

# ===== CẤU HÌNH =====
INPUT_CSV = "AIP_merged_all_labeled_cdhit90.csv"

X_TRAIN_OUT = "AIP_x_train.csv"
X_TEST_OUT  = "AIP_x_test.csv"
Y_TRAIN_OUT = "AIP_y_train.csv"
Y_TEST_OUT  = "AIP_y_test.csv"

ID_COL = "ID"
SEQ_COL = "Sequence"
LABEL_COL = "Label"

TEST_SIZE = 0.2
RANDOM_STATE = 42  # để tái lặp được kết quả

def main():
    print(f"Reading input file: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # Kiểm tra cột
    required_cols = {ID_COL, SEQ_COL, LABEL_COL}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Input CSV must contain columns: {required_cols}. "
            f"Found: {df.columns.tolist()}"
        )

    print(f"Total samples in input: {len(df)}")
    print("Label distribution:")
    print(df[LABEL_COL].value_counts(dropna=False))

    # Chia train/test (stratify theo label để giữ tỷ lệ class)
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COL]
    )

    print(f"\nTrain size: {len(train_df)}")
    print("Train label distribution:")
    print(train_df[LABEL_COL].value_counts(dropna=False))

    print(f"\nTest size: {len(test_df)}")
    print("Test label distribution:")
    print(test_df[LABEL_COL].value_counts(dropna=False))

    # X: chỉ id và sequence
    X_train = train_df[[ID_COL, SEQ_COL]].copy()
    X_test  = test_df[[ID_COL, SEQ_COL]].copy()

    # y: sequence (đứng đầu) + label
    y_train = train_df[[SEQ_COL, LABEL_COL]].copy()
    y_test  = test_df[[SEQ_COL, LABEL_COL]].copy()

    # Ghi file
    X_train.to_csv(X_TRAIN_OUT, index=False)
    X_test.to_csv(X_TEST_OUT, index=False)
    y_train.to_csv(Y_TRAIN_OUT, index=False)
    y_test.to_csv(Y_TEST_OUT, index=False)

    print("\n=== DONE ===")
    print(f"Saved X_train to: {X_TRAIN_OUT} (n={len(X_train)})")
    print(f"Saved X_test  to: {X_TEST_OUT}  (n={len(X_test)})")
    print(f"Saved y_train to: {Y_TRAIN_OUT} (n={len(y_train)})")
    print(f"Saved y_test  to: {Y_TEST_OUT}  (n={len(y_test)})")

if __name__ == "__main__":
    main()
