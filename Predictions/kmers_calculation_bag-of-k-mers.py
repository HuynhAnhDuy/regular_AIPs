import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# ===== 1. Generate k-mers from a single sequence =====
def generate_kmers(sequence, k=3):
    seq = str(sequence).strip().upper()
    if len(seq) < k:
        return []
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


# ===== 2. Read CSV and turn sequences into "tokenized strings" =====
def process_file(input_csv, k=3, seq_col="Sequence"):
    """
    Trả về:
      - danh sách sequence gốc (chuẩn hóa)
      - danh sách chuỗi token k-mer ("AAA AAC ACG ...")
    """
    df = pd.read_csv(input_csv)
    if seq_col not in df.columns:
        raise ValueError(f"Input CSV must contain a '{seq_col}' column.")
    
    sequences = df[seq_col].astype(str).str.strip().str.upper().tolist()
    sequences_as_tokens = [" ".join(generate_kmers(seq, k)) for seq in sequences]
    return sequences, sequences_as_tokens


# ===== 3. Vectorize k-mers =====
def vectorize_kmers(train_kmers, test_kmers, binary=True):
    """
    binary=True => 0/1 (có/không k-mer).
    Nếu muốn đếm số lần xuất hiện thì để binary=False.
    """
    vectorizer = CountVectorizer(
        analyzer=lambda x: x.split(),  # x là chuỗi token, split() ra list k-mer
        token_pattern=None,            # tắt regex mặc định
        binary=binary
    )

    X_train = vectorizer.fit_transform(train_kmers)
    X_test = vectorizer.transform(test_kmers)

    return X_train, X_test, vectorizer


# ===== 4. Main workflow =====
if __name__ == "__main__":
    x_train_file = "AIP_x_train.csv"
    x_test_file = "AIP_x_test.csv"
    x_train_output = "AIP_x_train_kmers.csv"
    x_test_output = "AIP_x_test_kmers.csv"
    k = 3  # k-mer length

    print("Processing x_train...")
    train_sequences, train_kmers = process_file(x_train_file, k=k, seq_col="Sequence")

    print("Processing x_test...")
    test_sequences, test_kmers = process_file(x_test_file, k=k, seq_col="Sequence")

    print("Vectorizing k-mers...")
    X_train, X_test, vectorizer = vectorize_kmers(train_kmers, test_kmers, binary=True)

    # Lấy tên feature = tên k-mer
    feature_names = vectorizer.get_feature_names_out()

    print("Saving results...")
    # thêm cột Sequence ở cột đầu tiên
    train_df = pd.DataFrame(X_train.toarray(), columns=feature_names)
    train_df.insert(0, "Sequence", train_sequences)
    train_df.to_csv(x_train_output, index=False)

    test_df = pd.DataFrame(X_test.toarray(), columns=feature_names)
    test_df.insert(0, "Sequence", test_sequences)
    test_df.to_csv(x_test_output, index=False)

    print(f"Train data saved to: {x_train_output} (shape: {train_df.shape})")
    print(f"Test data saved to: {x_test_output} (shape: {test_df.shape})")
    print("Processing complete.")
