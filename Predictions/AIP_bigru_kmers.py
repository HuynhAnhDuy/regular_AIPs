import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Bidirectional, GRU, Input, Embedding
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, f1_score, confusion_matrix, average_precision_score
)

# ====== BiGRU model cho k-mer IDs ======
def create_bigru(max_len, vocab_size, embed_dim=64):
    """
    BiGRU-based model cho input 2D: (batch, max_len),
    mỗi phần tử là ID của k-mer (int).
    
    - max_len: số vị trí k-mer sau khi pad/truncate
    - vocab_size: số lượng token trong vocab (k-mer + PAD + UNK)
    """
    model = Sequential([
        Input(shape=(max_len,), dtype="int32"),
        Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True,
            name="kmer_embedding"
        ),
        Bidirectional(GRU(64, return_sequences=True)),
        Bidirectional(GRU(32)),
        Dropout(0.5),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ====== Evaluate Metrics ======
def evaluate_model(y_true, y_pred_prob):
    y_pred = (y_pred_prob > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_pred_prob),
        "pr_auc": average_precision_score(y_true, y_pred_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred)
    }


# ====== Run BiGRU (thay cho stacking) ======
def run_bigru(X_train, y_train, X_test, y_test,
              vocab_size,
              n_repeats=3, epochs=50, batch_size=32, embed_dim=64):
    """
    Huấn luyện BiGRU n lần, mỗi lần init lại model từ đầu,
    rồi báo cáo mean ± std cho bộ metrics.

    - X_train, X_test: mảng int32 [N, max_len], mỗi phần tử là k-mer ID
    - vocab_size: số lượng token trong vocab (max_id + 1)
    """
    metrics_names = ["accuracy", "balanced_accuracy", "auc", "pr_auc",
                     "mcc", "precision", "recall", "specificity", "f1"]
    final_results = {metric: [] for metric in metrics_names}

    max_len = X_train.shape[1]

    for repeat in range(n_repeats):
        print(f"\n===== BiGRU Training Run {repeat+1}/{n_repeats} =====")

        model = create_bigru(max_len=max_len, vocab_size=vocab_size, embed_dim=embed_dim)
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        y_pred_prob = model.predict(X_test).flatten()
        metrics = evaluate_model(y_test, y_pred_prob)

        for metric in final_results:
            final_results[metric].append(metrics[metric])

        print(f"→ Accuracy: {metrics['accuracy']:.3f}, "
              f"AUROC: {metrics['auc']:.3f}, "
              f"AUPRC: {metrics['pr_auc']:.3f}, "
              f"MCC: {metrics['mcc']:.3f}")

    print("\n=== Final BiGRU Evaluation (Mean ± Std) ===")
    for metric, values in final_results.items():
        print(f"{metric.capitalize():18}: {np.mean(values):.3f} ± {np.std(values):.3f}")


# ====== Main Execution ======
def main():
    # Đọc dữ liệu k-mer IDs
    # File này do script make_kmer_for_bigru.py sinh ra:
    #   cột đầu: Sequence
    #   cột sau: pos_0, pos_1, ..., pos_{L-1}
    train_ids_df = pd.read_csv("AIP_x_train_kmer_ids.csv")
    test_ids_df  = pd.read_csv("AIP_x_test_kmer_ids.csv")

    # Nhãn y tách riêng
    y_train = pd.read_csv("AIP_y_train.csv")["Label"].values.astype("float32")
    y_test  = pd.read_csv("AIP_y_test.csv")["Label"].values.astype("float32")

    # X = chỉ lấy các cột pos_*
    X_train = train_ids_df.drop(columns=["Sequence"]).values.astype("int32")
    X_test  = test_ids_df.drop(columns=["Sequence"]).values.astype("int32")

    # max_len = số cột pos_*
    max_len = X_train.shape[1]

    # vocab_size: có thể lấy từ vocab file hoặc lấy max ID + 1
    max_id_train = X_train.max()
    max_id_test  = X_test.max()
    vocab_size = int(max(max_id_train, max_id_test) + 1)

    print("✅ Input summary:")
    print("  X_train shape:", X_train.shape)
    print("  X_test  shape:", X_test.shape)
    print("  max_len      :", max_len)
    print("  vocab_size   :", vocab_size)

    # Chạy BiGRU với k-mer IDs
    run_bigru(
        X_train, y_train,
        X_test, y_test,
        vocab_size=vocab_size,
        n_repeats=3,
        epochs=50,
        batch_size=32,
        embed_dim=64
    )


if __name__ == "__main__":
    main()
