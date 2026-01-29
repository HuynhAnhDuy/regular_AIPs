import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Bidirectional, GRU, Input
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, f1_score, confusion_matrix, average_precision_score
)

# ====== BiGRU model ======
def create_bigru(input_shape):
    """
    BiGRU-based model cho input 3D: (time_steps, features)
    Ở đây: time_steps = max_length, features = 20 (one-hot theo vị trí)
    """
    model = Sequential([
        Input(shape=input_shape),
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
def run_bigru(X_train, y_train, X_test, y_test, n_repeats=3, epochs=50, batch_size=32):
    """
    Huấn luyện BiGRU n lần, mỗi lần init lại model từ đầu,
    rồi báo cáo mean ± std cho bộ metrics.
    """
    metrics_names = ["accuracy", "balanced_accuracy", "auc", "pr_auc",
                     "mcc", "precision", "recall", "specificity", "f1"]
    final_results = {metric: [] for metric in metrics_names}

    for repeat in range(n_repeats):
        print(f"\n===== BiGRU Training Run {repeat+1}/{n_repeats} =====")

        model = create_bigru(X_train.shape[1:])
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
    # Đọc dữ liệu
    X_train_df = pd.read_csv("AIP_x_train_esm.csv", index_col=0)
    y_train = pd.read_csv("AIP_y_train.csv")["Label"].values
    X_test_df = pd.read_csv("AIP_x_test_esm.csv", index_col=0)
    y_test = pd.read_csv("AIP_y_test.csv")["Label"].values

    X_train = X_train_df.values
    X_test = X_test_df.values

    # Tính số chiều (flattened = max_length * 20)
    total_features = X_train.shape[1]
    assert total_features % 20 == 0, "⚠️ Tổng số đặc trưng không chia hết cho 20 → one-hot sai!"

    max_length = total_features // 20

    # Reshape về 3D: [samples, max_length, 20]
    X_train = X_train.reshape((-1, max_length, 20))
    X_test = X_test.reshape((-1, max_length, 20))

    print("✅ Input shapes:")
    print("  X_train:", X_train.shape)
    print("  X_test :", X_test.shape)

    # Chạy BiGRU (thay cho stacking)
    run_bigru(X_train, y_train, X_test, y_test, n_repeats=3, epochs=50, batch_size=32)


if __name__ == "__main__":
    main()
