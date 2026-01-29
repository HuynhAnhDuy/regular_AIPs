#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MLP classifier dùng ESM pooled embeddings cho AIP dataset.

Input:
    - AIP_x_train_esm.csv : cột đầu 'Sequence', các cột còn lại 'esm_0', 'esm_1', ...
    - AIP_x_test_esm.csv
    - AIP_y_train.csv      : có cột 'Label' (0/1)
    - AIP_y_test.csv

Output:
    - In ra AUROC, AUPRC, Balanced accuracy, MCC, v.v. (mean ± std qua n_repeats)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, f1_score, confusion_matrix, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from collections import Counter


# ====== MLP model ======
def create_mlp(input_dim):
    """
    MLP đơn giản cho vector ESM (global embedding).
    input_dim = số chiều ESM (vd 320).
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auroc'),
            tf.keras.metrics.AUC(name='auprc', curve='PR')
        ]
    )
    return model


# ====== Class weights ======
def compute_class_weights(y):
    """
    Trả về dict cho class_weight: {0: w0, 1: w1}
    weight_c = total / (n_class * n_c)
    """
    counts = Counter(y)
    total = len(y)
    num_classes = len(counts)
    class_weight = {}
    for c, n_c in counts.items():
        class_weight[int(c)] = total / (num_classes * n_c)
    print("[INFO] Class counts:", counts)
    print("[INFO] Class weights:", class_weight)
    return class_weight


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


# ====== Run MLP with ESM ======
def run_mlp_esm(X_train, y_train, X_test, y_test,
                n_repeats=3, epochs=100, batch_size=32):
    """
    Huấn luyện MLP n lần, mỗi lần init lại model từ đầu,
    rồi báo cáo mean ± std cho bộ metrics.
    """
    metrics_names = ["accuracy", "balanced_accuracy", "auc", "pr_auc",
                     "mcc", "precision", "recall", "specificity", "f1"]
    final_results = {metric: [] for metric in metrics_names}

    input_dim = X_train.shape[1]
    print(f"[INFO] Input dim (ESM): {input_dim}")

    # class weights cho imbalance
    class_weight = compute_class_weights(y_train)

    for repeat in range(n_repeats):
        print(f"\n===== MLP-ESM Training Run {repeat+1}/{n_repeats} =====")

        model = create_mlp(input_dim)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_auroc',
            mode='max',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auroc',
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        )

        model.fit(
            X_train, y_train,
            validation_split=0.15,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        y_pred_prob = model.predict(X_test, batch_size=batch_size).flatten()
        metrics = evaluate_model(y_test, y_pred_prob)

        for metric in final_results:
            final_results[metric].append(metrics[metric])

        print(f"→ Accuracy: {metrics['accuracy']:.3f}, "
              f"AUROC: {metrics['auc']:.3f}, "
              f"AUPRC: {metrics['pr_auc']:.3f}, "
              f"MCC: {metrics['mcc']:.3f}")

    print("\n=== Final MLP-ESM Evaluation (Mean ± Std) ===")
    for metric, values in final_results.items():
        print(f"{metric.capitalize():18}: {np.mean(values):.3f} ± {np.std(values):.3f}")


# ====== Main Execution ======
def main():
    # Đọc embeddings ESM (Sequence + esm_*)
    train_df = pd.read_csv("AIP_x_train_esm.csv")
    test_df  = pd.read_csv("AIP_x_test_esm.csv")

    # Đọc nhãn
    y_train = pd.read_csv("AIP_y_train.csv")["Label"].values.astype("float32")
    y_test  = pd.read_csv("AIP_y_test.csv")["Label"].values.astype("float32")

    # X = bỏ cột Sequence, giữ các cột esm_*
    X_train = train_df.drop(columns=["Sequence"]).values.astype("float32")
    X_test  = test_df.drop(columns=["Sequence"]).values.astype("float32")

    print("[INFO] Raw shapes:")
    print("  X_train:", X_train.shape)
    print("  X_test :", X_test.shape)

    # Chuẩn hoá (StandardScaler) theo train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_test  = scaler.transform(X_test).astype("float32")

    print("[INFO] After scaling:")
    print("  X_train:", X_train.shape)
    print("  X_test :", X_test.shape)

    # Chạy MLP với ESM
    run_mlp_esm(
        X_train, y_train,
        X_test, y_test,
        n_repeats=3,
        epochs=100,
        batch_size=32
    )


if __name__ == "__main__":
    main()
