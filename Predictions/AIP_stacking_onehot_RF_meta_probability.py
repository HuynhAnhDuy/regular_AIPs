import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add
)
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier

# ====== Define Base Models (CNN, Transformer) ======
def create_cnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_transformer(input_shape, embed_dim=128, num_heads=4, ff_dim=128):
    inputs = Input(shape=input_shape)
    x = Conv1D(embed_dim, kernel_size=1, activation='relu')(inputs)

    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    attn_output = Dropout(0.1)(attn_output)
    out1 = Add()([x, attn_output])
    out1 = LayerNormalization(epsilon=1e-6)(out1)

    ffn = Dense(ff_dim, activation='relu')(out1)
    ffn = Dense(embed_dim)(ffn)
    ffn = Dropout(0.1)(ffn)
    out2 = Add()([out1, ffn])
    out2 = LayerNormalization(epsilon=1e-6)(out2)

    x = GlobalAveragePooling1D()(out2)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ====== Define Meta Model ======
def create_meta_model_rf(n_estimators=100, max_depth=7, random_state=42):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

# ====== Main Execution ======
def main():
    # 1. Load training and test data (features + sequence)
    X_train_df = pd.read_csv("AIP_x_train_onehot_esm_candidate.csv")
    X_test_df  = pd.read_csv("AIP_x_test_onehot_esm_candidate.csv")

    # Lấy sequence của TEST để xuất ra output
    # (tên cột là 'sequence' như bạn nói)
    if "Sequence" not in X_train_df.columns or "Sequence" not in X_test_df.columns:
        raise ValueError("Both train and test feature files must contain a 'sequence' column.")

    sequences_test = X_test_df["Sequence"].values

    # Features: bỏ cột 'sequence', giữ lại các đặc trưng one-hot/ESM
    X_train = X_train_df.drop(columns=["Sequence"]).values
    X_test  = X_test_df.drop(columns=["Sequence"]).values

    # Đọc y_train: file AIP_y_train.csv có cột 'sequence' và 'label'
    y_train_df = pd.read_csv("AIP_y_train.csv")
    if "Label" not in y_train_df.columns:
        raise ValueError("AIP_y_train.csv must contain a 'label' column.")
    y_train = y_train_df["Label"].values

    print("✅ Loaded data:")
    print("  X_train raw shape:", X_train.shape)
    print("  X_test  raw shape:", X_test.shape)
    print("  y_train shape    :", y_train.shape)

    # 2. Reshape to (samples, max_length, 20)
    if X_train.shape[1] % 20 != 0:
        raise ValueError("Number of feature columns in X_train is not divisible by 20.")
    if X_test.shape[1] != X_train.shape[1]:
        raise ValueError("X_test and X_train must have the same number of feature columns.")

    max_length = X_train.shape[1] // 20
    X_train = X_train.reshape((-1, max_length, 20))
    X_test  = X_test.reshape((-1, max_length, 20))

    print("✅ After reshape:")
    print("  X_train:", X_train.shape)
    print("  X_test :", X_test.shape)

    # 3. Train base models on training data
    input_shape = X_train.shape[1:]
    cnn = create_cnn(input_shape)
    transformer = create_transformer(input_shape)

    print("\n🚀 Training CNN...")
    cnn.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)

    print("\n🚀 Training Transformer...")
    transformer.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)

    # 4. Lấy dự đoán của base models trên TRAIN để train meta-model (stacking đúng)
    print("\n🔁 Getting base model predictions on TRAIN for meta-model...")
    cnn_pred_train = cnn.predict(X_train).flatten()
    transformer_pred_train = transformer.predict(X_train).flatten()

    meta_input_train = np.column_stack([cnn_pred_train, transformer_pred_train])
    print("  meta_input_train shape:", meta_input_train.shape)
    print("  y_train shape          :", y_train.shape)

    # 5. Train meta model (Random Forest) trên train
    meta_model = create_meta_model_rf()
    meta_model.fit(meta_input_train, y_train)
    print("✅ Meta-model (Random Forest) trained.")

    # 6. Lấy dự đoán của base models trên TEST, rồi dùng meta-model để predict final
    print("\n🔁 Getting base model predictions on TEST for final prediction...")
    cnn_pred_test = cnn.predict(X_test).flatten()
    transformer_pred_test = transformer.predict(X_test).flatten()

    meta_input_test = np.column_stack([cnn_pred_test, transformer_pred_test])
    print("  meta_input_test shape:", meta_input_test.shape)

    final_prob = meta_model.predict_proba(meta_input_test)[:, 1]
    print("  final_prob shape:", final_prob.shape)

    # 7. Tạo nhãn dự đoán theo ngưỡng 0.5
    # > 0.5  -> "AIP"
    # <= 0.5 -> "non-AIP"
    predicted_label = np.where(final_prob > 0.5, "AIP", "non-AIP")

    # 8. Tạo DataFrame kết quả với chuỗi + xác suất + nhãn
    df_result = pd.DataFrame({
        "sequence": sequences_test,
        "Predicted probability": final_prob,
        "predicted_label": predicted_label
    })

    # 9. Lưu ra CSV
    out_file = "AIP_7_candidates_predictions_2.csv"
    df_result.to_csv(out_file, index=False)
    print(f"\n📦 Saved prediction scores with sequences and predicted labels to: {out_file}")

if __name__ == "__main__":
    main()
