import torch
import esm
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_sequences(input_csv):
    """Load sequences from a CSV file."""
    df = pd.read_csv(input_csv)
    if 'Sequence' not in df.columns:
        raise ValueError("Input CSV must contain a 'Sequence' column.")
    return df  # Trả về cả dataframe để lấy Sequence và index

def compute_local_esm_embeddings(sequences, model_name="esm2_t6_8M_UR50D"):
    """Compute local (per-sequence averaged) ESM embeddings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    embeddings = []
    model.eval()

    with torch.no_grad():
        for seq in tqdm(sequences, desc="Computing local ESM embeddings"):
            batch_labels, batch_strs, batch_tokens = batch_converter([("seq", seq)])
            batch_tokens = batch_tokens.to(device)

            results = model(batch_tokens, repr_layers=[6], return_contacts=False)
            token_representations = results["representations"][6]

            # Lấy embedding trung bình trên toàn bộ chuỗi (mean pooling)
            sequence_embedding = token_representations[0, 1:len(seq)+1].cpu().numpy()
            embeddings.append(sequence_embedding.mean(axis=0))

    return np.array(embeddings)

def save_embeddings(output_csv, df, embeddings):
    """Save only Sequence and ESM embeddings to CSV."""
    emb_df = pd.DataFrame(embeddings, columns=[f"esm_{i}" for i in range(embeddings.shape[1])])
    result_df = pd.concat([df[['Sequence']].reset_index(drop=True), emb_df], axis=1)
    result_df.to_csv(output_csv, index=False)  # Không lưu cột index, chỉ Sequence + ESM

def main():
    # File paths
    x_train_file = "AIP_x_train.csv"
    x_test_file = "AIP_x_test.csv"
    x_train_output = "AIP_x_train_esm.csv"
    x_test_output = "AIP_x_test_esm.csv"

    # Load sequences
    print("Loading sequences...")
    train_df = load_sequences(x_train_file)
    test_df = load_sequences(x_test_file)

    # Compute embeddings
    print("Computing ESM embeddings for training data...")
    train_embeddings = compute_local_esm_embeddings(train_df['Sequence'])

    print("Computing ESM embeddings for test data...")
    test_embeddings = compute_local_esm_embeddings(test_df['Sequence'])

    # Save embeddings
    print("Saving embeddings...")
    save_embeddings(x_train_output, train_df, train_embeddings)
    save_embeddings(x_test_output, test_df, test_embeddings)

    print(f"✅ Train embeddings saved to: {x_train_output}")
    print(f"✅ Test embeddings saved to: {x_test_output}")
    print("Processing complete.")

if __name__ == "__main__":
    main()
