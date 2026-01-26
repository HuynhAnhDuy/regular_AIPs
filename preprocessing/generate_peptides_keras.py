import numpy as np
import tensorflow as tf
from tensorflow import keras

GEN_MODEL_FILE = "wgan_gp_generator_keras.h5"
AA_MAP_FILE = "aa_mapping.npy"
Z_DIM = 128

# Load model + alphabet
generator = keras.models.load_model(GEN_MODEL_FILE, compile=False)
AA_ALPHABET = np.load(AA_MAP_FILE, allow_pickle=True)[0]
idx_to_aa = {i + 1: aa for i, aa in enumerate(AA_ALPHABET)}

# Lấy seq_len, vocab_size từ chính generator
# (data_dim = seq_len * vocab_size)
data_dim = generator.output_shape[1]
# Bạn phải biết seq_len từ dữ liệu train
# Easiest: đọc lại X để lấy shape
X_idx = np.load("cyclicpep_all_core_X.npy")
seq_len = X_idx.shape[1]
vocab_size = len(AA_ALPHABET) + 1

print("seq_len =", seq_len, "| vocab_size =", vocab_size, "| data_dim =", data_dim)

def decode_seq(idx_row):
    return "".join(idx_to_aa[i] for i in idx_row if i != 0)

def sample_peptides(n_samples=20):
    z = tf.random.normal(shape=(n_samples, Z_DIM))
    fake_flat = generator(z, training=False).numpy()
    fake_onehot = fake_flat.reshape(-1, seq_len, vocab_size)
    fake_idx = fake_onehot.argmax(axis=2)
    seqs = [decode_seq(row) for row in fake_idx]
    return seqs

if __name__ == "__main__":
    seqs = sample_peptides(100)
    for s in seqs:
        print(s)
