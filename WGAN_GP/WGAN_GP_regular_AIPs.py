import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam

# ======================
# CONFIG
# ======================
INPUT_CSV = "AIP_merged_all_WGAN_GP_pos.csv"   # <-- đổi đúng tên file của bạn
OUTPUT_CSV = "regularpep_AIPs_generated.csv"

# Train length range in your real data
MAX_LEN_TRAIN = 24
MAX_TOKENS = MAX_LEN_TRAIN + 1  # +1 for EOS slot

# Desired output length range
MIN_LEN_OUT = 5
MAX_LEN_OUT = 10
N_GENERATE = 1000

# WGAN-GP hyperparams (reasonable for N~372)
NOISE_DIM = 64
BATCH_SIZE = 32
EPOCHS = 6000
N_CRITIC = 5
GP_WEIGHT = 10.0

# Decode controls
TEMPERATURE = 1.0      # 0.8-1.2; lower = more conservative
EOS_BIAS = 1.2         # bias EOS after i>=MIN_LEN_OUT to increase short lengths
MAX_TRIES = 200000     # allow many tries to collect enough 10-15aa uniques

EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"

AA20 = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
}

TOKEN2ID = dict(AA20)
TOKEN2ID[EOS_TOKEN] = 20
TOKEN2ID[PAD_TOKEN] = 21
VOCAB_SIZE = 22
ID2TOKEN = {v: k for k, v in TOKEN2ID.items()}


# ======================
# 1) DATA LOADING from seq_tokens
# ======================
def parse_seq_tokens_cell(cell: str):
    """
    cell example: "F L K ... <EOS> <PAD> ..."
    return list of tokens
    """
    if pd.isna(cell):
        return None
    toks = str(cell).strip().split()
    return toks

def one_hot_tokens(tokens):
    x = np.zeros((MAX_TOKENS, VOCAB_SIZE), dtype=np.float32)
    for i in range(MAX_TOKENS):
        tok = tokens[i] if i < len(tokens) else PAD_TOKEN
        if tok not in TOKEN2ID:
            # unknown token -> treat as PAD (or raise)
            tok = PAD_TOKEN
        x[i, TOKEN2ID[tok]] = 1.0
    return x

def load_real_data(csv_path: str):
    df = pd.read_csv(csv_path)
    if "seq_tokens" not in df.columns:
        raise ValueError("CSV phải có cột 'seq_tokens'.")

    all_tokens = []
    for cell in df["seq_tokens"].tolist():
        toks = parse_seq_tokens_cell(cell)
        if toks is None:
            continue
        # enforce length = MAX_TOKENS (pad/cut)
        if len(toks) < MAX_TOKENS:
            toks = toks + [PAD_TOKEN] * (MAX_TOKENS - len(toks))
        else:
            toks = toks[:MAX_TOKENS]
        all_tokens.append(toks)

    if len(all_tokens) == 0:
        raise ValueError("Không đọc được token nào từ 'seq_tokens'.")

    X = np.stack([one_hot_tokens(t) for t in all_tokens], axis=0)  # (N, MAX_TOKENS, VOCAB)
    X = X.reshape((X.shape[0], MAX_TOKENS * VOCAB_SIZE))           # flatten
    return X, df

real_data, df_raw = load_real_data(INPUT_CSV)
print("Real data:", real_data.shape)  # (N, 33*22=726)


# ======================
# 2) MODELS
# ======================
def build_generator():
    z = Input(shape=(NOISE_DIM,))
    x = Dense(256)(z); x = LeakyReLU(0.2)(x)
    x = Dense(512)(x); x = LeakyReLU(0.2)(x)
    logits = Dense(MAX_TOKENS * VOCAB_SIZE)(x)  # logits
    return Model(z, logits, name="Generator")

def build_critic():
    inp = Input(shape=(MAX_TOKENS * VOCAB_SIZE,))
    x = Dense(256)(inp); x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)
    x = Dense(128)(x); x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)
    out = Dense(1)(x)  # linear
    return Model(inp, out, name="Critic")

G = build_generator()
C = build_critic()


# ======================
# 3) HELPERS: softmax per position
# ======================
def softmax_tokens(flat_logits: tf.Tensor) -> tf.Tensor:
    # (B, MAX_TOKENS*VOCAB) -> (B, MAX_TOKENS, VOCAB) -> softmax -> flatten
    x = tf.reshape(flat_logits, (-1, MAX_TOKENS, VOCAB_SIZE))
    p = tf.nn.softmax(x, axis=-1)
    return tf.reshape(p, (-1, MAX_TOKENS * VOCAB_SIZE))


def gradient_penalty(critic, real_x, fake_x):
    bsz = tf.shape(real_x)[0]
    alpha = tf.random.uniform((bsz, 1), 0.0, 1.0)
    inter = alpha * real_x + (1.0 - alpha) * fake_x
    with tf.GradientTape() as tape:
        tape.watch(inter)
        v = critic(inter, training=True)
    g = tape.gradient(v, inter)
    gn = tf.sqrt(tf.reduce_sum(tf.square(g), axis=1) + 1e-12)
    return tf.reduce_mean((gn - 1.0) ** 2)


# ======================
# 4) WGAN-GP TRAINER
# ======================
class WGAN_GP:
    def __init__(self, G, C, gp_weight=10.0):
        self.G = G
        self.C = C
        self.gp_weight = gp_weight
        # WGAN-GP commonly uses beta1=0, beta2=0.9
        self.optC = Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9)
        self.optG = Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9)

    @tf.function
    def train_critic_step(self, real_x):
        bsz = tf.shape(real_x)[0]
        z = tf.random.normal((bsz, NOISE_DIM))
        with tf.GradientTape() as tape:
            fake_logits = self.G(z, training=True)
            fake_x = softmax_tokens(fake_logits)
            real_v = self.C(real_x, training=True)
            fake_v = self.C(fake_x, training=True)
            gp = gradient_penalty(self.C, real_x, fake_x)
            lossC = tf.reduce_mean(fake_v) - tf.reduce_mean(real_v) + self.gp_weight * gp
        grads = tape.gradient(lossC, self.C.trainable_variables)
        self.optC.apply_gradients(zip(grads, self.C.trainable_variables))
        return lossC

    @tf.function
    def train_generator_step(self, bsz):
        z = tf.random.normal((bsz, NOISE_DIM))
        with tf.GradientTape() as tape:
            fake_logits = self.G(z, training=True)
            fake_x = softmax_tokens(fake_logits)
            fake_v = self.C(fake_x, training=True)
            lossG = -tf.reduce_mean(fake_v)
        grads = tape.gradient(lossG, self.G.trainable_variables)
        self.optG.apply_gradients(zip(grads, self.G.trainable_variables))
        return lossG

    def train(self, X, epochs, batch_size, n_critic=3):
        n = X.shape[0]
        X = X.astype(np.float32)

        for epoch in range(1, epochs + 1):
            for _ in range(n_critic):
                idx = np.random.randint(0, n, size=min(batch_size, n))
                real_batch = tf.convert_to_tensor(X[idx], dtype=tf.float32)
                lossC = self.train_critic_step(real_batch)

            lossG = self.train_generator_step(tf.constant(min(batch_size, n), dtype=tf.int32))

            if epoch % 200 == 0:
                uniq = estimate_unique(self.G, n_samples=200)
                print(f"Epoch {epoch}/{epochs} | C={lossC.numpy():.4f} | G={lossG.numpy():.4f} | unique@200={uniq}")


# ======================
# 5) DECODING / GENERATION
# ======================
def softmax_np(logits, temperature=1.0):
    x = logits / max(temperature, 1e-6)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def sample_from_probs(p):
    return int(np.random.choice(len(p), p=p))

def decode_from_logits(flat_logits: np.ndarray,
                       temperature=1.0,
                       eos_bias=0.0):
    """
    flat_logits: (MAX_TOKENS*VOCAB,)
    return peptide string decoded until EOS
    """
    mat = flat_logits.reshape(MAX_TOKENS, VOCAB_SIZE)
    toks = []

    for i in range(MAX_TOKENS):
        row = mat[i].copy()

        # Bias EOS once we've reached MIN_LEN_OUT positions (increase chance to stop)
        if i >= MIN_LEN_OUT:
            row[TOKEN2ID[EOS_TOKEN]] += eos_bias

        p = softmax_np(row, temperature=temperature)
        tid = sample_from_probs(p)
        tok = ID2TOKEN[tid]

        if tok == EOS_TOKEN:
            break
        if tok != PAD_TOKEN:
            toks.append(tok)

    # keep amino acid letters only
    return "".join([t for t in toks if len(t) == 1])

def generate_sequences(generator: Model, n_target=100, max_tries=200000):
    out = []
    seen = set()
    tries = 0

    while len(out) < n_target and tries < max_tries:
        tries += 1
        z = tf.random.normal((1, NOISE_DIM))
        flat_logits = generator(z, training=False).numpy()[0]
        seq = decode_from_logits(flat_logits, temperature=TEMPERATURE, eos_bias=EOS_BIAS)
        L = len(seq)

        if MIN_LEN_OUT <= L <= MAX_LEN_OUT and seq not in seen:
            seen.add(seq)
            out.append(seq)

    print(f"Generated {len(out)}/{n_target} unique sequences (10-15 aa) after {tries} tries.")
    return out

def estimate_unique(generator: Model, n_samples=200):
    z = tf.random.normal((n_samples, NOISE_DIM))
    logits = generator(z, training=False).numpy()
    seqs = [decode_from_logits(logits[i], temperature=1.0, eos_bias=EOS_BIAS) for i in range(n_samples)]
    seqs = [s for s in seqs if MIN_LEN_OUT <= len(s) <= MAX_LEN_OUT]
    return len(set(seqs))


# ======================
# 6) RUN
# ======================
wgan = WGAN_GP(G, C, gp_weight=GP_WEIGHT)
wgan.train(real_data, EPOCHS, BATCH_SIZE, n_critic=N_CRITIC)

new_seqs = generate_sequences(G, n_target=N_GENERATE, max_tries=MAX_TRIES)

df_out = pd.DataFrame({
    "Sequence": new_seqs,
    "length": [len(s) for s in new_seqs]
})
df_out.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)
