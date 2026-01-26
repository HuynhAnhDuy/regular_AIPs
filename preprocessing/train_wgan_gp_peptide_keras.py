import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ========= 0. Config & load data =========

DATA_NUMPY = "cyclicpep_all_core_X.npy"
AA_MAP_FILE = "aa_mapping.npy"

BATCH_SIZE = 64
Z_DIM = 128
GP_WEIGHT = 10.0
CRITIC_STEPS = 5
EPOCHS = 1000   # bắt đầu ít thôi để test

# ----- Load index matrix -----
X_idx = np.load(DATA_NUMPY)          # shape (N, MAX_LEN)
AA_ALPHABET = np.load(AA_MAP_FILE, allow_pickle=True)[0]
vocab_size = len(AA_ALPHABET) + 1   # +1 vì PAD=0
seq_len = X_idx.shape[1]

print("X_idx shape:", X_idx.shape)
print("seq_len =", seq_len, "| vocab_size =", vocab_size)

# ----- One-hot + flatten -----
X_onehot = tf.keras.utils.to_categorical(X_idx, num_classes=vocab_size)
# shape: (N, seq_len, vocab_size)
X_flat = X_onehot.reshape(X_onehot.shape[0], -1).astype("float32")
data_dim = X_flat.shape[1]

print("X_flat shape:", X_flat.shape, "| data_dim =", data_dim)

dataset = (
    tf.data.Dataset.from_tensor_slices(X_flat)
    .shuffle(buffer_size=min(10000, X_flat.shape[0]))
    .batch(BATCH_SIZE, drop_remainder=True)
)

# ========= 1. Define Generator & Critic =========

def make_generator(z_dim: int, data_dim: int, hidden_dim: int = 512):
    model = keras.Sequential(
        [
            layers.Input(shape=(z_dim,)),
            layers.Dense(hidden_dim, activation="relu"),
            layers.Dense(hidden_dim, activation="relu"),
            layers.Dense(data_dim),  # linear output
        ],
        name="generator",
    )
    return model


def make_critic(data_dim: int, hidden_dim: int = 512):
    model = keras.Sequential(
        [
            layers.Input(shape=(data_dim,)),
            layers.Dense(hidden_dim),
            layers.LayerNormalization(),
            layers.LeakyReLU(0.2),
            layers.Dense(hidden_dim),
            layers.LeakyReLU(0.2),
            layers.Dense(1),
        ],
        name="critic",
    )
    return model


generator = make_generator(Z_DIM, data_dim)
critic = make_critic(data_dim)

generator.summary()
critic.summary()

# ========= 2. WGAN-GP model =========

class WGAN_GP(keras.Model):
    def __init__(self, generator, critic, z_dim, gp_weight=10.0, critic_steps=5, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.critic = critic
        self.z_dim = z_dim
        self.gp_weight = gp_weight
        self.critic_steps = critic_steps

    def compile(self, g_optimizer, c_optimizer):
        super().compile()
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer

    def gradient_penalty(self, real_samples, fake_samples):
        batch_size = tf.shape(real_samples)[0]
        alpha = tf.random.uniform(shape=(batch_size, 1), minval=0.0, maxval=1.0)
        alpha = tf.broadcast_to(alpha, real_samples.shape)

        interpolated = alpha * real_samples + (1.0 - alpha) * fake_samples
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)
        grads = tape.gradient(pred, interpolated)
        grads_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
        gp = tf.reduce_mean((grads_norm - 1.0) ** 2)
        return gp

    def train_step(self, real_data):
        if isinstance(real_data, tuple):
            real_data = real_data[0]
        batch_size = tf.shape(real_data)[0]

        # ----- train critic -----
        for _ in range(self.critic_steps):
            z = tf.random.normal(shape=(batch_size, self.z_dim))
            with tf.GradientTape() as tape:
                fake_data = self.generator(z, training=True)

                real_logits = self.critic(real_data, training=True)
                fake_logits = self.critic(fake_data, training=True)

                c_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
                gp = self.gradient_penalty(real_data, fake_data)
                c_loss += self.gp_weight * gp

            c_grads = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(c_grads, self.critic.trainable_variables))

        # ----- train generator -----
        z = tf.random.normal(shape=(batch_size, self.z_dim))
        with tf.GradientTape() as tape:
            fake_data = self.generator(z, training=True)
            fake_logits = self.critic(fake_data, training=True)
            g_loss = -tf.reduce_mean(fake_logits)

        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return {"c_loss": c_loss, "g_loss": g_loss}


wgan = WGAN_GP(generator, critic, Z_DIM, gp_weight=GP_WEIGHT, critic_steps=CRITIC_STEPS)

g_opt = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9)
c_opt = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9)

wgan.compile(g_optimizer=g_opt, c_optimizer=c_opt)

# ========= 3. Callback để in vài sample =========

class SampleCallback(keras.callbacks.Callback):
    def __init__(self, every_n_epochs=100, num_samples=4):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        AA_ALPHABET = np.load(AA_MAP_FILE, allow_pickle=True)[0]
        self.idx_to_aa = {i + 1: aa for i, aa in enumerate(AA_ALPHABET)}
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def decode_seq(self, idx_row):
        return "".join(self.idx_to_aa[i] for i in idx_row if i != 0)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs != 0:
            return
        print("\n=== Sampling sequences at epoch", epoch + 1, "===")
        z = tf.random.normal(shape=(self.num_samples, Z_DIM))
        fake_flat = self.model.generator(z, training=False).numpy()
        fake_onehot = fake_flat.reshape(-1, self.seq_len, self.vocab_size)
        fake_idx = fake_onehot.argmax(axis=2)
        for row in fake_idx:
            print("Sample:", self.decode_seq(row))


# ========= 4. Train =========

wgan.fit(
    dataset,
    epochs=EPOCHS,
    callbacks=[SampleCallback(every_n_epochs=50, num_samples=4)],
)

generator.save("wgan_gp_generator_keras.h5")
print("Saved generator to wgan_gp_generator_keras.h5")
