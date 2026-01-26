import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====== CONFIG ======
INPUT_CSV      = 'AIP_merged_all_labeled_cleaned.csv'
OUTPUT_CSV     = 'AIP_merged_all_WGAN_GP_pos.csv'   # đặt tên rõ là chỉ positive
HISTOGRAM_FILE = 'AIP_pos.svg'

L_MIN      = 5
L_MAX_HARD = 50
EOS_TOKEN  = '<EOS>'
PAD_TOKEN  = '<PAD>'

# ====== 1. Load Data & chỉ giữ mẫu dương ======
df = pd.read_csv(INPUT_CSV)

required_cols = {'Sequence', 'Label'}
if not required_cols.issubset(df.columns):
    raise ValueError(f"File phải có các cột: {required_cols}, hiện có: {list(df.columns)}")

print("=== Phân bố Label trước khi lọc ===")
print(df['Label'].value_counts().sort_index())

# Chỉ giữ mẫu dương cho WGAN-GP
df = df[df['Label'] == 1].copy()
df.reset_index(drop=True, inplace=True)
print(f"\n[INFO] Chỉ giữ Label = 1 cho WGAN-GP, số chuỗi: {len(df)}")

if len(df) == 0:
    raise ValueError("Không còn mẫu dương (Label=1) nào sau khi lọc.")

# Chuẩn hoá Sequence
df['seq_clean'] = df['Sequence'].astype(str).str.upper().str.strip()

# ====== 2. Tính độ dài chuỗi ======
df['length'] = df['seq_clean'].str.len()
N_total = len(df)

print("\n=== Thống kê độ dài chuỗi (Label = 1) ===")
print(df['length'].describe())
print(f"Tổng số chuỗi (Label=1): {N_total}")

# Phân bố theo khoảng độ dài: <5, 5–10, 11–25, 26–50, >50
n_lt_5    = (df['length'] < 5).sum()
n_5_10    = df['length'].between(5, 10).sum()
n_11_25   = df['length'].between(11, 25).sum()
n_26_50   = df['length'].between(26, 50).sum()
n_gt_50   = (df['length'] > 50).sum()

print("\n=== Phân bố theo khoảng độ dài (Label = 1) ===")
print(f"< 5 aa          : {n_lt_5:4d} sequences ({n_lt_5 / N_total * 100:5.1f}%)")
print(f"5–10 aa         : {n_5_10:4d} sequences ({n_5_10 / N_total * 100:5.1f}%)")
print(f"11–25 aa        : {n_11_25:4d} sequences ({n_11_25 / N_total * 100:5.1f}%)")
print(f"26–50 aa        : {n_26_50:4d} sequences ({n_26_50 / N_total * 100:5.1f}%)")
print(f"> 50 aa         : {n_gt_50:4d} sequences ({n_gt_50 / N_total * 100:5.1f}%)")

# Kiểm tra tổng cho chắc
assert n_lt_5 + n_5_10 + n_11_25 + n_26_50 + n_gt_50 == N_total

# Đặc trưng thống kê
mean_len = df['length'].mean()
median_len = df['length'].median()
q95 = df['length'].quantile(0.95)
N_q95 = (df['length'] <= q95).sum()

print("\n=== Thống kê đặc trưng (Label = 1) ===")
print(f"Mean length    : {mean_len:.2f} aa")
print(f"Median length  : {median_len:.2f} aa")
print(f"95th percentile: {q95:.2f} aa → {N_q95} sequences ({N_q95 / N_total * 100:5.1f}%)")

# ====== 3. Vẽ histogram độ dài ======
lengths = df['length'].values
bins = np.arange(lengths.min(), lengths.max() + 2)

plt.figure(figsize=(6, 4))
plt.hist(lengths, bins=bins, color="#179C4C",
         edgecolor='black', linewidth=0.5, alpha=0.8)

ax = plt.gca()
ymax = ax.get_ylim()[1]
# Hai vạch tham chiếu theo L_MIN và L_MAX_HARD
ax.axvline(L_MIN, linestyle=':', linewidth=1.5, color="#2025C1")
ax.axvline(L_MAX_HARD, linestyle=':', linewidth=1.5, color="#C22626")

ax.text(L_MIN, ymax * 0.95, f"{L_MIN} aa",
        rotation=90, va='top', ha='right', fontsize=10)
ax.text(L_MAX_HARD, ymax * 0.95, f"{L_MAX_HARD} aa",
        rotation=90, va='top', ha='right', fontsize=10)

plt.xlabel('Peptide length (aa)', fontweight='bold', fontstyle='italic', fontsize=12)
plt.ylabel('Count', fontweight='bold', fontstyle='italic', fontsize=12)

# Text box với các khoảng mới
textstr = (
    f"Mean length: {mean_len:.1f} aa\n"
    f"Median length: {median_len:.1f} aa\n"
    f"< 5 aa:  {n_lt_5} ({n_lt_5 / N_total * 100:4.1f}%)\n"
    f"5–10 aa: {n_5_10} ({n_5_10 / N_total * 100:4.1f}%)\n"
    f"11–25 aa: {n_11_25} ({n_11_25 / N_total * 100:4.1f}%)\n"
    f"26–50 aa: {n_26_50} ({n_26_50 / N_total * 100:4.1f}%)\n"
    f"> 50 aa: {n_gt_50} ({n_gt_50 / N_total * 100:4.1f}%)"
)

plt.gcf().text(
    0.65, 0.7, textstr, fontsize=10,
    bbox=dict(boxstyle='round', facecolor='none', edgecolor='black', linewidth=0.8)
)

plt.tight_layout()
plt.savefig(HISTOGRAM_FILE, format='svg')
print(f"[✓] Đã lưu biểu đồ histogram (Label=1): {HISTOGRAM_FILE}")

# ====== 4. Độ dài phổ biến nhất ======
length_counts = df['length'].value_counts().sort_index()
most_common_len = length_counts.idxmax()
most_common_count = length_counts.max()

print("\n=== Độ dài phổ biến nhất (Label = 1) ===")
print(f"Độ dài phổ biến nhất: {most_common_len} aa ({most_common_count} chuỗi)")

# ====== 5. Thiết lập L_MAX và lọc chuỗi cho WGAN-GP ======
L_MAX = int(min(q95, L_MAX_HARD))
print(f"\nĐề xuất L_MAX = {L_MAX}")

df_filtered = df[(df['length'] >= L_MIN) & (df['length'] <= L_MAX)].copy()
df_filtered.reset_index(drop=True, inplace=True)
print(f"Số chuỗi dùng cho WGAN-GP (Label=1) [{L_MIN}, {L_MAX}]: {len(df_filtered)}")

# MAX_TOKENS = chiều dài token (aa + EOS + PAD) mà WGAN-GP sẽ thấy
MAX_TOKENS = L_MAX + 1
print(f"MAX_TOKENS (L_MAX + 1) = {MAX_TOKENS}")

# ====== 6. Thêm EOS + PAD ======
def add_eos_and_pad(seq: str, max_tokens: int) -> str:
    tokens = list(seq) + [EOS_TOKEN]
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    while len(tokens) < max_tokens:
        tokens.append(PAD_TOKEN)
    return ' '.join(tokens)

df_filtered['seq_tokens'] = df_filtered['seq_clean'].apply(
    lambda s: add_eos_and_pad(s, MAX_TOKENS)
)
df_filtered['seq_token_list'] = df_filtered['seq_tokens'].str.split(' ')

# ====== 7. Lưu output ======
df_filtered.to_csv(OUTPUT_CSV, index=False)
print(f"[✓] Đã lưu dữ liệu huấn luyện WGAN-GP (chỉ Label=1): {OUTPUT_CSV}")
print(f"L_MAX sử dụng = {L_MAX}")
print(f"MAX_TOKENS sử dụng (để đặt MAX_TOKENS trong WGAN) = {MAX_TOKENS}")
