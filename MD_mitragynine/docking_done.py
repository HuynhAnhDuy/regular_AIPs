#!/usr/bin/env python3
import os, subprocess, re, math, csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ==== Cấu hình ====
RECEPTOR = "/home/andy/andy/regular_AIPs/molecular_docking/Protein_clean/COX2_5IKR_clean.pdbqt"
LIG_DIR = Path("/home/andy/andy/regular_AIPs/molecular_docking/ligands_COX2_5IKR")
OUT_DIR = Path("/home/andy/andy/regular_AIPs/molecular_docking/Docking_results_COX2_5IKR")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Hộp docking (Å)
CENTER = (39.436, 20.551, 73.702)
SIZE   = (22, 22, 22)

# Tham số Vina
EXHAUSTIVENESS = 32
NUM_MODES = 5
ENERGY_RANGE = 3

# Tên lệnh vina (có thể là 'vina' hoặc 'autodock_vina')
VINA_BIN = "vina"

# Số luồng chạy song song
N_THREADS = 4

RT = 0.593  # kcal/mol ở 298K

def calc_Ki(deltaG_kcal):
    """Tính Ki (M) từ ΔG (kcal/mol)."""
    try:
        return math.exp(deltaG_kcal / RT)
    except Exception:
        return None

def run_vina(lig_path: Path):
    lig_base = lig_path.stem
    out_pose = OUT_DIR / f"{lig_base}_out.pdbqt"
    out_log  = OUT_DIR / f"{lig_base}.log"

    cmd = [
        VINA_BIN,
        "--receptor", RECEPTOR,
        "--ligand", str(lig_path),
        "--center_x", str(CENTER[0]), "--center_y", str(CENTER[1]), "--center_z", str(CENTER[2]),
        "--size_x", str(SIZE[0]), "--size_y", str(SIZE[1]), "--size_z", str(SIZE[2]),
        "--exhaustiveness", str(EXHAUSTIVENESS),
        "--num_modes", str(NUM_MODES),
        "--energy_range", str(ENERGY_RANGE),
        "--out", str(out_pose),
    ]
    try:
        with open(out_log, "w") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError:
        return lig_base, None

    best_pose = None
    if out_log.exists():
        poses = []
        with open(out_log, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = re.match(r"\s*(\d+)\s+([-\d\.]+)", line)
                if m:
                    rank = int(m.group(1))
                    score = float(m.group(2))
                    poses.append((rank, score))
        if poses:
            # Chọn pose có affinity nhỏ nhất (score thấp nhất)
            best_rank, best_score = min(poses, key=lambda x: x[1])
            Ki = calc_Ki(best_score)
            best_pose = (lig_base, best_rank, best_score, Ki)
    return lig_base, best_pose

def main():
    ligands = sorted(LIG_DIR.glob("*.pdbqt"))
    if not ligands:
        print("Không tìm thấy ligand .pdbqt trong thư mục 'ligands/'.")
        return

    all_results = []
    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        fut2lig = {ex.submit(run_vina, lig): lig for lig in ligands}
        for fut in as_completed(fut2lig):
            lig_base, best_pose = fut.result()
            if not best_pose:
                print(f"[FAIL] {lig_base}: docking failed.")
            else:
                lig, rank, score, Ki = best_pose
                if Ki:
                    print(f"[OK] {lig}: best pose {rank}, affinity = {score:.3f} kcal/mol, Ki ≈ {Ki:.3e} M")
                else:
                    print(f"[OK] {lig}: best pose {rank}, affinity = {score:.3f} kcal/mol")
                all_results.append(best_pose)

    # Xuất CSV (chỉ pose tốt nhất cho mỗi ligand)
    csv_path = OUT_DIR / "COX2_5IKR_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fw:
        writer = csv.writer(fw)
        writer.writerow(["ligand", "best_pose_rank", "affinity (kcal/mol)", "Ki_estimated (M)"])
        for lig, rank, score, Ki in sorted(all_results):
            writer.writerow([lig, rank, f"{score:.3f}", "" if Ki is None else f"{Ki:.3e}"])

    print(f"\nHoàn tất. Xem kết quả & log trong '{OUT_DIR}/'. Bảng điểm CSV: {csv_path}")

if __name__ == "__main__":
    main()
