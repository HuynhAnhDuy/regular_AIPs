#!/usr/bin/env python3
import os
from pathlib import Path
from openbabel import openbabel

# ==== Cấu hình ====
INPUT_DIR = Path("/home/andy/andy/regular_AIPs/molecular_docking/Best_pose_pdbqt")   # thư mục chứa các ligand_out.pdbqt
OUTPUT_DIR = Path("/home/andy/andy/regular_AIPs/molecular_docking/Best_pose_pdb")    # thư mục lưu file .pdb sau khi convert
OUTPUT_DIR.mkdir(exist_ok=True)

def extract_first_pose(in_file: Path, tmp_file: Path):
    """Lấy MODEL 1 (pose 1) từ pdbqt và lưu ra file tạm."""
    lines = []
    recording = False
    with open(in_file, "r") as f:
        for line in f:
            if line.startswith("MODEL 1"):
                recording = True
            if recording:
                lines.append(line)
            if line.startswith("ENDMDL") and recording:
                break
    if lines:
        with open(tmp_file, "w") as f:
            f.writelines(lines)
        return True
    return False

def convert_pdbqt_to_pdb(in_file: Path, out_file: Path):
    """Convert pose 1 từ pdbqt sang pdb bằng Open Babel."""
    tmp_file = in_file.with_suffix(".pose1.pdbqt")
    if not extract_first_pose(in_file, tmp_file):
        print(f"[FAIL] Không tách được pose 1 từ {in_file.name}")
        return
    
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdbqt", "pdb")
    mol = openbabel.OBMol()
    if obConversion.ReadFile(mol, str(tmp_file)):
        obConversion.WriteFile(mol, str(out_file))
        print(f"[OK] {in_file.name} (pose 1) → {out_file.name}")
    else:
        print(f"[FAIL] Không đọc được {tmp_file.name}")
    tmp_file.unlink(missing_ok=True)  # xóa file tạm

def main():
    pdbqt_files = sorted(INPUT_DIR.glob("*.pdbqt"))
    if not pdbqt_files:
        print(f"Không tìm thấy file .pdbqt trong {INPUT_DIR}")
        return
    
    for f in pdbqt_files:
        out_f = OUTPUT_DIR / f.with_suffix(".pdb").name
        convert_pdbqt_to_pdb(f, out_f)

if __name__ == "__main__":
    main()
