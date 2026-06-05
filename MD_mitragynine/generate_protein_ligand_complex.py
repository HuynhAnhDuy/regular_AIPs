#!/usr/bin/env python3
import os
from pathlib import Path

# ==== Cấu hình ====
PROTEIN_FILE = Path("/home/andy/andy/regular_AIPs/MD_mitragynine/Protein_clean/COX2_5IKR_clean.pdb")   # file protein đã chuẩn bị
LIGAND_DIR   = Path("/home/andy/andy/regular_AIPs/MD_mitragynine/Selection")       # thư mục chứa ligand_out.pdb
OUTPUT_DIR   = Path("/home/andy/andy/regular_AIPs/MD_mitragynine/Complexes")           # thư mục xuất complex.pdb
OUTPUT_DIR.mkdir(exist_ok=True)

def merge_protein_ligand(protein_file: Path, ligand_file: Path, out_file: Path):
    """Gộp protein.pdb + ligand.pdb thành complex.pdb"""
    with open(protein_file, "r") as f1, open(ligand_file, "r") as f2, open(out_file, "w") as fout:
        # copy protein (bỏ END nếu có)
        for line in f1:
            if line.strip().startswith("END"):  # bỏ END để không cắt file sớm
                continue
            fout.write(line)
        # thêm TER để ngăn cách
        fout.write("TER\n")
        # copy ligand
        for line in f2:
            if line.strip().startswith("END"):
                continue
            fout.write(line)
        # kết thúc file
        fout.write("END\n")
    print(f"[OK] {ligand_file.name} → {out_file.name}")

def main():
    ligands = sorted(LIGAND_DIR.glob("*.pdb"))
    if not ligands:
        print(f"Không tìm thấy ligand .pdb trong {LIGAND_DIR}")
        return
    
    for lig in ligands:
        out_name = lig.stem + "_complex_COX2_5IKR_chain B.pdb"
        out_file = OUTPUT_DIR / out_name
        merge_protein_ligand(PROTEIN_FILE, lig, out_file)

if __name__ == "__main__":
    main()
