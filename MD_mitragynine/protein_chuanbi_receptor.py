#!/usr/bin/env python3
import os
from openbabel import openbabel

# ==========================
# Function helpers
# ==========================
def change_pdb_pdbqt(input_file, output_file):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "pdbqt")
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, input_file)
    obConversion.WriteFile(mol, output_file)
    print(f"[OK] Receptor saved: {output_file}")

def clean_pdbqt(input_path, output_path):
    with open(input_path, 'r') as infile:
        lines = infile.readlines()
    with open(output_path, 'w') as outfile:
        for line in lines:
            if not line.startswith(('ROOT','ENDROOT','BRANCH','ENDBRANCH','TORSDOF')):
                outfile.write(line)
    print(f"[OK] Clean receptor saved: {output_path}")

# ==========================
# Main workflow
# ==========================
def main():
    receptor_dir = "Protein_clean"   # chứa file *_clean.pdb
    os.makedirs(receptor_dir, exist_ok=True)

    pdb_files = [f for f in os.listdir(receptor_dir) if f.endswith("_clean.pdb")]
    for pdb_file in pdb_files:
        receptor_name = os.path.splitext(pdb_file)[0]
        pdb_path = os.path.join(receptor_dir, pdb_file)
        pdbqt_path = os.path.join(receptor_dir, receptor_name + ".pdbqt")
        receptor_clean = os.path.join(receptor_dir, receptor_name + "_clean.pdbqt")

        change_pdb_pdbqt(pdb_path, pdbqt_path)
        clean_pdbqt(pdbqt_path, receptor_clean)

if __name__ == "__main__":
    main()
