from Bio.PDB import PDBParser, is_aa
import numpy as np
import os
import pandas as pd

protein_dir = "/home/andy/andy/Inflam_NP/molecular_docking/Protein_original"

ligand_map = {
    "COX1_1EQH.pdb": "FLP",
    "COX2_5IKR.pdb": "ID8",
    "COX2_3LN1.pdb": "CEL",    
    "5LOX_6NCF.pdb": "AF7",
    "mPGES1_5TL9.pdb": "7DN"
}

buffer_per_side = 4.0  # Å đệm mỗi bên
parser = PDBParser(QUIET=True)
rows = []

def residue_key(res):
    hetflag, resseq, icode = res.id
    return (res.parent.id, hetflag, resseq, icode)  # (chain, HET, seq, icode)

def residue_atoms_coords(res):
    coords = []
    for atom in res.get_atoms():
        alt = atom.get_altloc()
        occ = atom.get_occupancy()
        # Giữ altloc 'A' hoặc rỗng, loại occupancy=0
        if (alt in (' ', 'A')) and (occ is None or occ > 0):
            coords.append(atom.coord)
    return np.array(coords) if coords else None

for pdb_file, resname in ligand_map.items():
    pdb_path = os.path.join(protein_dir, pdb_file)
    structure = parser.get_structure(pdb_file, pdb_path)

    # Gom các residue đúng resname (HETATM)
    candidates = {}
    for model in structure:
        for chain in model:
            for res in chain:
                if is_aa(res, standard=True):  # bỏ amino acid
                    continue
                if res.get_resname().strip() != resname:
                    continue
                key = residue_key(res)
                coords = residue_atoms_coords(res)
                if coords is not None and len(coords) > 0:
                    candidates[key] = coords

    if not candidates:
        print(f"[WARN] {pdb_file}: ligand {resname} not found or empty after altloc filter")
        continue

    # Chọn residue có số atom lớn nhất (thường là bản đầy đủ)
    best_key = max(candidates, key=lambda k: len(candidates[k]))
    coords = candidates[best_key]

    # Hộp ngoại tiếp
    minc = coords.min(axis=0)
    maxc = coords.max(axis=0)

    # Center tại tâm hộp
    center = (minc + maxc) / 2.0

    # Size cộng đệm mỗi bên ⇒ + 2*buffer
    extent = (maxc - minc)
    size = extent + 2 * buffer_per_side

    # Cảnh báo thể tích
    volume = float(size[0] * size[1] * size[2])
    warn = "OK"
    if volume > 27000.0:
        warn = f"Vina warning: volume {volume:.0f} Å^3 (>27000). Consider reducing size or verifying center."

    print(f"[OK] {pdb_file} ({resname})  site={best_key}")
    print(f"     Center = ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    print(f"     Size = ({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})  | Volume ~ {volume:.0f} Å^3  [{warn}]")

    rows.append({
        "Protein": pdb_file,
        "Ligand": resname,
        "center_x": round(float(center[0]), 3),
        "center_y": round(float(center[1]), 3),
        "center_z": round(float(center[2]), 3),
        "size_x": round(float(size[0]), 2),
        "size_y": round(float(size[1]), 2),
        "size_z": round(float(size[2]), 2),
        "volume": round(volume, 0),
        "note": warn
    })

pd.DataFrame(rows).to_csv("grid_centers_2.csv", index=False)
print("\n[DONE] Results saved to grid_centers.csv")
