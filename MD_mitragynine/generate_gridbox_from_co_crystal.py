from Bio.PDB import PDBParser, is_aa
import numpy as np
import os
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================

protein_dir = "/home/andy/andy/regular_AIPs/MD_mitragynine/Protein_original"

ligand_map = {
    "COX1_1EQH.pdb": "FLP",
    "COX2_5IKR.pdb": "ID8",
    "COX2_3LN1.pdb": "CEL",
    "5LOX_6NCF.pdb": "AF7",
    "mPGES1_5TL9.pdb": "7DN"
}

buffer_per_side = 5.0  # Å

output_csv = "grid_centers_all_sites.csv"

# ============================================================
# FUNCTIONS
# ============================================================

parser = PDBParser(QUIET=True)

rows = []


def residue_key(res):
    """
    Unique residue identifier
    """
    hetflag, resseq, icode = res.id

    return (
        res.parent.id,   # chain ID
        hetflag,
        resseq,
        icode
    )


def residue_atoms_coords(res):
    """
    Extract coordinates while handling altlocs.

    Keep:
        altloc = ' ' or 'A'
        occupancy > 0
    """

    coords = []

    for atom in res.get_atoms():

        alt = atom.get_altloc()
        occ = atom.get_occupancy()

        if (alt in (" ", "A")) and (occ is None or occ > 0):
            coords.append(atom.coord)

    if len(coords) == 0:
        return None

    return np.array(coords)


# ============================================================
# MAIN
# ============================================================

for pdb_file, ligand_name in ligand_map.items():

    pdb_path = os.path.join(protein_dir, pdb_file)

    if not os.path.exists(pdb_path):
        print(f"[ERROR] File not found: {pdb_path}")
        continue

    structure = parser.get_structure(pdb_file, pdb_path)

    candidates = {}

    # --------------------------------------------------------
    # Find all ligand copies
    # --------------------------------------------------------

    for model in structure:
        for chain in model:
            for res in chain:

                if is_aa(res, standard=True):
                    continue

                if res.get_resname().strip() != ligand_name:
                    continue

                coords = residue_atoms_coords(res)

                if coords is None:
                    continue

                key = residue_key(res)

                candidates[key] = coords

    # --------------------------------------------------------
    # Check ligand found
    # --------------------------------------------------------

    if not candidates:

        print(
            f"[WARN] {pdb_file}: "
            f"ligand '{ligand_name}' not found."
        )

        continue

    print("\n" + "=" * 80)
    print(f"{pdb_file} | {ligand_name}")
    print(f"Detected {len(candidates)} binding site(s)")
    print("=" * 80)

    # --------------------------------------------------------
    # Process EVERY ligand copy
    # --------------------------------------------------------

    site_counter = 1

    for site_key, coords in candidates.items():

        chain_id = site_key[0]
        residue_number = site_key[2]

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        minc = coords.min(axis=0)
        maxc = coords.max(axis=0)

        extent = maxc - minc

        # Geometric center
        center = coords.mean(axis=0)

        # Box size
        size = extent + (2.0 * buffer_per_side)

        # ----------------------------------------------------
        # Check enclosure
        # ----------------------------------------------------

        lower = center - size / 2.0
        upper = center + size / 2.0

        inside = np.all(
            (coords >= lower) &
            (coords <= upper)
        )

        # ----------------------------------------------------
        # Volume
        # ----------------------------------------------------

        volume = float(
            size[0] *
            size[1] *
            size[2]
        )

        note = "OK"

        if not inside:
            note = "WARNING: ligand not fully enclosed"

        if volume > 27000:
            note += (
                f" | Vina volume warning "
                f"({volume:.0f} Å³)"
            )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(f"\nSite {site_counter}")
        print(
            f"Chain={chain_id} "
            f"Residue={residue_number}"
        )

        print(f"Atoms={len(coords)}")

        print(
            f"Center = "
            f"({center[0]:.3f}, "
            f"{center[1]:.3f}, "
            f"{center[2]:.3f})"
        )

        print(
            f"Size = "
            f"({size[0]:.2f}, "
            f"{size[1]:.2f}, "
            f"{size[2]:.2f})"
        )

        print(
            f"Volume = {volume:.0f} Å³"
        )

        print(
            f"Status = {note}"
        )

        # ----------------------------------------------------
        # Save row
        # ----------------------------------------------------

        rows.append({

            "Protein": pdb_file,
            "Ligand": ligand_name,

            "Site_ID": site_counter,

            "Chain": chain_id,

            "Residue_ID": residue_number,

            "Atoms": len(coords),

            "center_x": round(
                float(center[0]), 3
            ),
            "center_y": round(
                float(center[1]), 3
            ),
            "center_z": round(
                float(center[2]), 3
            ),

            "size_x": round(
                float(size[0]), 2
            ),
            "size_y": round(
                float(size[1]), 2
            ),
            "size_z": round(
                float(size[2]), 2
            ),

            "volume_A3": round(
                volume, 0
            ),

            "inside_box": inside,

            "note": note
        })

        site_counter += 1

# ============================================================
# SAVE CSV
# ============================================================

df = pd.DataFrame(rows)

df.to_csv(
    output_csv,
    index=False
)

print("\n" + "=" * 80)
print(f"[DONE] Results saved to: {output_csv}")
print("=" * 80)