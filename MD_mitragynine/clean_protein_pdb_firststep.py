from Bio.PDB import PDBParser, PDBIO, Select

class ProteinSelect(Select):
    def accept_residue(self, residue):
        # Loại bỏ nước (resname = HOH) và ligand (HETATM không phải amino acid)
        if residue.id[0] != " ":  # " " = amino acid chuẩn
            return False
        if residue.get_resname() == "HOH":  # nước
            return False
        return True

# Load PDB
parser = PDBParser(QUIET=True)
structure = parser.get_structure("mPGES1", "mPGES1_5TL9.pdb")

# Save file mới
io = PDBIO()
io.set_structure(structure)
io.save("mPGES1_5TL9_clean.pdb", ProteinSelect())
print("✅ Done! File đã được lưu thành clean.pdb")
