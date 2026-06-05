input_file = "/home/andy/andy/regular_AIPs/MD_mitragynine/Complexes/Ligand_1_out_complex_COX2_5IKR.pdb"
output_file = "/home/andy/andy/regular_AIPs/MD_mitragynine/Complexes/Ligand_1_out_complex_COX2_5IKR_hetatm.pdb"

after_ter = False

with open(input_file) as fin, open(output_file, "w") as fout:

    for line in fin:

        if line.startswith("TER"):
            after_ter = True
            fout.write(line)
            continue

        if after_ter and line.startswith("ATOM"):
            line = "HETATM" + line[6:]

        fout.write(line)