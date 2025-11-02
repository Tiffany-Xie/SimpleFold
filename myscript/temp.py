from Bio.PDB import MMCIFParser, PDBIO
from pathlib import Path

input_dir = Path("/hpf/projects/mtyers/ningrui/SimpleFold/results/monomer_100M/predictions_simplefold_100M")   
output_dir = Path("/hpf/projects/mtyers/ningrui/SimpleFold/results/monomer_100M/predictions_pdb")  

output_dir.mkdir(parents=True, exist_ok=True)


parser = MMCIFParser(QUIET=True)
io = PDBIO()

for cif_file in input_dir.glob("*.cif"):
    try:
        structure = parser.get_structure(cif_file.stem, cif_file)

        pdb_file = output_dir / f"{cif_file.stem}.pdb"

        io.set_structure(structure)
        io.save(pdb_file)

        print(f"Converted: {cif_file.name} -> {pdb_file.name}")

    except Exception as e:
        print(f"Failed to convert {cif_file.name}: {e}")

