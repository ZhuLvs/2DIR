import numpy as np
import Bio.PDB
import pandas as pd
import os
from multiprocessing import Pool, cpu_count

def calculate_min_distance(residue1, residue2):
    """Calculate the minimum distance between two residues, preferring CA atoms, else another atom."""
    ca1 = residue1['CA'] if 'CA' in residue1 else next(iter(residue1), None)
    ca2 = residue2['CA'] if 'CA' in residue2 else next(iter(residue2), None)

    if ca1 is not None and ca2 is not None:
        distance = ca1 - ca2
    else:
        distance = float('inf')
    return distance

def calculate_distance_matrix(structure):
    """Calculate the all-against-all CA atoms distance matrix for a given protein structure."""
    residues = [residue for residue in structure.get_residues()
                if Bio.PDB.is_aa(residue, standard=True) or residue.get_resname() in ['HSD', 'HSE', 'HSP', 'NLE', 'HIP', 'NLE', 'HIE']]

    distance_matrix = np.zeros((len(residues), len(residues)))

    for i, residue1 in enumerate(residues):
        for j, residue2 in enumerate(residues):
            distance_matrix[i, j] = calculate_min_distance(residue1, residue2)

    return distance_matrix, residues

def main(pdb_file_path, csv_file_path):
    """Main function to calculate the distance matrix and save it as a CSV file."""
    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    structure = pdb_parser.get_structure('protein_structure', pdb_file_path)

    distance_matrix, residues = calculate_distance_matrix(structure[0])

    residue_labels = [f"{residue.get_resname()} {residue.get_id()[1]}" for residue in residues]

    pd.DataFrame(distance_matrix, columns=residue_labels).to_csv(csv_file_path, index=False)

def process_single_file(args):
    """Helper function to allow Pool.map to work with multiple arguments."""
    pdb_file_path, csv_file_path = args
    main(pdb_file_path, csv_file_path)

def process_pdb_files(input_folder, output_folder):
    """Process all PDB files in the input folder and save the distance matrices to the output folder."""
    files_to_process = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.pdb'):
            pdb_file_path = os.path.join(input_folder, filename)
            csv_file_name = os.path.splitext(filename)[0] + '.csv'
            csv_file_path = os.path.join(output_folder, csv_file_name)
            files_to_process.append((pdb_file_path, csv_file_path))

    with Pool(processes=cpu_count()) as pool:
        pool.map(process_single_file, files_to_process)

if __name__ == '__main__':
    input_folder = 'PDB'
    output_folder = 'contact'
    process_pdb_files(input_folder, output_folder)
