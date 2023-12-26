import os
import warnings

import numpy as np
import prody
from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from scipy import spatial
from tqdm import tqdm

from datasets.process_mols import read_molecule
from utils.utils import read_strings_from_txt

cutoff = 10
data_dir = 'data/PDBBind_processed'
names = read_strings_from_txt(f'data/splits/timesplit_test')

io = PDBIO()
biopython_parser = PDBParser()
for name in tqdm(names):
    rec_path = os.path.join(data_dir, name, f'{name}_protein.pdb')
    lig = read_molecule(os.path.join(data_dir, name, f'{name}_ligand.sdf'), sanitize=True, remove_hs=False)
    if lig == None:
        lig = read_molecule(os.path.join(data_dir, name, f'{name}_ligand.mol2'), sanitize=True, remove_hs=False)
    if lig == None:
        print('ligand was none for ', name)
        with open('select_chains.log', 'a') as file:
            file.write(f'{name}\n')
        continue
    conf = lig.GetConformer()
    lig_coords = conf.GetPositions()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure('random_id', rec_path)
        rec = structure[0]
    min_distances = []
    coords = []
    valid_chain_ids = []
    lengths = []
    for i, chain in enumerate(rec):
        chain_coords = []  # num_residues, num_atoms, 3
        count = 0
        invalid_res_ids = []
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                invalid_res_ids.append(residue.get_id())
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
                residue_coords.append(list(atom.get_vector()))
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid and not some weired molecule that is part of the complex
                chain_coords.append(np.array(residue_coords))
                count += 1
            else:
                invalid_res_ids.append(residue.get_id())
        for res_id in invalid_res_ids:
            chain.detach_child(res_id)
        if len(chain_coords) > 0:
            all_chain_coords = np.concatenate(chain_coords, axis=0)
            distances = spatial.distance.cdist(lig_coords, all_chain_coords)
            min_distance = distances.min()
        else:
            min_distance = np.inf
        min_distances.append(min_distance)
        lengths.append(count)
        coords.append(chain_coords)
        if min_distance < cutoff:
            valid_chain_ids.append(chain.get_id())
    min_distances = np.array(min_distances)
    if len(valid_chain_ids) == 0:
        valid_chain_ids.append(np.argmin(min_distances))
    valid_coords = []
    valid_lengths = []
    invalid_chain_ids = []
    for i, chain in enumerate(rec):
        if chain.get_id() in valid_chain_ids:
            valid_coords.append(coords[i])
            valid_lengths.append(lengths[i])
        else:
            invalid_chain_ids.append(chain.get_id())

    try:
        prot = prody.parsePDB(rec_path)
        sel = prot.select(' or '.join(map(lambda c: f'chain {c}', valid_chain_ids)))
        prody.writePDB(os.path.join(data_dir,name,f'{name}_protein_chain_removed.pdb'),sel)
    except:
        io.set_structure(structure)
        io.save(os.path.join(data_dir,name,f'{name}_protein_chain_removed.pdb'))