import os
import warnings

import numpy as np
import prody as pr
from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from scipy import spatial
from tqdm import tqdm

from datasets.parse_chi import get_coords, get_onehot_sequence
from datasets.process_mols import read_molecule
from utils.utils import read_strings_from_txt

cutoff = 10
data_dir = 'data/BindingMOAD_2020_ab_processed_biounit'
new_data_dir = 'data/MOAD_new_test_processed'
names = np.load("data/BindingMOAD_2020_ab_processed_biounit/test_names.npy")

io = PDBIO()
biopython_parser = PDBParser()
for name in tqdm(names):
    rec_path = os.path.join(data_dir, 'pdb_protein', name[:6] + '_protein.pdb')
    lig = read_molecule(os.path.join(data_dir, 'pdb_ligand', name + '.pdb'), sanitize=True, remove_hs=False)

    conf = lig.GetConformer()
    lig_coords = conf.GetPositions()

    pdb = pr.parsePDB(rec_path)
    seq = pdb.ca.getSequence()
    coords = get_coords(pdb)
    ca_coords = coords[:, 1, :]
    res_chain_ids = pdb.ca.getChids()
    res_seg_ids = pdb.ca.getSegnames()

    distances = spatial.distance.cdist(ca_coords, lig_coords, 'euclidean')
    min_distances = np.min(distances, axis=1)
    valid_chain_ids = set()

    for i in range(len(ca_coords)):
        if min_distances[i] < cutoff:
            valid_chain_ids.add((res_seg_ids[i], res_chain_ids[i]))

    if len(valid_chain_ids) == 0:
        print('no valid chains for ', name)
    #print(valid_chain_ids)
    #query = 'chain A'
    query = ' or '.join([f'(segment {s} and chain {c})' for s,c in valid_chain_ids])
    #print(query)
    sel = pdb.select(query)
    pr.writePDB(os.path.join(new_data_dir,f'{name[:6]}_protein_chain_removed.pdb'),sel)