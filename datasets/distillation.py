import torch
from torch_geometric.data import Dataset

import os
import pickle
import numpy as np
import copy

class DistillationDataset(Dataset):
    def __init__(self, load_preinf=False, complexes_save_dir=None, cluster_name=None, results_path=None, confidence_cutoff=None, 
                 root=None, transform=None, multiplicity=1, max_complexes_per_couple=None, fixed_length=None, temperature=1.0, buffer_decay=0.2):
        super(DistillationDataset, self).__init__(root, transform)

        self.multiplicity = multiplicity
        self.complexes = []
        self.iteration = 0
        self.max_complexes_per_couple = max_complexes_per_couple
        self.fixed_length = fixed_length
        self.temperature = temperature
        self.buffer_decay = buffer_decay
        
        with open("data/BindingMOAD_2020_ab_processed_biounit/new_cluster_to_ligands.pkl", "rb") as f:
            self.cluster_to_ligands = pickle.load(f)
        if cluster_name is not None:
            self.ligand_names = self.cluster_to_ligands[cluster_name]
        else:
            print('Cluster name is None. Using all complexes in the validation dataset.')
            with open("./data/splits/MOAD_generalisation_splits.pkl", "rb") as f:  # TODO make parameter
                self.split_clusters = pickle.load(f)['val']

            self.ligand_names = []
            for c in self.split_clusters:
                self.ligand_names.extend(self.cluster_to_ligands[c])
        print(f'There are {len(self.ligand_names)} complexes in the cluster {cluster_name}')

        self.ligand_cnt = {ligand_name: 0 for ligand_name in self.ligand_names} # Dictionary to keep track of the number of each complex

        if load_preinf:
            print('Loading generated complexes from a previous inference run')
            assert cluster_name is not None
            assert not (complexes_save_dir is None or results_path is None or confidence_cutoff is None)
            
            self.load_preinf(complexes_save_dir, results_path, confidence_cutoff)
        
        self.print_statistics()
        print('SUCCESS| Distillation dataset initialized.')

    def load_preinf(self, complexes_save_dir, results_path, confidence_cutoff):
        
        save_path = os.path.join(complexes_save_dir, "ligands.pkl")
        assert os.path.exists(save_path)

        print('Loading generated complexes.')
        with open(save_path, 'rb') as f:
            self.generated_complexes = pickle.load(f)
            self.generated_complex_names = self.generated_complexes.keys()
        print('Generated complexes loaded.')

        # Load confidences
        confidence_path = os.path.join(results_path, "confidences.npy")
        name_path = os.path.join(results_path, "complex_names.npy")
        assert os.path.exists(save_path) and os.path.exists(name_path)
        
        confidences, names = np.load(confidence_path), np.load(name_path)
        
        print(f'There are {confidences.shape[0]} different complexes with {confidences.shape[1]} samples each.')
        self.name_to_conf = {}
        for idx in range(confidences.shape[0]):
            self.name_to_conf[names[idx]] = confidences[idx]
        
        print('Confidence scores loaded.')
        names = set()
        # Filtering and adding sampled complexes
        for ligand in self.ligand_names:
            if not ligand in self.generated_complex_names:
                print(f'Ligand {ligand} not found.')
                continue
            ligand_confidence = self.name_to_conf[ligand]
            sampled_complexes = self.generated_complexes[ligand]
    
            sampled_complexes = [sampled_complexes[i].cpu() for i in range(len(sampled_complexes)) if ligand_confidence[i] > confidence_cutoff]

            self.complexes.extend(sampled_complexes)
            if len(sampled_complexes)>0:
                names.add(ligand)
                self.ligand_names[ligand] += len(sampled_complexes)
        
        print(f'There are a total of {len(self.complexes)} samples above the confidence threshold from {len(names)} different complexes.')
        print('Complexes: ', list(names))
        
        for complex_graph in self.complexes:
            t = 0
            t_value = {'tr': t * torch.ones(1), 'rot': t * torch.ones(1), 'tor': t * torch.ones(1)}
        
            lig_node_t =  {'tr': t * torch.ones(complex_graph['ligand'].num_nodes),
                                            'rot': t * torch.ones(complex_graph['ligand'].num_nodes),
                                            'tor': t * torch.ones(complex_graph['ligand'].num_nodes)}
            rec_node_t = {'tr': t * torch.ones(complex_graph['receptor'].num_nodes),
                                                'rot': t * torch.ones(complex_graph['receptor'].num_nodes),
                                                'tor': t * torch.ones(complex_graph['receptor'].num_nodes)}
            
            complex_graph.complex_t = t_value
            complex_graph['ligand'].node_t = lig_node_t
            complex_graph['receptor'].node_t = rec_node_t

    def get(self, idx):
        if self.fixed_length is None:
            complex_graph = copy.deepcopy(self.complexes[idx % len(self.complexes)])
        else:
            confidences = np.asarray([complex_graph.confidence for complex_graph in self.complexes])
            weights = np.exp(confidences * self.temperature)
            weights = weights / np.sum(weights)
            idx = np.random.choice(len(self.complexes), p=weights)
            complex_graph = copy.deepcopy(self.complexes[idx])

        for a in ['confidence', 'iteration']:
            if hasattr(complex_graph, a):
                delattr(complex_graph, a)
            if hasattr(complex_graph['receptor'], a):
                delattr(complex_graph['receptor'], a)
            if hasattr(complex_graph['ligand'], a):
                delattr(complex_graph['ligand'], a)

        return complex_graph
    
    def len(self):
        return len(self.complexes) * self.multiplicity if self.fixed_length is None else self.fixed_length
    
    def print_statistics(self):
        # Prints how many of each complexes is contained in the dataset
        print(f'Distillation dataset starting with {len(self.complexes)} complexes.')
        for ligand, cnt in self.ligand_cnt.items():
            print(f'Ligand: {ligand} Cnt: {cnt}')

    def add_complexes(self, new_complex_list):
        print(f'Adding {len(new_complex_list)} new complexes to the distillation dataset.')
        for complex_graph, confidence in new_complex_list:
            complex_graph.confidence = confidence
            complex_graph.iteration = self.iteration
            t = 0
            t_value = {'tr': t * torch.ones(1), 'rot': t * torch.ones(1), 'tor': t * torch.ones(1)}
        
            lig_node_t =  {'tr': t * torch.ones(complex_graph['ligand'].num_nodes),
                                            'rot': t * torch.ones(complex_graph['ligand'].num_nodes),
                                            'tor': t * torch.ones(complex_graph['ligand'].num_nodes)}
            rec_node_t = {'tr': t * torch.ones(complex_graph['receptor'].num_nodes),
                                                'rot': t * torch.ones(complex_graph['receptor'].num_nodes),
                                                'tor': t * torch.ones(complex_graph['receptor'].num_nodes)}
            
            complex_graph.complex_t = t_value
            complex_graph['ligand'].node_t = lig_node_t
            complex_graph['receptor'].node_t = rec_node_t

            # update ligand_cnt dictionary
            self.ligand_cnt[complex_graph.name[0]] += 1
            complex_graph = complex_graph.cpu()

        self.iteration += 1
        self.complexes.extend([c for c, _ in new_complex_list])

        print(f'There are now {len(self.complexes)} complexes in the dataset.')
        if self.max_complexes_per_couple is not None:
            c_to_samples = {}
            for s in self.complexes:
                c_to_samples[s.name[0][:6]] = []

            for s in self.complexes:
                c_to_samples[s.name[0][:6]].append((s.confidence + self.buffer_decay * s.iteration, s)) # the policy is quite arbitrary here

            # Sort complexes by confidence and iteration and keep only the top ones
            for c in c_to_samples:
                if len(c_to_samples[c]) > self.max_complexes_per_couple:
                    c_to_samples[c] = sorted(c_to_samples[c], key=lambda x: x[0], reverse=True)
                    c_to_samples[c] = c_to_samples[c][:self.max_complexes_per_couple]

            self.complexes = []
            for c in c_to_samples:
                for _, s in c_to_samples[c]:
                    self.complexes.append(s)
            print(f'After filtering {len(self.complexes)} complexes in the dataset.')

        self.print_statistics()
