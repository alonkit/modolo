import copy
from datetime import datetime
from typing import List, Optional
import warnings
import numpy as np
import torch
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem, GetPeriodicTable, RemoveHs
from rdkit.Geometry import Point3D
from torch import cdist
from torch_cluster import knn_graph
import prody as pr
from torch_geometric.data import Data, HeteroData

import time
import torch.nn.functional as F
from datasets.process_chem.features import allowable_features, bonds
from datasets.process_chem.constants import aa_short2long, atom_order, three_to_one
from datasets.process_chem.parse_chi import get_chi_angles, get_coords, aa_idx2aa_short, get_onehot_sequence
from utils.logging_utils import get_logger
from torch_cluster import radius, radius_graph


periodic_table = GetPeriodicTable()


def lig_atom_featurizer(mol,extra_props={}):
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        chiral_tag = str(atom.GetChiralTag())
        if chiral_tag  in ['CHI_SQUAREPLANAR', 'CHI_TRIGONALBIPYRAMIDAL', 'CHI_OCTAHEDRAL']:
            chiral_tag = 'CHI_OTHER'

        atom_features_list.append([
            safe_index(allowable_features['possible_hole_ids'], extra_props['__holeIdx'][idx]),
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(chiral_tag)),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),
            allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),
            allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
            allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
            allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
            allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
            allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx, 8)),
            #g_charge if not np.isnan(g_charge) and not np.isinf(g_charge) else 0.
        ])
    return torch.tensor(atom_features_list)


def safe_index(l, e):
    """ Return index of element e in list l. If e is not present, return the last index """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def moad_extract_receptor_structure(pdb, complex_graph, neighbor_cutoff=20, max_neighbors=None, sequences_to_embeddings=None,
                                    knn_only_graph=False, lm_embeddings=None, all_atoms=False, atom_cutoff=None, atom_max_neighbors=None):
    # load the entire pdb file
    seq = pdb.ca.getSequence()[:1022]
    coords = get_coords(pdb)[:1022]
    one_hot = get_onehot_sequence(seq)

    chain_ids = np.zeros(len(one_hot))
    res_chain_ids = pdb.ca.getChids()[:1022]
    res_seg_ids = pdb.ca.getSegnames()[:1022]
    res_chain_ids = np.asarray([s + c for s, c in zip(res_seg_ids, res_chain_ids)])
    ids = np.unique(res_chain_ids)
    sequences = []
    lm_embeddings = lm_embeddings if sequences_to_embeddings is None else []

    for i, id in enumerate(ids):
        chain_ids[res_chain_ids == id] = i

        s = np.argmax(one_hot[res_chain_ids == id], axis=1)
        s = ''.join([aa_idx2aa_short[aa_idx] for aa_idx in s])
        sequences.append(s)
        if sequences_to_embeddings is not None:
            lm_embeddings.append(sequences_to_embeddings[s])

    complex_graph['receptor'].sequence = sequences
    complex_graph['receptor'].chain_ids = torch.from_numpy(np.asarray(chain_ids)).long()

    new_extract_receptor_structure(seq, coords, complex_graph, neighbor_cutoff=neighbor_cutoff, max_neighbors=max_neighbors,
                                   lm_embeddings=lm_embeddings, knn_only_graph=knn_only_graph, all_atoms=all_atoms,
                                   atom_cutoff=atom_cutoff, atom_max_neighbors=atom_max_neighbors)


def new_extract_receptor_structure(seq, all_coords, complex_graph, neighbor_cutoff=20, max_neighbors=None, lm_embeddings=None,
                                   knn_only_graph=False, all_atoms=False, atom_cutoff=None, atom_max_neighbors=None):
    chi_angles, one_hot = get_chi_angles(all_coords, seq, return_onehot=True)
    n_rel_pos, c_rel_pos = all_coords[:, 0, :] - all_coords[:, 1, :], all_coords[:, 2, :] - all_coords[:, 1, :]
    sidechain_vecs = torch.from_numpy(np.concatenate([chi_angles / 360, n_rel_pos, c_rel_pos], axis=1))

    # Build the k-NN graph
    coords = torch.tensor(all_coords[:, 1, :], dtype=torch.float)
    if len(coords) > 3000:
        raise ValueError(f'The receptor is too large {len(coords)}')
    if knn_only_graph:
        edge_index = knn_graph(coords, k=max_neighbors if max_neighbors else 32)
    else:
        distances = cdist(coords, coords)
        src_list = []
        dst_list = []
        for i in range(len(coords)):
            dst = list(np.where(distances[i, :] < neighbor_cutoff)[0])
            dst.remove(i)
            max_neighbors = max_neighbors if max_neighbors else 1000
            if max_neighbors != None and len(dst) > max_neighbors:
                dst = list(np.argsort(distances[i, :]))[1: max_neighbors + 1]
            if len(dst) == 0:
                dst = list(np.argsort(distances[i, :]))[1:2]  # choose second because first is i itself
                print(
                    f'The cutoff {neighbor_cutoff} was too small for one atom such that it had no neighbors. '
                    f'So we connected it to the closest other atom')
            assert i not in dst
            src = [i] * len(dst)
            src_list.extend(src)
            dst_list.extend(dst)
        edge_index = torch.from_numpy(np.asarray([dst_list, src_list]))

    res_names_list = [aa_short2long[seq[i]] if seq[i] in aa_short2long else 'misc' for i in range(len(seq))]
    feature_list = [[safe_index(allowable_features['possible_amino_acids'], res)] for res in res_names_list]
    node_feat = torch.tensor(feature_list, dtype=torch.float32)

    lm_embeddings = torch.tensor(np.concatenate(lm_embeddings, axis=0)) if lm_embeddings is not None else None
    complex_graph['receptor'].x = torch.cat([node_feat, lm_embeddings], axis=1) if lm_embeddings is not None else node_feat
    complex_graph['receptor'].pos = coords
    complex_graph['receptor'].sidechain_vecs = sidechain_vecs.float()
    complex_graph['receptor', 'rec_contact', 'receptor'].edge_index = edge_index
    if all_atoms:
        atom_coords = all_coords.reshape(-1, 3)
        atom_coords = torch.from_numpy(atom_coords[~np.any(np.isnan(atom_coords), axis=1)]).float()

        if knn_only_graph:
            atoms_edge_index = knn_graph(atom_coords, k=atom_max_neighbors if atom_max_neighbors else 1000)
        else:
            atoms_distances = cdist(atom_coords, atom_coords)
            argsorts = atoms_distances.argsort() 
            atom_src_list = []
            atom_dst_list = []
            for i in range(len(atom_coords)):
                dst = list(np.where(atoms_distances[i, :] < atom_cutoff)[0])
                dst.remove(i)
                max_neighbors = atom_max_neighbors if atom_max_neighbors else 1000
                if max_neighbors != None and len(dst) > max_neighbors:
                    dst = argsorts[i][1: max_neighbors + 1]
                if len(dst) == 0:
                    dst = argsorts[i][1:2]  # choose second because first is i itself
                    print(
                        f'The atom_cutoff {atom_cutoff} was too small for one atom such that it had no neighbors. '
                        f'So we connected it to the closest other atom')
                assert i not in dst
                src = [i] * len(dst)
                atom_src_list.extend(src)
                atom_dst_list.extend(dst)
            atoms_edge_index = torch.from_numpy(np.asarray([atom_dst_list, atom_src_list]))
        
        feats = [get_moad_atom_feats(res, all_coords[i]) for i, res in enumerate(seq)]
        atom_feat = torch.from_numpy(np.concatenate(feats, axis=0)).float()
        c_alpha_idx = np.concatenate([np.zeros(len(f)) + i for i, f in enumerate(feats)])
        np_array = np.stack([np.arange(len(atom_feat)), c_alpha_idx])
        atom_res_edge_index = torch.from_numpy(np_array).long()
        complex_graph['atom'].x = atom_feat
        complex_graph['atom'].pos = atom_coords
        assert len(complex_graph['atom'].x) == len(complex_graph['atom'].pos)
        complex_graph['atom', 'atom_contact', 'atom'].edge_index = atoms_edge_index
        complex_graph['atom', 'atom_rec_contact', 'receptor'].edge_index = atom_res_edge_index

    return


def get_moad_atom_feats(res, coords):
    feats = []
    res_long = aa_short2long[res]
    res_order = atom_order[res]
    for i, c in enumerate(coords):
        if np.any(np.isnan(c)):
            continue
        atom_feats = []
        if res == '-':
            atom_feats = [safe_index(allowable_features['possible_amino_acids'], 'misc'),
                     safe_index(allowable_features['possible_atomic_num_list'], 'misc'),
                     safe_index(allowable_features['possible_atom_type_2'], 'misc'),
                     safe_index(allowable_features['possible_atom_type_3'], 'misc')]
        else:
            atom_feats.append(safe_index(allowable_features['possible_amino_acids'], res_long))
            if i >= len(res_order):
                atom_feats.extend([safe_index(allowable_features['possible_atomic_num_list'], 'misc'),
                                   safe_index(allowable_features['possible_atom_type_2'], 'misc'),
                                   safe_index(allowable_features['possible_atom_type_3'], 'misc')])
            else:
                atom_name = res_order[i]
                try:
                    atomic_num = periodic_table.GetAtomicNumber(atom_name[:1])
                except:
                    print("element", res_order[i][:1], 'not found')
                    atomic_num = -1

                atom_feats.extend([safe_index(allowable_features['possible_atomic_num_list'], atomic_num),
                                   safe_index(allowable_features['possible_atom_type_2'], (atom_name + '*')[:2]),
                                   safe_index(allowable_features['possible_atom_type_3'], atom_name)])
        feats.append(atom_feats)
    feats = np.asarray(feats)
    return feats

def get_binding_pockets(protein_graph, ligand_graphs: List[Data], lig_rec_cutoff:float, lig_atom_cutoff: float):
    '''
    adds the binding pocket of the protein to the ligand graphs based on distance.
    computes multiple ligands at once for performance 
    '''
    lig_poses = torch.cat([g["ligand"].pos for g in ligand_graphs], dim=0)
        
    lig_receptor_batch = radius(
        protein_graph["receptor"].pos,
        lig_poses,
        lig_rec_cutoff,
        max_num_neighbors=9999,
    )

    lig_atom_batch = radius(
        protein_graph["atom"].pos,
        lig_poses,
        lig_atom_cutoff,
        max_num_neighbors=9999,
        num_workers=4,
    )

    for i, ligand_graph in enumerate(ligand_graphs):
        num_lig = ligand_graph['ligand'].num_nodes
        #receptor receptor & lig receptor
        lig_rec = lig_receptor_batch[:,lig_receptor_batch[0] < num_lig] # !!!!!!!!!!!!!!!!!!!!!!!
        curr_rec = torch.unique(lig_rec[1])
        rec_nodes_x = protein_graph['receptor'].x[curr_rec]
        rec_nodes_pos = protein_graph['receptor'].pos[curr_rec]
        rec_rec = protein_graph['receptor','receptor'].edge_index
        rec_rec = rec_rec[:,torch.all(torch.isin(rec_rec, curr_rec),dim=0)]
        rec_rec = torch.searchsorted(curr_rec,rec_rec)
        lig_rec[1] = torch.searchsorted(curr_rec,lig_rec[1])
        #atom atom & lig atom
        lig_atom = lig_atom_batch[:,lig_atom_batch[0] < num_lig] # batch edges
        curr_atom = torch.unique(lig_atom[1])
        atom_nodes_x = protein_graph['atom'].x[curr_atom]
        atom_nodes_pos = protein_graph['atom'].pos[curr_atom]
        atom_atom = protein_graph['atom','atom'].edge_index
        atom_atom = atom_atom[:,torch.all(torch.isin(atom_atom, curr_atom),dim=0)]
        atom_atom = torch.searchsorted(curr_atom,atom_atom)
        lig_atom[1] = torch.searchsorted(curr_atom,lig_atom[1])
        #atom receptor
        atom_receptor = protein_graph['atom','receptor'].edge_index
        relevant_edges = torch.isin(atom_receptor[0], curr_atom) & torch.isin(atom_receptor[1], curr_rec)
        atom_receptor = atom_receptor[:,relevant_edges]
        atom_receptor[0] = torch.searchsorted(curr_atom,atom_receptor[0])
        atom_receptor[1] = torch.searchsorted(curr_rec,atom_receptor[1])
        #building
        ligand_graph['receptor'].x = rec_nodes_x
        ligand_graph['receptor'].pos = rec_nodes_pos
        
        ligand_graph['atom'].x = atom_nodes_x
        ligand_graph['atom'].pos = atom_nodes_pos
        
        ligand_graph['receptor','receptor'].edge_index = rec_rec
        ligand_graph['atom','atom'].edge_index = atom_atom
        ligand_graph['atom','receptor'].edge_index = atom_receptor
        ligand_graph['ligand','receptor'].edge_index = lig_rec
        ligand_graph['ligand','atom'].edge_index = lig_atom
        #prepare for next
        lig_receptor_batch[0] = lig_receptor_batch[0]-num_lig
        lig_atom_batch[0] = lig_atom_batch[0]-num_lig
        lig_receptor_batch = lig_receptor_batch[:,lig_receptor_batch[0] >= 0] # batch edges
        lig_atom_batch = lig_atom_batch[:,lig_atom_batch[0] >= 0] # batch edges

def local_isin(elements, test_elements, both_relevant=False):
    res = torch.zeros_like(elements).bool()
    test_min, test_max = test_elements[0], test_elements[-1] # sorted
    arb_elements_mask = (test_min <= elements) & (elements <= test_max)
    if both_relevant:
        arb_elements_mask = torch.all(arb_elements_mask,dim=0)
        res[:,arb_elements_mask] = torch.isin(elements[:,arb_elements_mask],test_elements)
        return torch.all(res,dim=0)
    res[arb_elements_mask] = torch.isin(elements[arb_elements_mask],test_elements)
    return res
    
def get_sub_edges(edges, sub_nodes_0=None, sub_nodes_1=None):
    if sub_nodes_1 is None:
        edges = edges[:,local_isin(edges,sub_nodes_0, both_relevant=True)]
    else:
        edges = edges[:,local_isin(edges[0], sub_nodes_0) & local_isin(edges[1], sub_nodes_1)]
    return edges
    

def get_binding_pockets2(protein_graph, ligand_graph: List[Data], lig_rec_edges: torch.Tensor, lig_atom_edges: Optional[torch.Tensor]):
    '''
    adds the binding pocket of the protein to the ligand graphs based on distance.
    computes multiple ligands at once for performance 
    '''
    do_atoms = (lig_atom_edges is not None)
    lig_atom_edges = lig_atom_edges.clone()
    lig_rec_edges = lig_rec_edges.clone()
    times = []    
    #receptor receptor & lig receptor
    curr_rec = torch.unique(lig_rec_edges[1])
    rec_nodes_x = protein_graph['receptor'].x[curr_rec]
    rec_nodes_pos = protein_graph['receptor'].pos[curr_rec]
    rec_rec = protein_graph['receptor','receptor'].edge_index
    rec_rec = get_sub_edges(rec_rec, curr_rec)
    rec_rec = torch.searchsorted(curr_rec,rec_rec)
    lig_rec_edges[1] = torch.searchsorted(curr_rec,lig_rec_edges[1])

    ligand_graph['receptor'].x = rec_nodes_x
    ligand_graph['receptor'].pos = rec_nodes_pos
    ligand_graph['receptor','receptor'].edge_index = rec_rec
    ligand_graph['ligand','receptor'].edge_index = lig_rec_edges
    
    #atom atom & lig atom
    if do_atoms:
        curr_atom = torch.unique(lig_atom_edges[1])
        atom_nodes_x = protein_graph['atom'].x[curr_atom]
        atom_nodes_pos = protein_graph['atom'].pos[curr_atom]
        atom_atom = protein_graph['atom','atom'].edge_index
        atom_atom = get_sub_edges(atom_atom, curr_atom)
        atom_atom = torch.searchsorted(curr_atom,atom_atom)
        lig_atom_edges[1] = torch.searchsorted(curr_atom,lig_atom_edges[1])
        
        ligand_graph['atom'].x = atom_nodes_x
        ligand_graph['atom'].pos = atom_nodes_pos
        ligand_graph['atom','atom'].edge_index = atom_atom
        ligand_graph['ligand','atom'].edge_index = lig_atom_edges
        
        #atom receptor
        atom_receptor = protein_graph['atom','receptor'].edge_index
        atom_receptor = get_sub_edges(atom_receptor, curr_atom, curr_rec)
        atom_receptor[0] = torch.searchsorted(curr_atom,atom_receptor[0])
        atom_receptor[1] = torch.searchsorted(curr_rec,atom_receptor[1])
    
        ligand_graph['atom','receptor'].edge_index = atom_receptor
        
    

def hide_sidechains(graph:HeteroData, show_idx=0):
    # lig lig
    sidechain_mask = torch.tensor(graph.sidechains_mask <= show_idx)
    core_indices = torch.where(sidechain_mask)[0]
    
    graph = graph
    graph['ligand'].x = graph['ligand'].x[core_indices]
    graph['ligand'].pos = graph['ligand'].pos[core_indices]
    lig_lig_edges = graph['ligand','ligand'].edge_index
    core_lig_lig_indices = torch.all(torch.isin(lig_lig_edges, core_indices),dim=0)
    core_lig_lig_edges = lig_lig_edges[:, core_lig_lig_indices]
    core_lig_lig_edges = torch.searchsorted(core_indices,core_lig_lig_edges)
    graph['ligand','ligand'].edge_index = core_lig_lig_edges
    graph['ligand','ligand'].edge_attr = graph['ligand','ligand'].edge_attr[core_lig_lig_indices]
    
    #lig atom
    lig_atom = graph['ligand','atom'].edge_index
    lig_atom = lig_atom[:,torch.isin(lig_atom[0], core_indices)]
    lig_atom[0] = torch.searchsorted(core_indices,lig_atom[0])
    graph['ligand','atom'].edge_index = lig_atom
    #lig rec
    lig_rec = graph['ligand','receptor'].edge_index
    lig_rec = lig_rec[:,torch.isin(lig_rec[0], core_indices)]
    lig_rec[0] = torch.searchsorted(core_indices,lig_rec[0])
    graph['ligand','receptor'].edge_index = lig_rec
    return graph

def get_lig_graph(mol, complex_graph, lig_max_radius=None, extra_atom_feats={}):
    atom_feats = lig_atom_featurizer(mol, extra_atom_feats)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]] if bond.GetBondType() != BT.UNSPECIFIED else [0, 0]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    if mol.GetNumConformers() > 0:
        lig_coords = torch.from_numpy(mol.GetConformer().GetPositions()).float()
        complex_graph['ligand'].pos = lig_coords
    
    radius_edges = radius_graph(lig_coords, lig_max_radius)
    
    complex_graph['ligand'].x = atom_feats
    complex_graph['ligand', 'lig_bond', 'ligand'].edge_index = torch.cat((edge_index, radius_edges),dim=-1)
    complex_graph['ligand', 'lig_bond', 'ligand'].edge_attr = torch.cat((edge_attr, torch.zeros(radius_edges.shape[1],edge_attr.shape[1])),dim=0)
    
    return


def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        with Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False) as supplier:
            mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        raise ValueError('Expect the format of the molecule_file to be '
                         'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)

    except Exception as e:
        # Print stacktrace
        import traceback
        msg = traceback.format_exc()
        get_logger().warning(f"Failed to process molecule: {molecule_file}\n{msg}")
        return None

    return mol

