from collections import defaultdict
import numpy as np
import random
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd
import scaffoldgraph as sg
import re
import itertools
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch
try:
    from utils.logging_utils import get_logger
except:
    import logging
    def get_logger():
        return logging.getLogger(__name__)
from rdkit.Chem import rdmolops

'''
very important:
https://github.com/rdkit/rdkit/discussions/5458
https://github.com/rdkit/rdkit/discussions/6844
'''

logger = get_logger()
def get_num_fused_rings(mol):
    """rings that share at least one atom with another ring"""
    num_fused_rings = set()
    atom_rings = [set(cur_ring) for cur_ring in mol.GetRingInfo().AtomRings()]
    for i in range(len(atom_rings)):
        for j in range(i + 1, len(atom_rings)):
            if not atom_rings[i].isdisjoint(atom_rings[j]):
                num_fused_rings.update([i, j])
    return len(num_fused_rings)



def tree_frags_from_mol(mol, weight_ratio=0.5):
    scaffold = sg.core.Scaffold(sg.core.fragment.get_murcko_scaffold(mol))
    rdmolops.RemoveStereochemistry(scaffold.mol)
    parents = [scaffold]
    # fragmenter = sg.core.MurckoRingFragmenter(use_scheme_4=True)
    fragmenter = sg.core.MurckoRingSystemFragmenter()
    minimal_core_weight = AllChem.CalcExactMolWt(mol) * weight_ratio 
    # if AllChem.CalcExactMolWt(scaffold.mol) <= minimal_core_weight:
    #     return mol
    rules = sg.prioritization.original_ruleset
    original_rings_count = scaffold.rings.count

    def _next_scaffold(child):
        next_parents = [p for p in fragmenter.fragment(child) if (p and AllChem.CalcExactMolWt(p.mol)> minimal_core_weight)]
        if not next_parents:
            return
        next_parent = rules(child, next_parents)
        parents.append(next_parent)
        if next_parent.rings.count > 1:
            _next_scaffold(next_parent)
    try:
        _next_scaffold(scaffold)
    except Exception as e:
        logger.error(f"Error in tree_frags_from_mol: {e}, {Chem.MolToSmiles(mol)}")
        
    for p in reversed(parents):
        try:
            sidechains = Chem.ReplaceCore(mol, p.mol)
            Chem.SanitizeMol(sidechains)
            if original_rings_count == sidechains.GetRingInfo().NumRings() + p.rings.count:
                return p.mol
        except Exception as e:
            logger.error(f"Error processing scaffold {p.smiles}: {e}")
            continue
    return None

def get_mol_smiles(mol):
    if isinstance(mol,str):
        return Chem.CanonSmiles(mol)
    return Chem.MolToSmiles(mol)

def get_core_and_chains(m1,core_weight):
    error = [None] * 3
    if  isinstance(m1,str):
        m1 = Chem.MolFromSmiles(m1)
    if m1 is None:
        return error
    for a in m1.GetAtoms():
        a.SetIntProp("__origIdx", a.GetIdx())
    # clean_core = MurckoScaffold.GetScaffoldForMol(m1)
    clean_core = tree_frags_from_mol(m1,core_weight)
    if clean_core is None:
        return error
    
    core, sidechains = _get_core_and_chains(m1, clean_core)
    # core = Chem.ReplaceSidechains(m1, clean_core)
    # sidechains = Chem.ReplaceCore(m1, clean_core)
    # sidechains = [] if sidechains is None else list(Chem.GetMolFrags(sidechains, asMols=True))
    if core is None or len(sidechains)==0:
        return error
    return clean_core, sidechains , get_slicing_data(m1,clean_core, core, sidechains)

def _get_core_and_chains(mol, clean_core):
    core_idxs = set()
    for atom in clean_core.GetAtoms():
        if atom.HasProp("__origIdx"):
            core_idxs.add(atom.GetIntProp("__origIdx"))
    if not core_idxs:
        raise ValueError("Core atoms do not have __origIdx property")
    if len(core_idxs) == mol.GetNumAtoms():
        return mol, []

    bonds_to_break = []
    dummy_labels = [] # List of (label_for_u_side, label_for_v_side)
    for bond in mol.GetBonds():
        u = bond.GetBeginAtom()
        v = bond.GetEndAtom()
        u_idx = u.GetIntProp("__origIdx")
        v_idx = v.GetIntProp("__origIdx")
        u_is_core = u_idx in core_idxs
        v_is_core = v_idx in core_idxs
        
        # We only care if one is core and the other is NOT core (XOR)
        if u_is_core ^ v_is_core:
            bonds_to_break.append(bond.GetIdx())
            dummy_labels.append((len(bonds_to_break), len(bonds_to_break)))
            
    # 3. Perform the fragmentation
    # This returns a single molecule with broken bonds and dummy atoms added
    fragmented_mol = Chem.FragmentOnBonds(
        mol, 
        bonds_to_break, 
        dummyLabels=dummy_labels
    )
    
    # 4. Separate fragments
    frags = Chem.GetMolFrags(fragmented_mol, asMols=True)
    
    sidechains = []
    core = None
    seen_idxs = set()
    for frag in frags:
        # We need to distinguish the "Core Fragment" from "Sidechain Fragments".
        # A simple heuristic: Check if the atoms in this fragment are effectively the core atoms.
        # However, since we added dummy atoms, we check if ANY non-dummy atom in this fragment
        # is NOT in the core_idxs set.
        
        idxs = set(a.GetIntProp("__origIdx") for a in frag.GetAtoms() if a.HasProp("__origIdx"))
        assert len(seen_idxs.intersection(idxs)) == 0, "Fragments should be disjoint"
        seen_idxs.update(idxs)
        
        if all(idx in core_idxs for idx in idxs):
            # All atoms in this fragment are from the core
            core = frag
            continue    
        
        elif all(idx not in core_idxs for idx in idxs):
            sidechains.append(frag)
        else:
            # This fragment has a mix of core and sidechain atoms, which should not happen
            raise ValueError("Fragment contains a mix of core and sidechain atoms.")
    
    assert len(seen_idxs) == mol.GetNumAtoms(), "Some atoms were not assigned to core or sidechains"
    
    #  sidechains might not be sorted by the order i predicted, SO STUPIDDDDD of me aaaaaaaaa
    sidechains.sort(key=lambda s: int(re.search(r'\[(\d+)', Chem.MolToSmiles(s)).group(1)))
    
    return core, sidechains

# def get_mask_of_sidechains(full_mol,frags):
#     mask = np.zeros(full_mol.GetNumAtoms())
#     for i, frag in enumerate(frags):
#         frag_indices = [a.GetIntProp("__origIdx") for a in frag.GetAtoms() if a.HasProp("__origIdx")]
#         mask[frag_indices] = i + 1            
#     return mask

def get_slicing_data(mol, clean_core, core, frags):
    core_mask = np.zeros(mol.GetNumAtoms(), dtype=bool)
    core_mask[[a.GetIntProp("__origIdx") for a in clean_core.GetAtoms() if a.HasProp("__origIdx")]] = True
    
    hole_count = np.zeros(mol.GetNumAtoms(),dtype=int)
    frag_idxs = np.zeros(mol.GetNumAtoms(),dtype=int) - 1
    frag_hole = np.zeros(len(frags),dtype=int)
    for i, frag in enumerate(frags):
        frag_idxs[[a.GetIntProp("__origIdx") for a in frag.GetAtoms() if a.HasProp("__origIdx")]] = i
    assert frag_idxs[core_mask].max() == -1 , "Core atoms should not have fragment indices"
    for a in mol.GetAtoms():
        a_id = a.GetIntProp("__origIdx")
        if not core_mask[a_id]: # already in side chain
            continue
        for neighbor in a.GetNeighbors(): 
            n_id = neighbor.GetIntProp("__origIdx")
            if not core_mask[n_id]: # "a" not in sidechain, "n" is, Good
                n_frag_id = frag_idxs[n_id]
                frag_hole[n_frag_id] = a_id
                hole_count[a_id] += 1
    return  dict(
        core_mask=core_mask,
        hole_count=hole_count,
        frag_idxs=frag_idxs,
        frag_hole = frag_hole,
        num_frags=len(frags),
        core_smiles=Chem.MolToSmiles(core),
        frag_smiles = list(map(lambda f: Chem.MolToSmiles(f).split(']',1)[1], frags))
    )
    
def get_fake_slicing_data(extended_mol, mol):
    core_mask = np.zeros(mol.GetNumAtoms(), dtype=bool)
    core_mask[:] = True
    hole_count = np.zeros(mol.GetNumAtoms(),dtype=int)
    frag_idxs = np.zeros(mol.GetNumAtoms(),dtype=int) - 1
    
    fake_atoms = [a for a in extended_mol.GetAtoms() if not  a.HasProp("__origIdx")]
    
    frag_hole = np.zeros(len(fake_atoms),dtype=int)

    for i, a in enumerate(fake_atoms):
        for neighbor in a.GetNeighbors(): 
            n_id = neighbor.GetIntProp("__origIdx")
            frag_hole[i] = n_id
            hole_count[n_id] += 1

    return  dict(
        core_mask=core_mask,
        hole_count=hole_count,
        frag_idxs=frag_idxs,
        frag_hole = frag_hole,
        num_frags=len(fake_atoms),
        core_smiles=Chem.MolToSmiles(extended_mol),
        frag_smiles = ['']* len(fake_atoms)
    )
    
    
    
   
def get_holes(mol):
    hole_lst = np.zeros(mol.GetNumAtoms())
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.HasProp("__holeIdx"):
            hole_lst[i] = atom.GetIntProp("__holeIdx")
    return hole_lst

def smiles_valid(smiles,verbose=False):
    if smiles is None:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return True
    # print(smiles)
    return False

def get_fp(mol):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            return
    fp_obj = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048,
                                                   useChirality=False)
    fp = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp_obj, fp)
    return fp


def calc_tani_sim(mol1_smiles, mol2_smiles):
    mol1 = Chem.MolFromSmiles(mol1_smiles)
    mol2 = Chem.MolFromSmiles(mol2_smiles)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048, useChirality=False)
    tani_sim = DataStructs.FingerprintSimilarity(fp1, fp2, metric=DataStructs.TanimotoSimilarity)
    return tani_sim

def reconstruct_from_core_and_chains(core, chains):
    chains = Chem.MolFromSmiles(chains)
    core_clean = Chem.MolFromSmiles(core)
    if core_clean is None or chains is None:
        return None
    try:
        sidechain_mols = Chem.GetMolFrags(chains, asMols=True)
    except:
        return None
    for mol in sidechain_mols:
        if len([atom.GetSmarts() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"]) == 0:
            return None
    sidechain_tags = [[atom.GetSmarts() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"][0]
                       for mol in sidechain_mols]
    sidechain_indexes = [[atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"][0]
                          for mol in sidechain_mols]
    sidechain_dict = dict(zip(sidechain_tags, zip(sidechain_mols, sidechain_indexes)))

    core_sidechain_tags = [atom.GetSmarts() for atom in core_clean.GetAtoms() if atom.GetSymbol() == "*"]
    core_sidechain_tags = [re.sub(r'(\[\d+\*).*\]', r'\1]', x) for x in core_sidechain_tags]
    current_core = core_clean
    for tag in core_sidechain_tags:
        replacement = sidechain_dict.get(tag, None)
        if replacement is None:
            return None
        new_core = Chem.ReplaceSubstructs(current_core,
                                          Chem.MolFromSmiles(tag),
                                          replacement[0],
                                          replacementConnectionPoint=sidechain_dict[tag][1],
                                          useChirality=1)
        if new_core[0] is None:
            return None
        current_core = new_core[0]
    reconstructed_smiles = Chem.MolToSmiles(current_core)
    reconstructed_smiles_clean = re.sub(r'\[\d+\*\]', '', reconstructed_smiles)
    if not smiles_valid(reconstructed_smiles_clean):
        return None
    recon = Chem.MolToSmiles(Chem.MolFromSmiles(reconstructed_smiles_clean))
    canon = Chem.CanonSmiles(recon, useChiral=0)
    return canon


def isotopize_dummies(fragment, isotope):
    for atom in fragment.GetAtoms():
        if atom.GetSymbol() == "*":
            atom.SetIsotope(isotope)
    return fragment


def add_attachment_points(mol, n, seed=None, fg_weight=0, fg_list=[]):
    if mol is None:
        return None
    if seed is not None:
        random.seed(seed)
    if len(fg_list) == 0:
        fg_list = [(1.0, "[c,C][H]", "C*")]
    else:
        epsilon = 1.e-10
        fg_weights = itertools.accumulate([fg_weight / len(fg_list)
                                           for i in range(len(fg_list))]
                                          + [1.0 + epsilon - fg_weight])
        fg_list = list(zip(fg_weights,
                           [x[0] for x in fg_list] + ["[c,C][H]"],
                           [x[1] for x in fg_list] + ["C*"]))

    for a in mol.GetAtoms():
        a.SetIntProp("__origIdx", a.GetIdx())
    current_attachment_index = 1
    current_mol = mol
    for i in range(n):
        current_mol = Chem.AddHs(current_mol)
        current_mol.UpdatePropertyCache()
        next_mol = []
        max_tries = 100
        current_try = 0
        while len(next_mol) == 0:
            the_choice = [x for x in fg_list if x[0] >= random.random()][0]
            the_target = Chem.MolFromSmarts(the_choice[1])
            the_replacement = isotopize_dummies(Chem.MolFromSmiles(the_choice[2]), current_attachment_index)
            next_mol = Chem.ReplaceSubstructs(current_mol, the_target, the_replacement)
            current_try += 1
            if current_try >= max_tries:
                break  # we failed
        if current_try >= max_tries:
            continue  # skip and try again (we will return less than n attachment points)
        current_attachment_index += 1
        current_mol = random.choice(next_mol)
        current_mol = Chem.RemoveHs(current_mol)
        current_mol.UpdatePropertyCache()

        # a little stupid but i need to find after every iteration 
        # what is the new atom that was added to mark him with the old idx
        new_atom = None
        idxs = set(range(mol.GetNumAtoms()))
        for atom in current_mol.GetAtoms():
            if atom.GetSymbol() == "*":
                continue
            if atom.HasProp("__origIdx"):
                idxs.remove(atom.GetIntProp("__origIdx"))
            else:
                new_atom = atom
        new_atom.SetIntProp("__origIdx", idxs.pop())

    return mol, [], get_fake_slicing_data(current_mol, mol)


def add_frag_place_holder(graph):
    graph = graph.clone()
    frag_idxs = graph['ligand'].frag_idxs
    # centers = []
    # for i in range(graph['ligand'].frag_hole.shape[0]):
    #     mask = frag_idxs == i
    #     if mask.sum() == 0:
    #         continue
    #     center = graph['ligand'].pos[mask].mean(0, keepdim=True)
    #     centers.append(center)
    # graph['frag'].pos = torch.cat(centers, dim=0)
    graph['frag'].pos = graph['ligand'].pos[graph['ligand'].frag_hole]
    
    graph['frag'].x = torch.zeros(graph['frag'].pos.shape[0],1)
    graph['ligand','frag'].edge_index = torch.cat([graph['ligand'].frag_hole.unsqueeze(0), torch.arange(graph['ligand'].frag_hole.shape[0]).unsqueeze(0)], dim=0)
    #  edges
    for dst in ['atom', 'receptor']:
        edges = []
        for i in range(graph['ligand'].frag_hole.shape[0]):
            curr_frag_nodes = graph['ligand'].frag_idxs == i
            fe = graph['ligand',dst].edge_index
            fe = fe[:, curr_frag_nodes[fe[0]]]
            fe[0] = i
            edges.append(fe)
        if len(edges)>0:
            graph['frag', dst].edge_index = torch.cat(edges, dim=1)
            
            
    return graph
    
        
    