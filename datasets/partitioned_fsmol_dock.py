from collections.abc import Iterable
from copy import deepcopy
import os.path as osp
from collections import defaultdict
import pickle
import random
import time
import traceback
from typing import Callable, List, Tuple
import concurrent
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from multiprocessing import Pool
import re
import torch
from torch_geometric.data import Dataset, HeteroData, makedirs, Batch
from torch_geometric.data.dataset import files_exist
from torch_geometric.nn.pool import radius
import prody as pr
from tqdm import tqdm
from datetime import datetime

from datasets.process_chem.process_mols import (
    get_binding_pockets2,
    get_lig_graph,
    moad_extract_receptor_structure,
    read_molecule,
    get_binding_pockets,
)
from esm import FastaBatchedDataset, pretrained

from datasets.process_chem.process_sidechains import get_core_and_chains, get_holes, get_mask_of_sidechains, get_mol_smiles, set_hole_ids
from utils.esm_utils import compute_ESM_embeddings
from utils.logging_utils import get_logger
from utils.map_file_manager import MapFileManager
from utils.protein_utils import get_sequences_from_protein


class FsDockDatasetPartitioned(Dataset):
    """
    tasks =
    assay_id, target_id, protein_path, ligand_path, label, type
    """

    saved_protein_file = "proteins.pt"
    saved_esm_file = "esm_embd.pt"

    def __init__(
        self,
        root,
        tasks: pd.DataFrame,
        transform=None,
        receptor_radius=15,
        ligand_radius=20,
        c_alpha_max_neighbors=24,
        remove_hs=False,
        all_atoms=True,
        atom_radius=5,
        atom_max_neighbors=8,
        knn_only_graph=False,
        num_workers=1,
        tokenizer=None,
        load_mols=False,
        random_max_angle=None,
        core_weight=0.5,
    ):
        if isinstance(tasks, str):
            tasks = pd.read_csv(tasks)
        self.logger = get_logger()
        self.tasks_df = tasks
        self.num_workers = num_workers
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.all_atoms = all_atoms
        self.receptor_radius = receptor_radius
        self.ligand_radius = ligand_radius
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        self.knn_only_graph = knn_only_graph
        self.core_weight = core_weight
        self.tasks_file = f"tasks_rh{remove_hs}_cw{core_weight}.pt"
        self.tasks_metadata_file  = f"tasks_metadata_rh{remove_hs}_cw{core_weight}.pt"
        self.ligands_file = f"ligands_cw{core_weight}.pt"
        self.saved_protein_graph_file = f"protein_graphs_rr{receptor_radius}_camn{c_alpha_max_neighbors}_amn{atom_max_neighbors}_kog{knn_only_graph}_aa{all_atoms}_ar{atom_radius}.pt"
        self.saved_ligand_sub_protein_file = f"sub_protein_ligand_edges_lr{ligand_radius}_la{ligand_radius}_"\
            f"rr{receptor_radius}_camn{c_alpha_max_neighbors}_amn{atom_max_neighbors}_kog{knn_only_graph}_aa{all_atoms}_ar{atom_radius}_rh{remove_hs}_cw{core_weight}.npz"
        self.tokenizer = tokenizer
        self.tasks = {}
        self.load_mols = load_mols
        super().__init__(root, transform)
        # if not hasattr(self, "protein_graphs"):
        #     self.load()
        metadata = torch.load(osp.join(self.processed_dir, self.tasks_metadata_file))
        self._indices = metadata['indices']
        self.tasks_target = metadata['tasks_target']
        self.random_max_angle = random_max_angle
        
    def get_partition_and_idxs(self, world_size=None):
        world_size = world_size or torch.distributed.get_world_size()
        target_indices = defaultdict(list)
        for task_name,i in self._indices:
            target_indices[self.tasks_target[task_name]].append((task_name,i))
        sorted_targets = sorted([(target, len(indices)) for target,indices in target_indices.items()], key=lambda x:x[1], reverse=True)
        bucket_sums = [0]*world_size
        bucket_targets = [[] for _ in range(world_size)]
        bucket_indices = [[] for _ in range(world_size)]
        for target, size in sorted_targets:
            min_index = bucket_sums.index(min(bucket_sums))
            bucket_targets[min_index].append(target)
            bucket_indices[min_index].extend(target_indices[target])
            # Update the sum of this bucket
            bucket_sums[min_index] += size
        
        return bucket_targets, bucket_indices

    def get_indices(self):
        split_indexes = []
        for task_name, task in self.tasks.items():
            for i in range(len(task["graphs"])):
                split_indexes.append(
                    (task_name, i)
                )
        return split_indexes

    def tokenize_smiles(self, graph):
        if self.tokenizer:
            core_tokens = self.tokenizer.encode(graph.core_smiles).ids
            sidechain_tokens = self.tokenizer.encode(graph.sidechains_smiles).ids
            graph.core_tokens = torch.tensor(core_tokens).unsqueeze(0)
            graph.sidechain_tokens = torch.tensor(sidechain_tokens).unsqueeze(0)
            sidechains = re.sub('\[[0-9]+\*\]','', graph.sidechains_smiles).split('.')
            sidechains = torch.tensor(list(map(lambda sch: self.tokenizer.encode(sch).ids, sidechains)))
            graph.split_sidechain_tokens = sidechains

    def connect_ligand_to_protein(self, task_name, idx, data):
        task = self.tasks[task_name]
        protein_graph = self.protein_graphs[task["target"]]
        sub_protein = self.sub_proteins.load(f'{task["name"]}_{idx}')
        rec_id, lig_rec, rec_rec, atom_id, lig_atom, atom_atom, atom_rec = sub_protein
        data["receptor"].x = protein_graph["receptor"].x[rec_id]
        data["receptor"].pos = protein_graph["receptor"].pos[rec_id]
        data["receptor", "receptor"].edge_index = rec_rec
        data["ligand", "receptor"].edge_index = lig_rec
        if self.all_atoms:
            data["atom"].x = protein_graph["atom"].x[atom_id]
            data["atom"].pos = protein_graph["atom"].pos[atom_id]
            data["atom", "atom"].edge_index = atom_atom
            data["ligand", "atom"].edge_index = lig_atom
            data["atom", "receptor"].edge_index = atom_rec

        return data

    def len(self):
        return len(self._indices)

    def _create_sub_task(self, task, idxs):
        sub_task = {}
        for key, value in task.items():
            if isinstance(value, str) or not isinstance(value, Iterable):
                sub_task[key] = value
            else:
                if key == "graphs":
                    sub_task[key] = []
                    for i in idxs:
                        graph = deepcopy(task["graphs"][i])
                        graph.sidechains_mask = torch.from_numpy(graph.sidechains_mask)
                        self.connect_ligand_to_protein(task["name"], i, graph)
                        self.tokenize_smiles(graph)
                        sub_task[key].append(graph)
                else:
                    sub_task[key] = deepcopy([value[i] for i in idxs])
        return sub_task

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            task_name, i = idx
        elif isinstance(idx,list):
            items = [self[s_idx] for s_idx in idx]
            return items
        else:
            task_name, i = self._indices[idx]
        
        graph = deepcopy(self.tasks[task_name]["graphs"][i])
        if self.random_max_angle is not None:
            graph["ligand"].pos = self.rotate_point_cloud(graph["ligand"].pos, self.random_max_angle)
        graph.task = task_name
        graph.sidechains_mask = torch.from_numpy(graph.sidechains_mask).to(torch.int)
        graph.hole_neighbors = torch.tensor(graph.hole_neighbors)
        self.connect_ligand_to_protein(self.tasks[task_name]["name"], i, graph)
        graph.num_sidechains = int(graph.num_sidechains)
        self.tokenize_smiles(graph)
        return graph

    def random_small_rotation_matrix(self, max_angle):
        # Random unit axis
        axis = torch.randn(3)
        axis = axis / axis.norm()
        
        # Small angle in [-max_angle, max_angle]
        angle = (torch.rand(1) * 2 - 1) * max_angle

        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
        return R

    def rotate_point_cloud(self, pc, max_angle):
        centroid = pc.mean(dim=0)
        pc_centered = pc - centroid

        R = self.random_small_rotation_matrix(max_angle)
        pc_rotated = pc_centered @ R.T
        pc_rotated += centroid

        return pc_rotated

    def get_task_metadata(self, task_name):
        task = self.tasks[task_name]
        data = {}
        for key, val in task.items():
            if not isinstance(val, list):
                data[key] = val
        return data

    def task_names(self):
        if hasattr(self, "_task_names"):
            return self._task_names
        return self.tasks_df["assay_id"].unique()

    def processed_file_names(self):
        names = [
            self.saved_protein_file,
            self.saved_protein_graph_file,
            self.saved_esm_file,
            self.saved_ligand_sub_protein_file,
            self.tasks_file,
            self.ligands_file,
            self.tasks_metadata_file
        ]
        return names

    def load(self, targets=[]):
        '''
        data is partitioned by target protein, so we load by target
        '''
        self.logger.info("started load")
        
        targets = set(targets)
        tasks = [task for task, target in self.tasks_target.items() if (target in targets or len(targets) == 0)]
        self.logger.info("started load ligands")
        with MapFileManager(osp.join(self.processed_dir, self.ligands_file),'r')as mf:
            self.ligands = {task:mf[task] for task in tasks}
        self.logger.info("started load tasks")
        with MapFileManager(osp.join(self.processed_dir, self.tasks_file),'r') as mf:
            self.tasks = {task:mf[task] for task in tasks}
        self.logger.info("started load prots")
        with MapFileManager(osp.join(self.processed_dir, self.saved_protein_graph_file),'r') as mf:
            if len(targets):
                self.protein_graphs = {target:mf[target] for target in targets}
            else:
                self.protein_graphs = mf.load_all()
        with MapFileManager(osp.join(self.processed_dir, self.ligands_file),'r') as mf:
            self.ligands = {task:mf[task] for task in tasks}
        sub_proteins_path = osp.join(
            self.processed_dir, self.saved_ligand_sub_protein_file
        )
        self.sub_proteins = MapFileManager(sub_proteins_path, 'r').open()
        self.logger.info("finished load")

    def process(self):
        self.logger.info("started process_ligands")
        self.process_ligands()
        self.logger.info("started process_tasks")
        self.process_tasks()
        self.logger.info("started process_proteins")
        self.process_proteins()
        self.logger.info("started process_ligand_protein_edges")
        self.process_sub_proteins()
        self.logger.info("finished process")
        self.process_metadata()

    def process_ligands(self):
        path = osp.join(self.processed_dir, self.ligands_file)
        if files_exist([path]):
            with MapFileManager(path, 'r') as mf:
                self.ligands = mf.load_all()
            return

        task_groups = self.tasks_df.groupby("assay_id")

        ligand_build_params = []
        tasks_size = {}
        for assay_id, grouped_rows in task_groups:
            tasks_size[assay_id] = len(grouped_rows)
            for idx, (_, row) in enumerate(grouped_rows.iterrows()):
                ligand_build_params.append((assay_id, idx, row["ligand_path"], self.core_weight))
        ligands = {k: [None] * v for k, v in tasks_size.items()}
        with tqdm(total=len(ligand_build_params), desc="build ligands") as progress_bar:
            with torch.multiprocessing.Pool(self.num_workers) as pool:
                for task_name, idx, chem_data in pool.imap(
                    self.process_ligand, ligand_build_params
                ):
                    ligands[task_name][idx] = chem_data
                    progress_bar.update()
        self.ligands = ligands
        with MapFileManager(path, 'w') as mf:
            for task_name, task in ligands.items():
                mf.save(task, task_name)

    @staticmethod
    def process_ligand(args):
        res = {}
        try:
            task_name, idx, ligand_path, core_weight = args
            ligand = read_molecule(ligand_path, sanitize=True)
            if ligand is None:
                return task_name, idx, res
            smiles = get_mol_smiles(ligand)
            res['ligand']=ligand
            res['smiles']=smiles
            core, core_smiles, sidechains, sidechains_smiles, hole_neighbors = get_core_and_chains(
                ligand, core_weight
            )
            if core is None:
                get_logger().warning(
                    f"couldnt extract core: {task_name}, {idx}, {Chem.MolToSmiles(ligand)}"
                )
                return task_name, idx, res
            res['core']=core
            res['core_smiles']=core_smiles
            res['sidechains']=sidechains
            res['sidechains_smiles']=sidechains_smiles
            res['hole_neighbors']=hole_neighbors
            sidechains_mask = get_mask_of_sidechains(ligand, sidechains)
            res['num_sidechains']= sidechains_mask.max().item()
            hole_features = get_holes(ligand)
            extra_atom_feats = {'__holeIdx': hole_features}
            res['sidechains_mask']=sidechains_mask
            res['extra_atom_feats']=extra_atom_feats
            return (
                task_name,
                idx,
                res
            )
        except Exception as e:
            get_logger().error(
                f"Error processing ligand {task_name}, {idx}, {Chem.MolToSmiles(ligand)}"
            )
            get_logger().error(traceback.format_exc())
            return task_name, idx, res

    def process_sub_proteins(self):
        path = osp.join(self.processed_dir, self.saved_ligand_sub_protein_file)
        if files_exist([path]):
            self.sub_proteins = MapFileManager(path, 'r').open()
            return
        with MapFileManager(path, 'w') as mf:
            for task_name, task in tqdm(
                self.tasks.items(), desc="processsing sub proteins"
            ):
                protein_graph = self.protein_graphs[task["target"]]
                ligand_graphs = task["graphs"]
                task_sub_proteins = self.get_sub_prot_for_ligs(
                    protein_graph, ligand_graphs, self.ligand_radius, self.atom_radius
                )
                for i, sub_prot in enumerate(task_sub_proteins):
                    mf.save(sub_prot,f'{task_name}_{i}')
        self.sub_proteins = MapFileManager(path, 'r').open()
        

    @staticmethod
    def get_sub_prot_for_ligs(
        protein_graph, ligand_graphs, rec_cutoff_distance, atom_cutoff_distance=None
    ):
        all_atoms = atom_cutoff_distance is not None
        lig_poses = torch.cat([g["ligand"].pos for g in ligand_graphs], dim=0)
        lig_slices = torch.tensor([0, *(len(g["ligand"].pos) for g in ligand_graphs)])
        lig_slices = torch.cumsum(lig_slices, dim=0)
        lig_rec_batch = radius(
            protein_graph["receptor"].pos,
            lig_poses,
            rec_cutoff_distance,
            max_num_neighbors=9999,
        )
        if all_atoms:
            lig_atom_batch = radius(
                protein_graph["atom"].pos,
                lig_poses,
                atom_cutoff_distance,
                max_num_neighbors=9999,
            )
        lig_rec = None
        rec_id = None
        lig_atom = None
        atom_id = None
        rec_rec = None
        atom_rec = None
        atom_atom = None
        sub_proteins = []
        lig_rec_slices = torch.searchsorted(lig_rec_batch[0], lig_slices)
        lig_atom_slices = (
            torch.searchsorted(lig_atom_batch[0], lig_slices) if all_atoms else None
        )
        rec_rec_orig = protein_graph["receptor", "receptor"].edge_index
        rec_rec_orig = rec_rec_orig[
            :, torch.all(torch.isin(rec_rec_orig, lig_rec_batch[1]), dim=0)
        ]
        if all_atoms:
            atom_atom_orig = protein_graph["atom", "atom"].edge_index
            atom_atom_orig = atom_atom_orig[
                :, torch.all(torch.isin(atom_atom_orig, lig_atom_batch[1]), dim=0)
            ]
            atom_rec_orig = protein_graph["atom", "receptor"].edge_index
            atom_rec_orig_e = torch.isin(
                atom_rec_orig[0], lig_atom_batch[1]
            ) & torch.isin(atom_rec_orig[1], lig_rec_batch[1])
            atom_rec_orig = atom_rec_orig[:, atom_rec_orig_e]
        for i, lig_i_start in enumerate(lig_slices[:-1]):
            rec_e_start, rec_e_end = lig_rec_slices[i : i + 2]
            lig_rec = lig_rec_batch[:, rec_e_start:rec_e_end].clone()
            if lig_rec.numel():
                lig_rec[0] = lig_rec[0] - lig_i_start
            if all_atoms:
                atom_e_start, atom_e_end = lig_atom_slices[i : i + 2]
                lig_atom = lig_atom_batch[:, atom_e_start:atom_e_end].clone()
                if lig_atom.numel():
                    lig_atom[0] = lig_atom[0] - lig_i_start
            # now we have all the edges to the full protein
            rec_id = torch.unique(lig_rec[1])
            if all_atoms:
                atom_id = torch.unique(lig_atom[1])

            rec_rec = rec_rec_orig[
                :, torch.all(torch.isin(rec_rec_orig, rec_id), dim=0)
            ]
            rec_rec = torch.searchsorted(rec_id, rec_rec)
            lig_rec[1] = torch.searchsorted(rec_id, lig_rec[1])

            if all_atoms:
                atom_atom = atom_atom_orig[
                    :, torch.all(torch.isin(atom_atom_orig, atom_id), dim=0)
                ]
                atom_atom = torch.searchsorted(atom_id, atom_atom)
                lig_atom[1] = torch.searchsorted(atom_id, lig_atom[1])

                relevant_edges = torch.isin(atom_rec_orig[0], atom_id) & torch.isin(
                    atom_rec_orig[1], rec_id
                )
                atom_rec = atom_rec_orig[:, relevant_edges]
                atom_rec[0] = torch.searchsorted(atom_id, atom_rec[0])
                atom_rec[1] = torch.searchsorted(rec_id, atom_rec[1])
            sub_proteins.append(
                (rec_id, lig_rec, rec_rec, atom_id, lig_atom, atom_atom, atom_rec)
            )
        return sub_proteins

    def process_tasks(self):
        path = osp.join(self.processed_dir, self.tasks_file)
        if files_exist([path]):
            with MapFileManager(path, 'r') as mf:
                self.tasks = mf.load_all()
            return

        task_groups = self.tasks_df.groupby("assay_id")

        tasks = {}
        with torch.multiprocessing.Pool(self.num_workers) as p:
            for assay_id, grouped_rows in tqdm(task_groups, desc="processing tasks"):
                try:
                    task = self.process_task(
                        assay_id, grouped_rows, self.ligands[assay_id]
                    )
                    if task is not None:
                        tasks[assay_id] = task
                except Exception as e:
                    self.logger.error(
                        f"failed to process task {assay_id}, {traceback.format_exc()}"
                    )

        self.tasks = tasks
        with MapFileManager(path, 'w') as mf:
            for task_name, task in tasks.items():
                mf.save(task, task_name)
        

    def process_metadata(self):
        tasks_target = {}
        for task_name, task in self.tasks.items():
            tasks_target[task_name] = task['target']
        indices = self.get_indices()
        torch.save({'indices':indices, 'tasks_target':tasks_target} , osp.join(self.processed_dir, self.tasks_metadata_file))

    def process_task(self, assay_id, grouped_rows, ligands):
        task = {
            "name": assay_id,
            "target": "",
            "activity_type": "",
            "graphs": [],
            "labels": [],
        }
        for (idx, row), ligand_data in zip(grouped_rows.iterrows(), ligands):
            if ligand_data.get('sidechains_mask') is None:
                continue
            ligand_graph = HeteroData()
            get_lig_graph(ligand_data['ligand'], ligand_graph, self.ligand_radius, ligand_data['extra_atom_feats'])
            ligand_graph.smiles = ligand_data['smiles']
            ligand_graph.core_smiles = ligand_data['core_smiles']
            ligand_graph.sidechains_smiles = ligand_data['sidechains_smiles']
            ligand_graph.sidechains_mask = ligand_data['sidechains_mask']
            ligand_graph.hole_neighbors = ligand_data['hole_neighbors']
            ligand_graph.num_sidechains = ligand_data['num_sidechains']
            ligand_graph.activity_type = row["type"]
            ligand_graph.label = row["label"]
            task["activity_type"] = row["type"]
            task["target"] = row["target_id"]
            task["labels"].append(row["label"])
            task["graphs"].append(ligand_graph)
        if task["target"] == "":  # nothing is good :(
            return None
        return task

    def process_proteins(self):
        if files_exist([osp.join(self.processed_dir, self.saved_protein_graph_file)]):
            self.protein_graphs = self.generate_protein_graphs(None, None)
            return
        proteins = self.build_proteins_from_pdb()
        lm_embeddings = self.generate_ESM(proteins)
        self.protein_graphs = self.generate_protein_graphs(proteins, lm_embeddings)

    def build_proteins_from_pdb(self):
        protein_path = osp.join(self.processed_dir, self.saved_protein_file)
        if not files_exist([protein_path]):
            proteins = {}
            tasks = self.tasks_df.groupby("target_id")
            for protein_id, grouped_rows in tqdm(tasks):
                path = grouped_rows.iloc[0]["protein_path"]
                protein = pr.parsePDB(path)
                proteins[protein_id] = protein

            torch.save(proteins, protein_path)
        else:
            proteins = torch.load(protein_path)
        return proteins

    def generate_protein_graphs(self, proteins, lm_embeddings):
        protein_graph_path = osp.join(self.processed_dir, self.saved_protein_graph_file)
        if files_exist([protein_graph_path]):
            with MapFileManager(protein_graph_path, 'r') as mf:
                return mf.load_all()
        else:
            protein_graphs = {}
            for protein_id, protein in tqdm(
                proteins.items(), desc="Generating protein graphs"
            ):
                protein_graph = HeteroData()
                moad_extract_receptor_structure(
                    pdb=protein,
                    complex_graph=protein_graph,
                    neighbor_cutoff=self.receptor_radius,
                    max_neighbors=self.c_alpha_max_neighbors,
                    lm_embeddings=lm_embeddings[protein_id],
                    knn_only_graph=self.knn_only_graph,
                    all_atoms=self.all_atoms,
                    atom_cutoff=self.atom_radius,
                    atom_max_neighbors=self.atom_max_neighbors,
                )
                
                protein_graphs[protein_id] = protein_graph
            with MapFileManager(protein_graph_path, 'w') as mf:
                for prot_name, prot in protein_graphs.items():
                    mf.save(prot, prot_name)
        return protein_graphs

    def generate_ESM(self, proteins):
        esm_path = osp.join(self.processed_dir, self.saved_esm_file)
        if not files_exist([esm_path]):

            print("Generating ESM language model embeddings")
            model_location = "esm2_t33_650M_UR50D"
            model, alphabet = pretrained.load_model_and_alphabet(model_location)
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()

            protein_sequences = {
                target: get_sequences_from_protein(prot)
                for target, prot in proteins.items()
            }
            labels, sequences = [], []
            for target, sequence in protein_sequences.items():
                s = sequence.split(":")
                sequences.extend(s)
                labels.extend([target + "_chain_" + str(j) for j in range(len(s))])

            lm_embeddings = compute_ESM_embeddings(model, alphabet, labels, sequences)

            for target, sequence in protein_sequences.items():
                s = sequence.split(":")
                lm_embeddings[target] = [
                    lm_embeddings[f"{target}_chain_{j}"] for j in range(len(s))
                ]
            torch.save(lm_embeddings, esm_path)
        else:
            lm_embeddings = torch.load(esm_path)
        return lm_embeddings
