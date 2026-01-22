from collections.abc import Iterable
from copy import deepcopy
import os.path as osp
from collections import defaultdict
import pickle
import random
import re
import time
import traceback
from typing import Callable, List, Tuple
import concurrent
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from multiprocessing import Pool

import torch
from torch_geometric.data import Dataset, HeteroData, makedirs, Batch
from torch_geometric.data.dataset import files_exist
from torch_geometric.nn.pool import radius
import prody as pr
from tqdm import tqdm
from datasets.process_chem.process_mols import (
    get_binding_pockets2,
    get_lig_graph,
    moad_extract_receptor_structure,
    read_molecule,
    get_binding_pockets,
    get_sub_prot_for_ligs
)
from esm import FastaBatchedDataset, pretrained

from datasets.process_chem.process_sidechains import add_frag_place_holder, get_core_and_chains, get_holes, get_mol_smiles
from utils.esm_utils import compute_ESM_embeddings
from utils.logging_utils import get_logger
from utils.map_file_manager import MapFileManager
from utils.protein_utils import get_sequences_from_protein


class FsDockDataset(Dataset):
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
        core_weight=0.5,
        random_max_angle=None,
        random_translation=None,
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
        self.ligands_file = f"ligands_cw{core_weight}.pt"
        self.saved_protein_graph_file = f"protein_graphs_rr{receptor_radius}_camn{c_alpha_max_neighbors}_amn{atom_max_neighbors}_kog{knn_only_graph}_aa{all_atoms}_ar{atom_radius}.pt"
        # self.saved_ligand_sub_protein_file = f"sub_protein_ligand_edges_lr{ligand_radius}_la{ligand_radius}_"\
        #     f"rr{receptor_radius}_camn{c_alpha_max_neighbors}_amn{atom_max_neighbors}_kog{knn_only_graph}_aa{all_atoms}_ar{atom_radius}_rh{remove_hs}_cw{core_weight}.npz"
        self.tokenizer = tokenizer
        self.tasks = {}
        self.random_max_angle = random_max_angle
        self.random_translation = random_translation
        self.load_mols = load_mols
        super().__init__(root, transform)
        if not hasattr(self, "protein_graphs"):
            self.load()
        self.task_sizes = {k: len(v["graphs"]) for k, v in self.tasks.items()}
        self._indices = self.get_indices()
        # self._indices = self.get_indices()[:1000]
    

    def random_small_rotation_matrix(self, max_angle):
        # Random unit axis
        axis = torch.randn(3)
        axis = axis / axis.norm()
        
        # Small angle in [-max_angle, max_angle]
        angle = (torch.rand(1) * 2 - 1) * max_angle
        # angle = max_angle

        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
        return R

    def translate_point_cloud(self, pc, max_translation):
        direction = torch.rand(3)
        direction = direction / direction.norm()
        
        translation = direction * max_translation
        pc_translated = pc + translation
        return pc_translated

    def rotate_point_cloud(self, pc, max_angle):
        centroid = pc.mean(dim=0)
        pc_centered = pc - centroid

        R = self.random_small_rotation_matrix(max_angle)
        pc_rotated = pc_centered @ R.T
        pc_rotated += centroid

        return pc_rotated

    def get_indices(self):
        split_indexes = []
        for task_name, task in self.tasks.items():
            for i in range(len(task["graphs"])):
                split_indexes.append(
                    (task_name, i)
                )
        rng = random.Random(42)
        rng.shuffle(split_indexes)
        return split_indexes
    
    def tokenize_smiles(self, graph):
        if self.tokenizer:
            sidechains = torch.tensor(list(map(lambda sch: self.tokenizer.encode(sch).ids, graph['ligand'].frag_smiles)))
            graph['ligand'].frag_tokens = sidechains

            graph['ligand'].mol_tokens = torch.tensor([self.tokenizer.encode(graph['ligand'].smiles).ids])


    def connect_ligand_to_protein(self, task_name, idx, data, use_cache=True):
        task = self.tasks[task_name]
        protein_graph = self.protein_graphs[task["target"]]
        use_cache = False
        if not hasattr(self,'reported'):
            print('using cache' if use_cache else 'Warning: not using cache')
            self.reported = True
        if use_cache:
            sub_protein = self.sub_proteins.load(f'{task["name"]}_{idx}')
        else: 

            sub_protein = get_sub_prot_for_ligs(
                    protein_graph, [data], self.ligand_radius, self.atom_radius
                )[0]
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
        if self.random_translation is not None:
            graph["ligand"].pos = self.translate_point_cloud(graph["ligand"].pos, self.random_translation)
        graph.task = task_name
        graph['ligand'].frag_hole = torch.from_numpy(graph['ligand'].frag_hole)
        graph['ligand'].frag_idxs = torch.from_numpy(graph['ligand'].frag_idxs)
        graph['ligand'].core_mask = torch.from_numpy(graph['ligand'].core_mask)
        self.connect_ligand_to_protein(self.tasks[task_name]["name"], i, graph)
        graph = add_frag_place_holder(graph)
        self.tokenize_smiles(graph)
        return graph


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
            # self.saved_ligand_sub_protein_file,
            self.tasks_file,
            self.ligands_file,
        ]
        return names

    def load(self):
        self.logger.info("started load")
        self.ligands = torch.load(osp.join(self.processed_dir, self.ligands_file))
        self.logger.info("started load tasks")
        self.tasks = torch.load(osp.join(self.processed_dir, self.tasks_file))
        self.logger.info("started load proteins")
        self.protein_graphs = torch.load(
            osp.join(self.processed_dir, self.saved_protein_graph_file)
        )
        # self.logger.info("started load ligand_protein_edges")
        # # sub_proteins_path = osp.join(
        #     self.processed_dir, self.saved_ligand_sub_protein_file
        # )
        # self.sub_proteins = MapFileManager(sub_proteins_path, 'r').open()
        self.logger.info("finished load")

    def process(self):
        self.logger.info("started process_ligands")
        self.process_ligands()
        self.logger.info("started process_tasks")
        self.process_tasks()
        self.logger.info("started process_proteins")
        self.process_proteins()
        self.logger.info("started process_ligand_protein_edges")
        # self.process_sub_proteins()
        self.logger.info("finished process")

    def process_ligands(self):
        if files_exist([osp.join(self.processed_dir, self.ligands_file)]):
            self.ligands = torch.load(osp.join(self.processed_dir, self.ligands_file))
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
        torch.save(self.ligands, osp.join(self.processed_dir, self.ligands_file))
        
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
            core, sidechains, slicing_data = get_core_and_chains(
                ligand, core_weight
            )
            if core is None:
                get_logger().warning(
                    f"couldnt extract core: {task_name}, {idx}, {Chem.MolToSmiles(ligand)}"
                )
                return task_name, idx, res
            res['core']=core
            res['sidechains']=sidechains
            res.update(slicing_data)
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

    # def process_sub_proteins(self):
    #     path = osp.join(self.processed_dir, self.saved_ligand_sub_protein_file)
    #     if files_exist([path]):
    #         self.sub_proteins = MapFileManager(path, 'r').open()
    #         return
    #     with MapFileManager(path, 'w') as mf:
    #         for task_name, task in tqdm(
    #             self.tasks.items(), desc="processsing sub proteins"
    #         ):
    #             protein_graph = self.protein_graphs[task["target"]]
    #             ligand_graphs = task["graphs"]
    #             task_sub_proteins = self.get_sub_prot_for_ligs(
    #                 protein_graph, ligand_graphs, self.ligand_radius, self.atom_radius
    #             )
    #             for i, sub_prot in enumerate(task_sub_proteins):
    #                 mf.save(sub_prot,f'{task_name}_{i}')
    #     self.sub_proteins = MapFileManager(path, 'r').open()
        
    

    def process_tasks(self):
        if files_exist([osp.join(self.processed_dir, self.tasks_file)]):
            self.tasks = torch.load(osp.join(self.processed_dir, self.tasks_file))
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
        torch.save(self.tasks, osp.join(self.processed_dir, self.tasks_file))

    def process_task(self, assay_id, grouped_rows, ligands):
        task = {
            "name": assay_id,
            "target": "",
            "activity_type": "",
            "graphs": [],
            "labels": [],
        }
        for (idx, row), ligand_data in zip(grouped_rows.iterrows(), ligands):
            if "core_mask" not in ligand_data:
                continue
            ligand_graph = HeteroData()
            get_lig_graph(ligand_data['ligand'], ligand_graph, self.ligand_radius, {"__holeIdx": ligand_data['hole_count']})
            ligand_graph['ligand'].smiles = ligand_data['smiles']
            ligand_graph['ligand'].frag_smiles = ligand_data['frag_smiles']
            ligand_graph['ligand'].frag_hole = ligand_data['frag_hole']
            ligand_graph['ligand'].core_smiles = ligand_data['core_smiles']
            ligand_graph['ligand'].frag_idxs = ligand_data['frag_idxs']
            ligand_graph['ligand'].core_mask = ligand_data['core_mask']
            ligand_graph['ligand'].num_frags = ligand_data['num_frags']
            
            
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

    @staticmethod
    def process_protein_graph(args):
        protein_graph = HeteroData()
        moad_extract_receptor_structure(
            pdb=args['protein'],
            complex_graph=protein_graph,
            neighbor_cutoff=args['receptor_radius'],
            max_neighbors=args['c_alpha_max_neighbors'],
            lm_embeddings=args['lm_embeddings'],
            knn_only_graph=args['knn_only_graph'],
            all_atoms=args['all_atoms'],
            atom_cutoff=args['atom_radius'],
            atom_max_neighbors=args['atom_max_neighbors'],
        )
        return args['protein_id'], protein_graph
        
    def generate_protein_graphs(self, proteins, lm_embeddings):
        protein_graph_path = osp.join(self.processed_dir, self.saved_protein_graph_file)
        if not files_exist([protein_graph_path]):
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
            torch.save(protein_graphs, protein_graph_path)
        else:
            protein_graphs = torch.load(protein_graph_path)
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



