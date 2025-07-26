import random
import traceback
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import torch
from datasets.fsmol_dock import FsDockDataset, files_exist, osp
from datasets.process_chem.process_mols import read_molecule
from datasets.process_chem.process_sidechains import (
    add_attachment_points,
    get_core_and_chains,
    get_fp,
    get_holes,
    get_mask_of_sidechains,
    get_mol_smiles,
)
from utils.logging_utils import get_logger
from rdkit import Chem, DataStructs


class FsDockClfDataset(FsDockDataset):
    def __init__(
        self,
        root,
        tasks,
        min_clf_samples=300,
        test_fraction=0.2,
        max_depth=2,
        num_estimators=100,
        min_roc_auc=0.75,
        only_active=False,
        only_inactive=False,
        *args,
        **kwargs,
    ):
        assert not (
            only_active and only_inactive
        ), "only_active and only_inactive cannot be True at the same time"
        self.only_active = only_active
        self.only_inactive = only_inactive
        self.min_clf_samples = min_clf_samples
        self.test_fraction = test_fraction
        self.max_depth = max_depth
        self.num_estimators = num_estimators
        self.min_roc_auc = min_roc_auc
        self.clfs_file = f"clfs_nsamples{min_clf_samples}_mdep{max_depth}_nest{num_estimators}_mroc{min_roc_auc}.pt"
        super().__init__(root, tasks, *args, **kwargs)

    def processed_file_names(self):
        return super().processed_file_names() + [self.clfs_file]

    def process(self):
        super().process()
        self.logger.info("started proccessing clf")
        self.process_clf()
        self.logger.info("finished proccessing clf")
        self.remove_unwanted()

    def load(self):
        super().load()
        self.load_clfs()
        self.remove_unwanted()

    def remove_unwanted(self):
        if not (self.only_active or self.only_inactive):
            return
        for task_name, task in self.tasks.items():
            new_graphs = []
            new_labels = []
            for i in range(len(task["labels"])):
                if self.only_active and task["labels"][i] == 1:
                    new_graphs.append(task["graphs"][i])
                    new_labels.append(task["labels"][i])
                elif self.only_inactive and task["labels"][i] == 0:
                    new_graphs.append(task["graphs"][i])
                    new_labels.append(task["labels"][i])
            task["graphs"] = new_graphs
            task["labels"] = new_labels

    def load_clfs(self):
        if not getattr(self, "clfs", None):
            self.clfs = torch.load(osp.join(self.processed_dir, self.clfs_file))
        new_tasks = {}
        for task in self.tasks:
            if task not in self.clfs:
                continue
            else:
                new_tasks[task] = self.tasks[task]
                new_tasks[task]["clf"] = self.clfs[task]
        self.tasks = new_tasks

    def process_clf(self):
        if files_exist([osp.join(self.processed_dir, self.clfs_file)]):
            self.clfs = torch.load(osp.join(self.processed_dir, self.clfs_file))
            return
        self.clfs = {}
        for task in self.ligands:
            if len(self.ligands[task]) < self.min_clf_samples:
                continue
            labels = self.tasks[task]["labels"]
            mols = [d['ligand'] for d in self.ligands[task] if d.get('ligand') is not None]
            clf, roc_auc, best_thresh = self.get_clf(mols, labels)
            if roc_auc < self.min_roc_auc:
                self.logger.info(
                    f"roc_auc for {task} is {roc_auc}, which is less than {self.min_roc_auc}"
                )
                continue
            self.clfs[task] = (clf, best_thresh)
        self.load_clfs()

        torch.save(self.clfs, osp.join(self.processed_dir, self.clfs_file))

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
                core_smiles, hole_neighbors = add_attachment_points(ligand, 2)
                if core_smiles is None:
                    get_logger().warning(
                        f"couldnt extract core: {task_name}, {idx}, {Chem.MolToSmiles(ligand)}"
                    )
                sidechains_mask = np.zeros(ligand.GetNumAtoms()).astype(int)
                sidechains_smiles = ''
            else:
                sidechains_mask = get_mask_of_sidechains(ligand, sidechains)
            res['core']=core
            res['core_smiles']=core_smiles
            res['sidechains']=sidechains
            res['sidechains_smiles']=sidechains_smiles
            res['hole_neighbors'] = hole_neighbors
            hole_features = get_holes(ligand)
            extra_atom_feats = {'__holeIdx': hole_features}

            res['sidechains_mask']=sidechains_mask
            
            res['num_sidechains']= sidechains_mask.max().item() if sidechains_mask.max().item() > 0 else 2
            res['extra_atom_feats']=extra_atom_feats
            return task_name, idx, res
        except Exception as e:
            get_logger().error(
                f"Error processing ligand {task_name}, {idx}, {Chem.MolToSmiles(ligand)}"
            )
            get_logger().error(traceback.format_exc())
            return task_name, idx, res

    def get_clf(self, mols, labels):
        positive_fp = [get_fp(mol) for l, mol in zip(labels, mols) if l == 1]
        negative_fp = [get_fp(mol) for l, mol in zip(labels, mols) if l == 0]
        random.shuffle(positive_fp)
        random.shuffle(negative_fp)
        pos_ratio = len(positive_fp) / (len(negative_fp) + len(positive_fp))
        num_test = int(self.test_fraction * (len(negative_fp) + len(positive_fp)))
        pos_test_fp = positive_fp[: int(pos_ratio * num_test)]
        pos_train_fp = positive_fp[int(pos_ratio * num_test) :]
        neg_test_fp = negative_fp[: int((1 - pos_ratio) * num_test)]
        neg_train_fp = negative_fp[int((1 - pos_ratio) * num_test) :]
        X = np.stack(pos_train_fp + neg_train_fp, axis=0)
        y = np.concatenate([np.ones(len(pos_train_fp)), np.zeros(len(neg_train_fp))])
        sample_weights = [
            1 if cur_label == 0 else len(neg_train_fp) / len(pos_train_fp)
            for cur_label in y
        ]
        clf = RandomForestClassifier(
            max_depth=self.max_depth, random_state=0, n_estimators=self.num_estimators
        )
        clf.fit(X, y, sample_weight=sample_weights)
        positives = np.ones(len(pos_test_fp))
        negatives = np.zeros(len(neg_test_fp))
        labels = np.concatenate([positives, negatives], axis=0)
        test_samples = np.concatenate([pos_test_fp, neg_test_fp], axis=0)
        probs = clf.predict_proba(test_samples)
        roc_auc = roc_auc_score(labels, probs[:, 1])
        fpr, tpr, thresholds = roc_curve(labels, probs[:, 1], pos_label=1)
        best_thresh = thresholds[np.argmax(tpr - fpr)]
        return clf, roc_auc, best_thresh
