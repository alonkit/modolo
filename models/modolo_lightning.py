from collections import defaultdict
import copy
from datetime import datetime
import os
import random
from typing import Literal
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

from torch.optim import Adam
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets.custom_distributed_sampler import CustomDistributedSampler
from datasets.fsmol_dock import FsDockDataset
from datasets.fsmol_dock_clf import FsDockClfDataset
from datasets.partitioned_fsmol_dock import FsDockDatasetPartitioned
from datasets.process_chem.process_sidechains import (
    calc_tani_sim,
    get_fp,
    reconstruct_from_core_and_chains,
)
from models.modolo import Modolo
from utils.logging_utils import configure_logger, get_logger
from rdkit import Chem


class ModoloLightning(pl.LightningModule):
    def __init__(
        self,
        model: Modolo,
        tokenizer,
        lr,
        weight_decay,
        num_gen_samples,
        loss=None,
        test_clfs=None,
        similarity_threshold=0.4,
        gen_meta_params=None,
        name=None,
        handle_inactive: Literal['flag','penalty','hide'] = 'flag'
    ):
        super().__init__()
        self.model = model
        self.num_gen_samples = num_gen_samples
        self.tokenizer = tokenizer
        if loss is None:
            loss = nn.CrossEntropyLoss(reduction='none')
        self.loss = loss
        self.noise_loss = nn.MSELoss()
        self._reset_eval_step_outputs()
        self.weight_decay = weight_decay
        self.lr = lr
        self.validation_clfs = None
        self.test_clfs = test_clfs
        self.similarity_threshold = similarity_threshold
        self.gen_meta_params = gen_meta_params or {"p":1.}
        # self.save_hyperparameters(
        #     ignore=["model", "loss", "tokenizer", "validation_clfs", "test_clfs", 'side_']
        # )
        self.name = (name or "") + f'{datetime.today().strftime("%Y-%m-%d-%H_%M_%S")}'
        self.name = f'modolo_{self.name}'
        self.test_result_path = f'test_stats/{self.name}'
        self.freeze_layers = self.model.freeze_layers
        self.use_freeze=False
        self.unfreeze_start = 1
        self.unfreeze_step = 1
        assert handle_inactive in ['flag','penalty','hide']
        self.handle_inactive = handle_inactive

    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.sub_proteins.open()

    def _reset_eval_step_outputs(self):
        self.eval_step_outputs = defaultdict(lambda: defaultdict(list))
    
    
    def t_to_sigma(self, t):
        return 0.05 ** (1-min(t,1)) * 0.2 ** t
    
    @staticmethod
    def extend_tensors(l1,l2):
        if l1.shape[0] == l2.shape[0]:
            return l1,l2
        elif l1.shape[0] > l2.shape[0]:
            long,short = l1,l2
        else:
            long,short = l2,l1
        repeat_factor = len(long) // len(short)
        remainder = len(long) % len(short)

        # Repeat and concatenate
        short = torch.cat([
            short.repeat(repeat_factor),
            short[:remainder]
        ])
        return (long,short) if l1.shape[0] > l2.shape[0] else (short,long)
            
    
    def get_loss(self,data):
        if self.handle_inactive == 'hide':
            if data.label.any(): # exist active and inactive
                # filter actives
                gs = data.to_data_list()
                gs = list(filter(lambda x: x.label.item(), gs))
                data = type(data).from_data_list(gs)
            else: # all inactive
                return torch.tensor(0.0, device=data.label.device,), {}
            
        if self.handle_inactive == 'flag':
            flag = data.label
        else:
            # flag is not interesting so..
            flag = torch.zeros_like(data.label,device=data.label.device)+1

        logits = self.model(
            data.core_tokens,
            data.split_sidechain_tokens[:, :-1],
            data,
            (data.activity_type, flag), 
            molecule_sidechain_mask_idx=1
        )
        logits = logits.transpose(1, -1)
        tgt = data.split_sidechain_tokens[:, 1:]
        losses = self.loss(logits, tgt)
        if self.handle_inactive in ('penalty','hide'):
            good_spots = torch.nonzero(data.label.repeat_interleave(data.num_sidechains)==1, as_tuple=True)[0]
            bad_spots = torch.nonzero(data.label.repeat_interleave(data.num_sidechains)==0, as_tuple=True)[0]
            
            recon_loss = losses.mean()
            if len(good_spots) == 0 or len(bad_spots) == 0:
                active_loss = losses[good_spots].mean().nan_to_num(0)
                inactive_loss = losses[bad_spots].mean().nan_to_num(0)
                return recon_loss, {'recon':recon_loss, 'active':active_loss, 'inactive': inactive_loss}
            good_spots, bad_spots = self.extend_tensors(good_spots,bad_spots)
            # maybe use distances
            active_loss = losses[good_spots].mean(1)
            inactive_loss = losses[bad_spots].mean(1)
            
            margin = 0.05
            margin_loss = F.relu(margin + active_loss - inactive_loss).mean()
            
            loss = recon_loss + 2 * margin_loss
            return loss, {'recon':recon_loss, 'margin': margin_loss, 'active':active_loss.mean(), 'inactive': inactive_loss.mean()}
        if self.handle_inactive == 'flag':
            return losses.mean() , {}
    
    def unfreeze(self):
        if self.current_epoch == 0:
            for layers in self.freeze_layers:
                if not isinstance(layers,list):
                    layers = [layers]
                for layer in layers:
                    if layer is None:
                        continue
                    for param in layer.parameters():
                        param.requires_grad=False
        if self.current_epoch < self.unfreeze_start:
            return
        elif (self.current_epoch - self.unfreeze_start) % self.unfreeze_step == 0:
            layer_idx = len(self.freeze_layers) - (self.current_epoch - self.unfreeze_start) // self.unfreeze_step - 1
            if layer_idx < 0:
                return
            layers = self.freeze_layers[layer_idx]

            if not isinstance(layers,list):
                layers = [layers]
            for layer in layers:
                if layer is None:
                    continue
                for param in layer.parameters():
                    param.requires_grad=True
    
    def on_train_epoch_start(self):
        if self.use_freeze:
            self.unfreeze()
        
            
    def training_step(self, data, batch_idx):
        loss, loss_dict = self.get_loss(data)
        # loss_weights = data.label * alpha + (1- data.label) * (1-alpha)
        # loss = loss * loss_weights.unsqueeze(-1)
        if self.handle_inactive in ('penalty','hide'):
            self.log("train_loss", loss,prog_bar=True, sync_dist=True)
            for k,v in loss_dict.items():
                self.log(f"train_loss_{k}",v, sync_dist=True)
        else:
            self.log("train_loss", loss,prog_bar=True, sync_dist=True)
        # self.log("alpha", alpha, )
        return loss
            
            
    def generate_samples(self, data):
        mols_sidechains_batches = self.model.optimized_generate_samples(
            self.num_gen_samples,
            data,
            (data.activity_type, [1] * len(data)),
            self.tokenizer,
            **self.gen_meta_params
        )
        # we want to genenerate good samples so we give label=1
        new_mols = []
        for core, old_smile, task, mol_sidechains_batches in zip(data.core_smiles, data.smiles, data.task, mols_sidechains_batches):
            for chains in zip(*mol_sidechains_batches):
                # chains = self.tokenizer.decode_batch(chains, skip_special_tokens=True)
                chains = [f'[{i+1}*]{ch}' for i,ch in enumerate(chains)]
                chains_smiles = '.'.join(chains)
                new_smile = reconstruct_from_core_and_chains(core, chains_smiles)
                old_smile = self.removeChirality(old_smile)
                if new_smile is None:
                    new_mols.append((task, None, old_smile, None))
                else:
                    
                    new_smile = self.removeChirality(new_smile)
                    new_mols.append((task, new_smile, old_smile, get_fp(new_smile)))
        return new_mols
    
    @staticmethod
    def removeChirality(smiles):
        mol = Chem.MolFromSmiles(smiles)
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol)

    def validation_step(self, graph, batch_idx):
        loss, loss_dict = self.get_loss(graph)
        if self.handle_inactive in ('penalty','hide'):
            self.log("valid_loss", loss,batch_size=len(graph), sync_dist=True)
            for k,v in loss_dict.items():
                self.log(f"valid_loss_{k}",v,batch_size=len(graph), sync_dist=True)
        else:
            self.log("valid_loss", loss,batch_size=len(graph), sync_dist=True)
        if not self.validation_clfs:
            return
        if graph.label.all().item() == True: # all active
            return
        if graph.label.all().any() == True: # exist active and inactive
            # filter actives
            gs = graph.to_data_list()
            gs = list(filter(lambda x: not x.label.item(), gs))
            graph = type(graph).from_data_list(gs)
        
        gen_res = self.generate_samples(graph)
        for (task_name, new_sm, old_sm, new_fp) in gen_res:
            self.eval_step_outputs[task_name][old_sm].append((new_sm, new_fp))

    def test_step(self, graph, batch_idx):
        if not self.test_clfs:
            return
        
        gen_res = self.generate_samples(graph)
        for (task_name, new_sm, old_sm, new_fp) in gen_res:
            self.eval_step_outputs[task_name][old_sm].append((new_sm, new_fp))
        

    def evaluate_single_task(
        self, task_name, opt_molecules, clf, threshold, similarity_threshold, log=True
    ):
        all_success_rates, all_diversities, all_similarities, all_scores = [], [], [], []
        
        all_valid_samples, num_molecules = [], 0
        for old_sm in opt_molecules.keys():
            for new_sm, _ in opt_molecules[old_sm]:
                num_molecules += 1
                if new_sm is not None and new_sm != old_sm:
                    all_valid_samples.append(new_sm)
        
        for _ in range(self.num_gen_samples):
            chosen_mols, similarities, tot_success, scores = [], [], 0, []
            for old_sm in opt_molecules.keys():
                candidates = [
                        (new_mol,new_fp)
                        for new_mol, new_fp in opt_molecules[old_sm]
                        if new_mol and new_mol != old_sm
                    ]
                if len(candidates) == 0:
                    continue
                candidates, cand_fps = zip(*candidates)
                chosen_candidate_i = random.randint(0, len(candidates)-1)
                chosen_candidate = candidates[chosen_candidate_i]
                cand_fp = cand_fps[chosen_candidate_i]
                chosen_mols.append(chosen_candidate)
                cur_sim = calc_tani_sim(old_sm, chosen_candidate)
                similarities.append(cur_sim)
                cur_score = clf.predict_proba(np.reshape(cand_fp, (1, -1)))
                cur_score = cur_score[0][1]
                scores.append(cur_score)
                if log:
                    self.log('1_score', cur_score, sync_dist=True)
                if cur_score > threshold and cur_sim > similarity_threshold:
                    tot_success += 1
                    if log:
                        self.log('1_success', 1, sync_dist=True)
                else:
                    if log:
                        self.log('1_success', 0, sync_dist=True)
            all_success_rates.append(tot_success / len(opt_molecules.keys()))
            all_diversities.append(len(set(chosen_mols)) / max(1, len(chosen_mols)))
            all_similarities.append(sum(similarities) / max(1, len(chosen_mols)))
            all_scores.append(sum(scores) / max(1, len(chosen_mols)))
        avg_diversity, std_diversity = np.mean(all_diversities), np.std(all_diversities)
        avg_similarity, std_similarity = np.mean(all_similarities), np.std(
            all_similarities
        )
        avg_score, std_score = np.mean(all_scores), np.std(
            all_scores
        )
        avg_success, std_success = np.mean(all_success_rates), np.std(all_success_rates)
        validity = len(all_valid_samples) / num_molecules
        return (
            validity,
            avg_diversity,
            std_diversity,
            avg_similarity,
            std_similarity,
            avg_success,
            std_success,
            avg_score,
            std_score
        )

    def on_test_start(self):
        path = self.test_result_path
        if path[-1] == '/':
            path = path[:-1]
        base_path = path
        idx = 1
        while os.path.exists(path):
            path = f"{base_path}_{idx}"
            idx += 1
        os.makedirs(path)
        self.test_result_path = path

    def on_test_end(self):
        
        for name, opt_molecules in self.eval_step_outputs.items():
            with open(f'{self.test_result_path}/{name}', 'w') as f:
                for orig_mol in opt_molecules:
                    for new_mol,_ in opt_molecules[orig_mol]:
                        if new_mol is not None:
                            f.write(f'{orig_mol} {new_mol}\n')
        
        results = defaultdict(list)
        for task_name in sorted(self.eval_step_outputs.keys()):
            opt_molecules = self.eval_step_outputs[task_name]
            (
                validity,
                avg_diversity,
                std_diversity,
                avg_similarity,
                std_similarity,
                avg_success,
                std_success,
                avg_score,
                std_score
            ) = self.evaluate_single_task(
                task_name,
                opt_molecules,
                self.test_clfs[task_name][0],
                self.test_clfs[task_name][1],
                self.similarity_threshold,
                log=False
            )
            results['task'].append(task_name)
            results['validity'].append(validity)
            results['diversity'].append(avg_diversity)
            results['std_diversity'].append(std_diversity)
            results['similarity'].append(avg_similarity)
            results['std_similarity'].append(std_similarity)
            results['score'].append(avg_score)
            results['std_score'].append(std_score)
            results['success'].append(avg_success)
            results['std_success'].append(std_success)
            results['total'].append(sum(map(len, opt_molecules.values())))
        pd.DataFrame(results).to_csv(f'{self.test_result_path}/results.csv')
        self._reset_eval_step_outputs()

    def on_validation_epoch_end(self):
        tot_avg_success, tot_score = [], []
        for task_name, opt_molecules in self.eval_step_outputs.items():
            (
            validity,
            avg_diversity,
            std_diversity,
            avg_similarity,
            std_similarity,
            avg_success,
            std_success,
            avg_score,
            std_score
            )  = self.evaluate_single_task(
                task_name,
                opt_molecules,
                self.validation_clfs[task_name][0],
                self.validation_clfs[task_name][1],
                self.similarity_threshold,
            )
            self.log(f"{task_name}_validity", validity, sync_dist=True)
            self.log(f"{task_name}_diversity", avg_diversity, sync_dist=True)
            # self.log(f"{task_name}_std_diversity", std_diversity, sync_dist=True)
            self.log(f"{task_name}_similarity", avg_similarity, sync_dist=True)
            # self.log(f"{task_name}_std_similarity", std_similarity, sync_dist=True)
            self.log(f"{task_name}_success", avg_success, sync_dist=True)
            # self.log(f"{task_name}_std_success", std_success, sync_dist=True)
            self.log(f"{task_name}_score", avg_score, sync_dist=True)
            
            tot_avg_success.append(avg_success)
            tot_score.append(avg_score)
        self.log("validation_avg_success", sum(tot_avg_success) / max(len(tot_avg_success), 1), sync_dist=True)
        self.log("validation_avg_score", sum(tot_score) / max(1, len(tot_score)), sync_dist=True)
        self._reset_eval_step_outputs()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, min_lr=self.lr / 100)
        return {
                        "optimizer": optimizer,
                        "lr_scheduler": sched,
                        "monitor": "train_loss"
                    }
