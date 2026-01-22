import datetime
import os
import subprocess
import random
import string
from rdkit.Chem import AllChem, DataStructs
from typing import List
from easydict import EasyDict
from rdkit import Chem, DataStructs
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
import multiprocessing
from proc_run import run_step
import pandas as pd
from loguru import logger
# /home/alon.kitin/docking_cfom/data/proteins/pdbs/Q07869.pdb	/home/alon.kitin/docking_cfom/data/CHEMBL1001152/docks/b1/rank1.sdf

import warnings
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from importlib.util import find_spec
from math import pi

import prolif as plf
import MDAnalysis as mda
from MDAnalysis.converters import RDKitInferring

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)


def get_fp_similarity(smile1, smile2):
    try:
            mol1 = Chem.MolFromSmiles(smile1)
            mol2 = Chem.MolFromSmiles(smile2)
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
    except: 
        return 0.0
    

class QVinaDockingTask:
    
    def __init__(self, orig_ligand_path, protein_path,optimized_smiles, tmp_dir='./tmp', use_uff=False, center=None, cpus=1, silent=False):
        
        self.name = protein_path.split('/')[-1].replace('.pdb','') + '__' + orig_ligand_path.split('/')[-1].replace('.sdf','')
        self.task_id = self.name+"_"  #  +QVinaDockingTask.get_random_id(5)
        self.tmp_dir = os.path.realpath(tmp_dir+'/'+self.task_id)
        self.use_uff = use_uff
        self.log_path = os.path.join(self.tmp_dir, 'log.txt')

        self.cpus = cpus
        self.silent = silent
        self.orig_ligand_path = orig_ligand_path
        self.protein_path = protein_path
        self.optimized_smiles = optimized_smiles

        self.ligand_id = os.path.splitext(os.path.basename(orig_ligand_path))[0]
        self.protein_id = os.path.splitext(os.path.basename(protein_path))[0]
        
        
        ligand_rdmol = next(iter(Chem.SDMolSupplier(orig_ligand_path)))

        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)
        if use_uff:
            try:
                UFFOptimizeMolecule(ligand_rdmol)
            except:pass
        self.ligand_rdmol = ligand_rdmol
        self.original_smiles = Chem.MolToSmiles(ligand_rdmol) 
        pos = ligand_rdmol.GetConformer(0).GetPositions()
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center

        self.proc = None
        self.results = None
        self.output = None
        self.docked_sdf_path = None

    
    @staticmethod
    def get_random_id(length=30):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length)) 

    @staticmethod
    def load_pdb(path):
        with open(path, 'r') as f:
            return f.read()

    def run_many_tasks(task_list: List["QVinaDockingTask"], cpu_per_task)->pd.DataFrame:
        cpus = int(os.environ.get("SLURM_CPUS_ON_NODE","1"))
        logger.info("Num workers:", cpus//cpu_per_task, "num tasks:", len(task_list))
        with multiprocessing.Pool(processes=cpus//cpu_per_task) as pool:
            results = pool.map(QVinaDockingTask.run_task_wrapper, task_list)
        dct = defaultdict(list)
        for task, task_results in zip(task_list, results):
            # logger.info(task.name, len(task_results) if task_results is not None else 0)
            original_affinity = None
            for result in task_results:
                i, smi, aff, prolif_sim = result
                if i==0:
                    original_affinity = aff
                else:
                    dct["protein_pdb"].append(task.protein_path)
                    dct["original_sdf"].append(task.orig_ligand_path)
                    dct["original_smiles"].append(task.original_smiles)
                    dct["optimized_smiles"].append(smi)
                    dct["original_affinity"].append(original_affinity)
                    dct["optimized_affinity"].append(aff)
                    dct["similarity"].append(get_fp_similarity(task.original_smiles, smi))
                    # dct["prolif_similarity"].append(prolif_sim)
        return pd.DataFrame(dct)

    @staticmethod
    def run_task_wrapper(task):
        logger.info(f'Running task {task.name}')
        return task.run()

    @staticmethod
    def parse_qvina_outputs(docked_sdf_path):

        suppl = Chem.SDMolSupplier(docked_sdf_path,sanitize=False)
        results = []
        for i, mol in enumerate(suppl):
            if mol is None:
                continue
            line = mol.GetProp('REMARK').splitlines()[0].split()[2:]
            results.append(EasyDict({
                'rdmol': mol,
                'mode_id': i,
                'affinity': float(line[0]),
                'rmsd_lb': float(line[1]),
                'rmsd_ub': float(line[2]),
            }))

        return results

    def smiles_to_sdf(self,smiles_list,output_path):
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        mols = [Chem.AddHs(mol) for mol in mols]
        if self.use_uff:
            for mol in mols:
                try:
                    UFFOptimizeMolecule(mol)
                except:pass
        paths = [f'{output_path}/mol{i+1}.sdf' for i in range(len(mols))]
        for mol,path in (zip(mols,paths)):
            writer = Chem.SDWriter(path)
            writer.write(mol)
            writer.close()
        return paths
   
    def qvina_commands(self,i):
        obabel_input =  self.orig_ligand_path if i==0 else f'-:"{self.optimized_smiles[i-1]}"'
        return [
            f'cd {self.tmp_dir}',
            f"obabel {obabel_input} -O mol{i}.pdb  --partialcharge gasteiger --gen3d",
            f"prepare_ligand4 -l mol{i}.pdb -A checkhydrogen -o mol{i}.pdbqt",
            f"qvina2 \
                --receptor {self.protein_id}.pdbqt \
                --ligand mol{i}.pdbqt \
                --center_x {self.center[0]:.4f} \
                --center_y {self.center[1]:.4f} \
                --center_z {self.center[2]:.4f} \
                --size_x 20 --size_y 20 --size_z 20 \
                --cpu {self.cpus} \
                --exhaustiveness 8 ",
            f"obabel mol{i}_out.pdbqt -O mol{i}_out.sdf -h"
        ]
    
    def get_proc(self):
        self.proc = subprocess.Popen(
            '/bin/bash', 
            shell=False, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
        )
        return self.proc
    
    def run_prepare_prot(self):
        if os.path.exists(f'{self.tmp_dir}/{self.protein_id}.pdbqt'):
            logger.info(f'Protein pdbqt already exists: {self.protein_id}.pdbqt')
            return
        
        commands = [
            f'cd {self.tmp_dir}',
            f"prepare_receptor4 -r {self.protein_path} -o {self.protein_id}.pdbqt",
            # f"prepare_receptor4 -r {self.protein_path} -o {self.protein_id}.pdbqt -A checkhydrogens",
            f"obabel {self.protein_id}.pdbqt -O {self.protein_id}.pdb -h",
            ]
        self.proc = self.get_proc()
        for command in commands:
            logger.debug(command)
            out, err = run_step(self.proc, command, timeout=120)
            logger.debug(out)
            logger.debug(err)
            logger.debug('---')
    
    def sub_run(self, mol_idx):
        if os.path.exists(os.path.join(self.tmp_dir, f"mol{mol_idx}_out.sdf")):
            logger.info(f'Docking output already exists: mol{mol_idx}_out.sdf')
            return self.get_results(mol_idx)
        
        commands = self.qvina_commands(mol_idx)
        self.proc = self.get_proc()
        for i, command in enumerate(commands):
            
            logger.debug(command)
            out, err = run_step(self.proc, command, timeout=240) # TODO: adjust timeouts
            logger.debug(out)
            logger.debug(err)
            logger.debug('---')
        return self.get_results(mol_idx)

    def prolif_similarity(self, dock_results):
        RDKitInferring.MDAnalysisInferrer.sanitize = False
        rdkit_prot = Chem.MolFromPDBFile(f'{self.tmp_dir}/{self.protein_id}.pdb', removeHs=False, sanitize=False)
        protein_mol = plf.Molecule(rdkit_prot)
        ligand_mols = []
        for mol_idx, _, aff in dock_results:
            if aff is None:
                continue
            ligand_mol = plf.sdf_supplier(f'{self.tmp_dir}/mol{mol_idx}_out.sdf')[0]
            ligand_mols.append(ligand_mol)
        fp = plf.Fingerprint(interactions=[
            "HBDonor",
            "HBAcceptor",
            "PiStacking",
            "Anionic",
            "Cationic",
            "CationPi",
            "PiCation",
            "XBAcceptor",
            "XBDonor",
        ], parameters={
            "HBAcceptor": {
                "distance": 3.7,
                # modified nitrogen pattern (replaced valence specs with charge
                # to account for broken peptide bond during pocket extraction)
                # otherwise HBonds with backbone nitrogen wouldn't be detected
                "donor": "[$([O,S,#7;+0]),$([N+1])]-[H]",
            },
            "HBDonor": {"distance": 3.7},
            "CationPi": {"distance": 5.5},
            "PiCation": {"distance": 5.5},
            "Anionic": {"distance": 5},
            "Cationic": {"distance": 5},
        },count=True)
        fp.run_from_iterable(ligand_mols, protein_mol)
        bv = fp.to_bitvectors()
        j = 0
        for i, (mol_idx, smi, aff) in enumerate(dock_results):
            if aff is None:
                similarity = None
            else:
                similarity = DataStructs.TanimotoSimilarity(bv[0], bv[j])
                j+=1
                logger.debug(f'Similarity of docked molecule {mol_idx} to original ligand: {similarity:.4f}')
            dock_results[i] = (mol_idx, smi, aff, similarity)
        return dock_results
            

    def run(self):
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        results = []
        try:
            self.run_prepare_prot()
            logger.info(f"Number of molecules to dock: {len(self.optimized_smiles)+1}")
            for i in range(len(self.optimized_smiles)+1):
                logger.info(f'{self.name} {i}')                
                try:
                    res = self.sub_run(i)
                except subprocess.TimeoutExpired as e:
                    logger.error(e)
                    res = None
                results.append(
                    (   
                        i,
                        self.original_smiles if i==0 else self.optimized_smiles[i-1],
                        res,
                    )
                )
            results = [(*res, 0) for res in results]  # add placeholder for prolif similarity
            # results = self.prolif_similarity(results)
            logger.info(f'Finished task {self.name}, with {len(results)} results')
            logger.debug(results)
            ## delete tmp dir:
            self.proc = self.get_proc()
            # run_step(self.proc, f'rm -rf {self.tmp_dir}', timeout=30)
            return results
        except Exception as e:
            logger.error(e)
            import traceback
            traceback.print_exc()
            return []
        

    def get_results(self,i):
        try:
            res_path = os.path.join(self.tmp_dir, f"mol{i}_out.sdf")
            self.results = self.parse_qvina_outputs(res_path)
        except:
            logger.error('[Error] Vina output error: %s' % f"mol{i}_out.sdf")
            return None
        return self.results[0]['affinity']
    
if __name__ == '__main__':
    with open('/home/alon.kitin/vinaTest/data/example/smiles.txt','r') as f:
        optimized_smiles = [line.strip() for line in f.readlines()]
    task = QVinaDockingTask(
        orig_ligand_path = '/home/alon.kitin/crossdocked_test_set/ABL2_HUMAN_274_551_0/4xli_B_rec_4xli_1n1_lig_tt_min_0.sdf',
        protein_path = '/home/alon.kitin/crossdocked_test_set/ABL2_HUMAN_274_551_0/4xli_B_rec_4xli_1n1_lig_tt_min_0_pocket10.pdb',
        optimized_smiles=optimized_smiles,
        tmp_dir = '/home/alon.kitin/vinaTest/data/example/tmp',
        cpus = 2,
        silent=True        
    )
    QVinaDockingTask.run_many_tasks([task], cpu_per_task=2).to_csv('/home/alon.kitin/vinaTest/data/example/results.csv', index=False)
    # print(task.run())
    exit()
    results = task.run_all()
    print('Done. Found %d modes' % len(results))