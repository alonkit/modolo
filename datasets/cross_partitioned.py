import torch
from datasets.partitioned_fsmol_dock import FsDockDatasetPartitioned
from utils.map_file_manager import MapFileManager
import os.path as osp
from copy import deepcopy

class CrossPartitionedFsDockDataset(FsDockDatasetPartitioned):
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
        self.logger.info(f"started load tasks- {sum([len(ligs) for ligs in self.ligands.values()])}")
        with MapFileManager(osp.join(self.processed_dir, self.tasks_file),'r') as mf:
            self.tasks = {task:mf[task] for task in tasks}
        self.logger.info(f"started load prots- {sum([len(t['labels']) for t in self.tasks.values()])}")
        self.logger.info("finished load")
        
    def __getitem__(self, idx):
        if not hasattr(self, 'protein_graphs'):
            self.protein_graphs = MapFileManager(osp.join(self.processed_dir, self.saved_protein_graph_file),'r').open()
        return super().__getitem__(idx)

if __name__ == "__main__":
    ds = CrossPartitionedFsDockDataset(
        'data/cross/train','data/cross/train_tasks.csv', 
        num_workers=torch.get_num_threads(), core_weight=0.7)
    ds.load()
    for i in range(len(ds)):
        t = ds[i]
    exit()