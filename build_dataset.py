
import sys
import scipy.spatial # very important, does not work without it, i don't know why
from datetime import datetime
import numpy as np
from tqdm import tqdm
from datasets.custom_distributed_sampler import CustomDistributedSampler
from datasets.fsmol_dock import FsDockDataset
from datasets.fsmol_dock import FsDockDataset
from torch_geometric.loader import DataLoader

import torch
import os.path as osp

from datasets.fsmol_dock_clf import FsDockClfDataset
from datasets.partitioned_fsmol_dock import FsDockDatasetPartitioned
from datasets.samplers import TaskRandomSampler, TaskSequentialSampler
from datasets.task_data_loader import TaskDataLoader
torch.multiprocessing.set_sharing_strategy('file_system')
from torch_geometric.data import Dataset, HeteroData, makedirs, Batch


def make_datasets(core_weight):
    ds = FsDockDatasetPartitioned('data/fsdock/valid','data/fsdock/valid_tasks.csv', num_workers=torch.get_num_threads(), core_weight=core_weight)
    ds = FsDockDatasetPartitioned('data/fsdock/test','data/fsdock/test_tasks.csv', num_workers=torch.get_num_threads(), core_weight=core_weight)
    ds = FsDockDatasetPartitioned('data/fsdock/train','data/fsdock/train_tasks.csv', num_workers=torch.get_num_threads(), core_weight=core_weight)
    
    
    ds = FsDockClfDataset('data/fsdock/clfs/valid','data/fsdock/valid_tasks.csv', num_workers=torch.get_num_threads(), min_roc_auc=0.7, core_weight=core_weight)
    ds = FsDockClfDataset('data/fsdock/clfs/test','data/fsdock/test_tasks.csv', num_workers=torch.get_num_threads(), min_roc_auc=0.7, core_weight=core_weight)

    
    
if __name__ == "__main__":
    print(torch.get_num_threads())
    make_datasets(0.7)



