
import sys
import scipy.spatial # very important, does not work without it, i don't know why
from datasets.cross_partitioned import CrossPartitionedFsDockDataset
from datasets.fsmol_dock import FsDockDataset

import torch
torch.multiprocessing.set_sharing_strategy('file_system')


def make_datasets(core_weight):
    ds = FsDockDataset('data/cross/valid','data/split/valid_tasks.csv', num_workers=torch.get_num_threads(), core_weight=core_weight)
    ds = FsDockDataset('data/cross/test','data/split/test_tasks.csv', num_workers=torch.get_num_threads(), core_weight=core_weight)
    ds = CrossPartitionedFsDockDataset('data/cross/train','data/split/train_tasks.csv', num_workers=torch.get_num_threads(), core_weight=core_weight)

if __name__ == "__main__":
    print(torch.get_num_threads())
    make_datasets(0.7)



