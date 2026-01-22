from collections import defaultdict
import random
import torch
from torch.utils.data import Sampler, DistributedSampler, BatchSampler, RandomSampler
import math
from typing import Optional, Iterator

import torch
import torch.distributed as dist

class CustomDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.targets, self.indices = dataset.get_partition_and_idxs(num_replicas)
        self.indices_len = list(map(len,self.indices))
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and list(map(len,self.indices)).count(len(self.indices[0])) != len(self.indices):  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = min(self.indices_len)
        else:
            self.num_samples = max(self.indices_len)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.dataset.load(self.targets[rank])

    def __iter__(self) -> Iterator:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = [self.indices[self.rank][i] for i in torch.randperm(len(self.indices[self.rank]), generator=g).tolist()]  # type: ignore[arg-type]
        else:
            indices = list(self.indices[self.rank])  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = max(self.indices_len) - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:min(self.indices_len)]

        bad = set(['A95216', 'A70694', 'A86475', 'A64763', 'A26624'])
        indices = [x for x in indices if x[0] not in bad]  # TEMPORARY FIX FOR CORRUPTED DATA
        return iter(indices)


class CustomTaskDistributedSampler(CustomDistributedSampler):
    def __init__(self, dataset,query_size: int=None, support_size:int=None, task_size: int=None, stream:bool=True, **kwargs):
        assert task_size or (query_size and support_size)
        
        super().__init__(dataset, **kwargs)
        self.stream=stream
        self.tasks = defaultdict(list)
        self.task_size = task_size if task_size else support_size + query_size
        self.query_size = query_size
        self.support_size = support_size
        self.set_tasks()
        self.num_tasks = self.num_samples // self.task_size,
        
        self.tasks_good = {}
        self.tasks_bad = {}
        
    
    def set_tasks(self):
        self.tasks_good = defaultdict(list)
        self.tasks_bad = defaultdict(list)
        for task, idx in super().__iter__():
            if self.dataset.tasks[task]['labels'][idx] == 1:
                self.tasks_good[task].append(idx)
            else:
                self.tasks_bad[task].append(idx)
        self.tasks = defaultdict(list)
        
        for task in self.tasks_good:
            for i in range(min(len(self.tasks_good[task]), len(self.tasks_bad[task]))):
                self.tasks[task].append(self.tasks_good[task][i])
                self.tasks[task].append(self.tasks_bad[task][i])
        
    
    def __iter__(self):
        self.set_tasks()
                    
        self.task_samplers = {}
        for task, idxs in self.tasks.items():
            self.task_samplers[task] = BatchSampler(idxs, self.task_size, drop_last=self.drop_last)
        self.num_tasks = self.num_samples // self.task_size
        task_iters = None
        i=0
        while i < self.num_tasks:
            if not task_iters:
                task_iters = {task: iter(sampler) for task,sampler in self.task_samplers.items() }
            try:
                task = random.choice(list(task_iters.keys()))
                idxs = next(task_iters[task])
                i += 1
                task = list(zip([task]*len(idxs), idxs))
                if self.stream:
                    for t in task:
                        yield t
                elif self.support_size:        
                    yield [task[:self.support_size], task[self.support_size:]]
                else:
                    yield [task]
            except StopIteration:
                del task_iters[task]
    
    def __len__(self):
        if self.stream:
            return self.num_tasks * self.task_size
        return self.num_tasks