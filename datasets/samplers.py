import random
from torch.utils.data import Sampler, SequentialSampler, RandomSampler, BatchSampler

class TaskSequentialSampler(BatchSampler):
    def __init__(self, task_sizes, batch_size, drop_last=False):
        super().__init__(None, batch_size, drop_last)
        self.task_sizes = task_sizes
        self._len = 0
        
        self.task_samplers = {}
        for task,size in self.task_sizes.items():
            self.task_samplers[task] = BatchSampler(SequentialSampler(range(size)), self.batch_size, drop_last)
        
        self._len = 0
        for task,sampler in self.task_samplers.items():
            self._len += len(sampler)
        
        
    def __iter__(self):
        task_iters = {task: iter(sampler) for task,sampler in self.task_samplers.items() }
        for task in task_iters:
            for idxs in task_iters[task]:
                yield list(zip([task]*len(idxs), idxs))
    
    def __len__(self):
        return self._len

class TaskRandomSampler(BatchSampler):
    def __init__(self, task_sizes, batch_size, drop_last=False):
        super().__init__(None, batch_size, drop_last)
        self.batch_size = batch_size
        self.task_sizes = task_sizes
        
        self.task_samplers = {}
        for task,size in self.task_sizes.items():
            self.task_samplers[task] = BatchSampler(RandomSampler(range(size)), self.batch_size, drop_last)
        
        self._len = 0
        for task,sampler in self.task_samplers.items():
            self._len += len(sampler)
        
        
    def __iter__(self):
        task_iters = {task: iter(sampler) for task,sampler in self.task_samplers.items() }
        while task_iters:
            try:
                task = random.choice(list(task_iters.keys()))
                idxs = next(task_iters[task])
                yield list(zip([task]*len(idxs), idxs))
            except StopIteration:
                del task_iters[task]
    
    def __len__(self):
        return self._len


if __name__ == "__main__":
    task_sizes = {'a': 10, 'b': 20, 'c': 30}
    sub_task_size = 5
    sampler = TaskRandomSampler(task_sizes, sub_task_size)
    for x in sampler:
        print(x)
    print(len(sampler))
    sampler = TaskSequentialSampler(task_sizes, sub_task_size)
    for x in sampler:
        print(x)
    print(len(sampler))