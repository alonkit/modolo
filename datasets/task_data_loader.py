from torch_geometric.loader import DataLoader

class TaskDataLoader(DataLoader):
    def __init__(self, ds, *args, **kwargs):
        super().__init__(ds, *args, **kwargs)
        
    @staticmethod
    def is_list_identical(l):
        return all(x == l[0] for x in l)

    def __iter__(self):
        for res in super().__iter__():
            assert self.is_list_identical(res.task), 'batch has samples from different tasks'
            task = self.dataset.get_task_metadata(res.task[0])
            task['graphs'] = res
            yield task