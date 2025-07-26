import numpy as np
import torch
import pickle
import zipfile
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class MapFileManager:
    def __init__(self, f_name, mode=None):
        self.mode = mode
        self.f_name = f_name
        self.zipf : zipfile.ZipFile = None
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
    def open(self):
        if self.zipf is not None:
            self.zipf.close()
        self.zipf = zipfile.ZipFile(self.f_name, self.mode)
        return self
    
    def close(self):
        self.zipf.close()
        self.zipf = None
    
    @staticmethod
    def _file_name(name:str):
        if name.endswith('.pkl'):
            return name
        return name + '.pkl'
    
    @staticmethod
    def _unfile_name(name:str):
        if name.endswith('.pkl'):
            return name[:-4]
        return name
    
    def save(self, obj, name):
        assert 'w' in self.mode , "manager must be in save mode"    
        with self.zipf.open(self._file_name(name), self.mode) as obj_f:
                torch.save(obj, obj_f)
    
    
    def load(self, name):
        assert 'r' in self.mode , "manager must be in load mode"
        with self.zipf.open(self._file_name(name), self.mode) as obj_f:
            return torch.load(obj_f)
    
    def load_all(self,):
        names = self.zipf.namelist()
        res_dct = {}
        for name in names:
            res_dct[self._unfile_name(name)] = self.load(name)
        return res_dct
    
            
    def __getitem__(self, key):
        return self.load(key)

    def __setitem__(self, key, value):
        self.save(value, key)
    
    def __len__(self):
        assert 'r' in self.mode , "manager must be in load mode"
        return len(self.zipf.namelist())

if __name__ == '__main__':
    with MapFileManager('objects.zip', 'w') as mf:
        for i in tqdm(range(10)):
            mf[f'v{i}'] = i
        
    # with MapFileManager('objects.zip', 'r') as mf:
    #     for i in tqdm(range(1000)):
    #         mf.load(f'v{i}')
    