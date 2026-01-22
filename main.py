import random
import sys
import scipy.spatial # very important, does not work without it, i don't know why
import resource

import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from pytorch_lightning.tuner import Tuner
from utils.logging_utils import get_logger
from hydra.utils import instantiate, get_class
from utils.omega_utils import load_config
import torch.multiprocessing as mp

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.manual_seed(0)
random.seed(0)

torch.multiprocessing.set_sharing_strategy('file_system')
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
# mp.set_start_method("spawn", force=True)


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # dataset.sub_proteins.open()   


def train_model(config, smol=False, ckpt=None):
    metadata = config.metadata
    wandb_logger = instantiate(config.logger)

    tokenizer = instantiate(config.tokenizer)
    model = instantiate(config.model)

    lit_model = instantiate(config.lightning, model=model, name=metadata.name)
    lit_model.test_result_path = f'{metadata.experiment_folder}/test_results/{type(lit_model).__name__}/'
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        monitor="valid_loss/dataloader_idx_0",
        mode="min",
        save_on_train_epoch_end=True,
        dirpath=f'{config.metadata.experiment_folder}/checkpoints/{type(lit_model).__name__}/',
        filename= "{valid_loss/dataloader_idx_0:.5f}_{epoch:02d}",
    )
    trainer : pl.Trainer = instantiate(config.train.trainer, logger=wandb_logger, callbacks=[checkpoint_callback])

    dst = instantiate(config.train.train_dataset, tokenizer=tokenizer
                      )
    
    if "sampler" in config.train:
        trainer.strategy.setup_environment()
        sampler = instantiate(config.train.sampler, dataset=dst)
    else:
        sampler = None
    dlt = DataLoader(dst, batch_size=config.train.batch_size, 
                        sampler=sampler,
                        num_workers=min(torch.get_num_threads(),5), 
                        worker_init_fn=worker_init_fn)
    
    dsv = instantiate(config.train.val_dataset, tokenizer=tokenizer)
    dlv = DataLoader(dsv, batch_size=config.train.batch_size,
                # num_workers=torch.get_num_threads()//2, 
                worker_init_fn=worker_init_fn)
    
    dsv_corrupt = instantiate(config.train.corrupt_val_dataset, tokenizer=tokenizer)
    dlv_corrupt = DataLoader(dsv_corrupt, batch_size=config.train.batch_size,
                # num_workers=torch.get_num_threads()//2, 
                worker_init_fn=worker_init_fn)
    
    
    if isinstance(ckpt,str) and len(ckpt)==0:
        ckpt = None
    print('ckpt=',ckpt)
    try:
        lit_model.validation_clfs=dsv.clfs
    except:
        lit_model.validation_clfs=None
    trainer.fit(lit_model, 
                train_dataloaders=dlt, 
                val_dataloaders=[dlv, dlv_corrupt],
                ckpt_path=ckpt)
    

if __name__ == "__main__":
    config= load_config(config_path=sys.argv[1] if len(sys.argv) > 1 else 'config.yaml')
    ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    train_model(smol=bool(os.environ.get("SMOL")), config=config, ckpt=ckpt)
    