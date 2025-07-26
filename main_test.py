import sys
import scipy.spatial # very important, does not work without it, i don't know why
import resource

from datasets.custom_distributed_sampler import CustomDistributedSampler, CustomTaskDistributedSampler
from datasets.partitioned_fsmol_dock import FsDockDatasetPartitioned
from models.dock_lightning import DockLightning
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import pytorch_lightning as pl
from tokenizers import Tokenizer
import torch
from datasets.fsmol_dock import FsDockDataset
from datasets.fsmol_dock_clf import FsDockClfDataset
from datasets.samplers import TaskSequentialSampler
from datasets.task_data_loader import TaskDataLoader
from models.modolo import Modolo
from models.modolo_lightning import ModoloLightning
from models.graph_embedder import GraphEmbedder
from models.graph_encoder import GraphEncoder
from models.interaction_encoder import InteractionEncoder
import datasets.process_chem.features as features
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from models.transformer import TransformerDecoder, TransformerEncoder
from pytorch_lightning.loggers import WandbLogger
import os
from pytorch_lightning.tuner import Tuner
from utils.logging_utils import get_logger
from hydra.utils import instantiate, get_class
from utils.omega_utils import load_config

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.manual_seed(0)

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.sub_proteins.open()   


def test_model(config, path,  smol=False):
    metadata = config.metadata

    tokenizer = instantiate(config.tokenizer)
    model = instantiate(config.model)
    
    lit_model = instantiate(config.lightning, model=model, tokenizer=tokenizer, name=metadata.name)
    lit_model.test_result_path = f'{metadata.experiment_folder}/test_results/{type(lit_model).__name__}/'
    
    trainer = instantiate(config.train.trainer,devices=1)

    print(config)

    dst = instantiate(config.test.dataset, tokenizer=tokenizer
                      )

    dlt = DataLoader(dst, batch_size=30, 
                        num_workers=torch.get_num_threads(), 
                        worker_init_fn=worker_init_fn)
    
    lit_model.test_clfs=dst.clfs
    trainer.test(lit_model,dlt, ckpt_path=path)
    

if __name__ == "__main__":
    config= load_config(config_path=sys.argv[1])
    test_model(path=sys.argv[2], config=config)
    