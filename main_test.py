import random
import sys
import scipy.spatial # very important, does not work without it, i don't know why
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import torch
from torch_geometric.loader import DataLoader
import os
from utils.logging_utils import get_logger
from hydra.utils import instantiate, get_class
from utils.omega_utils import load_config

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.manual_seed(0)
random.seed(0)
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # dataset.sub_proteins.open()   


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
                        num_workers=2, 
                        worker_init_fn=worker_init_fn)
    try:
        lit_model.test_clfs=dst.clfs
    except:
        pass
    trainer.test(lit_model,dlt, ckpt_path=path)
    

if __name__ == "__main__":
    config= load_config(save=True, config_path=sys.argv[1])
    test_model(path=sys.argv[2], config=config)
    