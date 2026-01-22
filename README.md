# Modolo
 
Official implementation of **Modolo**
### Conda environment
```bash
conda env create -f environment.yml
conda activate Modolo
```

### Pre-trained models
Pre-trained model is available at 
```
experiments/final/final.ckpt
```

### Data
The train, test and evaluation data splits are available under `data/splits`. For the model to run, `.sdf` and `.pdb` files of the molecule and the target protein are required.

### Dataset
The construction of the dataset is ran with `build_dataset.py`. The dataset classes `CrossPartitionedFsDockDataset` and `FsDockDataset` require a .csv file with the following format:

|assay_id|target_id|protein_path|ligand_path|label|type|
|--------|---------|------------|-----------|-----|----|
|CHEMBL1000314|P14222|data/proteins/pdbs/P14222.pdb|data/CHEMBL1000314/docks/b6.sdf|0|B|

### Training 
To train the model, run 
```
python main.py experiments/final/config.yaml
```

### Inference
To generate new molecules, run
```
python main.py experiments/final/config.yaml experiments/final/final.ckpt
> [!TIP]
> Note that the config file holds the route to the dataset that the model relies on.

### Evaluation
The code we used for Evaluation is available in the `evaluation` directory.
