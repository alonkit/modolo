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
experiments/modolo_margin_v2/2025_06_24_23_34/checkpoints/ModoloLightning/validation_avg_success=0.23591_epoch=49.ckpt
```

### Data
The train, test and evaluation data splits are available under `data/splits`. For the model to run, `.sdf` and `.pdb` files of the molecule and the target protein are required.

### Dataset
The construction of the dataset is ran with `build_dataset.py`. The dataset classes `FsDockDatasetPartitioned` and `FsDockClfDataset` require a .csv file with the following format:

|assay_id|target_id|protein_path|ligand_path|label|type|
|--------|---------|------------|-----------|-----|----|
|CHEMBL1000314|P14222|data/proteins/pdbs/P14222.pdb|data/CHEMBL1000314/docks/b6.sdf|0|B|

### Training 
To train the model, run 
```
python main.py experiments/modolo_margin_v2/2025_06_24_23_34/config.yaml
```

### Inference
To generate new molecules, run
```
python main.py experiments/modolo_margin_v2/2025_06_24_23_34/config.yaml experiments/modolo_margin_v2/2025_06_24_23_34/checkpoints/ModoloLightning/validation_avg_success=0.23591_epoch=49.ckpt
```
> [!TIP]
> Note that the config file holds the route to the dataset that the optimization relies on.

### Evaluation
The code we used for Evaluation is available in the `Ipynb` notebook `evaluation.ipynb`.
