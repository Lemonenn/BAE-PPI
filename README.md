# T-GAE: Transformer-based Graph Autoencoder for Protein-Protein Interaction

## Dependencies
Install the required dependencies and activate the running environment with conda:
```bash
conda env create -f environment.yml
conda activate T-GAE
```
The default PyTorch version and cudatoolkit version are configured in environment.yml, which can be modified according to your local CUDA version and hardware environment.

## Dataset
Raw Data
The raw datasets used in this work include SHS27k, SHS148k, and STRING for Protein-Protein Interaction (PPI) prediction, containing protein sequence dictionaries, PPI network topology and protein structure files:
```bash
protein.SHS27k.sequences.dictionary.csv Protein sequences of SHS27k dataset
protein.SHS148k.sequences.dictionary.csv Protein sequences of SHS148k dataset
protein.STRING.sequences.dictionary.csv Protein sequences of STRING dataset
protein.actions.SHS27k.txt PPI network edge list of SHS27k
protein.actions.SHS148k.txt PPI network edge list of SHS148k
protein.actions.STRING.txt PPI network edge list of STRING
```

## Data Preprocessing
Preprocess the raw data to generate protein feature matrices and adjacency matrices required for model training (the script is applicable to custom PPI datasets):
```bash
python ./src/dataloader.py --dataset data_name
```
where data_name should be one of SHS27k, SHS148k, STRING.

## Usage
Standard Training and Inference on SHS27k/SHS148k/STRING
Run the following command to train the T-GAE model and perform PPI prediction on the target dataset. The model will automatically complete training, validation and test process:
```bash
python -B ./src/train.py --dataset STRING --split_mode bfs
```
Key parameter description:
--dataset: Select the training dataset, optional: SHS27k, SHS148k, STRING
--split_mode: Graph data partition strategy, default is bfs (Breadth-First Search)
