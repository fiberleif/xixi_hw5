# Homework5.7: Pytorch 

## Dependency Installation
```bash
pip install -r requirements.txt
```

## Data Preparation
```bash
sh dataset/scripts/get_data.sh
sh dataset/scripts/fetch_flowers17.sh
sh dataset/scripts/fetch_flowers102.sh
```
For convenience, you can also use the following script to automatically generate all the datasets.
```bash
sh prepareall.sh
```

## Code Usage
### 7.1 Train a neural network in PyTorch
```bash
python code/main.py --dataset-name (select one from ["NIST26", "NIST36", "MNIST", "EMNIST"]) --model-architecture (select one from ["FCN", "CNN"])
```
### 7.2 Fine Tuning
```bash
python code/main.py --dataset-name (select one from ["flowers17", "flowers102"]) --pretrain True
```
Further, you can use the following script to automatically train all the tasks and collect experiments results (including both 7.1, 7.2).
```bash
sh runall.sh
```
