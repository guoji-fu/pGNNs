# $p$GNNs
This repository provides a reference implementation of $^p$GNNs as described in the paper "**$p$-Laplacian Based Graph Neural Networks**". 

### Requirements
Install the following packages:

- [pytorch 1.9.0](https://pytorch.org/get-started/locally/)
- [torch_geometric 1.7.1](https://github.com/pyg-team/pytorch_geometric)
- scikit-learn 0.24.2
- networkx 2.5.1

### Basic Usage
```
$ python main.py --input cora --train_rate 0.025 --val_rate 0.025 --model pgnn --mu 0.1  --p 2 --K 4 --num_hid 16 --lr 0.01 --epochs 1000 
```

### Testing Examples
```
$ bash run_test.sh
```