# pGNNs
This repository provides a reference implementation of <img src="http://latex.codecogs.com/gif.latex?^p">GNNs as described in the paper "**p-Laplacian Based Graph Neural Networks**". 

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

### Citing
If you find *pGNNs* useful in your research, please cite our paper:
```
@article{DBLP:journals/corr/abs-2111-07337,
  author    = {Guoji Fu and
               Peilin Zhao and
               Yatao Bian},
  title     = {p-Laplacian Based Graph Neural Networks},
  journal   = {CoRR},
  volume    = {abs/2111.07337},
  year      = {2021},
  url       = {https://arxiv.org/abs/2111.07337},
  eprinttype = {arXiv},
  eprint    = {2111.07337},
  timestamp = {Tue, 16 Nov 2021 12:12:31 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2111-07337.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
