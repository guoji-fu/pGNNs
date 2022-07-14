# pGNNs
This repository provides a reference implementation of <img src="http://latex.codecogs.com/gif.latex?^p">GNNs as described in the paper "**[p-Laplacian Based Graph Neural Networks](https://proceedings.mlr.press/v162/fu22e.html)**" published at ICML'2022.

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
If you find <img src="http://latex.codecogs.com/gif.latex?^p">*GNNs* useful in your research, please cite our paper:
```
@inproceedings{DBLP:conf/icml/FuZB22,
  author    = {Guoji Fu and
               Peilin Zhao and
               Yatao Bian},
  editor    = {Kamalika Chaudhuri and
               Stefanie Jegelka and
               Le Song and
               Csaba Szepesv{\'{a}}ri and
               Gang Niu and
               Sivan Sabato},
  title     = {p-Laplacian Based Graph Neural Networks},
  booktitle = {International Conference on Machine Learning, {ICML} 2022, 17-23 July
               2022, Baltimore, Maryland, {USA}},
  series    = {Proceedings of Machine Learning Research},
  volume    = {162},
  pages     = {6878--6917},
  publisher = {{PMLR}},
  year      = {2022},
  url       = {https://proceedings.mlr.press/v162/fu22e.html},
  timestamp = {Tue, 12 Jul 2022 17:36:52 +0200},
  biburl    = {https://dblp.org/rec/conf/icml/FuZB22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
