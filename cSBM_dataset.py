#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#
# Distributed under terms of the MIT license.

"""
This is a script for contexual SBM model and its dataset generator.
contains functions:
        ContextualSBM
        parameterized_Lambda_and_mu
        save_data_to_pickle
    class:
        dataset_ContextualSBM

"""
import numpy as np
import torch
from torch_geometric.data import Data
import pickle
from datetime import datetime
import os.path as osp
import os
import ipdb
import argparse

import torch
from torch_geometric.data import InMemoryDataset



def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag == 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(
            rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data


def ContextualSBM(n, d, Lambda, p, mu, train_percent=0.025, val_percent=0.025):
    # n = 800 #number of nodes
    # d = 5 # average degree
    # Lambda = 1 # parameters
    # p = 1000 # feature dim
    # mu = 1 # mean of Gaussian
    gamma = n/p

    c_in = d + np.sqrt(d)*Lambda
    c_out = d - np.sqrt(d)*Lambda
    y = np.ones(n)
    y[int(n/2)+1:] = -1
    y = np.asarray(y, dtype=int)

    # creating edge_index
    edge_index = [[], []]
    for i in range(n-1):
        for j in range(i+1, n):
            if y[i]*y[j] > 0:
                Flip = np.random.binomial(1, c_in/n)
            else:
                Flip = np.random.binomial(1, c_out/n)
            if Flip > 0.5:
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_index[0].append(j)
                edge_index[1].append(i)

    # creating node features
    x = np.zeros([n, p])
    u = np.random.normal(0, 1/np.sqrt(p), [1, p])
    for i in range(n):
        Z = np.random.normal(0, 1, [1, p])
        x[i] = np.sqrt(mu/n)*y[i]*u + Z/np.sqrt(p)
    data = Data(x=torch.tensor(x, dtype=torch.float32),
                edge_index=torch.tensor(edge_index),
                y=torch.tensor((y + 1) // 2, dtype=torch.int64))
    # order edge list and remove duplicates if any.
    data.coalesce()

    num_class = len(np.unique(y))
    val_lb = int(n * val_percent)
    percls_trn = int(round(train_percent * n / num_class))
    data = random_planetoid_splits(data, num_class, percls_trn, val_lb)

    # add parameters to attribute
    data.Lambda = Lambda
    data.mu = mu
    data.n = n
    data.p = p
    data.d = d
    data.train_percent = train_percent
    data.val_percent = val_percent

    return data


def parameterized_Lambda_and_mu(theta, p, n, epsilon=0.1):
    '''
    based on claim 3 in the paper, 

        lambda^2 + mu^2/gamma = 1 + epsilon.

    1/gamma = p/n
    longer axis: 1
    shorter axis: 1/gamma.
    =>
        lambda = sqrt(1 + epsilon) * sin(theta * pi / 2)
        mu = sqrt(gamma * (1 + epsilon)) * cos(theta * pi / 2)
    '''
    from math import pi
    gamma = n / p
    assert (theta >= -1) and (theta <= 1)
    Lambda = np.sqrt(1 + epsilon) * np.sin(theta * pi / 2)
    mu = np.sqrt(gamma * (1 + epsilon)) * np.cos(theta * pi / 2)
    return Lambda, mu


def save_data_to_pickle(data, p2root='../data/', file_name=None):
    '''
    if file name not specified, use time stamp.
    '''
    now = datetime.now()
    surfix = now.strftime('%b_%d_%Y-%H:%M')
    if file_name is None:
        tmp_data_name = '_'.join(['cSBM_data', surfix])
    else:
        tmp_data_name = file_name
    p2cSBM_data = osp.join(p2root, tmp_data_name)
    if not osp.isdir(p2root):
        os.makedirs(p2root)
    with open(p2cSBM_data, 'bw') as f:
        pickle.dump(data, f)
    return p2cSBM_data


class dataset_ContextualSBM(InMemoryDataset):
    r"""Create synthetic dataset based on the contextual SBM from the paper:
    https://arxiv.org/pdf/1807.09596.pdf

    Use the similar class as InMemoryDataset, but not requiring the root folder.

       See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_dataset.html#creating-in-memory-datasets>`__ for the accompanying
    tutorial.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset if not specified use time stamp.

        for {n, d, p, Lambda, mu}, with '_' as prefix: intial/feed in argument.
        without '_' as prefix: loaded from data information

        n: number nodes
        d: avg degree of nodes
        p: dimenstion of feature vector.

        Lambda, mu: parameters balancing the mixture of information, 
                    if not specified, use parameterized method to generate.

        epsilon, theta: gap between boundary and chosen ellipsoid. theta is 
                        angle of between the selected parameter and x-axis.
                        choosen between [0, 1] => 0 = 0, 1 = pi/2

        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

#     url = 'https://github.com/kimiyoung/planetoid/raw/master/data'

    def __init__(self, root, name=None,
                 n=800, d=5, p=100, Lambda=None, mu=None,
                 epsilon=0.1, theta=0.5,
                 train_percent=0.01, val_percent=0.01,
                 transform=None, pre_transform=None):

        now = datetime.now()
        surfix = now.strftime('%b_%d_%Y-%H:%M')
        if name is None:
            # not specifing the dataset name, create one with time stamp.
            self.name = '_'.join(['cSBM_data', surfix])
        else:
            self.name = name

        self._n = n
        self._d = d
        self._p = p

        self._Lambda = Lambda
        self._mu = mu
        self._epsilon = epsilon
        self._theta = theta

        self._train_percent = train_percent
        self._val_percent = val_percent

        root = osp.join(root, self.name)
        if not osp.isdir(root):
            os.makedirs(root)
        super(dataset_ContextualSBM, self).__init__(
            root, transform, pre_transform)

#         ipdb.set_trace()
        self.data, self.slices = torch.load(self.processed_paths[0])
        # overwrite the dataset attribute n, p, d, Lambda, mu
        self.Lambda = self.data.Lambda.item()
        self.mu = self.data.mu.item()
        self.n = self.data.n.item()
        self.p = self.data.p.item()
        self.d = self.data.d.item()
        self.train_percent = self.data.train_percent.item()
        self.val_percent = self.data.val_percent.item()

#     @property
#     def raw_dir(self):
#         return osp.join(self.root, self.name, 'raw')

#     @property
#     def processed_dir(self):
#         return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        for name in self.raw_file_names:
            p2f = osp.join(self.raw_dir, name)
            if not osp.isfile(p2f):
                # file not exist, so we create it and save it there.
                if self._Lambda is None or self._mu is None:
                    # auto generate the lambda and mu parameter by angle theta.
                    self._Lambda, self._mu = parameterized_Lambda_and_mu(self._theta,
                                                                         self._p,
                                                                         self._n,
                                                                         self._epsilon)
                tmp_data = ContextualSBM(self._n,
                                         self._d,
                                         self._Lambda,
                                         self._p,
                                         self._mu,
                                         self._train_percent,
                                         self._val_percent)

                _ = save_data_to_pickle(tmp_data,
                                        p2root=self.raw_dir,
                                        file_name=self.name)
            else:
                # file exists already. Do nothing.
                pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phi', type=float, default=1)
    parser.add_argument('--epsilon', type = float , default = 3.25)
    parser.add_argument('--root', default = 'data/')
    parser.add_argument('--name', default = 'cSBM_demo')
    parser.add_argument('--num_nodes', type = int, default = 800)
    parser.add_argument('--num_features', type = int, default = 1000)
    parser.add_argument('--avg_degree', type = float, default = 5)
    parser.add_argument('--train_percent', type = float, default = 0.025)
    parser.add_argument('--val_percent', type = float, default = 0.025)

    args = parser.parse_args()

    dataset_ContextualSBM(root = args.root,
            name = args.name,
            theta = args.phi,
            epsilon = args.epsilon, 
            n = args.num_nodes,
            d = args.avg_degree,
            p = args.num_features,
            train_percent = args.train_percent,
            val_percent=args.val_percent)
    
