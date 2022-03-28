#! /bin/bash
#
# create_cSBM_dataset.sh

python cSBM_dataset.py --phi -1.0 \
    --name csbm_phi_dense_-1 \
    --root ./dataset \
    --num_nodes 5000 \
    --num_features 2000 \
    --avg_degree 5 \
    --epsilon 3.25 \
    --train_percent 0.6 \
    --val_percent 0.2


python cSBM_dataset.py --phi -0.75 \
    --name csbm_phi_dense_-0.75 \
    --root ./dataset \
    --num_nodes 5000 \
    --num_features 2000 \
    --avg_degree 5 \
    --epsilon 3.25 \
    --train_percent 0.6 \
    --val_percent 0.2


python cSBM_dataset.py --phi -0.5 \
    --name csbm_phi_dense_-0.5 \
    --root ./dataset \
    --num_nodes 5000 \
    --num_features 2000 \
    --avg_degree 5 \
    --epsilon 3.25 \
    --train_percent 0.6 \
    --val_percent 0.2


python cSBM_dataset.py --phi -0.25 \
    --name csbm_phi_dense_-0.25 \
    --root ./dataset \
    --num_nodes 5000 \
    --num_features 2000 \
    --avg_degree 5 \
    --epsilon 3.25 \
    --train_percent 0.6 \
    --val_percent 0.2


python cSBM_dataset.py --phi 0 \
    --name csbm_phi_dense_0 \
    --root ./dataset \
    --num_nodes 5000 \
    --num_features 2000 \
    --avg_degree 5 \
    --epsilon 3.25 \
    --train_percent 0.6 \
    --val_percent 0.2


python cSBM_dataset.py --phi 0.25 \
    --name csbm_phi_dense_0.25 \
    --root ./dataset \
    --num_nodes 5000 \
    --num_features 2000 \
    --avg_degree 5 \
    --epsilon 3.25 \
    --train_percent 0.6 \
    --val_percent 0.2


python cSBM_dataset.py --phi 0.5 \
    --name csbm_phi_dense_0.5 \
    --root ./dataset \
    --num_nodes 5000 \
    --num_features 2000 \
    --avg_degree 5 \
    --epsilon 3.25 \
    --train_percent 0.6 \
    --val_percent 0.2


python cSBM_dataset.py --phi 0.75 \
    --name csbm_phi_dense_0.75 \
    --root ./dataset \
    --num_nodes 5000 \
    --num_features 2000 \
    --avg_degree 5 \
    --epsilon 3.25 \
    --train_percent 0.6 \
    --val_percent 0.2


python cSBM_dataset.py --phi 1.0 \
    --name csbm_phi_dense_1 \
    --root ./dataset \
    --num_nodes 5000 \
    --num_features 2000 \
    --avg_degree 5 \
    --epsilon 3.25 \
    --train_percent 0.6 \
    --val_percent 0.2