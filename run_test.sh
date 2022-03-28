#!/bin/bash

function run_exp(){
    python main.py --input $1 \
                        --train_rate $2 \
                        --val_rate $3 \
                        --model $4 \
                        --num_hid $5 \
                        --lr $6 \
                        --epochs $7 \
                        --weight_decay $8 \
                        --dropout $9 \
                        --mu ${10} \
                        --p ${11} \
                        --K ${12} \
                        --alpha ${13} \
                        --dprate ${14} \
                        --runs ${15}
}

run_exp cora 0.025 0.025 pgnn 16 0.01 1000 5e-4 0.5 0.1 1 4 0.1 0.5 10
run_exp cora 0.025 0.025 pgnn 16 0.01 1000 5e-4 0.5 0.1 1.5 4 0.1 0.5 10
run_exp cora 0.025 0.025 pgnn 16 0.01 1000 5e-4 0.5 0.1 2 4 0.1 0.5 10
run_exp cora 0.025 0.025 pgnn 16 0.01 1000 5e-4 0.5 0.1 2.5 4 0.1 0.5 10