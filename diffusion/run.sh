#!/bin/bash
# Author: qichen@cs.toronto.ca
# File Name: run_toy.sh
# Description:

seed=$1
T=$2
R=$3
data=$4
n_train=$5
n_iter=$6

python src/img_bound_fast_estimation.py --dataset $data --print_every $n_iter --num_steps 1000 --n_train $n_train --n_test 100 --mc_samples 100 \
        --sm_training_iter $n_iter --sample_every $n_iter --checkpoint_every $n_iter --real=True --debias=False --lr=0.0001 \
        --seed $seed --T0 $T --R $R --train_batch_size 128 --test_batch_size 128
