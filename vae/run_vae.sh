#!/bin/sh
# Author: qichen@cs.toronto.ca
# Created Time : Mon Nov 18 02:06:27 2024
# File Name: run_vae.sh
# Description:


SCRIPT_DIR="./scripts"
SUMMARY_DIR="./summaries"
OUTPUT_DIR="./output"
RESULT_DIR="./results"

RESULT_DIR_BMNIST_LR3=$RESULT_DIR"/binarized_mnist_extra_lr3/results"
RESULT_DIR_CIFAR10=$RESULT_DIR"/cifar10/results"
RESULT_DIR_CELEBA=$RESULT_DIR"/celeba/results"

CKPT_DIR_BMNIST_LR3=$RESULT_DIR"/binarized_mnist_extra_lr3/checkpoints"

python $SCRIPT_DIR/memorization.py  \
    --dataset BinarizedMNIST \
    --model BernoulliMLPVAE \
    --mode full \
    --latent-dim 16 \
    --batch-size 64 \
    --learning-rate 1e-3 \
    --epochs 100 \
    --repeats 1 \
    --seed 42 \
    --compute-px-every 5 \
    --result-dir $RESULT_DIR_BMNIST_LR3 \
    --checkpoint-every 20 \
    --checkpoint-dir $CKPT_DIR_BMNIST_LR3