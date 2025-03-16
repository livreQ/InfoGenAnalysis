#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-02:00     # DD-HH:MM:SS


'''
add your cluster settings here
'''

# Start training

seed=$1
T=$2
R=$3
n_train=$4
n_iter=$5
data="cifar10"

sh run.sh $seed $T $R $data $n_train $n_iter



