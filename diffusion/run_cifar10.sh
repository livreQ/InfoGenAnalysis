#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-02:00     # DD-HH:MM:SS
#SBATCH --account=def-florian7
module load python/3.10 cuda cudnn

SOURCEDIR=~/scratch/diffusion_bound/

# Prepare virtualenv
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
export PYTHONPATH="/home/cq92/scratch/DCML/:$PYTHONPATH"

# Start training

seed=$1
T=$2
R=$3
n_train=$4
n_iter=$5
data="cifar10"

sh run.sh $seed $T $R $data $n_train $n_iter

#python src/img_bound_estimation.py --dataset=mnist --print_every 2000 --num_steps 1000 --n_train 100 --n_test 100 --mc_samples 100 \
#	--sm_training_iter 10000 --sample_every 2000 --checkpoint_every 5000 --real=True --debias=False --lr=0.0001 \
#	--seed $seed --T0 $T -R $R --train_batch_size 128 --test_batch_size 128


