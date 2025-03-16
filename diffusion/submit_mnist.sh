#!/bin/bash

for seed in $(seq 0 1 2)
    do
        for T in $(seq 0.2 0.2 2)
            do
	        echo $T
	        echo $seed
                sbatch run_mnist.sh $seed $T 4000 16 10000
        done        
done
