#!/bin/bash

for seed in $(seq 0 1 2)
    do
        for T in $(seq 0.2 0.2 2)
            do
		echo $T
		echo $seed
                sh run.sh $seed $T 4000 "mnist" 16 10000
        done        
done

#seed=$1
#T=$2
#R=$3
#data=$4
#n_train=$5
