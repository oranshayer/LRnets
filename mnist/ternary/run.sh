#!/bin/bash

BASE_PATH="/home/oran/logdir/cifar10_ternary_first_layer_fp/"
c=0
for wd in 0.00000000001 0.0
do
	for dropout in 0.5 1
	do
		(( c++ ))
		NEW_PATH=$BASE_PATH$c
		python3 cifar10_multi_gpu_train.py --train_dir $NEW_PATH --wd $wd --dropout $dropout --first_layer_ternary False
	done
done
