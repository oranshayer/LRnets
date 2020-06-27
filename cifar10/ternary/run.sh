#!/bin/bash

BASE_PATH="/home/oran/logdir/cifar10_ternary_no_init/"
c=0
for wd in 0.00000000001 0.0
do
	for dropout in 0.5 1
	do
		(( c++ ))
		NEW_PATH=$BASE_PATH$c
		python3 cifar10_multi_gpu_train.py --train_dir $NEW_PATH --wd $wd --dropout $dropout --hot_start False --lr_decay_epochs 250 --epochs 320
	done
done
