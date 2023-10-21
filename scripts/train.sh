#!/usr/bin/env sh
mkdir -p logs
now=$(date +"%m%d_%H%M")
log_name="LOG_Train_$2_$3_$now"
CUDA_VISIBLE_DEVICES=$1 nohup python3 -u main.py --prefix $3 --benchmark $2 ${@:4} 2>&1|tee logs/$log_name.log > nohup.txt 2>&1 &

# bash train.sh TSA FineDiving 0,1 [--resume]
