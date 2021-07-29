#!/bin/bash

ROOT_DIR=$1
NUM_GPUS=$2
python -u -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port 9901 training/pipelines/train_classifier.py \
 --distributed --config configs/deit_distill.json --freeze-epochs 0 --test_every 1 --opt-level O1 --label-smoothing 0.01 --folds-csv folds.csv  --fold 0 --seed 111 --data-dir $ROOT_DIR --prefix deit_d_111_ > final/Efficient_ViT_Distill/logs/deit_d_111

python -u -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port 9901 training/pipelines/train_classifier.py \
 --distributed --config configs/deit_distill.json --freeze-epochs 0 --test_every 1 --opt-level O1 --label-smoothing 0.01 --folds-csv folds.csv  --fold 0 --seed 555 --data-dir $ROOT_DIR --prefix deit_d_555_ > final/Efficient_ViT_Distill/logs/deit_d_555

python -u -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port 9901 training/pipelines/train_classifier.py \
 --distributed --config configs/deit_distill.json --freeze-epochs 0 --test_every 1 --opt-level O1 --label-smoothing 0.01 --folds-csv folds.csv  --fold 0 --seed 777 --data-dir $ROOT_DIR --prefix deit_d_777_ > final/Efficient_ViT_Distill/logs/deit_d_777

python -u -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port 9901 training/pipelines/train_classifier.py \
 --distributed --config configs/deit_distill.json --freeze-epochs 0 --test_every 1 --opt-level O1 --label-smoothing 0.01 --folds-csv folds.csv  --fold 0 --seed 888 --data-dir $ROOT_DIR --prefix deit_d_888_ > final/Efficient_ViT_Distill/logs/deit_d_888

python -u -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port 9901 training/pipelines/train_classifier.py \
 --distributed --config configs/deit_distill.json --freeze-epochs 0 --test_every 1 --opt-level O1 --label-smoothing 0.01 --folds-csv folds.csv  --fold 0 --seed 999 --data-dir $ROOT_DIR --prefix deit_d_999_ > final/Efficient_ViT_Distill/logs/deit_d_999
