#!/bin/bash

# Run your python code
python src/supervised_swav_module.py \
    --max_epochs 500 \
    --arch resnet26_w32 \
    --batch_size 256 \
    --data_dir ./datasets \
    --dataset cifar100 \
    --fast_dev_run 0 \
    --gaussian_blur True \
    --gpus 1 \
    --hidden_mlp 256 \
    --learning_rate 0.5 \
    --max_scale_crops [1] \
    --min_scale_crops [0.14] \
    --nmb_crops [2] \
    --nmb_prototypes 300 \
    --norm_layer group_norm \
    --optimizer sgd \
    --size_crops [32] \
    --supervised_transforms default \
    --supervised_weight 0.3 \
    --temperature 0.1 \
    --weight_decay 1e-05 \
    --freeze_prototypes_epochs 0 \
    --queue_length 0 \
    --supervised_hidden_mlp 0 \
    --prototype_entropy_regularization_type same_optimizer \
    --prototype_entropy_regularization_weight 0.1 \
    --wandb_log_dir ./wandb_logs \
    --wandb_project example_project_cifar100 \
    --wandb_run_name joint-entropy \
    --supervised_head_after_proj_head True


# print end date
echo "Finished execution at $(date)"
