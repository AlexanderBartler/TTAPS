#!/bin/bash

python src/tta_validation_module.py \
    --adapt_exclusively layer3 \
    --artifact_dir ./wandb_artifacts \
    --artifact_name model-xxxxxxxx:v1 \
    --batch_size 32 \
    --corruption_type brightness \
    --data_dir ./datasets \
    --epsilon 1 \
    --fast_dev_run 0 \
    --gpus 1 \
    --loss_type swav \
    --lr_scaling 0.2 \
    --num_nodes 1 \
    --num_steps 10 \
    --num_workers 8 \
    --q_generator softmax_normalized \
    --temperature 0.75 \
    --wandb_entity <wandb_username> \
    --wandb_log_dir ./wandb_logs \
    --wandb_project example_project


# print end date
echo "Finished execution at $(date)"