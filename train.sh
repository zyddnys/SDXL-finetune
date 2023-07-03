#!/bin/bash

BASE_MODEL="/mnt/nvme1/sdxl_ft/stable-diffusion-xl-base-0.9"
RUN_NAME="danbooru2022-ft"
DATASET="/mnt/nvme1/dataset/danbooru2022-full"
N_GPU=2 # <-- change to your number of GPUs
N_EPOCHS=2
BATCH_SIZE=4 # <-- works on RTX A6000
LOG_STEPS=500
GRADACC=1
RES=896 # change to 1024 if you have the VRAM
SAVE_STEP=2000

if [ $N_GPU -eq 1 ]
then
    python trainer.py --run_name=$RUN_NAME  --bucket_side_min=64 --use_8bit_adam=True --gradient_checkpointing=False --wandb False --save_steps=$SAVE_STEP --batch_size=$BATCH_SIZE --dataset=$DATASET --fp16=True --image_log_steps=$LOG_STEPS --epochs=$N_EPOCHS --resolution=$RES --use_ema=False --clip_penultimate=True --train_text_encoder  --model=$BASE_MODEL --gradient_accumulation $GRADACC
else
    python3 -m torch.distributed.run --nproc_per_node=$N_GPU trainer.py --run_name=$RUN_NAME  --bucket_side_min=64 --use_8bit_adam=True --save_steps=$SAVE_STEP --gradient_checkpointing=False --wandb False --batch_size=$BATCH_SIZE --dataset=$DATASET --fp16=True --image_log_steps=$LOG_STEPS --epochs=$N_EPOCHS --resolution=$RES --use_ema=False --clip_penultimate=True --train_text_encoder  --model=$BASE_MODEL --gradient_accumulation $GRADACC
fi
# and to resume... just add the --resume flag and supply it with the path to the checkpoint.
