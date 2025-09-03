export MODEL_NAME="stabilityai/sdxl-turbo"
export VAE="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="./turbo_dpo_dataset"


# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~30 hours / 200 steps

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataset_name=$DATASET_NAME \
  --max_train_steps=3000 \
  --dataloader_num_workers=0 \
  --gradient_accumulation_steps=16 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=5 \
  --learning_rate=1e-8 --scale_lr \
  --cache_dir="~/.cache" \
  --checkpointing_steps 10 \
  --beta_dpo 5000 \
  --sdxl --resolution 512 --proportion_empty_prompts 0 \
  --output_dir="try-improve-2-more-steps"
