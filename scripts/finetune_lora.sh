

JSON_FOLDER=None
IMAGE_FOLDER=None
VIDEO_FOLDER=None
DATA_PATH="./data/data.json"

# NCCL_P2P_LEVEL=NVL

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --include localhost:5 --master_addr 127.0.0.1 --master_port 28459 train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --lora_dropout 0.1 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path Vitron-base \
    --version v1 \
    --data_path  ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER}  \
    --image_tower LanguageBind/LanguageBind_Image \
    --video_folder ${VIDEO_FOLDER} \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --tune_region_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/vitron-7b-lora-4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048  --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"
