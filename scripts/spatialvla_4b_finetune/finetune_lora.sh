set -x

DEBUG=true
if [ "$DEBUG" = true ]; then
  GPUS=1
  GPUS_PER_NODE=1
  PER_DEVICE_BATCH_SIZE=2
  shuffle_buffer_size=2
  mixture=simpler_env
  NUM_WORKERS=0
  TORCH_RUN_ARGS="--standalone --nnodes=1"
  freeze_llm=True
  save_steps=5
fi

GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-32}
BATCH_SIZE=${BATCH_SIZE:-$((GPUS * PER_DEVICE_BATCH_SIZE))}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

mixture="fractal"
min_sigma=0
mixture=${mixture:-oxe_magic_soup_plus}
NUM_WORKERS=${NUM_WORKERS:-1}
shuffle_buffer_size=${shuffle_buffer_size:-131072}
tsfm_thread_muti=1
read_thread_muti=1

lr=5e-4
use_all_lora=32
lora_alpha=32
lora_target="all-linear"
modules_to_save="none"

epoch=50
freeze_llm=${freeze_llm:-False}
save_steps=${save_steps:-10000}
use_flash_attn2=True

# ADAPT_ARGS="--adapt_emb scripts/gaussian_statistic_fractal.json" # use spatial embedding adaption
adpt_feature=False

# cur_time=$(date "+%H-%M-%S")
cur_time=$(date "+%H")
date_dir=$(date "+%Y-%m-%d")

# resume training from ckpt
model_name_or_path=../pretrained/spatialvla-4b-224-pt
note=$(basename $model_name_or_path)_lr${lr}_bs${PER_DEVICE_BATCH_SIZE}_node$((GPUS / GPUS_PER_NODE))_gpu${GPUS}_r${use_all_lora}_a${lora_alpha}_ep${epoch}_${lora_target}
# fix_raw_length=${resume_path:+xxx}
fix_raw_length=${fix_raw_length:-0}
OUTPUT_DIR=${resume_path:-outputs/spatialvla_v1_paligemma2_3b_collected/$date_dir/${cur_time}_${mixture}_${note}}
mkdir -p $OUTPUT_DIR

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TF_CPP_MIN_LOG_LEVEL=3
export TF_USE_LEGACY_KERAS=False
export LD_PRELOAD=../libtcmalloc.so.4.5.3
export TRITON_CACHE_DIR=~/.triton

# NOTE: set CONDA
export PATH="/cpfs01/shared/optimal/vla_ptm/miniconda3/bin:$PATH"
. /cpfs01/shared/optimal/vla_ptm/miniconda3/etc/profile.d/conda.sh
export LD_LIBRARY_PATH="/cpfs01/shared/optimal/vla_ptm/miniconda3/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/cpfs01/shared/optimal/vla_ptm/miniconda3/envs/internvla/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/cpfs01/shared/optimal/vla_ptm/miniconda3/envs/internvla/lib/python3.10/site-packages/nvidia/cusparse/lib:${LD_LIBRARY_PATH}"
LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
CFLAGS="/cpfs01/shared/optimal/vla_ptm/miniconda3/include"
LDFLAGS="/cpfs01/shared/optimal/vla_ptm/miniconda3/lib"
source /cpfs01/shared/optimal/vla_ptm/miniconda3/bin/activate internvla

cp $(realpath "$0") ${OUTPUT_DIR}
log_path=${OUTPUT_DIR}/log_$(date +%Y-%m-%d-%H-%M-%S)_${mixture}.out

export LAUNCHER="pytorch"
TORCH_RUN_ARGS=${TORCH_RUN_ARGS:-"--nnodes $NODES --nproc-per-node $GPUS_PER_NODE --master_addr $MASTER_ADDR --master_port $MASTER_PORT"}

torchrun $TORCH_RUN_ARGS \
  train/spatialvla_finetune.py \
  --model_name_or_path ${model_name_or_path} \
  ${ADAPT_ARGS} \
  --adpt_feature ${adpt_feature} \
  --use_all_lora ${use_all_lora} \
  --lora_alpha ${lora_alpha} \
  --lora_target ${lora_target} \
  --min_sigma ${min_sigma} \
  --modules_to_save ${modules_to_save} \
  --fix_raw_length ${fix_raw_length} \
  --ignore_data_skip True \
  --data_root_dir /oss/vla_ptm_hwfile/DATA/fine_tune \
  --data_mix ${mixture} \
  --shuffle_buffer_size ${shuffle_buffer_size} \
  --tsfm_thread_muti ${tsfm_thread_muti} \
  --read_thread_muti ${read_thread_muti} \
  --data_augment True \
  --obs_backward_steps 0 \
  --obs_backward_delta 1 \
  --action_forward_steps 3 \
  --vision_zoe_path ../pretrained/zoedepth-nyu-kitti \
  --vlm_path ../pretrained/paligemma2-3b-pt-224 \
  --use_vision_zoe True \
  --use_flash_attn2 ${use_flash_attn2} \
  --output_dir ${OUTPUT_DIR} \
  --overwrite_output_dir False \
  --force_image_size 224 \
  --vision_attn_dropout 0.0 \
  --freeze_llm ${freeze_llm} \
  --unfreeze_lm_head True \
  --freeze_llm_embed True \
  --un_tie_weight True \
  --freeze_vision_tower False \
  --freeze_projector False \
  --n_freqs 8 \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers ${NUM_WORKERS} \
  --bf16 True \
  --tf32 True \
  --num_train_epochs ${epoch} \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy no \
  --eval_accumulation_steps 64 \
  --save_strategy steps \
  --save_steps ${save_steps} \
  --save_total_limit 3 \
  --learning_rate ${lr} \
  --weight_decay 0.0 \
  --warmup_ratio 0.005 \
  --lr_scheduler_type linear \
  --logging_steps 500 \
  --max_seq_length 2048 \
  --do_train True \
  --grad_checkpoint True \
  --ps_version v2 \
  --deepspeed scripts/zero_stage1_config.json \
  --intrinsic_config_path scripts/intrinsics_uni.json \
  --report_to tensorboard \
  --use_raw_dataloader True \
  --eval_on_start False \
  --train_only True \
  --log_level warning