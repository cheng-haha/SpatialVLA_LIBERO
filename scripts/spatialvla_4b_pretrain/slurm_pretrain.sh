set -x

DEBUG=false
if [ "$DEBUG" = true ]; then
  GPUS=8
  GPUS_PER_NODE=8
  PER_DEVICE_BATCH_SIZE=2
  shuffle_buffer_size=2
  mixture=simpler_env
  NUM_WORKERS=0
  QUOTA_TYPE=reserved
  PARTITION="EAItmp"
  CPUS_PER_TASK=10
  freeze_llm=True
  save_steps=5
fi

PARTITION=${PARTITION:-"EAItmp"}
GPUS=${GPUS:-64}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
SRUN_ARGS=${SRUN_ARGS:-""}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-32}
BATCH_SIZE=${BATCH_SIZE:-$((GPUS * PER_DEVICE_BATCH_SIZE))}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

mixture=oxe_spatial_vla
NUM_WORKERS=${NUM_WORKERS:-1}
shuffle_buffer_size=${shuffle_buffer_size:-65536}
tsfm_thread_muti=1
read_thread_muti=1
lr=2e-5
min_sigma=0.5
freeze_llm=${freeze_llm:-False}
save_steps=${save_steps:-20000}

note=paligemma3b_vis_zoe_obs14_untie_gaussN8194_unicam_lr${lr}_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACC}_node$((GPUS / GPUS_PER_NODE))_gpu${GPUS}
cur_time=$(date "+%H-%M-%S")
date_dir=$(date "+%Y-%m-%d")

# resume training from ckpt
resume_path=
model_name_or_path=
fix_raw_length=${fix_raw_length:-0}
OUTPUT_DIR=${resume_path:-outputs/spatialvla_v1_paligemma2_3b_pretrain/$date_dir/${cur_time}_${mixture}_${note}}
mkdir -p $OUTPUT_DIR

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=$((1024 + $RANDOM % 64536))
export TF_CPP_MIN_LOG_LEVEL=3
export TF_USE_LEGACY_KERAS=False
export LD_PRELOAD=../libtcmalloc_minimal.so.4

cp $(realpath "$0") ${OUTPUT_DIR}

srun -p ${PARTITION} \
  -o ${OUTPUT_DIR}/log_$(date +%Y-%m-%d-%H-%M-%S)_${mixture}_%j.out \
  -J spatialvla_N8194_gs_obs14${mixture}_${note} \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  --quotatype=${QUOTA_TYPE} \
  --time=25-00:00:00 \
  ${SRUN_ARGS} \
  python -u train/spatialvla_pretrain.py \
  --fix_raw_length ${fix_raw_length} \
  --ignore_data_skip True \
  --data_root_dir ../DATA/open_x_embodiment_converted \
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
  --use_flash_attn2 True \
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
  --num_train_epochs 1 \
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
  --action_config scripts/action_config_N8194.json \
  --intrinsic_config_path scripts/intrinsics_uni.json \
  --normalized_statistic_path scripts/gaussian_statistic_spatialvla_plus.json \
  --min_sigma ${min_sigma} \
  --report_to tensorboard \
  --use_raw_dataloader True \
  --eval_on_start False \
  --train_only True \
  --log_level warning