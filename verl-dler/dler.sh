#!/usr/bin/env bash
set -euxo pipefail
# DAPO (w/o Dynamic Sampling)

project_name='DAPO-verl'
exp_name='DAPO-wo-DS-r1-1.5B'

adv_estimator=reinforce_plus_plus_baseline

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.0005

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=1024
max_response_length=4000
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=False
# train_prompt_bsz=512
train_prompt_bsz=256
n_resp_per_prompt=16
train_prompt_mini_bsz=64

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"/lustre/fsw/portfolios/nvr/users/shihyangl/reasoning_trace_distill/verl-main/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-16}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"/lustre/fsw/portfolios/nvr/users/shihyangl/reasoning_trace_distill/verl-main/verl"}
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CKPTS_DIR="/lustre/fsw/portfolios/nvr/users/shihyangl/test_dler"
TRAIN_FILE="/lustre/fsw/portfolios/nvr/users/mingjiel/data/deepscaler/train.parquet"
TEST_FILE="/lustre/fsw/portfolios/nvr/users/mingjiel/data/deepscaler/aime.parquet"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=1.0

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
offload=True
gen_tp=2

# ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
#     --working-dir "${WORKING_DIR}" \
#     -- 

python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=384 \
    data.val_batch_size=512 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0 \
    actor_rollout_ref.actor.use_kl_loss=True\
    actor_rollout_ref.actor.kl_loss_coef=0.0005 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=seq_reward \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.correct_sample_advantage_boost_value=0.1 \
    stop_properly_penalty.penalty_coef=0.1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=5 \
    trainer.save_freq=5 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto
