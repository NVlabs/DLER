#!/bin/bash

#SBATCH --account ACCOUNT
#SBATCH --partition="YOUR PARTITION"
#SBATCH --time 04:00:00
#SBATCH --nodes 4
#SBATCH --gpus-per-node=8
#SBATCH --job-name YOUR JOB NAME
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --priority="TOP"

set -x

# load necessary modules

# replace these information with your own
verl_workdir=/path/to/verl
train_files=/path/to/gsm8k/train.parquet
val_files=/path/to/gsm8k/test.parquet
apptainer_image_path=/path/to/verl-ngc.sif
# replace these information with your own

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



# Paths
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TRAIN_FILE="./data/deepscaler/train.parquet"
TEST_FILE="./data/deepscaler/aime.parquet"

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





export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_DOCKER_CPU_WARNING=1

export WANDB_API_KEY="YOUR KEY"

export HF_HOME="YOUR ADDRESS"
export HF_TOKEN="YOUR KEY"

GPFS="/lustre/fsw/portfolios/nvr/users/shihyangl/reasoning_trace_distill/verl-main"
PROJECT="public_verl_1.5b_dler"
# EXPNAME="math-code-gym-stem-ifeval-$ALGO-1.5b-8k-8192-dapo-latest"
EXPNAME="TEST"
CKPT_DIR="./1.5b_public_verl/results/$PROJECT/$EXPNAME/ckpt"
RESULTS_DIR="./1.5b_public_verl/results/$PROJECT/$EXPNAME/$SLURM_JOB_ID"
mkdir -p $RESULTS_DIR
mkdir -p $CKPT_DIR


# TODO: add image name
container_name="YOUR CONTAINER NAME"

MOUNTS="--container-mounts=${GPFS}:${GPFS},/lustre:/lustre,${GPFS}:/verl"






# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)


port=6379
ip_head=$head_node_ip:$port
export ip_head

headnode_cmd="ray start --head --node-ip-address=$head_node_ip --port=$port "' --num-cpus=$(lscpu -p=CPU | grep -v "#" | wc -l) --block --disable-usage-stats --include-dashboard=true --dashboard-port=8265 --ray-client-server-port 6478 --redis-shard-ports 6580 --dashboard-grpc-port 6681 --node-manager-port 6782 --object-manager-port 6883 --runtime-env-agent-port 6984 --dashboard-agent-grpc-port 7085 --dashboard-agent-listen-port 7186 --metrics-export-port 7297'

worker_cmd="ray start --address $ip_head "' --num-cpus=$(lscpu -p=CPU | grep -v "#" | wc -l)'

srun --overlap --nodes=1 --ntasks=1 -w $head_node -o "$RESULTS_DIR/output-%j-head-node.out" -e "$RESULTS_DIR/output-%j-head-node.err" --no-container-mount-home --container-image="$container_name" $MOUNTS $EXPORTS bash -c "$headnode_cmd" &
sleep 30s

worker_num=$((SLURM_JOB_NUM_NODES))
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes_array[$i]}
    base_port=$((30000 + (i-1)*100))
    worker_min_port=$base_port
    worker_max_port=$((base_port + 4999))
    srun --overlap --nodes=1 --ntasks=1 -w $node_i -o "$RESULTS_DIR/output-%j-worker-node-$i.out" -e "$RESULTS_DIR/output-%j-worker-node-$i.err" --no-container-mount-home --container-image="$container_name" $MOUNTS $EXPORTS bash -c "$worker_cmd --block --disable-usage-stats --node-manager-port 6782 --object-manager-port 6883 --runtime-env-agent-port 6984 --dashboard-agent-grpc-port 7085 --dashboard-agent-listen-port 7186 --metrics-export-port 7297" &
done

check_cmd="while true; do
    num_nodes=\$(ray list nodes | grep node_id | wc -l)
    echo found \$num_nodes
    if [ \$num_nodes -eq $worker_num ]; then
        break
    fi
    echo sleeping
    sleep 3s
done"
srun --overlap --nodes=1 --ntasks=1 -w $head_node -o "$RESULTS_DIR/checker-%j-head-node.out" -e "$RESULTS_DIR/checker-%j-head-node.err" --no-container-mount-home --container-image="$container_name" $MOUNTS $EXPORTS bash -c "$check_cmd"

srun --overlap --nodes=1 --ntasks=1 -w $head_node -o "$RESULTS_DIR/command-%j-head-node.out" -e "$RESULTS_DIR/command-%j-head-node.err" --no-container-mount-home --container-image="$container_name" $MOUNTS $EXPORTS bash -c \
"ray status && ray job submit --address=http://localhost:8265 \
    --runtime-env-json='{\"working_dir\": \"/verl\"}' \
    -- python3 -m recipe.dapo.main_dapo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.train_batch_size=$train_prompt_bsz \
    data.gen_batch_size=384 \
    data.val_batch_size=512 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0 \
    actor_rollout_ref.actor.use_kl_loss=True\
    actor_rollout_ref.actor.kl_loss_coef=0.0005 \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=seq_reward \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$train_prompt_mini_bsz \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=$loss_agg_mode \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$gen_tp \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=$top_k \
    actor_rollout_ref.rollout.val_kwargs.temperature=$temperature \
    actor_rollout_ref.rollout.val_kwargs.top_p=$val_top_p \
    actor_rollout_ref.rollout.val_kwargs.top_k=$top_k \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.correct_sample_advantage_boost_value=0.1 \
    stop_properly_penalty.penalty_coef=0.1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$SLURM_JOB_NUM_NODES \
    trainer.val_before_train=False \
    trainer.test_freq=5 \
    trainer.save_freq=5 \
    trainer.total_epochs=100 \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.resume_mode=auto"
