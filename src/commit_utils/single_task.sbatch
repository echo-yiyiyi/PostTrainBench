#!/bin/bash
#SBATCH --job-name=ptb_%j
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1:00:00

# 从环境变量读取参数（由提交脚本传入）
EVAL="${EVAL}"
AGENT="${AGENT}"
MODEL_TO_TRAIN="${MODEL_TO_TRAIN}"
NUM_HOURS="${NUM_HOURS}"
AGENT_CONFIG="${AGENT_CONFIG}"

# Slurm 的 JOB ID 对应 HTCondor 的 Cluster ID
CLUSTER_ID="${SLURM_JOB_ID}"

# 执行主流程
bash src/run_task.sh "$EVAL" "$AGENT" "$MODEL_TO_TRAIN" "$CLUSTER_ID" "$NUM_HOURS" "$AGENT_CONFIG"