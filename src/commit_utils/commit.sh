#!/bin/bash
source src/commit_utils/set_env_vars.sh
source .env

models=(
    # "google/gemma-3-4b-pt"
    # "Qwen/Qwen3-4B-Base"
    "Qwen/Qwen3-1.7B-Base"
    # "HuggingFaceTB/SmolLM3-3B-Base"
)

evals=(
    "aime2025"
    # "arenahardwriting"
    # "bfcl"
    # "gpqamain"
    # "gsm8k"
    # "humaneval"
    # "healthbench"
)
export POST_TRAIN_BENCH_EXPERIMENT_NAME="_run_test"
for model in "${models[@]}"; do
    for eval in "${evals[@]}"; do
        echo ""
        echo $model on $eval
        if [ "${POST_TRAIN_BENCH_JOB_SCHEDULER}" = "htcondor_mpi-is" ]; then
            condor_submit_bid 100 -a "agent=codex" -a "agent_config=gpt-5.1-codex-max" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 50 -a "agent=codex" -a "agent_config=gpt-5.3-codex" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "agent=claude" -a "agent_config=claude-sonnet-4-5" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "agent=claude" -a "agent_config=claude-opus-4-5" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 50 -a "agent=claude" -a "agent_config=claude-opus-4-6" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 50 -a "agent=claude" -a "agent_config=claude-sonnet-4-5" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=1" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "agent=gemini" -a "agent_config=models/gemini-3-pro-preview" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "agent=gemini" -a "agent_config=models/gemini-3-flash-preview" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            sleep 10
        elif [ "${POST_TRAIN_BENCH_JOB_SCHEDULER}" = "htcondor" ]; then
            condor_submit_bid -a "agent=codex" -a "agent_config=gpt-5.1-codex-max" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid -a "agent=codex" -a "agent_config=gpt-5.3-codex" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid -a "agent=claude" -a "agent_config=claude-sonnet-4-5" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid -a "agent=claude" -a "agent_config=claude-opus-4-5" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid -a "agent=claude" -a "agent_config=claude-opus-4-6" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid -a "agent=claude" -a "agent_config=claude-sonnet-4-5" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=1" src/commit_utils/single_task.sub
            condor_submit_bid -a "agent=gemini" -a "agent_config=models/gemini-3-pro-preview" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid -a "agent=gemini" -a "agent_config=models/gemini-3-flash-preview" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            sleep 10
        elif [ "${POST_TRAIN_BENCH_JOB_SCHEDULER}" = "slurm" ]; then
            # sbatch \
            #     --export=ALL,EVAL="$eval",AGENT="codex",AGENT_CONFIG="gpt-5.1-codex-max",MODEL_TO_TRAIN="$model",NUM_HOURS="1",SKIP_GPU_CHECK="1" \
            #     src/commit_utils/single_task.sbatch
            sbatch \
                --export=ALL,EVAL="$eval",AGENT="claude",AGENT_CONFIG="claude-sonnet-4-5",MODEL_TO_TRAIN="$model",NUM_HOURS="1",SKIP_GPU_CHECK="1",CLAUDE_CREDENTIALS_DIR="$HOME/.claude" \
                src/commit_utils/single_task.sbatch
        else
            echo ERROR: job scheduler "${POST_TRAIN_BENCH_JOB_SCHEDULER}" is not supported.
        fi
    done
done
