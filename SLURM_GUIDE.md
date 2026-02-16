# PostTrainBench Slurm 集群复现指南

## 一、前置条件

| 需求 | 说明 |
|------|------|
| **GPU** | NVIDIA H100 80GB（每个任务 1 张） |
| **内存** | 每个任务 128GB |
| **CPU** | 每个任务 16 核 |
| **磁盘** | 每个任务 ~400GB 临时空间（`/tmp` 或 `$TMPDIR`） |
| **时间** | 每个任务约 11-12 小时（10h 训练 + 评估） |
| **软件** | Apptainer/Singularity ≥ 1.1、fuse-overlayfs、Python 3 |
| **API Keys** | `ANTHROPIC_API_KEY`、`OPENAI_API_KEY`、`GEMINI_API_KEY`（按需） |
| **网络** | 计算节点需要能访问外网（agent 会调用 API、可能联网搜索） |

## 二、总体工作量

全量复现 = 4 模型 × 7 评估任务 × N 个 agent 配置。

原仓库跑了 8 种 agent 配置，即 4×7×8 = **224 个任务**。建议先挑一个子集试跑。

## 三、具体步骤

### Step 1: 构建容器

在有 root/fakeroot 权限的节点上（或登录节点）构建：

```bash
cd /path/to/PostTrainBench
bash containers/build_container.sh standard
bash containers/build_container.sh vllm_debug  # 评估阶段需要
```

生成 `containers/standard.sif` 和 `containers/vllm_debug.sif`。

如果登录节点没有构建权限，可以在本地构建后 `scp` 到集群共享存储。

### Step 2: 下载 HuggingFace 缓存

```bash
export HF_HOME="/path/to/shared/storage/huggingface"  # 共享存储路径
export POST_TRAIN_BENCH_CONTAINERS_DIR="containers"
export POST_TRAIN_BENCH_CONTAINER_NAME="standard"
bash containers/download_hf_cache/download_hf_cache.sh
```

这会下载所有需要的模型和数据集。放在共享文件系统上（如 Lustre/GPFS），所有节点可访问。

### Step 3: 创建 Slurm 批处理脚本

将 HTCondor 的 `single_task.sub` 改写为 Slurm sbatch 脚本。

创建 `src/commit_utils/single_task.sbatch`：

```bash
#!/bin/bash
#SBATCH --job-name=ptb_%j
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err
#SBATCH --partition=gpu           # 改为你集群的 GPU 分区名
#SBATCH --gres=gpu:1              # 1 块 GPU
#SBATCH --constraint=h100         # 如有 H100 约束标签（按集群配置修改）
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00           # 12 小时上限
#SBATCH --tmp=400G                # 如果集群支持本地临时空间申请

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
```

### Step 4: 改写提交脚本

在 `commit.sh` 的循环中增加 `slurm` 分支：

```bash
elif [ "${POST_TRAIN_BENCH_JOB_SCHEDULER}" = "slurm" ]; then
    sbatch \
        --export=ALL,EVAL="$eval",AGENT="codex",AGENT_CONFIG="gpt-5.1-codex-max",MODEL_TO_TRAIN="$model",NUM_HOURS="10" \
        src/commit_utils/single_task.sbatch

    sbatch \
        --export=ALL,EVAL="$eval",AGENT="claude",AGENT_CONFIG="claude-opus-4-6",MODEL_TO_TRAIN="$model",NUM_HOURS="10" \
        src/commit_utils/single_task.sbatch

    # ... 其他 agent 配置类似
    sleep 2
```

### Step 5: 设置环境变量

提交前 export 或写入脚本：

```bash
export POST_TRAIN_BENCH_JOB_SCHEDULER="slurm"
export POST_TRAIN_BENCH_RESULTS_DIR="/path/to/shared/results"
export POST_TRAIN_BENCH_CONTAINERS_DIR="/path/to/containers"
export POST_TRAIN_BENCH_CONTAINER_NAME="standard"
export HF_HOME="/path/to/shared/huggingface"
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export POST_TRAIN_BENCH_EXPERIMENT_NAME="_slurm_run1"
```

### Step 6: `run_task.sh` 注意事项

`run_task.sh` 本身基本不需要改动，但要注意：

| 问题 | 解决方案 |
|------|---------|
| **`/tmp` 空间** | Slurm 的 `$TMPDIR` 通常指向节点本地 SSD。在 sbatch 脚本中可设置 `export TMP_SUBDIR="$TMPDIR/posttrain_..."` 或确保 `/tmp` 有 400GB+ 空间 |
| **fuse-overlayfs** | 需要计算节点安装 `fuse-overlayfs` 和 `fusermount`，且用户有 FUSE 权限。如果不支持，可改为直接复制 HF cache（更慢但更兼容） |
| **Apptainer 版本** | 确认集群的 Apptainer ≥ 1.1，支持 `--nv`、`--writable-tmpfs`、`-c` 等参数 |
| **网络访问** | 计算节点需访问外网（API 调用）。部分集群默认禁止，需找管理员开通或用代理 |
| **`uuidgen`** | `run_task.sh` 用到了 `uuidgen`，确认节点有 `util-linux` 包 |

## 四、最小方案（不改仓库代码）

如果只想快速跑通，可以写一个独立的 wrapper 脚本：

```bash
#!/bin/bash
# submit_one.sh - 提交单个任务到 Slurm
EVAL=$1; AGENT=$2; AGENT_CONFIG=$3; MODEL=$4; HOURS=$5

sbatch <<EOF
#!/bin/bash
#SBATCH -J ptb_${EVAL}_${AGENT}
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o slurm_%j.out -e slurm_%j.err

export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."
export HF_HOME="/shared/huggingface"
export POST_TRAIN_BENCH_RESULTS_DIR="results"
export POST_TRAIN_BENCH_CONTAINERS_DIR="containers"
export POST_TRAIN_BENCH_CONTAINER_NAME="standard"
export POST_TRAIN_BENCH_JOB_SCHEDULER="slurm"
export POST_TRAIN_BENCH_EXPERIMENT_NAME="_slurm"

cd /path/to/PostTrainBench
bash src/run_task.sh "$EVAL" "$AGENT" "$MODEL" "\$SLURM_JOB_ID" "$HOURS" "$AGENT_CONFIG"
EOF
```

用法：

```bash
bash submit_one.sh gsm8k claude claude-opus-4-6 "Qwen/Qwen3-1.7B-Base" 10
```

## 五、常见坑和排查

1. **FUSE 权限** — 很多集群默认不允许用户态 FUSE。如果 `fuse-overlayfs` 报权限错误，联系管理员启用 `/dev/fuse`，或改为 `cp -r $HF_HOME $TMP_SUBDIR/hf_cache` 替代 overlay。
2. **Apptainer `--home` 冲突** — Slurm 可能设置了 `$HOME`，导致 Apptainer 的 `--home` 行为异常。可在 sbatch 中显式 `export HOME=/home/username`。
3. **GPU 可见性** — 确认 `apptainer exec --nv` 能看到 GPU：先跑 `apptainer exec --nv containers/standard.sif nvidia-smi`。
4. **磁盘空间不足** — 训练产生的 checkpoint + HF cache 副本会很大。确保 `/tmp` 或 `$TMPDIR` 有足够空间。
5. **API 网络超时** — 长时间任务中 API 可能间歇性超时，这是正常的，agent 内部有重试逻辑。

## 六、推荐的试跑顺序

1. **测试容器** — `apptainer exec --nv containers/standard.sif nvidia-smi`
2. **测试 baseline** — `bash src/baselines/run_baseline.sh gsm8k "Qwen/Qwen3-1.7B-Base"`（不需要 API key）
3. **小时间测 agent** — 用 `NUM_HOURS=1` 跑一个 `gsm8k + Qwen3-1.7B` 组合
4. **全量提交** — 确认上述都没问题后再批量提交
