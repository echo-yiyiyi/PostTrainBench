# PostTrainBench: Measuring AI Ability to Perform LLM Post-Training

We introduce PostTrainBench, a benchmark which measures the ability of AI agents to post-train pre-trained LLMs. In this evaluation scenario, the agent is tasked to improve the performance of a target LLM on some benchmark task. The agent gets access to an evaluation script of the benchmark and 10 hours on an H100 GPU. Performance is measured as the benchmark score of the resulting model. By measuring post-training ability, PostTrainBench measures the AI R&D capabilities of the agent.

**We are actively looking for collaborators to gather more tasks and agent scaffolds. Collaborators can become co-authors or our paper. More information below.**

## Roadmap

Goal: Mid / End of January: release v1.0 of the benchmark

Until then, we gather:
- more tasks
- more agent scaffolds / different agents
- more advanced data decontamination

## Contributing
Get in touch with us through a pull request, by opening an issue or by writing an email.

## Requirements
- `apptainer`

## Installation

```bash
git clone ...

```

## Usage
Export your api keys.
```bash
export OPENAI_API_KEY="Your API key"
export ANTHROPIC_API_KEY="Your API key"
export GEMINI_API_KEY="Your API key"
```

## Configuration
todo

Download the huggingface cache
todo
```bash
bash containers/download_hf_cache/download_hf_cache.sh
```

## Code structure
todo
`agents`: agents live here

`containers`: container definition, download of cache

`src`: main codebase

`src/commit_utils`: utilities to commit a job to the cluster.
E.g. you can run `bash src/commit_utils/commit.sh` for commiting one job

`src/baselines`: scripts to compute baseline scores inside the standard container

`src/eval`: the evaluations.
Each evaluation folder consists of:
- `benchmark.txt`: The official name of the benchmark (e.g. "American Invitational Mathematics Examination (AIME) 2024")
- `evaluate.py`: the evaluation script
- (Optional) `task_context` directory: Other files which the agent can use. This could e.g. include some starting code or papers which the agent can read.

`results`: Evaluation results automatically go here (baseline runs are prefixed with `baseline_`)

## Usage
todo
