# PostTrainBench: Measuring AI Ability to Perform LLM Post-Training
http://posttrainbench.com/

We introduce PostTrainBench, a benchmark which measures the ability of AI agents to post-train pre-trained large language models (LLMs). In PostTrainBench the agent is tasked to improve the performance of a target LLM on some benchmark. The agent gets access to an evaluation script and 10 hours on an H100 GPU. Performance is measured as the benchmark score of the post-trained LLM. This setup naturally measures the ability of an AI agent to perform AI R&D.

**We are actively looking for collaborators to gather more tasks and agent scaffolds. Collaborators can become co-authors or our paper. More information below.**

## Roadmap
Goal: Mid / End of January: release v1.0 of the benchmark

Our goal with v1.0 is to have a simple, yet effective way to measure the performance of AI agents on performing AI R&D.
For this we want to add:
- more tasks
- more agent scaffolds and different agents
- more advanced data decontamination
- ablation studies, e.g. using more or less compute for training

## Contributing
If you want to contribute to this vision, get in touch with us through a pull request, by opening an issue or by writing an email.
We are especially interested in people who can contribute more tasks and agents.

People with substantial contributions can become co-authors on our paper.
### Adding Tasks
If you want to add a task, make sure the following conditions hold:
- The task is not too difficult for the human post-trained versions of the four models we test on (TODO). It should achieve significantly above random chance or simple baselines.
- Make sure that the default parameters allow the agent to run the evaluation on the H100 rather fast. 15 minutes is a good guideline. For minimal evaluation time, it is advisable to use vllm for inference. Additionally, you can subsample the benchmark.

### Adding Agents
When implementing agents, your code should go into a directory `agents/agent_name/`, where `agent_name` is your new agent. You then need to implement a script `agents/agent_name/solve.sh`, which calls the agent to solve the task.
See `agents/codex/` and `agents/claude/` for examples.

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
