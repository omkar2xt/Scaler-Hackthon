---
title: MisinfoGuard Env
emoji: "🛡️"
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# MisinfoGuard-Env

MisinfoGuard-Env is a production-style OpenEnv reinforcement learning environment for misinformation defense in social networks. It simulates adversarial spread and defender interventions over a small-world graph.

## Features

- Multi-agent misinformation simulation with scripted attacker bots
- OpenEnv-compatible environment API: reset(), step(), state()
- SIR-style post propagation on a NetworkX Watts-Strogatz graph
- Fully verifiable reward function with component-level grader output
- Stable-Baselines3 PPO training pipeline
- Hugging Face-compatible inference entrypoint

## Project Structure

- environment.py: Core MisinfoGuardEnv class
- network.py: Graph generation and SIR spread dynamics
- agents/defender.py: Defender policy interfaces and heuristic baseline
- agents/spreader.py: FloodBot, TargetBot, BurstBot scripted adversaries
- rewards.py: Programmatically verifiable reward computation
- grader.py: 10-episode scoring logic and GraderResult schema
- inference.py: Deployment entrypoint for one-episode evaluation
- train.py: PPO training and checkpointing
- demo.py: CLI walkthrough for policy behavior
- tests/test_env.py: Unit tests for reset/step/state APIs

## Setup

1. Create and activate a Python 3.11+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Demo

```bash
python demo.py --episodes 3 --policy heuristic
```

Optional policies:

- random
- ppo (loads checkpoints/best_model.zip if available)

## Train PPO

```bash
python train.py
```

Short sanity run (5k timesteps):

```bash
python train.py --sanity
```

Training outputs:

- Best model: checkpoints/best_model.zip
- Sanity model: checkpoints/sanity_model.zip
- TensorBoard logs: logs/

## Run Grader

```bash
python -c "from grader import grade_policy; print(grade_policy().to_json())"
```

## Run Inference Entrypoint

```bash
python inference.py
```

Environment variables supported by inference.py:

- API_BASE_URL
- MODEL_NAME
- HF_TOKEN

## Run Tests

```bash
pytest tests/test_env.py
```

## Deploy to Hugging Face Spaces

1. Build image:

```bash
docker build -t misinfoguard-env .
```

2. Run locally (Spaces-compatible port):

```bash
docker run --rm -p 7860:7860 misinfoguard-env
```

The container starts inference.py by default and prints grader JSON.

## CI

GitHub Actions workflow is included at .github/workflows/ci.yml.

It runs:

- ruff check .
- pytest tests/test_env.py -q
