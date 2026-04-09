---
title: MisinfoGuard-Env
emoji: 🛡
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Scaler-Hackthon: MisinfoGuard-Env

An OpenEnv reinforcement learning environment for misinformation defense in social networks.

## Live Deployment

- **Hugging Face Space**: https://huggingface.co/spaces/omkar2xt/misinfoguard-env
- **Space Runtime URL**: https://omkar2xt-misinfoguard-env.hf.space
- **Health Endpoint**: `GET /health`
- **OpenEnv Endpoints**: `POST /reset`, `POST /step`, `GET /state`

## Project Overview

**MisinfoGuard-Env** simulates adversarial misinformation spreading and defender interventions on a small-world social network graph. This environment is designed for:

- RL researchers building moderation policies
- Safety teams evaluating intervention strategies  
- Platform engineers benchmarking LLM or heuristic moderation agents

## Quick Links

- **Main Project**: See [misinfoguard_env/README.md](misinfoguard_env/README.md) for full documentation
- **Docker**: [Dockerfile](misinfoguard_env/Dockerfile) for deployment
- **CI/CD**: [GitHub Actions](misinfoguard_env/.github/workflows/ci.yml)
- **OpenEnv Spec**: [openenv.yaml](openenv.yaml)

## Key Features

- Multi-agent misinformation simulation with scripted attacker bots
- OpenEnv-compatible environment API
- Fully verifiable reward function with component-level grader output
- Stable-Baselines3 PPO training pipeline
- Hugging Face-compatible inference entrypoint
- Docker containerization for deployment

## Project Structure

```
Scaler-Hackthon/
├── misinfoguard_env/           # Main RL environment package
│   ├── environment.py          # Core environment class
│   ├── network.py              # Graph and SIR dynamics
│   ├── agents/                 # Defender and spreader policies
│   ├── graders/                # Task-specific scoring
│   ├── inference.py            # Deployment entrypoint
│   ├── train.py                # PPO training pipeline
│   ├── requirements.txt         # Python dependencies
│   ├── Dockerfile              # Container image
│   └── README.md               # Full documentation
├── .gitignore
└── README.md (this file)
```

## Getting Started

### Setup

1. **Install dependencies:**
```bash
cd misinfoguard_env
pip install -r requirements.txt
```

2. **Run a demo:**
```bash
python demo.py --episodes 3 --policy heuristic
```

3. **Train a model:**
```bash
python train.py --sanity  # Quick test run
python train.py           # Full training
```

4. **Run tests:**
```bash
pytest tests/
```

## Environment Details

See [misinfoguard_env/README.md](misinfoguard_env/README.md) for:
- Complete API documentation
- Task descriptions and baselines
- Training and inference guides
- Deployment instructions

## Submission Status

- Space is configured with Docker SDK and runs on port `7860`.
- OpenEnv API routes are live and responding from the deployed container.
- Submission contract files are included: `openenv.yaml`, `models.py`, and `graders/*`.

## License

This project is open source. See `LICENSE` for details.

## Contributors

- Omkar Raju Gurav 
