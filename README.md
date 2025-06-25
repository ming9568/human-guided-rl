# Human-Guided Attention Masking in Reinforcement Learning

This project explores an interactive reinforcement learning (RL) setup where **humans guide the agent's attention** using natural language feedback to control feature masking. The agent is trained in the classic `CartPole-v1` environment using the PPO algorithm from `stable-baselines3`.

## Features

- **Baseline PPO** agent (full observation)
- **Static Mask PPO** agent (predefined input masking)
- **Feedback Mask PPO** agent (masked dynamically using human-guided instructions)
- **Evaluation plots**, training reward curves, and convergence tracking
- Supports **real-time masking via Gradio UI**
- Produces reproducible performance across multiple random seeds

## Code Overview

- `human_guided_attention.py`: Main training and evaluation script
- `MaskedObservationWrapper`: Applies element-wise mask to the observation vector
- `RewardLogger`: Custom callback to log episode rewards
- `feedback_schedule`: Simulated feedback mask updates at specific timesteps

## Training Setup

The model variants are trained on:

- **Environment**: `CartPole-v1`
- **Algorithm**: `PPO` (Proximal Policy Optimization)
- **Timesteps**: `10000` per agent
- **Seeds**: `42`, `100`, `123`

Performance and convergence are compared across:

- `Baseline PPO`
- `Static Mask PPO`
- `Feedback Mask PPO`

## Usage

### Install dependencies

```bash
pip install gymnasium stable-baselines3 matplotlib

