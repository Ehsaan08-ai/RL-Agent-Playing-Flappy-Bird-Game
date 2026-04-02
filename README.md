# 🐦 RL Agent Playing Flappy Bird

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-0081A5?style=for-the-badge&logo=openai&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

**A Deep Q-Network (DQN) agent trained to play Flappy Bird using Reinforcement Learning.**

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Hyperparameters](#hyperparameters)
- [Installation](#installation)
- [Usage](#usage)
- [Training Details](#training-details)
- [Dependencies](#dependencies)

---

<a id="overview"></a>
## 🧠 Overview

This project implements a **Deep Q-Network (DQN)** reinforcement learning agent that learns to play the classic Flappy Bird game autonomously. The agent interacts with the [`flappy-bird-gymnasium`](https://github.com/markub3327/flappy-bird-gymnasium) environment and learns a control policy entirely from raw game state observations — no human demonstrations required.

Key features of this implementation:
- 🎮 **Custom Gymnasium environment** integration via `FlappyBird-v0`
- 🧠 **Double-network DQN** (Policy + Target Network) for stable training
- 🔁 **Experience Replay** with a fixed-size circular buffer for sample efficiency
- ⚡ **GPU/MPS/CPU** auto-detection for flexible hardware support
- 📈 **Epsilon-greedy exploration** with configurable decay
- 💾 **Best-model checkpointing** based on episode reward

---

<a id="how-it-works"></a>
## ⚙️ How It Works

The agent follows the standard **DQN algorithm**:

```
1. Observe the current game state (12 features from the environment)
2. Choose an action (flap or do nothing) using ε-greedy policy
3. Execute the action, receive reward, observe next state
4. Store transition (s, a, s', r, done) in Replay Memory
5. Sample a random mini-batch from memory
6. Compute target Q-values using the Target Network
7. Minimize MSE loss between predicted and target Q-values
8. Periodically sync the Target Network with the Policy Network
```

The **Policy Network** is updated every step, while the **Target Network** is synced every `network_sync_rate` steps — this separation stabilizes training by reducing oscillations.

---

<a id="project-structure"></a>
## 📁 Project Structure

```
RL-Agent-Playing-Flappy-Bird-Game/
│
├── agent.py                  # Core DQN Agent — training & inference loop
├── dqn.py                    # Neural network model definition (Policy/Target)
├── experience_replay.py      # Replay Memory buffer (FIFO deque)
├── game_flappy_bird.py       # Manual play script (keyboard-controlled)
├── parameters.yaml           # Hyperparameter configuration
│
└── runs/                     # Auto-generated training artifacts
    ├── flappybirdv0.pt       # Saved best model weights
    └── flappybirdv0_log      # Training log (best rewards per episode)
```

---

<a id="architecture"></a>
## 🏗️ Architecture

### DQN Model (`dqn.py`)

A lightweight fully-connected neural network:

```
Input Layer   → 12 nodes  (game state features)
Hidden Layer  → 256 nodes (ReLU activation)
Output Layer  → 2 nodes   (Q-values for: do nothing / flap)
```

### Agent (`agent.py`)

| Component | Details |
|---|---|
| **Loss Function** | Mean Squared Error (MSE) |
| **Optimizer** | Adam |
| **Policy** | ε-greedy (exploration → exploitation) |
| **Replay Buffer** | FIFO deque with max capacity |
| **Target Network Sync** | Every `network_sync_rate` steps |
| **Model Saving** | Saves best model whenever a new high reward is achieved |

### Experience Replay (`experience_replay.py`)

A fixed-capacity **double-ended queue (deque)** that stores transitions `(state, action, next_state, reward, terminated)`. Random mini-batches are sampled during optimization to break temporal correlations between consecutive transitions.

---

<a id="hyperparameters"></a>
## 🎛️ Hyperparameters

Configured via `parameters.yaml`:

| Parameter | Value | Description |
|---|---|---|
| `epsilon_init` | `1.0` | Initial exploration rate (100% random) |
| `epsilon_min` | `0.05` | Minimum exploration rate |
| `epsilon_decay` | `0.9995` | Multiplicative decay per episode |
| `alpha` (lr) | `0.001` | Adam optimizer learning rate |
| `gamma` | `0.99` | Discount factor for future rewards |
| `replay_memory_size` | `100,000` | Maximum transitions stored |
| `mini_batch_size` | `64` | Transitions sampled per optimization step |
| `network_sync_rate` | `100` | Steps between target network updates |
| `reward_threshold` | `1000` | Max reward per episode (early stop) |

---

<a id="installation"></a>
## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Ehsaan08-ai/RL-Agent-Playing-Flappy-Bird-Game.git
cd RL-Agent-Playing-Flappy-Bird-Game
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install torch gymnasium flappy-bird-gymnasium pygame pyyaml
```

---

<a id="usage"></a>
## 🚀 Usage

### 🎓 Train the Agent

```bash
python agent.py flappybirdv0 --train
```

Training will:
- Run indefinitely until manually stopped (`Ctrl+C`)
- Print episode number, total reward, and current epsilon each episode
- Save the best model to `runs/flappybirdv0.pt` whenever a new high reward is achieved
- Log best reward milestones to `runs/flappybirdv0_log`

### 🎮 Run the Trained Agent (Inference)

```bash
python agent.py flappybirdv0
```

This loads the saved model from `runs/flappybirdv0.pt` and runs one episode with the game rendered visually.

### 🕹️ Play Manually

```bash
python game_flappy_bird.py
```

Launches the game in human-playable mode. Press **Space** to flap, **Q** / close the window to quit.

---

<a id="training-details"></a>
## 📊 Training Details

- **State Space**: 12 continuous features (bird position, velocity, pipe distances, etc.) from the `FlappyBird-v0` environment with LIDAR disabled
- **Action Space**: 2 discrete actions — `0` (do nothing) and `1` (flap)
- **Reward**: Positive reward for surviving and passing pipes; negative reward (termination) on collision
- **Hardware**: Automatically uses **CUDA** (NVIDIA GPU) → **MPS** (Apple Silicon) → **CPU**

---

<a id="dependencies"></a>
## 📦 Dependencies

| Package | Purpose |
|---|---|
| `torch` | Neural network and tensor operations |
| `gymnasium` | RL environment interface |
| `flappy-bird-gymnasium` | Flappy Bird environment |
| `pygame` | Game rendering and keyboard input |
| `pyyaml` | Hyperparameter configuration loading |

---

<div align="center">

Made with ❤️ using PyTorch & Gymnasium

</div>
