# Multi-Agent Pursuit–Evasion 

This repository implements a **Multi-Agent Reinforcement Learning (MARL)** framework for a pursuit–evasion problem with **obstacle avoidance**, all contained in a single script: `train.py`.

---

## 🚀 Overview

We study a **two-agent zero-sum game**:

* **Attacker (A):** tries to reach a target boundary
* **Defender (D):** tries to capture the attacker
* **Environment:** continuous 2D space with a circular obstacle

Agents learn **continuous vector actions**:

[
u \in \mathbb{R}^2, \quad |u| \le 1
]

The implementation is inspired by:

* Differential games (HJI formulation)
* Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

---

## 📂 Project Structure

```
.
├── train.py        # Full implementation (env + model + training + visualization)
└── README.md
```

Everything (environment, neural networks, training loop, visualization) is implemented inside **one file** for simplicity and reproducibility.

---

## ⚙️ Features

* Continuous state and action space
* Centralized critic (multi-agent setup)
* Obstacle-aware dynamics (projection-based avoidance)
* Reward shaping for improved learning
* Trajectory visualization
* Minimal dependencies

---

## 📦 Installation

```bash
git clone https://github.com/irfan995/toy_peg.git
cd toy_peg
pip install numpy matplotlib torch
```

---

## ▶️ How to Run

```bash
python train.py
```

This will:

1. Train both agents for multiple episodes
2. Plot training rewards
3. Visualize one trajectory

---

## 🧠 Environment Details

| Component | Description                    |
| --------- | ------------------------------ |
| State     | $(x_A, y_A, x_D, y_D)$         |
| Action    | Continuous velocity vector     |
| Dynamics  | $x_{t+1} = x_t + v u \Delta t$ |
| Obstacle  | Circular (non-traversable)     |
| Capture   | Distance-based                 |
| Goal      | Reach bottom boundary          |

---

## 🎯 Reward Structure

* **+1** → attacker reaches goal
* **-1** → attacker is captured
* **0** → timeout

Additional shaping:

* Encourages moving toward goal
* Encourages staying away from defender

---

## 📊 Outputs

### Training Curve

Displays reward progression over episodes.

### Trajectory Plot

* 🔵 Attacker
* 🔴 Defender
* ⚫ Obstacle

---

## 📜 License

MIT License

---

