# 🐍 Deep Q-Learning Snake AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-333333?style=for-the-badge&logo=pygame&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> An Artificial Intelligence agent that learns to play the classic Snake game using **Deep Q-Networks (DQN)** and Reinforcement Learning.

---

## 📖 Overview

This project implements an Artificial Intelligence (AI) agent that learns to play the classic Snake game using the **Deep Q-Network (DQN)** algorithm, a powerful technique in Reinforcement Learning (RL). The agent trains itself by maximizing rewards (eating apples) and minimizing penalties (crashing into walls or its own tail).

The AI uses a **Linear QNet** with **Experience Replay**, optimized to solve the "spinning loop" problem common in snake AI agents.

## ✨ Key Features

* **🧠 Deep Q-Network (DQN):** Uses a Feed Forward Neural Network to predict the best action based on the game state.
* **💾 Experience Replay:** Stores past moves in memory to train on random batches, breaking correlation between consecutive steps.
* **🎯 Optimized Reward Shaping:**
    * Positive reward for moving towards food (0.1).
    * Heavy penalty for moving away from food (-0.5) or wasting time.
    * Critical penalty for death (-50).
* **🤖 Dual Modes:** Includes both a training mode (fast-paced learning) and a play mode (human-watchable speed).
* **📊 State Persistence:** Automatically saves the model (`model.pth`) when a new high score is reached.

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [[https://github.com/yourusername/snake-ai-pytorch.git](https://github.com/yourusername/snake-ai-pytorch.git)](https://github.com/AhmetBeratKocyigit/SnakeAI-DeepLearning)
    cd SnakeAI-DeepLearning
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *(Dependencies: `pygame`, `torch`, `numpy`)*

---

## 🚀 How to Run

### 1. Training Mode (`main.py`)
Train the AI from scratch. You will see the agent improve over time.

```bash
python main.py
```

# 🐍 Snake AI – Deep Q-Learning

This project uses Deep Q-Learning with PyTorch to train an AI that plays Snake.  
The training runs at high speed and the model is saved to:

```
./model/model.pth
```

---

## 🚀 1. Training Mode (`main.py`)

The game runs extremely fast for efficient training.  
The model is automatically saved during training.

---

## 🎮 2. Demonstration Mode (`play_mode.py`)

Watch the trained model play Snake at normal speed:

```bash
python play_mode.py
```

**Controls**
- **R** → Start / Restart  
- **ESC** → Quit  

Exploration is disabled in this mode → `epsilon = 0`.

---

## 📂 Project Structure

```plaintext
├── agent.py           # The Brain: Agent class, Q-Network, and training logic
├── snake_game_ai.py   # The Body: Pygame environment, collision detection, UI
├── main.py            # The Trainer: Main training loop
├── play_mode.py       # The Player: Loads and runs the trained model
├── model/             # Contains saved models (model.pth)
└── README.md          # Documentation
```

---

## ⚙️ Hyperparameters & Configuration

Located in `agent.py`:

| Parameter      | Value     | Description |
|----------------|-----------|-------------|
| Batch Size     | 1000      | Number of samples drawn from memory per update |
| Learning Rate  | 0.001     | Rate at which network weights are updated |
| Gamma          | 0.95      | Discount factor for future rewards |
| Memory Size    | 100,000   | Maximum replay memory size |
| Epsilon Floor  | 10⁻²⁰     | Minimum randomness to avoid agent loops |

---

## 🤝 Contributing

Contributions are welcome!  
Ideas such as improved reward functions or a CNN-based full-screen agent are appreciated.

1. Fork the project  
2. Create your feature branch → `git checkout -b feature/AmazingFeature`  
3. Commit your changes → `git commit -m "Add AmazingFeature"`  
4. Push to your branch → `git push origin feature/AmazingFeature`  
5. Open a Pull Request  

---

## 📝 License

Distributed under the **MIT License**.  
See the LICENSE file for more information.

---

**Developed with ❤️ using PyTorch and Pygame.**



