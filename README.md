# Deep Convolutional Q-Learning for Ms. Pac-Man

This repository contains an implementation of a Deep Q-Network (DQN) agent trained to play
Ms. Pac-Man directly from raw pixel inputs using the Gymnasium Arcade Learning Environment (ALE).
The project was developed for the Foundations of Artificial Intelligence (CS5100) course at
Northeastern University.

---

## Project Overview

This project demonstrates a complete Deep Reinforcement Learning pipeline:
- Capture raw RGB frames from the Ms. Pac-Man ALE environment
- Preprocess frames into 128×128 tensors
- Extract spatial features using a convolutional neural network (CNN)
- Train a Deep Q-Network using:
  - Experience Replay
  - Target Network
  - One-step Temporal Difference Learning
  - Epsilon-greedy exploration
- Evaluate the trained agent through rendered gameplay episodes

The agent learns navigation, pellet collection, and basic ghost avoidance without any handcrafted features.

---

## Training Results

Training was configured to stop automatically when the 100-episode running average exceeded 500.

### Key Milestones

| Episode | Average Score |
|---------|----------------|
| 100     | 293.9          |
| 200     | 374.0          |
| 300     | 378.1          |
| 400     | 400.8          |
| 500     | 475.8          |
| 533     | 500.7          |

The environment was solved at **Episode 433**, defined as achieving a 100-episode running average of at least 500. This is a strong performance for a baseline DQN without additional algorithmic enhancements.

The repository includes the training curve images and a video of the AI agent while training.

---

## Model Architecture

The DQN uses a convolutional neural network with the following structure:
Conv1: 32 filters, 8×8 kernel, stride 4, BatchNorm, ReLU
Conv2: 64 filters, 4×4 kernel, stride 2, BatchNorm, ReLU
Conv3: 64 filters, 3×3 kernel, stride 1, BatchNorm, ReLU
Conv4: 128 filters, 3×3 kernel, stride 1, BatchNorm, ReLU
FC1: 512 units, ReLU
FC2: 256 units, ReLU
Output layer: |A| Q-values (size = action space)

Hyperparameters:
- Optimizer: Adam (learning rate = 5e-4)  
- Replay buffer size: 10,000  
- Batch size: 64  
- Discount factor: γ = 0.99  
- Epsilon decay: 0.995, minimum epsilon = 0.01  




