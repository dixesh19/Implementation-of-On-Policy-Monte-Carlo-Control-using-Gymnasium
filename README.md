# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium

---

## Aim

To implement **On-Policy Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description

The **FrozenLake-v1** environment is a grid-based reinforcement learning environment provided by Gymnasium.

The default 4 × 4 environment can be represented as:

```text
S F F F
F H F H
F F F H
H F F G
```

Where:

* **S** – Starting state
* **F** – Frozen/safe tile
* **H** – Hole
* **G** – Goal

The agent has four possible actions:

| Action | Direction |
| ------ | --------- |
| 0      | Left      |
| 1      | Down      |
| 2      | Right     |
| 3      | Up        |

The objective is to move from the starting state to the goal without falling into a hole.

In this implementation, `is_slippery=False` is used, making the environment deterministic.

---

## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule used in this experiment is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha [G_t - Q(s,a)]
$$

Where:

| Symbol   | Meaning                   |
| -------- | ------------------------- |
| $s$      | Current state             |
| $a$      | Action taken in state $s$ |
| $G_t$    | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate     |
| $\alpha$ | Learning rate             |
| $\gamma$ | Discount factor           |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses an **epsilon-greedy policy** to balance exploration and exploitation.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

The value of epsilon is gradually decreased during training so that the agent explores more during the beginning and exploits the learned knowledge later.

The epsilon update is:

$$
\epsilon = \max(\epsilon_{min},\epsilon \times \epsilon_{decay})
$$

---

## Algorithm

1. Initialize the FrozenLake environment.
2. Initialize the Q-table with zeros.
3. Set the learning rate $\alpha$, discount factor $\gamma$, and epsilon parameters.
4. Generate a complete episode using the epsilon-greedy policy.
5. Store the state, action, and reward for every step.
6. Start from the last step of the episode.
7. Calculate the return $G$ backwards.
8. Update the corresponding Q-value using the Monte Carlo update rule.
9. Decrease epsilon gradually.
10. Repeat the process for the specified number of episodes.
11. Extract the greedy policy from the learned Q-table.
12. Calculate the state-value function.
13. Display the final Q-table, state-value function, learned policy, and average reward.
14. Plot the learning curve.

---

## Python Program

---

#### Monte Carlo Control

---

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create Environment
# -------------------------------------------------

env = gym.make("FrozenLake-v1", is_slippery=False)

num_states = env.observation_space.n
num_actions = env.action_space.n


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 20000
gamma = 0.99
alpha = 0.1

epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

max_steps_per_episode = 100


# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((num_states, num_actions))

episode_rewards = []

epsilon = epsilon_start


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):
    """
    Selects an action using epsilon-greedy policy.
    """

    if np.random.random() < epsilon:
        # Exploration
        return env.action_space.sample()

    else:
        # Exploitation
        return np.argmax(Q[state])


# -------------------------------------------------
# Generate One Complete Episode
# -------------------------------------------------

def generate_episode(epsilon):
    """
    Generates one episode using the current
    epsilon-greedy policy.

    Returns a list of:
    (state, action, reward)
    """

    episode = []

    state, info = env.reset()

    for _ in range(max_steps_per_episode):

        action = epsilon_greedy_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))

        state = next_state

        if terminated or truncated:
            break

    return episode


# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------

for episode_num in range(num_episodes):

    # Generate a complete episode
    episode = generate_episode(epsilon)

    # Calculate total reward
    total_reward = sum(step[2] for step in episode)

    episode_rewards.append(total_reward)

    # Initialize return
    G = 0

    # Process the episode backwards
    for state, action, reward in reversed(episode):

        # Calculate return
        G = reward + gamma * G

        # Incremental Monte Carlo update
        Q[state, action] = Q[state, action] + \
            alpha * (G - Q[state, action])

    # Reduce epsilon gradually
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


# -------------------------------------------------
# Extract Greedy Policy
# -------------------------------------------------

optimal_policy = np.argmax(Q, axis=1)

state_values = np.max(Q, axis=1)


# -------------------------------------------------
# Display Results
# -------------------------------------------------

def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("Name:          ")
    print("Register Number:      ")

    print("\nLearned Policy:")
    print(policy_grid)


def print_value_function(values):

    print("\nEstimated State-Value Function:")

    print(
        np.round(
            values.reshape(4, 4),
            3
        )
    )


print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)

print_policy(optimal_policy)


success_rate = np.mean(
    episode_rewards[-1000:]
)

print(
    "\nAverage reward over last 1000 episodes:",
    success_rate
)


# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))

plt.plot(moving_average)

plt.xlabel("Episode")

plt.ylabel("Average Reward")

plt.title(
    "Monte Carlo Control Learning Curve"
)

plt.grid(True)

plt.show()


env.close()
```

---


## Output
<img width="969" height="745" alt="image" src="https://github.com/user-attachments/assets/32aa3211-d137-4a08-a565-87a7d2baaac4" />
<img width="1071" height="585" alt="image" src="https://github.com/user-attachments/assets/3fc394e8-bcd8-4893-a0d2-fe0860dfcc8c" />


<img width="608" height="745" alt="image" src="https://github.com/user-attachments/assets/564da0a6-865e-4ae6-a98b-c51ae7643fff" />
<img width="938" height="568" alt="image" src="https://github.com/user-attachments/assets/b242377e-3f78-488c-8fdf-4a4618859932" />

## Result

The **On-Policy Monte Carlo Control algorithm** was successfully implemented using the Gymnasium `FrozenLake-v1` environment. The agent learned the action-value function $Q(s,a)$ from complete episodes and improved its policy using an epsilon-greedy strategy.


## Inference
The experiment demonstrates that **On-Policy Monte Carlo Control** can learn an effective policy through repeated interaction with the environment. Initially, the agent explores different actions using a high epsilon value. As training progresses, epsilon decreases and the agent increasingly exploits the learned Q-values.

The learned Q-table provides the estimated value of each action in every state, while the extracted greedy policy indicates the preferred action for reaching the goal. The learning curve shows the improvement in the agent's performance over successive training episodes.
