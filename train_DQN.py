import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from pandemic_sim import TravelEnv

# ---------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------
NUM_EPISODES = 100
MAX_STEPS_PER_EPISODE = 500

BATCH_SIZE = 64
GAMMA = 0.99           # discount factor
LR = 1e-3              # learning rate for the optimizer
EPS_START = 1.0        # starting value of epsilon
EPS_END = 0.01         # final value of epsilon
EPS_DECAY = 0.995      # decay rate of epsilon
TARGET_UPDATE_FREQ = 10  # how often to update the target network
REPLAY_MEMORY_SIZE = 10_000
MIN_REPLAY_SIZE = 1_000  # minimum replay size before training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# Q-Network Definition
# ---------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(QNetwork, self).__init__()
        # Simple 2-layer MLP
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=REPLAY_MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        sample_batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*sample_batch))
        
        # Convert to torch tensors
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
def main():
    # -----------------------------------------------------
    # 1. Initialize environment
    # -----------------------------------------------------
    env = TravelEnv()
    obs = env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    print("[DEBUG] Environment initialized.")
    print(f"[DEBUG] Observation dimension: {obs_dim}")
    print(f"[DEBUG] Action dimension (number of discrete actions): {act_dim}")

    # -----------------------------------------------------
    # 2. Create main DQN and target DQN
    # -----------------------------------------------------
    q_net = QNetwork(obs_dim, act_dim).to(DEVICE)
    target_q_net = QNetwork(obs_dim, act_dim).to(DEVICE)
    target_q_net.load_state_dict(q_net.state_dict())  # ensure they start the same

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    
    print("[DEBUG] Networks and optimizer initialized.")

    # -----------------------------------------------------
    # 3. Initialize replay buffer
    # -----------------------------------------------------
    replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)

    # -----------------------------------------------------
    # 4. Epsilon-greedy parameters
    # -----------------------------------------------------
    epsilon = EPS_START
    print(f"[DEBUG] Initial epsilon: {epsilon}")

    # -----------------------------------------------------
    # 5. Pre-fill Replay Buffer with random actions
    # -----------------------------------------------------
    print(f"[DEBUG] Pre-filling replay buffer with {MIN_REPLAY_SIZE} transitions...")
    obs = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        replay_buffer.push(obs, action, reward, next_obs, done)
        
        if done:
            obs = env.reset()
        else:
            obs = next_obs
    print(f"[DEBUG] Replay buffer pre-filled. Current size: {len(replay_buffer)}")

    # -----------------------------------------------------
    # 6. Training Loop
    # -----------------------------------------------------
    episode_rewards = []
    
    for episode in range(NUM_EPISODES):
        obs = env.reset()
        episode_reward = 0
        print(f"\n[DEBUG] Starting episode {episode+1}/{NUM_EPISODES} with epsilon={epsilon:.3f}")
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # -------------------------------------------------
            # 6.1. Epsilon-greedy action selection
            # -------------------------------------------------
            if step % 100 == 0:
                target_q_net.load_state_dict(q_net.state_dict())

            if random.random() < epsilon:
                action = env.action_space.sample()
                print(f"[DEBUG] Step={step} | Chose RANDOM action={action} (epsilon={epsilon:.3f})")
            else:
                # Convert obs to torch and get Q values
                state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                q_values = q_net(state_tensor)
                action = q_values.argmax(dim=1).item()
                print(f"[DEBUG] Step={step} | Chose GREEDY action={action} (epsilon={epsilon:.3f})")
            
            # -------------------------------------------------
            # 6.2. Interact with environment
            # -------------------------------------------------
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            print(f"[DEBUG] Step={step} | Reward={reward:.2f}, Done={done}, Info={info}")
            # Uncomment below to see the environment's debug print for each step
            # env.render()
            
            # -------------------------------------------------
            # 6.3. Store transition in replay buffer
            # -------------------------------------------------
            replay_buffer.push(obs, action, reward, next_obs, done)
            # Update current obs
            obs = next_obs

            # -------------------------------------------------
            # 6.4. Sample from replay & update Q-network
            # -------------------------------------------------
            if len(replay_buffer) > BATCH_SIZE:
                states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(BATCH_SIZE)

                # Q(s,a)
                q_values_b = q_net(states_b)
                # gather relevant Q-values for the chosen actions
                q_values_b = q_values_b.gather(1, actions_b.unsqueeze(1)).squeeze(1)

                # Q_target(s', a') for the next states using target network
                with torch.no_grad():
                    max_next_q_values = target_q_net(next_states_b).max(dim=1)[0]

                # Bellman backup: y = r + gamma * max Q(s',a')
                target_q_values = rewards_b + GAMMA * max_next_q_values * (1 - dones_b)

                # -------------------------------------------------
                # 6.5. Compute loss (MSE)
                # -------------------------------------------------
                loss = nn.SmoothL1Loss()(q_values_b, target_q_values)
                
                # -------------------------------------------------
                # 6.6. Gradient descent
                # -------------------------------------------------
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
                optimizer.step()
                
                print(f"[DEBUG] Step={step} | Loss={loss.item():.4f}")

            # -------------------------------------------------
            # 6.7. Check if episode is done
            # -------------------------------------------------
            if done:
                print(f"[DEBUG] Episode ended by done condition at step={step}.")
                break

        # -------------------------------------------------
        # 6.8. Update target network occasionally
        # -------------------------------------------------
        if episode % TARGET_UPDATE_FREQ == 0:
            target_q_net.load_state_dict(q_net.state_dict())
            print("[DEBUG] Target network updated.")

        # -------------------------------------------------
        # 6.9. Decay epsilon
        # -------------------------------------------------
        if epsilon > EPS_END:
            epsilon = max(EPS_END, EPS_START - (episode / (NUM_EPISODES * 0.8)) * (EPS_START - EPS_END))


        # -------------------------------------------------
        # 6.10. Log and wrap-up
        # -------------------------------------------------
        episode_rewards.append(episode_reward)
        print(f"[DEBUG] Episode {episode+1} finished. Total episode reward={episode_reward:.2f}")

    # -----------------------------------------------------
    # 7. Post-training output
    # -----------------------------------------------------
    print("\n[DEBUG] Training complete.")
    print("[DEBUG] Episode rewards:", episode_rewards)
    print("[DEBUG] Final epsilon:", epsilon)
    
    # Optionally save your Q-network
    torch.save(q_net.state_dict(), "q_network.pth")
    print("[DEBUG] Q-network saved to q_network.pth")

if __name__ == "__main__":
    main()
