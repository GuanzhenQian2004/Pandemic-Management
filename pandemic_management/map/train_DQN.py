import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F

from .pandemic_sim import TravelEnv

# -----------------------------------------
# 1. Configuration / Hyperparameters
# -----------------------------------------
CONFIG = {
    # Training
    "num_episodes": 100,        # For demonstration, fewer episodes
    "max_steps_per_episode": 500,

    # Replay Buffer
    "replay_memory_size": 20000,
    "min_replay_size": 2000,

    # Neural Network
    "batch_size": 64,
    "gamma": 0.99,
    "lr": 1e-10,

    # Exploration
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_episodes": 0.6,  # fraction of total episodes to decay over

    # Target Network
    "target_soft_update_tau": 0.01,  # how much to blend new weights each step
    "update_frequency": 1,          # how often to do a learning step
    "warmup_episodes": 1,           # episodes before we start learning

    # Logging
    "print_freq": 10,
}


# -----------------------------------------
# 2. Dueling Q-Network (Double DQN-friendly)
# -----------------------------------------
class DuelingQNetwork(nn.Module):
    """
    Dueling DQN:
    Splits the network into:
      - A shared 'feature' backbone,
      - A 'value' stream (outputs a single value),
      - An 'advantage' stream (outputs advantage for each action).
    Final Q: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    """
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.feature_layer = nn.Sequential(
        nn.Linear(obs_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU()
    )
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, act_dim)

    def forward(self, x):
        feats = self.feature_layer(x)
        value = self.value_stream(feats)           # shape: [batch_size, 1]
        advantage = self.advantage_stream(feats)   # shape: [batch_size, act_dim]
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        # Q = V + (A - mean(A))
        q_values = value + (advantage - advantage_mean)
        return q_values

def log_cosh_loss(pred, target):
        """
        pred, target: Tensors of the same shape
        Returns the average log-cosh loss over the batch
        """
        # log(cosh(x)) = log((e^x + e^-x)/2)
        # We'll compute x = pred - target, then log(cosh(x)).
        x = pred - target
        return torch.mean(torch.log(torch.cosh(x)))
# -----------------------------------------
# 3. Replay Buffer
# -----------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Debug: store transition
        # You might comment out this debug if it floods the console
        # print(f"[DEBUG] Pushing transition to replay buffer: A={action}, R={reward:.2f}, Done={done}")
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        # Convert to Torch
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device) / 1000.0 
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# -----------------------------------------
# 4. Training Loop
# -----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def soft_update(target_net, source_net, tau):
    """
    Soft-update target network parameters:
      theta_target = tau * theta_source + (1 - tau) * theta_target
    """
    for target_param, src_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(
            tau * src_param.data + (1 - tau) * target_param.data
        )


def main():
    cfg = CONFIG

    # 4.1. Create environment
    print("[DEBUG] Initializing the TravelEnv...")
    env = TravelEnv()
    print("[DEBUG] Environment created. Resetting environment to get initial observation.")
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    obs_dim = env.observation_space.shape[0]
    print(f"Observation Dimension (obs_dim): {obs_dim}")


    act_dim = env.action_space.n

    print(f"[DEBUG] Initial obs dimension = {obs_dim}, Action dimension = {act_dim}")

    # 4.2. Create main and target networks
    print("[DEBUG] Creating DuelingQNetwork for online_net and target_net...")
    online_net = DuelingQNetwork(obs_dim, act_dim).to(device)
    target_net = DuelingQNetwork(obs_dim, act_dim).to(device)
    target_net.load_state_dict(online_net.state_dict())  # start same
    optimizer = optim.Adam(online_net.parameters(), lr=cfg["lr"])
    print("[DEBUG] Networks created and optimizer initialized.")

    # 4.3. Replay Buffer & Epsilon
    replay_buffer = ReplayBuffer(cfg["replay_memory_size"])
    epsilon = cfg["epsilon_start"]
    print(f"[DEBUG] Epsilon starting at {epsilon:.2f}")

    # 4.4. Pre-fill Replay Buffer
    print(f"[DEBUG] Pre-filling replay buffer with {cfg['min_replay_size']} random transitions...")
    while len(replay_buffer) < cfg["min_replay_size"]:
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.push(obs, action, reward, next_obs, done)

        obs = next_obs
        if done:
            obs = env.reset()

        if len(replay_buffer) % 500 == 0:
            print(f"[DEBUG] Replay buffer size: {len(replay_buffer)} / {cfg['min_replay_size']}")

    print("[DEBUG] Buffer pre-fill complete.")

    # 4.5. Training
    episode_rewards = []
    total_episodes = cfg["num_episodes"]
    steps_done = 0

    print(f"[DEBUG] Beginning training for {total_episodes} episodes.")

    for episode in range(total_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_losses = []  # Initialize loss tracking for each episode
        print(f"\033[1;31m\n[DEBUG] ---- EPISODE {episode + 1}/{total_episodes} START ----\033[0m")
        print(f"[DEBUG] Environment reset. Epsilon={epsilon:.3f}, BufferSize={len(replay_buffer)}")

        # Epsilon decay calculation
        decay_episodes = int(cfg["num_episodes"] * cfg["epsilon_decay_episodes"])
        epsilon = np.interp(
            episode, 
            [0, decay_episodes], 
            [cfg["epsilon_start"], cfg["epsilon_end"]]
        ) if episode < decay_episodes else cfg["epsilon_end"]

        for t in range(cfg["max_steps_per_episode"]):
            steps_done += 1
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = online_net(torch.FloatTensor(obs).unsqueeze(0).to(device))
                action = q_values.argmax().item()

            # Environment step
            next_obs, reward, done, _ = env.step(action)
            episode_reward += reward
            replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs

            # Learning phase
            if (episode >= cfg["warmup_episodes"] and 
                steps_done % cfg["update_frequency"] == 0 and 
                len(replay_buffer) >= cfg["batch_size"]):
                
                # Sample and process batch
                states, actions, rewards, next_states, dones = replay_buffer.sample(cfg["batch_size"])
                
                # Calculate target Q-values
                with torch.no_grad():
                    next_actions = online_net(next_states).argmax(1, keepdim=True)
                    target_q = rewards + cfg["gamma"] * target_net(next_states).gather(1, next_actions).squeeze() * (1 - dones)
                
                # Calculate current Q-values and loss
                current_q = online_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                criterion = nn.MSELoss()
                loss = criterion(current_q, target_q)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online_net.parameters(), 1.0)
                optimizer.step()
                
                # Track loss and update target network
                episode_losses.append(loss.item())
                soft_update(target_net, online_net, cfg["target_soft_update_tau"])

            if done:
                break

        # Episode conclusion
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        episode_rewards.append(episode_reward)
        
        print(f"[DEBUG] ---- EPISODE {episode + 1} END ----")
        print(f"Reward: {episode_reward:.2f} | Avg Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f}")
        
        # Logging
        with open("episode_rewards.log", "a") as f:
            f.write(f"Episode {episode+1}: Reward={episode_reward:.2f}, AvgLoss={avg_loss:.4f}\n")
        
        # Save model after each episode
        model_filename = f"models/dueling_dqn_episode_{episode+1}.pth"
        torch.save(online_net.state_dict(), model_filename)
        print(f"[DEBUG] Model saved as {model_filename}")
        
        # Periodic summary
        if (episode + 1) % cfg["print_freq"] == 0:
            avg_reward = np.mean(episode_rewards[-cfg["print_freq"]:])
            print(f"[SUMMARY] Last {cfg['print_freq']} episodes: Avg Reward {avg_reward:.2f}")

    # 4.6. Post-training
    print("\n[DEBUG] Training complete!")
    if len(episode_rewards) >= 50:
        print("[DEBUG] Final average reward (last 50 episodes):", np.mean(episode_rewards[-50:]))
    else:
        print("[DEBUG] Final average reward (all episodes):", np.mean(episode_rewards))
    print("[DEBUG] Saving model state_dict to 'dueling_dqn.pth'...")
    torch.save(online_net.state_dict(), "dueling_dqn.pth")
    print("[DEBUG] Done.")

    



if __name__ == "__main__":
    main()
