import torch
import numpy as np
import matplotlib.pyplot as plt
from pandemic_sim import TravelEnv
from collections import deque
import random

class QNetwork(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, act_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class RealtimeLearner:
    def __init__(self, env, device, model_path=None):
        self.env = env
        self.device = device
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n
        
        # Initialize networks
        self.q_net = QNetwork(self.obs_dim, self.act_dim).to(device)
        self.target_net = QNetwork(self.obs_dim, self.act_dim).to(device)
        
        if model_path:
            self.q_net.load_state_dict(torch.load(model_path, map_location=device))
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 0.1  # Fixed exploration rate during deployment
        
        # Training statistics
        self.loss_history = []
        self.update_counter = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def update_network(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Calculate target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Calculate current Q-values
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Compute loss
        loss = torch.nn.MSELoss()(current_q, target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.loss_history.append(loss.item())

    def run(self, num_episodes=5, max_steps=1000):
        plt.figure(figsize=(15, 10))
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            step = 0
            
            # Tracking variables
            metrics = {
                'infections': {c: [] for c in range(self.env.n_cities)},
                'budgets': [],
                'actions': [],
                'rewards': []
            }

            print(f"\n=== Episode {episode+1} ===")
            
            while not done and step < max_steps:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Store experience
                self.replay_buffer.append((state, action, reward, next_state, done))
                
                # Update network continuously
                self.update_network()
                
                # Track metrics
                self._update_metrics(metrics, state, action, reward)
                
                state = next_state
                total_reward += reward
                step += 1

            # Visualize and save results
            if episode == 1 or episode == 450:
                self._visualize(metrics, episode)
            self._save_model(episode)
                
            print(f"\nEpisode {episode+1} Summary:")
            print(f"Total Steps: {step}")
            print(f"Final Budget: {self.env.budget:.2f}")
            print(f"Avg Loss: {np.mean(self.loss_history[-100:]):.4f}")

    def _update_metrics(self, metrics, state, action, reward):
        for city in range(self.env.n_cities):
            metrics['infections'][city].append(state[1 + city*5])
        metrics['budgets'].append(self.env.budget)
        metrics['actions'].append(action)
        metrics['rewards'].append(reward)

    def _visualize(self, metrics, episode):
        plt.clf()
        
        # Infection Plot
        plt.subplot(2, 2, 1)
        for city, data in metrics['infections'].items():
            plt.plot(data, label=f'City {city}')
        plt.title('Infection Trends')
        plt.legend()
        
        # Budget Plot
        plt.subplot(2, 2, 2)
        plt.plot(metrics['budgets'])
        plt.title('Budget Over Time')
        
        # Action Distribution
        plt.subplot(2, 2, 3)
        action_counts = np.bincount(metrics['actions'], minlength=self.act_dim)
        plt.bar(range(self.act_dim), action_counts)
        plt.title('Action Distribution')
        
        # Loss Tracking
        plt.subplot(2, 2, 4)
        plt.plot(self.loss_history)
        plt.title('Training Loss')
        
        plt.tight_layout()
        plt.savefig(f'episode_{episode+1}_realtime.png')
        plt.pause(0.1)

    def _save_model(self, episode):
        torch.save(self.q_net.state_dict(), f'realtime_model_ep{episode+1}.pth')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TravelEnv()
    
    learner = RealtimeLearner(
        env=env,
        device=device,
        model_path="q_network.pth"  # Optional pretrained model
    )
    
    learner.run(
        num_episodes=500,
        max_steps=200
    )