import matplotlib.pyplot as plt
import numpy as np
import torch
from pandemic_sim import TravelEnv
from train_DQN import DuelingQNetwork  # Ensure this matches your network's file name
# Add to imports
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import deque
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F

from pandemic_sim import TravelEnv

class PandemicVisualizer:
    def __init__(self, model_path='dueling_dqn.pth'):
        # Initialize environment first
        self.env = TravelEnv()
        self.obs = self.env.reset()
        
        # Then get observation dimensions
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n
        self.marketing_cleanliness_factors = self.env.marketing_cleanliness_factors 

        # Initialize visualization with proper grid layout
        self.fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 2, figure=self.fig, height_ratios=[2, 1, 1])
        self.ax_map = self.fig.add_subplot(gs[:, 0])  # Map takes left column
        self.ax_action = self.fig.add_subplot(gs[0, 1])

        # Load model
        self.model = DuelingQNetwork(self.obs_dim, self.act_dim)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Initialize tracking structures
        self.last_actions = deque(maxlen=5)
        self.q_values_history = []
        
        # Setup visual elements
        self.setup_visual_elements()
        self.stats = {
            'total_infected': [],
            'budget': [],
            'actions': []
        }

        


    def _explain_marketing_action(self, city_id):
        current_factor = self.marketing_cleanliness_factors[city_id]
        infected = self.env.city_states[city_id]['infected']
        pop = self.env.city_populations[city_id]
        density = self.env.city_manager.get_city_density(city_id)
        
        explanation = (
            f"Improving marketing/cleanliness in {self.env.city_manager.city_data[city_id]['name']}\n"
            f"Current marketing factor: {current_factor:.1f}/1.0\n"
            f"Infection rate: {infected/pop:.1%}\n"
            f"Population density: {density:.1f} people/sqmi"
        )
        return explanation
        
    def setup_visual_elements(self):
        """Initialize all visual components including the city scatter plot"""
        # Initialize city points
        city_data = self.env.city_manager.city_data
        self.lons = np.array([city['lon'] for city in city_data.values()])
        self.lats = np.array([city['lat'] for city in city_data.values()])
        self.populations = np.array([city['pop'] for city in city_data.values()])

        # Create main scatter plot (THIS WAS MISSING)
        self.scat = self.ax_map.scatter(
            self.lons, self.lats,
            s=np.sqrt(self.populations)/10,
            c=np.zeros(len(city_data)),
            cmap='Reds', alpha=0.6,
            edgecolors='k', linewidths=0.5
        )

        # Initialize edge lines
        self.edge_lines = []
        for edge in self.env.edges:
            city1, city2 = edge
            lon1 = city_data[city1]['lon']
            lat1 = city_data[city1]['lat']
            lon2 = city_data[city2]['lon']
            lat2 = city_data[city2]['lat']
            line, = self.ax_map.plot([lon1, lon2], [lat1, lat2], 'k-', alpha=0.2)
            self.edge_lines.append({'line': line, 'edge': edge})

        # Set up map aesthetics
        self.ax_map.set_title("Pandemic Spread Simulation")
        self.ax_map.set_xlabel("Longitude")
        self.ax_map.set_ylabel("Latitude")
        self.cbar = self.fig.colorbar(self.scat, ax=self.ax_map)
        self.cbar.set_label('Infection Rate')

        # Info text
        self.info_text = self.ax_map.text(
            0.02, 0.95, '',
            transform=self.ax_map.transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8)
        )


        # Action explanation text
        self.action_text = self.ax_action.text(
            0.05, 0.8, "", 
            transform=self.ax_action.transAxes,
            fontsize=10, 
            verticalalignment='top'
        )
        self.ax_action.axis('off')

        # Action type legend
        action_types = [
            ("Edge Closure", 'lightgray'),
            ("Vaccination", 'lightgreen'), 
            ("Marketing", 'lightblue')
        ]
        patches = [mpatches.Patch(color=c, label=l) for l, c in action_types]
        self.ax_action.legend(
            handles=patches, 
            loc='lower left',
            title="Action Types", 
            fontsize=8
        )

        
    def update_visualization(self, step, action, q_values):
        # Existing visualization updates
        self._update_action_explanation(action)
        #(action)
        self._update_infection_curve(step)
        


    def _update_action_explanation(self, action, current_step):
        """Update action explanation with proper step tracking"""
        explanation = self._explain_action(action)
        
        # Store both step number and explanation
        self.last_actions.appendleft( (current_step, explanation) )
        
        # Generate text with actual step numbers
        text = "Recent Actions:\n" + "\n".join([
            f"Step {step}: {explanation}" 
            for step, explanation in self.last_actions
        ])
        
        self.action_text.set_text(text)
        self.ax_action.set_title("Action Explanations")

    # Then update the call in update_visualization
    def update_visualization(self, step, action, q_values):
        self._update_action_explanation(action, step)  # Pass current step
        #self._highlight_current_action(action)

    def _explain_action(self, action):
        # Get current state information
        edge_actions = len(self.env.edges)
        vacc_actions = self.env.n_cities
        
        if action < edge_actions:
            return self._explain_edge_action(action)
        elif action < edge_actions + vacc_actions:
            return self._explain_vaccination_action(action - edge_actions)
        else:
            return self._explain_marketing_action(action - edge_actions - vacc_actions)

    def _explain_edge_action(self, action_idx):
        city1, city2 = self.env.edges[action_idx]
        status = self.env.edge_status.get((city1, city2), True)
        
        c1_inf = self.env.city_states[city1]['infected']
        c2_inf = self.env.city_states[city2]['infected']
        pop1 = self.env.city_populations[city1]
        pop2 = self.env.city_populations[city2]
        
        explanation = (
            f"{'Closing' if status else 'Reopening'} edge between "
            f"{self.env.city_manager.city_data[city1]['name']} "
            f"({c1_inf/pop1:.1%} infected) and "
            f"{self.env.city_manager.city_data[city2]['name']} "
            f"({c2_inf/pop2:.1%} infected)"
        )
        return explanation

    def _explain_vaccination_action(self, city_id):
        infected = self.env.city_states[city_id]['infected']
        vaccinated = self.env.city_states[city_id]['vaccinated']
        pop = self.env.city_populations[city_id]
        density = self.env.city_manager.get_city_density(city_id)
        
        explanation = (
            f"Vaccinating {self.env.city_manager.city_data[city_id]['name']}\n"
            f"Infection rate: {infected/pop:.1%}, "
            f"Vaccinated: {vaccinated/pop:.1%}, "
            f"Density: {density:.1f} people/sqmi"
        )
        return explanation

    def _highlight_current_action(self, action):
        # Highlight affected cities/edges
        if action < len(self.env.edges):
            # Edge action - highlight the connection
            city1, city2 = self.env.edges[action]
            for edge_info in self.edge_lines:
                if edge_info['edge'] == (city1, city2):
                    edge_info['line'].set_linewidth(3)
                    edge_info['line'].set_zorder(3)
        else:
            # City action - highlight the city
            city_id = action - len(self.env.edges) if action < len(self.env.edges) + self.env.n_cities \
                      else action - len(self.env.edges) - self.env.n_cities
            sizes = self.scat.get_sizes()
            sizes[city_id] *= 2  # Make the city marker bigger
            self.scat.set_sizes(sizes)

    def run_simulation(self, max_steps=1000):
        step = 0
        done = False
        
        while not done and step < max_steps:
            # Get action and Q-values from model
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(self.obs).unsqueeze(0)
                q_values = self.model(obs_tensor).cpu().numpy().flatten()
                action = q_values.argmax().item()
            
            # Step environment
            self.obs, reward, done, _ = self.env.step(action)
            
            # Update visualization with Q-values and action info
            self.update_visualization(step, action, q_values)
            
            # Track statistics
            self.stats['total_infected'].append(sum(c['infected'] for c in self.env.city_states.values()))
            self.stats['budget'].append(self.env.budget)
            self.stats['actions'].append(action)
            
            step += 1
            plt.pause(0.2)  # Slightly slower for better observation
        
        plt.ioff()
        self.show_summary_statistics()

def main():
    # Initialize visualizer with your trained model
    model_path = '/Users/zachariahrisheq/Desktop/HoyaHacksTwo/Pandemic-Management/pandemic_management/map/models/dueling_dqn_episode_69.pth'
    
    # Create visualizer instance
    visualizer = PandemicVisualizer(model_path=model_path)
    
    # Run the simulation with visualization
    visualizer.run_simulation(max_steps=500)

if __name__ == '__main__':
    main()