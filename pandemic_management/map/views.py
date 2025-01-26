import os
import json
import logging
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render

import json
import torch
from django.shortcuts import render
from .pandemic_sim import TravelEnv    # Ensure TravelEnv is accessible
from .train_DQN import DuelingQNetwork
import numpy as np


logger = logging.getLogger(__name__)

# JSON View to load data
def json_view(request):
    # Dynamically construct the file path
    file_path = os.path.join(settings.BASE_DIR, 'map', 'static', 'data', 'combined_data.json')
    
    try:
        # Try reading the file with utf-8 encoding
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except UnicodeDecodeError:
        # Fallback to latin-1 encoding if utf-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON from {file_path}: {str(e)}")
            return JsonResponse({'error': 'Invalid file encoding'}, status=500)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return JsonResponse({'error': 'File not found'}, status=404)
    except Exception as e:
        logger.error(f"Unexpected error loading JSON: {str(e)}")
        return JsonResponse({'error': 'Unexpected error'}, status=500)
    
    # Return the loaded JSON data
    return JsonResponse(data, safe=False)

# View to render the map template
# simulation/views.py


def county_map(request):
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize the environment
    env = TravelEnv()
    obs = env.reset()
    
    # 2. Define observation and action dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # 3. Initialize the Q-Network
    online_net = DuelingQNetwork(obs_dim, act_dim).to(device)
    
    print(f"Current obs_dim: {env.observation_space.shape[0]}")
    print(f"Original obs_dim: 20821")  # From error message
    print(f"Current act_dim: {env.action_space.n}")
    print(f"Original act_dim: 8865")  # From error message

    # 4. Load the trained model weights
    model_path = '/Users/zachariahrisheq/Desktop/HoyaHacksTwo/Pandemic-Management/pandemic_management/map/models/dueling_dqn_episode_1.pth'  # Update this path accordingly
    online_net.load_state_dict(torch.load(model_path, map_location=device))
    online_net.eval()  # Set to evaluation mode
    
    # 5. Run the model on the environment for a certain number of steps
    done = False
    steps = 0
    max_steps = 100  # Define how many steps you want to simulate
    simulation_data = []  # To store data for passing to frontend
    
    while not done and steps < max_steps:
        steps += 1
        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)  # Shape: [1, obs_dim]
        
        with torch.no_grad():
            q_values = online_net(state_tensor)  # Shape: [1, act_dim]
            action = q_values.argmax(dim=1).item()  # Select the best action
        
        # Take the action in the environment
        next_obs, reward, done, info = env.step(action)
        
        # Collect data (e.g., city infection states)
        city_data = {}
        for city_id, state in env.city_states.items():
            city_data[city_id] = {
                "susceptible": state["susceptible"],
                "infected": state["infected"],
                "recovered": state["recovered"],
                "vaccinated": state["vaccinated"],
                "marketing": env.marketing_cleanliness_factors.get(city_id, 0.0)
            }
        
        simulation_data.append({
            "step": steps,
            "action": action,
            "reward": reward,
            "done": done,
            "city_data": city_data,
            "budget": env.budget,
            "active_edges": list(env.edge_status.keys()),
            "closed_edges": [edge for edge, status in env.edge_status.items() if not status]
        })
        
        obs = next_obs
    
    # Convert simulation data to JSON
    simulation_json = json.dumps(simulation_data)
    # After running the simulation:
    # Process data for visualization
    infection_data = {}
    for city_id in range(env.n_cities):
        final_state = simulation_data[-1]['city_data'].get(city_id, {})
        infection_data[city_id] = {
            'infected': final_state.get('infected', 0),
            'total_population': env.city_populations.get(city_id, 0)
        }
    
    context = {
        'simulation_data': simulation_json,
        'infection_data': json.dumps(infection_data),
        'city_data': json.dumps({
            city_id: {
                'lat': env.city_manager.get_city_data(city_id)['lat'],
                'lng': env.city_manager.get_city_data(city_id)['lon'],
                'name': env.city_manager.get_city_data(city_id)['name']
            }
            for city_id in range(env.n_cities)
        })
    }
    
    return render(request, 'map.html', context)

