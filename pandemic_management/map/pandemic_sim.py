import numpy as np
import networkx as nx
import gym
from gym import spaces
from CityManager import CityManager
import logging


class TravelEnv(gym.Env):
    """Gym environment for simulating pandemic spread across cities with population-based infections."""
    
    BASE_BUDGET_PER_CITY = 100000
    BASE_LOCKDOWN_COST = 5000
    BASE_VACCINATION_COST = 10000
    MEAN_BETA = 0.5
    GAMMA = 0.05
    TRANSMISSION_THRESHOLD = 0.000001  # Minimum infection percentage for transmission
    
    def __init__(self,  map_data_path='/Users/zachariahrisheq/Desktop/HoyaHacksTwo/Pandemic-Management/pandemic_management/map/static/data/combined_data.json'):
        super().__init__()
        print("[DEBUG] Initializing TravelEnv...")
        self._initialize_city_data(map_data_path)
        self.graph = nx.Graph()
        self._setup_graph()
        self._setup_spaces()
        self._initialize_states()
        self._calculate_budget()
        self.edge_status = {edge: True for edge in self.edges}
        print("[DEBUG] Environment initialized successfully.\n")

        logging.basicConfig(
        filename='pandemic.log',              # Name of the log file
        filemode='a',                    # Mode: 'a' for append, 'w' for overwrite
        level=logging.DEBUG,             # Logging level
        datefmt='%Y-%m-%d %H:%M:%S'      # Date format
    )


    def _initialize_city_data(self, map_data_path):
        print(f"[DEBUG] Loading CityManager from: {map_data_path}")
        self.city_manager = CityManager(map_data_path)
        self.n_cities = self.city_manager.n_cities
        print(f"[DEBUG] Number of valid cities: {self.n_cities}")

        self.city_populations = {
            city_id: self.city_manager.get_city_data(city_id)["pop"]
            for city_id in range(self.n_cities)
        }
        print("[DEBUG] City populations loaded.")

        self.marketing_cleanliness_factors = {
            city_id: 0.2  # For example, each city can have a different factor
            for city_id in range(self.n_cities)
        }
        print("[DEBUG] Marketing/Cleanliness factors initialized.\n")

    def _setup_graph(self):
        """Create graph based on real county adjacency data"""
        print("[DEBUG] Setting up graph structure...")
        # Add nodes
        for city_id in range(self.n_cities):
            self.graph.add_node(city_id)
        print(f"[DEBUG] Added {self.n_cities} nodes to the graph.")

        # Add edges based on adjacency
        edge_count = 0
        for city_id in range(self.n_cities):
            city_data = self.city_manager.get_city_data(city_id)
            if city_data and 'neighbors' in city_data:
                # 'neighbors' is a list of neighbor FIPS
                for neighbor_fips in city_data['neighbors']:
                    # Now find the city ID that has 'fips' == neighbor_fips
                    for other_id, other_city in self.city_manager.city_data.items():
                        if other_city['fips'] == neighbor_fips:
                            distance = self.city_manager.get_city_distance(city_id, other_id)  # example
                            # Add edge if not already present
                            if not self.graph.has_edge(city_id, other_id):
                                self.graph.add_edge(city_id, other_id, weight=distance)
                                edge_count += 1
                            break

        self.edges = list(self.graph.edges())
        print(f"[DEBUG] Added {edge_count} edges to the graph.\n")

    def _setup_spaces(self):
        # Observation space:
        # Each city contributes 5 features (susceptible, infected, recovered, vaccinated, marketing factor)
        # Plus 1 feature per edge (indicating whether the edge is active or not)
        # If you want to keep both sets:
        obs_dim = self.n_cities * 5 + len(self.edges)
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)


        # Action space:
        # Close an edge: len(self.edges) actions
        # Vaccinate a city: self.n_cities actions
        # Improve marketing/cleanliness: self.n_cities actions
        self.action_space = spaces.Discrete(len(self.edges) + 2 * self.n_cities)

        print(f"[DEBUG] Observation space dimension: {obs_dim}")
        print(f"[DEBUG] Action space dimension: {self.action_space.n}\n")

    def _initialize_states(self):
        """Initialize city states with actual infected population counts."""
        print("[DEBUG] Initializing city states (S, I, R, V)...")
        self.city_states = {}
        for city_id in range(self.n_cities):
            total_pop = self.city_populations[city_id]
            self.city_states[city_id] = {
                "susceptible": total_pop,
                "infected": 0,
                "recovered": 0,
                "vaccinated": 0
            }
        self.current_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.total_infections = 0
        print("[DEBUG] City states initialization complete.\n")

    def _calculate_budget(self):
        city_densities = [city["density"] for city in self.city_manager.city_data.values()]
        total_density = sum(city_densities) if city_densities else 1
        max_density = max(city_densities) if city_densities else 1
        self.budget = 30000
        print(f"[DEBUG] Budget calculated: {self.budget:.2f}\n")

    # -------------------------------------------------------------------
    # Infection & Transmission
    # -------------------------------------------------------------------
    def get_infection_rate(self, city_id, base_rate=None):
        """
        Generate an adjusted infection rate for a city.
        """
        if base_rate is None:
            base_rate = self.MEAN_BETA

        vaccinated_fraction = 0.0
        if self.city_populations[city_id] > 0:
            vaccinated_fraction = (
                self.city_states[city_id]["vaccinated"] / self.city_populations[city_id]
            )

        marketing_factor = self.marketing_cleanliness_factors.get(city_id, 0.0)
        
        # Adjust base rate by vaccination and marketing
        effective_rate = base_rate
        effective_rate *= (1.0 - 0.7 * vaccinated_fraction)
        effective_rate *= (1.0 - 0.3 * marketing_factor)

        # Random Poisson variation
        poisson_lambda = effective_rate * 10
        final_rate = np.random.poisson(poisson_lambda) / 10

        final_rate = min(final_rate, 1.0)
        # Debug
        # print(f"[DEBUG] City {city_id} infection rate={final_rate:.4f} (vaccinated={vaccinated_fraction:.3f}, marketing={marketing_factor:.3f})")
        return final_rate

    def calculate_transmission_probability(self, source_city, target_city):
        src_infected = self.city_states[source_city]["infected"]
        src_pop = self.city_populations[source_city]
        if src_pop == 0:
            return 0.0
        
        edge = (source_city, target_city)
        if not self.edge_status.get(edge, False):
            return 0.0  # Edge is closed; no transmission

        source_infection_percentage = src_infected / src_pop
        if source_infection_percentage < self.TRANSMISSION_THRESHOLD:
            return 0.0

        distance = self.city_manager.get_city_distance(source_city, target_city)
        base_probability = 1 / (1 + (distance / 100.0))
        final_probability = base_probability * source_infection_percentage
        # print(f"[DEBUG] Transmission prob. from {source_city} to {target_city} = {final_probability:.4f}")
        return final_probability

    def update_infections(self):
        """Update infection states for all cities."""
        new_infections = {}
        
        # Calculate new infections from city-to-city transmission
        for city_id in range(self.n_cities):
            neighbors = list(self.graph.neighbors(city_id))
            total_new_infected = 0
            
            # Internal spread
            if self.city_populations[city_id] > 0:
                current_rate = self.get_infection_rate(city_id)
                fraction_infected = self.city_states[city_id]["infected"] / self.city_populations[city_id]
                internal_new = int(
                    self.city_states[city_id]["susceptible"] * current_rate * fraction_infected
                )
                total_new_infected += internal_new

            # External spread from neighbors
            for neighbor in neighbors:
                probability = self.calculate_transmission_probability(neighbor, city_id)
                # Corrected condition: check if random < probability
                if np.random.random() < probability:
                    external_rate = self.get_infection_rate(neighbor)
                    # Calculate based on neighbor's infection rate and current susceptible
                    external_new = int(self.city_states[city_id]["susceptible"] * external_rate)
                    total_new_infected += external_new

            # Recovery
            recovered = int(self.city_states[city_id]["infected"] * self.GAMMA)

            new_infections[city_id] = {
                "new_infected": min(total_new_infected, self.city_states[city_id]["susceptible"]),
                "recovered": recovered
            }

        # Apply updates
        for city_id, updates in new_infections.items():
            self.city_states[city_id]["infected"] += updates["new_infected"]
            self.city_states[city_id]["susceptible"] -= updates["new_infected"]
            self.city_states[city_id]["infected"] -= updates["recovered"]
            self.city_states[city_id]["recovered"] += updates["recovered"]

    # -------------------------------------------------------------------
    # Costs
    # -------------------------------------------------------------------
    def calculate_lockdown_cost(self, city_id):
        city_density = self.city_manager.get_city_data(city_id)["density"]
        max_density = max(city["density"] for city in self.city_manager.city_data.values())
        density_factor = city_density / max_density if max_density > 0 else 0.0
        return self.BASE_LOCKDOWN_COST * density_factor

    def calculate_vaccination_cost(self, city_id):
        city_density = self.city_manager.get_city_data(city_id)["density"]
        # Filter out zero densities to avoid division by zero
        valid_densities = [city["density"] for city in self.city_manager.city_data.values() if city["density"] > 0]
        if not valid_densities:
            return self.BASE_VACCINATION_COST  # fallback
        max_density = max(valid_densities)
        density_factor = city_density / max_density
        return self.BASE_VACCINATION_COST * density_factor
    
    # -------------------------------------------------------------------
    # Step / Reset / Render
    # -------------------------------------------------------------------
    def step(self, action):
        """
        Execute one time step:
        1) Budget increment
        2) Interpret action
        3) Deduct cost
        4) Update infections
        5) Compute reward
        6) Check done
        7) Build obs
        """
        print(f"[DEBUG] step() called with action={action}")
        # 1. Budget increment
        self.budget += 1500
        print(f"[DEBUG] Budget increment -> New budget={self.budget:.2f}")

        # 2. Interpret action => cost
        cost = 0.0
        done = False
        info = {}
        economic_boost = 0.0  # Track economic benefit from reopening edges

        num_edge_actions = len(self.edges)  # close-edge actions
        num_city_actions = self.n_cities

        # Debug info about action range
        print(f"[DEBUG] num_edge_actions={num_edge_actions}, num_city_actions={num_city_actions}, total_actions={self.action_space.n}")

        if action < num_edge_actions:
            edge_to_toggle = self.edges[action]
            previous_status = self.edge_status[edge_to_toggle]
            self.edge_status[edge_to_toggle] = not previous_status  # Toggle status
            
            if previous_status:
                # Edge was open, now closed: incur cost
                cost += 2000.0
                print(f"[DEBUG] Closed edge {edge_to_toggle}, cost=2000.0")
            else:
                # Edge was closed, now reopened: add economic boost
                city1, city2 = edge_to_toggle
                pop1 = self.city_populations[city1]
                pop2 = self.city_populations[city2]
                # Economic boost proportional to combined population
                economic_boost = (pop1 + pop2) / 1000  # Scale appropriately
                print(f"[DEBUG] Reopened edge {edge_to_toggle}, economic_boost={economic_boost:.2f}")
        elif action < num_edge_actions + num_city_actions:
            # Vaccinate a city
            city_id = action - num_edge_actions
            susceptible = self.city_states[city_id]["susceptible"]
            print(f"[DEBUG] Vaccinate city={city_id}, susceptible={susceptible}")
            if susceptible > 0:
                new_vacc = int(susceptible * 0.1)
                self.city_states[city_id]["vaccinated"] += new_vacc
                self.city_states[city_id]["susceptible"] -= new_vacc
                vacc_cost = self.calculate_vaccination_cost(city_id)
                cost += vacc_cost
                print(f"[DEBUG] Vaccinated {new_vacc} in city {city_id}, cost={vacc_cost:.2f}")
            else:
                print("[DEBUG] No susceptibles left to vaccinate, cost=0.")
        else:
            # Marketing improvement
            city_id = action - (num_edge_actions + num_city_actions)
            current_factor = self.marketing_cleanliness_factors[city_id]
            increment = 0.1
            new_factor = min(current_factor + increment, 0.4)
            change_in_factor = new_factor - current_factor
            self.marketing_cleanliness_factors[city_id] = new_factor
            cost_to_improve = 1000.0 * change_in_factor
            cost += cost_to_improve
            print(f"[DEBUG] Improve marketing city={city_id}, from={current_factor:.2f} to={new_factor:.2f}, cost={cost_to_improve:.2f}")

        # 3. Deduct cost from budget
        old_budget = self.budget
        self.budget -= cost
        print(f"[DEBUG] Deducted cost={cost:.2f} from budget. Old budget={old_budget:.2f} -> New budget={self.budget:.2f}")

        # 4. Update infections
        total_infected_before = sum(city["infected"] for city in self.city_states.values())
        self.update_infections()
        total_infected_after = sum(city["infected"] for city in self.city_states.values())

         # 5. Compute reward
        infection_change = total_infected_after - total_infected_before
        scaled_infection_change = infection_change / 1000.0
        scaled_cost = cost / 1000.0
        # 1. Economic reward for open edges
        num_open_edges = sum(self.edge_status.values())
        economic_reward = num_open_edges * 0.01  # Reduced from 0.1
        
        # 2. Infection control bonus
        infection_reduction = total_infected_before - total_infected_after
        infection_bonus = infection_reduction / 100.0  # Increased from 500


        # 3. Scaled cost penalty (original)
        scaled_cost = cost / 1000.0
        
        # Final reward
        reward = economic_reward + infection_bonus - scaled_cost

        print(f"[DEBUG] Economic: {economic_reward:.2f} | Infection Bonus: {infection_bonus:.2f} | Cost Penalty: {-scaled_cost:.2f}")

        # New detailed debug information
        num_closed_edges = sum(1 for status in self.edge_status.values() if not status)
        print("\n[DEBUG] === ECONOMIC & HEALTH METRICS ===")
        print(f"[DEBUG] Total infections: {total_infected_after}")
        print(f"[DEBUG] Infection change: {infection_change} (Δ)")
        with open('app.log', 'a') as log_file:
            log_file.write(f"[DEBUG] Total infections: {total_infected_after}\n")
            log_file.write(f"[DEBUG] Infection change: {infection_change} (Δ)\n")
        print(f"[DEBUG] Active edges: {len(self.edges) - num_closed_edges}/{len(self.edges)}")
        print(f"[DEBUG] Closed edges: {num_closed_edges}")
        print(f"[DEBUG] Cost penalty: {-scaled_cost:.4f}")
        print(f"[DEBUG] Infection penalty: {-scaled_infection_change * 2.0:.4f}")
        print(f"[DEBUG] Final reward: {reward:.4f}")
        # 6. Check done
        if self.budget < 0:
            done = True
            info["reason"] = "budget_depleted"
            print("[DEBUG] Budget depleted -> Episode DONE.")
        if total_infected_after == 0:
            done = True
            info["reason"] = "no_infections_left"
            print("[DEBUG] All infections cleared -> Episode DONE.")

        # 7. Build final observation
        observation = self._build_observation()
        print("[DEBUG] step() complete.\n")
        return observation, reward, done, info

    def reset(self):
        print("[DEBUG] reset() called. Re-initializing environment state...")
        # 1. Reset city states
        self.city_states.clear()
        for city_id in range(self.n_cities):
            pop = self.city_populations[city_id]
            self.city_states[city_id] = {
                "susceptible": pop,
                "infected": 0,
                "recovered": 0,
                "vaccinated": 0
            }

        # 2. Reset graph
        self.graph.clear()
        for city_id in range(self.n_cities):
            self.graph.add_node(city_id)
        for edge in self.edges:
            self.graph.add_edge(edge[0], edge[1])

        # 3. Reset marketing factors
        for cid in range(self.n_cities):
            self.marketing_cleanliness_factors[cid] = 0.2

        # 4. Reset edge status to all open (True)
        self.edge_status = {edge: True for edge in self.edges}

        # 5. Reset budget
        self.budget =  30000
        print(f"[DEBUG] Budget reset to {self.budget:.2f}")

        # 6. Seed an initial infection in city 0
        initial_infected = min(500, self.city_populations[0])
        self.city_states[0]["infected"] = initial_infected
        self.city_states[0]["susceptible"] -= initial_infected
        print(f"[DEBUG] Seeded city 0 with {initial_infected} infected.\n")

        # 7. Return initial observation
        obs = self._build_observation()
        print("[DEBUG] reset() complete. Returning initial observation.\n")
        return obs
    
    def _build_observation(self):
        """Build observation array with both normalized and absolute values"""
        obs = []
        absolute_values = {}  # New: Store absolute values for visualization
        
        for city_id in range(self.n_cities):
            pop = self.city_populations[city_id]
            cstate = self.city_states[city_id]
            
            # Existing normalized values for DQN
            fraction_sus = cstate["susceptible"] / pop if pop > 0 else 0.0
            fraction_inf = cstate["infected"] / pop if pop > 0 else 0.0
            fraction_rec = cstate["recovered"] / pop if pop > 0 else 0.0
            fraction_vac = cstate["vaccinated"] / pop if pop > 0 else 0.0
            marketing = self.marketing_cleanliness_factors[city_id]

            # Store absolute values for visualization
            absolute_values[city_id] = {
                "susceptible": cstate["susceptible"],
                "infected": cstate["infected"],
                "recovered": cstate["recovered"],
                "vaccinated": cstate["vaccinated"],
                "population": pop,
                "lat": self.city_manager.get_city_data(city_id)['lat'],
                "lng": self.city_manager.get_city_data(city_id)['lon']
            }

            # Maintain existing observation structure
            obs.extend([fraction_sus, fraction_inf, fraction_rec, fraction_vac, marketing])

        # Edge states (existing)
        edge_states = [1.0 if self.edge_status[e] else 0.0 for e in self.edges]
        obs.extend(edge_states)

        # Store absolute values as additional info (doesn't affect DQN input)
        self.current_absolute_values = absolute_values
        
        return np.array(obs, dtype=np.float32)

    def render(self, mode="human"):
        """Enhanced render with edge status"""
        print("[DEBUG] render() called. Current State:")
        # City states
        for city_id in range(self.n_cities):
            state = self.city_states[city_id]
            print(
                f"  City {city_id}: "
                f"S={state['susceptible']}, I={state['infected']}, "
                f"R={state['recovered']}, V={state['vaccinated']}, "
                f"Marketing={self.marketing_cleanliness_factors[city_id]:.2f}"
            )
        # Edge states
        num_closed = sum(1 for status in self.edge_status.values() if not status)
        print(f"  Budget: {self.budget:.2f}")
        print(f"  Edges: {len(self.edges)} total, {num_closed} closed")
        print("-" * 30)


if __name__ == "__main__":
    # Initialize the environment
    env = TravelEnv()
    
    # Reset the environment to get the initial observation
    observation = env.reset()
    print("\n[DEBUG] Initial Observation (truncated if large):", observation[:50], "...\n")
    
    # Simulate a few steps
    done = False
    steps = 0
    total_reward = 0
    
    while not done and steps < 10:  # Run for a maximum of 10 steps
        print(f"[DEBUG] === Step {steps + 1} ===")
        
        # Random action for testing
        action = env.action_space.sample()
        print(f"[DEBUG] Taking action: {action}")
        
        # Step through the environment
        observation, reward, done, info = env.step(action)
        
        # Render the environment (prints the state to the console)
        env.render()
        
        print(f"[DEBUG] Reward={reward:.4f}, Done={done}, Info={info}\n")
        total_reward += reward
        steps += 1
    
    print("[DEBUG] Simulation ended.")
    print(f"[DEBUG] Total reward after {steps} steps: {total_reward:.4f}")
