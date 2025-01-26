import numpy as np
import networkx as nx
import gym
from gym import spaces
from CityManager import CityManager

class TravelEnv(gym.Env):
    """Gym environment for simulating pandemic spread across cities with population-based infections."""
    
    BASE_BUDGET_PER_CITY = 100000
    BASE_LOCKDOWN_COST = 5000
    BASE_VACCINATION_COST = 10000
    MEAN_BETA = 0.3
    GAMMA = 0.1
    TRANSMISSION_THRESHOLD = 0.00001  # Minimum infection percentage for transmission
    
    def __init__(self,  map_data_path='/Users/zachariahrisheq/Desktop/HoyaHacksTwo/Pandemic-Management/pandemic_management/map/static/data/combined_data.json'):
        super().__init__()
        self._initialize_city_data(map_data_path)
        self.graph = nx.Graph()
        self._setup_graph()
        self._setup_spaces()
        self._initialize_states()
        self._calculate_budget()

    def _initialize_city_data(self, map_data_path):
        self.city_manager = CityManager(map_data_path)
        self.n_cities = self.city_manager.n_cities
        self.city_populations = {
            city_id: self.city_manager.get_city_data(city_id)["pop"]
            for city_id in range(self.n_cities)
        }

        self.marketing_cleanliness_factors = {
            city_id: 0.2  # For example, each city can have a different factor
            for city_id in range(self.n_cities)
        }


    def _setup_graph(self):
        """Create graph based on real county adjacency data"""
        # Add nodes
        for city_id in range(self.n_cities):
            self.graph.add_node(city_id)
            
        # Add edges based on adjacency
        for city_id in range(self.n_cities):
            city_data = self.city_manager.get_city_data(city_id)
            if city_data and 'neighbors' in city_data:
                # 'neighbors' is a list of neighbor FIPS
                for neighbor_fips in city_data['neighbors']:
                    # Now find the city ID that has 'fips' == neighbor_fips
                    for other_id, other_city in self.city_manager.city_data.items():
                        if other_city['fips'] == neighbor_fips:
                            distance = self.city_manager.get_city_distance(city_id, other_id)  # example
                            self.graph.add_edge(city_id, other_id, weight=distance)
                            break

        
        self.edges = list(self.graph.edges())

    def _setup_spaces(self):
        # Observation space:
        # Each city contributes 5 features (susceptible, infected, recovered, vaccinated, marketing factor)
        # Plus 1 feature per edge (indicating whether the edge is active or not)
        obs_dim = self.n_cities * 5 + len(self.edges)

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )

        # Action space:
        # Close an edge: len(self.edges) actions
        # Vaccinate a city: self.n_cities actions
        # Improve marketing/cleanliness: self.n_cities actions
        self.action_space = spaces.Discrete(len(self.edges) + 2 * self.n_cities)


    def _initialize_states(self):
        """Initialize city states with actual infected population counts."""
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

    def _calculate_budget(self):
        city_densities = [city["density"] for city in self.city_manager.city_data.values()]
        total_density = sum(city_densities)
        max_density = max(city_densities)
        self.budget = self.BASE_BUDGET_PER_CITY * self.n_cities * (total_density / max_density)

    def get_infection_rate(self, city_id, base_rate=None):
        """
        Generate an *adjusted* infection rate for a city,
        factoring in:
        - The base rate (mean beta)
        - The fraction of the city that's vaccinated
        - The marketing/cleanliness factor
        - (Optionally) random variation (e.g., Poisson)
        """
        if base_rate is None:
            base_rate = self.MEAN_BETA

        # Fraction vaccinated in this city
        vaccinated_fraction = 0.0
        if self.city_populations[city_id] > 0:
            vaccinated_fraction = (
                self.city_states[city_id]["vaccinated"] / self.city_populations[city_id]
            )

        # Marketing/cleanliness factor for this city (assumed in [0,1])
        # The higher this factor, the more it reduces infection rate.
        marketing_factor = self.marketing_cleanliness_factors.get(city_id, 0.0)
        
        # Simple model: reduce infection rate by some proportion of vaccinated
        # plus some proportion from marketing. 
        # Feel free to tweak the formula for your needs.
        # For example:
        #   - vaccination can reduce the base rate up to 70% if everyone is vaccinated
        #   - marketing can reduce up to 30% if marketing_factor=1
        effective_rate = base_rate
        effective_rate *= (1.0 - 0.7 * vaccinated_fraction)  # e.g. 70% max reduction
        effective_rate *= (1.0 - 0.3 * marketing_factor)     # e.g. 30% max reduction

        # Then you might still want random variation around that effective rate:
        poisson_lambda = effective_rate * 10
        final_rate = np.random.poisson(poisson_lambda) / 10

        # Or skip Poisson if you prefer a direct approach
        # final_rate = effective_rate

        return min(final_rate, 1.0)
    

    def calculate_transmission_probability(self, source_city, target_city):
        """Calculate probability of disease transmission between cities."""
        source_infection_percentage = (
            self.city_states[source_city]["infected"] / 
            self.city_populations[source_city]
        )
        

        # If infection percentage is below threshold, no external transmission
        
        if source_infection_percentage < self.TRANSMISSION_THRESHOLD:
            return 0.0

            
        distance = self.city_manager.get_city_distance(source_city, target_city)
        base_probability = 1 / (1 + distance/100)
        return base_probability * source_infection_percentage

    def update_infections(self):
        """Update infection states for all cities."""
        new_infections = {}
        
        # Calculate new infections from city-to-city transmission
        for city_id in range(self.n_cities):
            neighbors = list(self.graph.neighbors(city_id))
            total_new_infected = 0
            
            # Internal spread
            current_rate = self.get_infection_rate(city_id)
            internal_new_cases = int(
                self.city_states[city_id]["susceptible"] * 
                current_rate * 
                (self.city_states[city_id]["infected"] / self.city_populations[city_id])
            )
            total_new_infected += internal_new_cases
            
            # External spread from neighbors
            for neighbor in neighbors:
                if np.random.random() * 0.003 > self.calculate_transmission_probability(neighbor, city_id):
                    transmission_rate = self.get_infection_rate(neighbor)  # Reduced rate for external transmission
                    external_new_cases = int(
                        self.city_states[city_id]["susceptible"] * 
                        transmission_rate
                    )
                    total_new_infected += external_new_cases
            
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

    def calculate_lockdown_cost(self, city_id):
        city_density = self.city_manager.get_city_data(city_id)["density"]
        max_density = max(city["density"] for city in self.city_manager.city_data.values())
        density_factor = city_density / max_density
        return self.BASE_LOCKDOWN_COST * density_factor

    def calculate_vaccination_cost(self, city_id):
        city_density = self.city_manager.get_city_data(city_id)["density"]
        max_density = max(city["density"] for city in self.city_manager.city_data.values())
        density_factor = city_density / max_density
        return self.BASE_VACCINATION_COST * density_factor
        
    def step(self, action):
        """
        Execute one time step within the environment. 
        Each step the agent will get more money and take an action that will cost them money.

        Actions:
            1) Closing an edge (travel restriction)
            2) Vaccinating a city
            3) Improving marketing/cleanliness in a city

        Action mapping:
        - 0 to (len(self.edges) - 1): Close one of the edges.
        - len(self.edges) to (len(self.edges) + self.n_cities - 1): Vaccinate a city.
        - len(self.edges) + self.n_cities to (len(self.edges) + 2*self.n_cities - 1): Improve marketing/cleanliness in a city.

        Args:
            action (int): The index representing the chosen action. 

        Returns:
            observation (np.ndarray): A numpy array representing the new state of the 
                environment after the action, usually in normalized form 
                (e.g., susceptible/infected/recovered fractions per city).
            reward (float): The immediate reward (or negative cost) for taking the given 
                action (e.g., cost of vaccines, cost of marketing, etc.).
            done (bool): A boolean indicating whether the current episode has ended. 
                (You can define custom conditions such as if budget is exceeded or 
                infection is zero in all cities.)
            info (dict): A dictionary with additional information useful for debugging 
                or logging, typically not used by most RL algorithms.
        """
        # ----------------------
        # 1. Add money to budget (for example, the agent gets 5,000 each turn)
        self.budget += 1500

        # 2. Initialize cost/reward parameters
        cost = 0.0
        done = False
        info = {}

        # 3. Interpret the action
        num_edge_actions = len(self.edges)
        num_city_actions = self.n_cities

        if action < num_edge_actions:
            # (1) Close an edge
            edge_to_close = self.edges[action]
            if self.graph.has_edge(edge_to_close[0], edge_to_close[1]):
                self.graph.remove_edge(edge_to_close[0], edge_to_close[1])
                cost += 2000  # Arbitrary cost for closing an edge
        elif action < num_edge_actions + num_city_actions:
            # (2) Vaccinate a city
            city_id = action - num_edge_actions
            if self.city_states[city_id]["susceptible"] > 0:
                # Vaccinate 10% of remaining susceptibles (example)
                new_vacc = int(self.city_states[city_id]["susceptible"] * 0.1)
                self.city_states[city_id]["vaccinated"] += new_vacc
                self.city_states[city_id]["susceptible"] -= new_vacc
                # Cost of vaccinating
                cost += self.calculate_vaccination_cost(city_id)
        else:
            # (3) Improve marketing/cleanliness
            city_id = action - (num_edge_actions + num_city_actions)
            current_factor = self.marketing_cleanliness_factors[city_id]
            increment = 0.1  # example increment
            new_factor = min(current_factor + increment, 1.0)
            self.marketing_cleanliness_factors[city_id] = new_factor
            # Cost of improving marketing, e.g., scale by how much factor changed
            cost += 1000 * (new_factor - current_factor)

        # 4. Deduct cost from budget
        self.budget -= cost

        # 6. Build the new observation
        observation = self._build_observation()

        # 7. Define reward
        #    Typically you might want to penalize large infection spread or 
        #    reward lower infections. Here we demonstrate a simple negative cost:
        total_infected_before = sum(self.city_states[c]["infected"] for c in self.city_states)
        self.update_infections()
        total_infected_after = sum(self.city_states[c]["infected"] for c in self.city_states)
        
        infection_change = total_infected_after -total_infected_before 
        reward = -cost - infection_change * 10

        # 8. Check if done (simple examples below)
        # Condition 1: If budget is depleted, end episode
        if self.budget < 0:
            done = True
            info["reason"] = "budget_depleted"

        # Condition 2: If all cities have 0 infected, we could consider it "win/controlled"
        total_infected = sum(self.city_states[c]["infected"] for c in self.city_states)
        if total_infected == 0:
            done = True
            info["reason"] = "no_infections_left"

        return observation, reward, done, info


    def reset(self):
        """
        Reset the environment to an initial state and return the initial observation.

        Returns:
            observation (np.ndarray): The initial state of the environment, 
                often normalized (e.g., susceptible/infected/recovered fractions 
                in each city).
        """
        # 1. Re-initialize city states
        self.city_states = {}
        for city_id in range(self.n_cities):
            total_pop = self.city_populations[city_id]
            self.city_states[city_id] = {
                "susceptible": total_pop,
                "infected": 0,
                "recovered": 0,
                "vaccinated": 0
            }

        # 2. Reset the graph (in case edges were removed)
        self.graph.clear()
        for edge in self.edges:
            self.graph.add_edge(edge[0], edge[1])

        # 3. Reset marketing/cleanliness factors
        self.marketing_cleanliness_factors = {
            city_id: 0.2 for city_id in range(self.n_cities)
        }

        # 4. Reset budget (example: might depend on your initial formula)
        self.budget = self.BASE_BUDGET_PER_CITY * self.n_cities

        # 5. Optionally seed an infection in one or more cities
        #    (Example: Infect 100 people in city 0)
        initial_infected = min(100, self.city_populations[0])
        self.city_states[0]["infected"] = initial_infected
        self.city_states[0]["susceptible"] = self.city_populations[0] - initial_infected

        # 6. Build initial observation
        observation = self._build_observation()
        return observation


    def _build_observation(self):
        obs = []
        for city_id in range(self.n_cities):
            pop = self.city_populations[city_id]
            city_obs = [
                self.city_states[city_id]["susceptible"] / pop,
                self.city_states[city_id]["infected"] / pop,
                self.city_states[city_id]["recovered"] / pop,
                self.city_states[city_id]["vaccinated"] / pop,
                self.marketing_cleanliness_factors[city_id]
            ]
            obs.extend(city_obs)
        
        # Add edge states
        edge_states = [1.0 if self.graph.has_edge(e[0], e[1]) else 0.0 for e in self.edges]
        obs.extend(edge_states)
        
        return np.array(obs, dtype=np.float32)
    

    def render(self, mode="human"):
        """
        Render the current state of the environment.

        Args:
            mode (str): The mode to render with. "human" is the default.
        
        Example:
            Prints out the current state of each city, including:
            - Susceptible, Infected, Recovered, Vaccinated populations
            - Marketing/Cleanliness factors
        """
        print("Current State:")
        for city_id in range(self.n_cities):
            state = self.city_states[city_id]
            print(
                f"City {city_id}: "
                f"S={state['susceptible']}, I={state['infected']}, "
                f"R={state['recovered']}, V={state['vaccinated']}, "
                f"Marketing={self.marketing_cleanliness_factors[city_id]:.2f}"
            )
        print(f"Budget: {self.budget}")
        print("-" * 30)




if __name__ == "__main__":
    # Initialize the environment
    env = TravelEnv()
    
    # Reset the environment to get the initial observation
    observation = env.reset()
    print("Initial Observation:", observation)
    
    # Simulate a few steps
    done = False
    steps = 0
    total_reward = 0
    
    while not done and steps < 10:  # Run for a maximum of 10 steps
        print(f"Step {steps + 1}:")
        
        # Random action for testing
        action = env.action_space.sample()
        print(f"Taking action: {action}")
        
        # Step through the environment
        observation, reward, done, info = env.step(action)
        
        # Render the environment (prints the state to the console)
        env.render()
        
        print(f"Reward: {reward}")
        total_reward += reward
        
        steps += 1
    
    print("Simulation ended.")
    print(f"Total reward after {steps} steps: {total_reward}")