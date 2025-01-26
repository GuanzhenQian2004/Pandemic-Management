import json

class CityManager:
    def __init__(self, map_data_path='/Users/zachariahrisheq/Desktop/HoyaHacksTwo/Pandemic-Management/pandemic_management/map/static/data/combined_data.json'):
        self.map_data = self.load_map_data(map_data_path)
        self.city_data = self.initialize_city_data()
        self.n_cities = len(self.city_data)

    def load_map_data(self, path):
        """Load the raw geo-JSON (or similar) data from file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def initialize_city_data(self):
        city_data = {}
        features = self.map_data.get('features', [])

        city_index = 0
        for county in features:
            # Initialize props with empty dict first
            props = {}
            geometry = {}
            
            try:
                # Safely get properties and geometry
                props = county.get('properties', {})
                geometry = county.get('geometry', {})
            except AttributeError:
                print(f"Skipping invalid county entry: {county}")
                continue

            # Rest of your processing code...
            coordinates = geometry.get('coordinates', [])
            lats, lons = [], []
            
            if geometry.get('type') == 'Polygon' and coordinates:
                for point in coordinates[0]:
                    lons.append(point[0])
                    lats.append(point[1])
            if (props.get('Total Population', 0) == 0 ):
                density = 1
            else:
                density = props.get('AREALAND_SQMI', 0)/props.get('Total Population', 0)

            avg_lat = sum(lats)/len(lats) if lats else 0.0
            avg_lon = sum(lons)/len(lons) if lons else 0.0

            city_data[city_index] = {
                'fips': props.get('STATE', '') + props.get('COUNTY', ''),
                'name': props.get('NAME', ''),
                'pop': props.get('Total Population', 0),
                'area': props.get('AREALAND_SQMI', 0),
                'density' : density,
                'lat': avg_lat,
                'lon': avg_lon,
                'neighbors': props.get('neighbors', []),
                'coordinates': coordinates
            }
            city_index += 1
        
        return city_data


    # -----------------------------------------------------------------
    # Data Retrieval
    # -----------------------------------------------------------------
    def get_city_data(self, city_id):
        """Retrieve dictionary of all data for a city by internal ID."""
        return self.city_data.get(city_id, None)

    def get_city_population(self, city_id):
        city = self.get_city_data(city_id)
        return city['pop'] if city else None

    def get_city_density(self, city_id):
        city = self.get_city_data(city_id)
        return city['density'] if city else None

    # -----------------------------------------------------------------
    # Adjacency
    # -----------------------------------------------------------------
    def get_adjacent_cities(self, city_id):
        """
        Return a list of city-IDs that are adjacent to the given city_id.

        Since your JSON has 'neighbors' as a list of FIPS codes,
        we'll look for other city entries that share those FIPS.
        """
        if city_id not in self.city_data:
            return []
        
        city_fips = self.city_data[city_id]['fips']
        neighbor_fips_list = self.city_data[city_id]['neighbors']
        
        # Collect any city whose 'fips' is in 'neighbor_fips_list'
        adjacent_ids = []
        for other_id, other_city in self.city_data.items():
            if other_id != city_id and other_city['fips'] in neighbor_fips_list:
                adjacent_ids.append(other_id)
        return adjacent_ids

    # -----------------------------------------------------------------
    # Distance
    # -----------------------------------------------------------------
    def get_city_distance(self, city_id1, city_id2):
        """
        Calculate a simple Euclidean distance between the
        first coordinate pairs of each city.
        NOTE: This is simplistic; if you have polygons, you might
        want a more robust approach. For demonstration only.
        """
        c1 = self.get_city_data(city_id1)
        c2 = self.get_city_data(city_id2)
        if not c1 or not c2:
            return float('inf')

        # For simplicity, pick the first coordinate if it exists
        coords1 = c1['coordinates']
        coords2 = c2['coordinates']

        if not coords1 or not coords2:
            return float('inf')

        # Typically, for a simple polygon: coords1[0][0] => first ring, first point
        # coords1 = [ [ [x1, y1], [x2, y2], ... ] ]
        try:
            x1, y1 = coords1[0][0][:2]
            x2, y2 = coords2[0][0][:2]
            return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5
        except (IndexError, TypeError):
            return float('inf')


    # -----------------------------------------------------------------
    # Data Updates
    # -----------------------------------------------------------------
    def set_city_data(self, city_id, name=None, pop=None, area=None):
        """
        Update the name, population, or area for a specific city
        (and recalc density).
        """
        if city_id not in self.city_data:
            raise ValueError(f"City ID {city_id} does not exist.")

        if name is not None:
            self.city_data[city_id]['name'] = name
        if pop is not None:
            self.city_data[city_id]['pop'] = pop
        if area is not None:
            self.city_data[city_id]['area'] = area

        # Recalc density
        p = self.city_data[city_id]['pop']
        a = self.city_data[city_id]['area']
        self.city_data[city_id]['density'] = p / a if a else 0

    def update_city_population(self, city_id, new_population):
        """Update population for a city, recalc density."""
        if city_id not in self.city_data:
            raise ValueError(f"City ID {city_id} does not exist.")
        self.city_data[city_id]['pop'] = new_population
        a = self.city_data[city_id]['area']
        self.city_data[city_id]['density'] = new_population / a if a else 0

    def update_city_area(self, city_id, new_area):
        """Update area for a city, recalc density."""
        if city_id not in self.city_data:
            raise ValueError(f"City ID {city_id} does not exist.")
        p = self.city_data[city_id]['pop']
        self.city_data[city_id]['area'] = new_area
        self.city_data[city_id]['density'] = p / new_area if new_area else 0

    # -----------------------------------------------------------------
    # City Management
    # -----------------------------------------------------------------
    def add_city(self, city_id, name, pop, area):
        """Add a new city record."""
        if city_id in self.city_data:
            raise ValueError(f"City ID {city_id} already exists.")
        self.city_data[city_id] = {
            'name': name,
            'pop': pop,
            'area': area,
            'density': pop / area if area else 0,
            'neighbors': [],
            'coordinates': [],
            'fips': "",
        }
        self.n_cities += 1

    def remove_city(self, city_id):
        """Remove a city from the data."""
        if city_id in self.city_data:
            del self.city_data[city_id]
            self.n_cities -= 1
        else:
            raise ValueError(f"City ID {city_id} does not exist.")

    # -----------------------------------------------------------------
    # Aggregate Stats
    # -----------------------------------------------------------------
    def total_population(self):
        """Sum of all city populations."""
        return sum(c['pop'] for c in self.city_data.values())

    def total_area(self):
        """Sum of all city areas."""
        return sum(c['area'] for c in self.city_data.values())

    def average_density(self):
        """Average density across all cities."""
        if self.n_cities == 0:
            return 0
        return sum(c['density'] for c in self.city_data.values()) / self.n_cities

    def city_names(self):
        """Return a list of all city names."""
        return [c['name'] for c in self.city_data.values()]


def main():
    map_path = "/Users/zachariahrisheq/Desktop/HoyaHacksTwo/Pandemic-Management/pandemic_management/map/static/data/combined_data.json"  # Path to your JSON
    manager = CityManager(map_path)

    print("Total cities:", manager.n_cities)
    print("All city names:", manager.city_names())

    # Print data for all cities
    # Print data for all cities with error handling
    for i in range(manager.n_cities):
        c = manager.get_city_data(i)
        if c:  # Check if the city data exists
            print(f"\nCity ID: {i}")
            print(f" Name      : {c['name']}")
            print(f" FIPS      : {c['fips']}")
            print(f" Population: {c['pop']}")
            print(f" Area      : {c['area']}")
            print(f" Density   : {c['density']}")
            print(f" Neighbors (FIPS) : {c['neighbors']}")
            # Print the adjacent city IDs in our manager:
            adj_ids = manager.get_adjacent_cities(i)
            print(f" => Adjacent city IDs: {adj_ids}")
        else:
            print(f"\nCity ID: {i} - No data available.")



if __name__ == "__main__":
    main()
