import json
import re

# -------------------------------------------------------
# Step 1: Parse adjacency.txt into a Python dictionary
# -------------------------------------------------------
adjacency_dict = {}  # { "55007": ["27075", "55003", ...], ... }

with open('adjacency.txt', 'r') as f:
    for line in f:
        # Skip empty or whitespace-only lines
        line = line.strip()
        if not line:
            continue
        
        # Using a regex to capture pairs of: "Some County, ST"  [FIPS]
        # Pattern explanation:
        #   "([^"]+)"  => capture anything inside quotes as a group
        #   \s+        => one or more whitespace
        #   (\d+)      => capture one or more digits as FIPS
        pattern = r'"([^"]+)"\s+(\d+)'
        pairs = re.findall(pattern, line)
        
        if not pairs:
            # If we can't parse a line for pairs, skip
            continue
        
        # The first pair in the line is the "source" county
        source_county_name, source_fips = pairs[0]
        
        # All subsequent pairs in that line are neighbors of the source county
        neighbors = pairs[1:]
        
        # Ensure we have a list in adjacency_dict for the source FIPS
        if source_fips not in adjacency_dict:
            adjacency_dict[source_fips] = []
        
        for neighbor_name, neighbor_fips in neighbors:
            # Add neighbor_fips to source_fips' adjacency list
            if neighbor_fips not in adjacency_dict[source_fips]:
                adjacency_dict[source_fips].append(neighbor_fips)
            
            # Also make it bidirectional: neighbor_fips -> source_fips
            if neighbor_fips not in adjacency_dict:
                adjacency_dict[neighbor_fips] = []
            if source_fips not in adjacency_dict[neighbor_fips]:
                adjacency_dict[neighbor_fips].append(source_fips)
        # After building adjacency_dict (before using it):
    for fips_code, neighbors in adjacency_dict.items():
        # If a county lists itself as a neighbor, remove it
        if fips_code in neighbors:
            neighbors.remove(fips_code)



# -----------------------------------------------------------------
# Step 2: Read your existing JSON and add adjacency information
# -----------------------------------------------------------------
with open('combined_data.json', 'r') as f:
    data = json.load(f)

# data should be a dict with structure { "type": "...", "features": [ ... ] }
for feature in data["features"]:
    # 2a) Figure out how you want to extract the county's FIPS.
    #     Example: if GEO_ID = "0500000US01001" => we take the last 5 digits:
    #         "01001" for Autauga County, AL
    #     Or you could combine "STATE" + "COUNTY" if each is 2-digit / 3-digit strings.

    geo_id_full = feature["properties"]["GEO_ID"]  # e.g. "0500000US01001"
    # The last 5 digits often represent the County FIPS
    county_fips = geo_id_full[-5:]  # "01001"
    
    # 2b) Attach the adjacency list if it exists
    if county_fips in adjacency_dict:
        feature["properties"]["neighbors"] = adjacency_dict[county_fips]
    else:
        feature["properties"]["neighbors"] = []  # or None, or skip entirely

# ----------------------------------------------------------------
# Step 3: Write out the augmented JSON to a new file
# ----------------------------------------------------------------
with open('combined.json', 'w') as f:
    # Ensure ASCII=False if you want any special characters to remain
    # Indent for readability if you like
    json.dump(data, f, indent=2)
