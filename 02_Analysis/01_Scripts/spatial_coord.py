#spatial_coord.py
# Define the zone boundaries
zone_boundaries = {
    0: {"upper_left": (372982.0, 6867160.0), "lower_right": (375482.0, 6864660.0)},
    1: {"upper_left": (377982.0, 6854660.0), "lower_right": (380482.0, 6852160.0)},
    2: {"upper_left": (372982.0, 6869660.0), "lower_right": (375482.0, 6867160.0)},
    3: {"upper_left": (380482.0, 6857160.0), "lower_right": (382982.0, 6854660.0)},
    4: {"upper_left": (372982.0, 6872160.0), "lower_right": (375482.0, 6869660.0)},
    5: {"upper_left": (380482.0, 6859660.0), "lower_right": (382982.0, 6857160.0)},
    6: {"upper_left": (380482.0, 6862160.0), "lower_right": (382982.0, 6859660.0)},
    7: {"upper_left": (372982.0, 6864660.0), "lower_right": (375482.0, 6862160.0)},
    8: {"upper_left": (375482.0, 6867160.0), "lower_right": (377982.0, 6864660.0)},
    9: {"upper_left": (375482.0, 6869660.0), "lower_right": (377982.0, 6867160.0)},
    10: {"upper_left": (375482.0, 6872160.0), "lower_right": (377982.0, 6869660.0)},
    11: {"upper_left": (375482.0, 6859660.0), "lower_right": (377982.0, 6857160.0)},
    12: {"upper_left": (375482.0, 6862160.0), "lower_right": (377982.0, 6859660.0)},
    13: {"upper_left": (375482.0, 6864660.0), "lower_right": (377982.0, 6862160.0)},
    14: {"upper_left": (377982.0, 6867160.0), "lower_right": (380482.0, 6864660.0)},
    15: {"upper_left": (377982.0, 6857160.0), "lower_right": (380482.0, 6854660.0)},
    16: {"upper_left": (377982.0, 6859660.0), "lower_right": (380482.0, 6857160.0)},
    17: {"upper_left": (377982.0, 6862160.0), "lower_right": (380482.0, 6859660.0)},
    18: {"upper_left": (377982.0, 6864660.0), "lower_right": (380482.0, 6862160.0)},
    19: {"upper_left": (370482.0, 6867160.0), "lower_right": (372982.0, 6864660.0)},
    20: {"upper_left": (370482.0, 6869660.0), "lower_right": (372982.0, 6867160.0)}
}

def generate_spatial_indices(data, zone_id):
    """
    Generate row_idx and col_idx columns for spatial data based on zone boundaries.
    
    Args:
        data (DataFrame): The data for a specific zone
        zone_id (int): The ID of the zone
    
    Returns:
        DataFrame: The data with spatial indices added
    """
    if zone_id not in zone_boundaries:
        logging.warning(f"Zone ID {zone_id} not found in boundaries dictionary, spatial indices not generated")
        return data
    
    # Get boundaries
    ul_x, ul_y = zone_boundaries[zone_id]["upper_left"]
    lr_x, lr_y = zone_boundaries[zone_id]["lower_right"]
    
    # Calculate dimensions
    width = lr_x - ul_x
    height = ul_y - lr_y
    
    # Calculate resolution (assuming square pixels)
    # This assumes your data points are on a regular grid
    n_points = len(data)
    points_per_side = int(math.sqrt(n_points))
    
    # Generate row and column indices
    rows = []
    cols = []
    
    for i in range(points_per_side):
        for j in range(points_per_side):
            rows.append(i)
            cols.append(j)
    
    # Add remaining points if n_points is not a perfect square
    remaining = n_points - len(rows)
    for i in range(remaining):
        rows.append(points_per_side)
        cols.append(i)
    
    # Add indices to dataframe
    data['row_idx'] = rows
    data['col_idx'] = cols
    
    return data