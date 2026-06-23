import os
import sys
import datetime
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory


# Resolve paths relative to the script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import recommendation_engine
from recommendation_engine import run_integrated_traffic_pipeline, process_single_incident
from digital_twin import (load_bengaluru_graph_with_priors, load_empirical_traffic_priors,
                           compute_dynamic_network_states, compare_scenarios)
import osmnx as ox

app = Flask(__name__, static_folder='static', static_url_path='')

# Load reference datasets
historical_df = pd.DataFrame()
theme_path = os.path.join(BASE_DIR, "theme2.csv")
if os.path.exists(theme_path):
    dtype_dict = {
        'id': 'str', 'event_type': 'str', 'requires_road_closure': 'str', 'priority': 'str',
        'corridor': 'str', 'zone': 'str', 'police_station': 'str',
        'veh_type': 'str', 'junction': 'str', 'event_cause': 'str',
        'latitude': 'float64', 'longitude': 'float64'
    }
    historical_df = pd.read_csv(theme_path, dtype=dtype_dict)
    historical_df = historical_df.dropna(subset=['latitude', 'longitude', 'start_datetime'])
    historical_df['start_datetime'] = pd.to_datetime(historical_df['start_datetime'], format='ISO8601', utc=True, errors='coerce')

# Station supply state
STATION_SUPPLY_POOLS = {
    'peenya': {'cops': 15, 'barricades': 20},
    'hsr layout': {'cops': 20, 'barricades': 25},
    'wilson garden': {'cops': 12, 'barricades': 15},
    'sadashivanagar': {'cops': 10, 'barricades': 12},
    'cubbon park': {'cops': 25, 'barricades': 30},
    'kengeri': {'cops': 12, 'barricades': 15},
    'hebbala': {'cops': 18, 'barricades': 22},
    'unknown': {'cops': 8, 'barricades': 10}
}
recommendation_engine.STATION_SUPPLY_POOLS = STATION_SUPPLY_POOLS

# Load physical network graph & temporal priors
print("[API] Loading OSM Bengaluru graph map & traffic logs priors matrix...")
G_base = load_bengaluru_graph_with_priors(cache_filename=os.path.join(BASE_DIR, "bengaluru_network.graphml"))
mined_traffic_matrix, fallback_speed, junctions_registry = load_empirical_traffic_priors(os.path.join(BASE_DIR, "road_network_priors.csv"))
print("[API] Network graph loaded successfully.")

# Expose UI Categories dynamically matching historical records
def get_categorical_bounds(col_name, defaults):
    if not historical_df.empty and col_name in historical_df.columns:
        vals = historical_df[col_name].dropna().astype(str).str.strip().unique().tolist()
        return sorted([v for v in vals if v.lower() != 'unknown'])
    return defaults

CATEGORY_LISTS = {
    'event_cause': get_categorical_bounds('event_cause', ['vehicle_breakdown', 'accident', 'tree_fall', 'public_event', 'others']),
    'event_type': ['unplanned', 'planned'],
    'priority': ['High', 'Low'],
    'veh_type': get_categorical_bounds('veh_type', ['heavy_vehicle', 'bmtc_bus', 'private_bus', 'lcv', 'car', 'unknown']),
    'requires_road_closure': ['TRUE', 'FALSE'],
    'police_station': get_categorical_bounds('police_station', ['peenya', 'hsr layout', 'wilson garden', 'sadashivanagar', 'cubbon park', 'kengeri', 'hebbala']),
    'corridor': get_categorical_bounds('corridor', ['tumkur road', 'orr east 1', 'cbd 2', 'non-corridor'])
}

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/meta_data', methods=['GET'])
def get_metadata():
    return jsonify({
        'categories': CATEGORY_LISTS,
        'junctions': junctions_registry,
        'station_pools': STATION_SUPPLY_POOLS,
        'diagnostics': {
            'mined_segments_count': len(mined_traffic_matrix),
            'fallback_speed_kph': fallback_speed,
            'graph_nodes_count': len(G_base.nodes),
            'graph_edges_count': len(G_base.edges)
        }
    })

@app.route('/api/historical_heatmap', methods=['GET'])
def get_heatmap_points():
    if historical_df.empty:
        return jsonify([])
    # Sample subset to prevent front-end browser lag
    sample_size = min(3000, len(historical_df))
    subset = historical_df.sample(n=sample_size)[['latitude', 'longitude']].dropna()
    coords = [{"lat": float(r['latitude']), "lng": float(r['longitude'])} for _, r in subset.iterrows()]
    return jsonify(coords)

@app.route('/api/predict_incident', methods=['POST'])
def predict_single():
    data = request.json
    try:
        # Build mock record
        incident_row = {
            'id': data.get('id', 'INC_01'),
            'start_datetime': data.get('start_datetime', datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S+00')),
            'latitude': float(data['latitude']),
            'longitude': float(data['longitude']),
            'requires_road_closure': data.get('requires_road_closure', 'FALSE'),
            'priority': data.get('priority', 'Low'),
            'veh_type': data.get('veh_type', 'car'),
            'corridor': data.get('corridor', 'non-corridor'),
            'event_cause': data.get('event_cause', 'others'),
            'event_type': data.get('event_type', 'unplanned'),
            'zone': data.get('zone', 'unknown'),
            'police_station': data.get('police_station', 'unknown'),
            'junction': data.get('junction', 'unknown')
        }
        res = process_single_incident(incident_row, historical_df)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/optimize_queue', methods=['POST'])
def optimize_queue():
    data = request.json  # List of incident payloads
    if not data or len(data) == 0:
        return jsonify([])
    try:
        queue_df = pd.DataFrame(data)
        # Ensure correct coordinates data types
        queue_df['latitude'] = queue_df['latitude'].astype(float)
        queue_df['longitude'] = queue_df['longitude'].astype(float)
        
        result_table = run_integrated_traffic_pipeline(queue_df, historical_df)
        
        # Format results without emojis in labels
        records = result_table.to_dict('records')
        formatted = []
        for r in records:
            # Strip emojis from status string
            status_clean = r['Status'].replace('✅ ', '').replace('⚠️ ', '')
            formatted.append({
                'id': r['Incident'],
                'impact': r['Impact'],
                'demanded_cops': int(r['Demanded Cops']),
                'allocated_cops': int(r['Allocated Cops']),
                'demanded_barricades': int(r['Demanded Barricades']),
                'allocated_barricades': int(r['Allocated Barricades']),
                'signs': int(r['Signs']),
                'status': status_clean
            })
        return jsonify(formatted)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/station_config', methods=['GET', 'POST'])
def handle_station_config():
    global STATION_SUPPLY_POOLS
    if request.method == 'POST':
        data = request.json  # New station bounds dict
        if data:
            STATION_SUPPLY_POOLS.update(data)
            recommendation_engine.STATION_SUPPLY_POOLS = STATION_SUPPLY_POOLS
            return jsonify({"status": "applied", "pools": STATION_SUPPLY_POOLS})
    return jsonify(STATION_SUPPLY_POOLS)

@app.route('/api/find_route', methods=['POST'])
def calculate_routes():
    data = request.json
    try:
        orig_lat = float(data['orig_lat'])
        orig_lon = float(data['orig_lon'])
        dest_lat = float(data['dest_lat'])
        dest_lon = float(data['dest_lon'])
        sim_hour = int(data.get('sim_hour', 18))
        sim_day = int(data.get('sim_day', 0))
        
        active_inc = data.get('active_incident') # Optional active incident dictionary
        active_inc_df = None
        if active_inc:
            active_inc_df = pd.DataFrame([active_inc])
            active_inc_df['latitude'] = active_inc_df['latitude'].astype(float)
            active_inc_df['longitude'] = active_inc_df['longitude'].astype(float)

        # Calculate dynamic edge penalties under simulation hour/day and incident shockwaves
        G_dynamic = compute_dynamic_network_states(
            G_base, 
            current_hour=sim_hour, 
            current_day=sim_day, 
            active_incident_df=active_inc_df,
            prior_traffic_matrix=mined_traffic_matrix,
            global_speed_fallback=fallback_speed,
            junctions_registry=junctions_registry
        )
        
        comparison_df, path_dijkstra, path_astar = compare_scenarios(
            G_dynamic, orig_lat, orig_lon, dest_lat, dest_lon
        )
        
        # Convert path node lists into coordinate polyline streams
        G_unproj = ox.project_graph(G_dynamic, to_crs="EPSG:4326")
        dijkstra_coords = [[float(G_unproj.nodes[n]['y']), float(G_unproj.nodes[n]['x'])] for n in path_dijkstra]
        astar_coords = [[float(G_unproj.nodes[n]['y']), float(G_unproj.nodes[n]['x'])] for n in path_astar]
        
        # Parse comparison metrics table and strip emojis
        metrics = []
        for _, row in comparison_df.iterrows():
            metric_name = str(row['Operational Metric Report'])
            dij_val = str(row['🔵 Dijkstra (Optimal)'])
            ast_val = str(row['🟢 A* (Heuristic)'])
            metrics.append({
                'report': metric_name,
                'dijkstra': dij_val,
                'astar': ast_val
            })
            
        return jsonify({
            'dijkstra_route': dijkstra_coords,
            'astar_route': astar_coords,
            'metrics_table': metrics
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
