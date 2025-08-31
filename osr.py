import pandas as pd
import openrouteservice
import time

client = openrouteservice.Client(key = #get_it_from_osr_website)

df = pd.read_excel(#insert_file_path, engine="openpyxl")

def get_route_info(load_lat, load_lon, dump_lat, dump_lon):
    try:
        coords = ((load_lon, load_lat), (dump_lon, dump_lat))
        route = client.directions(coords)
        summary = route['routes'][0]['summary']
        distance_km = summary['distance'] / 1000
        duration_min = summary['duration'] / 60
        return distance_km, duration_min
    except Exception as e:
        print(f"Error for coordinates {coords}: {e}")
        return None, None

distances = []
durations = []

for index, row in df.iterrows():
    print(f"Processing row {index + 1}/{len(df)}")
    distance, duration = get_route_info(row['Load_Lat'], row['Load_Lon'], row['Dump_Lat'], row['Dump_Lon'])
    distances.append(distance)
    durations.append(duration)
    time.sleep(2)

df['Routed_Distance_km'] = distances
df['Travel_Time_min'] = durations

df.to_excel("route_time_27_07.xlsx", index=False)
print("✅ Routed distances and travel times saved to 'routed_distances_with_time.xlsx'")
