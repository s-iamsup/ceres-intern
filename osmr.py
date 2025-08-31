import pandas as pd
import requests
from datetime import datetime

# === CONFIGURATION ===
input_excel =   # Replace with your file path
output_excel = 

# === Load Excel File ===
df = pd.read_excel(input_excel)
df["Actual_StartTime"] = pd.to_datetime(df["Actual_StartTime"], errors='coerce')
df["Actual_EndTime"] = pd.to_datetime(df["Actual_EndTime"], errors='coerce')
df["Actual_TravelTime_min"] = (df["Actual_EndTime"] - df["Actual_StartTime"]).dt.total_seconds() / 60

# === OSRM Route Function ===
def get_osrm_route(lat1, lon1, lat2, lon2):
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            route = data['routes'][0]
            distance_km = round(route['distance'] / 1000, 2)
            duration_min = round(route['duration'] / 60, 2)
            return distance_km, duration_min
    except Exception as e:
        print(f"[ERROR] OSRM route failed: {e}")
    return None, None

# === Run Routing for Each Trip ===
routed_distances = []
routed_times = []

for idx, row in df.iterrows():
    dist_km, time_min = get_osrm_route(
        row["Load_Lat"], row["Load_Lon"],
        row["Dump_Lat"], row["Dump_Lon"]
    )
    routed_distances.append(dist_km)
    routed_times.append(time_min)

df["Routed_Distance_km"] = routed_distances
df["Routed_TravelTime_min"] = routed_times
df["Time_Efficiency_%"] = round((df["Routed_TravelTime_min"] / df["Actual_TravelTime_min"]) * 100, 2)

# === Save to Excel ===
df.to_excel(output_excel, index=False)
print(f"✅ Output saved to: {output_excel}")

