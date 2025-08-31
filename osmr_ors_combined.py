import pandas as pd
import requests
import time
from datetime import datetime
import openrouteservice
import statsmodels.api as sm # Import statsmodels for detailed regression output
import statsmodels.formula.api as smf # For formula-based OLS
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # Keep for consistency in MAE/MSE if needed, though statsmodels provides similar
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os # Import os module for path manipulation

# === CONFIGURATION ===
# IMPORTANT: Replace with your actual file paths
INPUT_EXCEL_PATH = 
OUTPUT_EXCEL_PATH = 

# IMPORTANT: Replace with your OpenRouteService API key
# Get one from: https://openrouteservice.org/sign-up/
OPENROUTESERVICE_API_KEY =  # <--- ENSURE THIS IS YOUR VALID API KEY!

# Additional time adjustments for round trip
LOADING_DUMPING_TIME_MIN = 60 # minutes to add for loading and dumping
BUFFER_PERCENTAGE = 0.10    # 10% buffer (0.10 for 10%)

# Initialize OpenRouteService client
try:
    ors_client = openrouteservice.Client(key=OPENROUTESERVICE_API_KEY)
    if OPENROUTESERVICE_API_KEY ==  or not OPENROUTESERVICE_API_KEY:
        print("[WARNING] OpenRouteService API Key is not set or is the default placeholder. ORS calls will likely fail with 403 errors.")
except Exception as e:
    print(f"[ERROR] Failed to initialize OpenRouteService client. Please check your API key: {e}")
    ors_client = None # Set to None if initialization fails

# === Load Excel File and Initial Data Processing ===
print(f"Loading data from: {INPUT_EXCEL_PATH}")
try:
    df = pd.read_excel(INPUT_EXCEL_PATH, engine="openpyxl")
    # Convert time columns to datetime objects, coercing errors to NaT (Not a Time)
    df["Actual_StartTime"] = pd.to_datetime(df["Actual_StartTime"], errors='coerce')
    df["Actual_EndTime"] = pd.to_datetime(df["Actual_EndTime"], errors='coerce')

    # Calculate Actual_TravelTime_min
    # Ensure both start and end times are valid before calculating
    valid_times_mask = df["Actual_StartTime"].notna() & df["Actual_EndTime"].notna()
    df.loc[valid_times_mask, "Actual_TravelTime_min"] = (
        (df.loc[valid_times_mask, "Actual_EndTime"] - df.loc[valid_times_mask, "Actual_StartTime"]).dt.total_seconds() / 60
    )
    df["Actual_TravelTime_min"] = df["Actual_TravelTime_min"].round(2)
    print("Initial data loaded and 'Actual_TravelTime_min' calculated.")

except FileNotFoundError:
    print(f"[ERROR] Input Excel file not found at: {INPUT_EXCEL_PATH}")
    exit()
except Exception as e:
    print(f"[ERROR] Error loading or processing Excel file: {e}")
    exit()

# === OSRM Routing Function (One-way and Round Trip) ===
def get_osrm_route(lat1, lon1, lat2, lon2):
    """
    Fetches one-way and round-trip distance and duration from OSRM public demo server.
    """
    base_url = "http://router.project-osrm.org/route/v1/driving/"
    one_way_dist_km, one_way_duration_min = None, None
    round_trip_dist_km, round_trip_duration_min = None, None

    # One-way trip (A to B)
    url_ab = f"{base_url}{lon1},{lat1};{lon2},{lat2}?overview=false"
    # print(f"DEBUG: OSRM One-way URL: {url_ab}") # Added for debugging OSRM 400 error
    try:
        response_ab = requests.get(url_ab, timeout=10) # Add timeout
        response_ab.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data_ab = response_ab.json()
        if data_ab and 'routes' in data_ab and len(data_ab['routes']) > 0:
            route_ab = data_ab['routes'][0]
            one_way_dist_km = round(route_ab['distance'] / 1000, 2)
            one_way_duration_min = round(route_ab['duration'] / 60, 2)
    except requests.exceptions.RequestException as e:
        print(f"[OSRM ERROR] One-way route (A->B) failed for ({lat1},{lon1}) to ({lat2},{lon2}): {e}")
    except Exception as e:
        print(f"[OSRM ERROR] One-way route (A->B) parsing error: {e}")

    # Round trip (A to B and B to A)
    if one_way_dist_km is not None and one_way_duration_min is not None:
        url_ba = f"{base_url}{lon2},{lat2};{lon1},{lat1}?overview=false"
        # print(f"DEBUG: OSRM Return URL: {url_ba}") # Added for debugging OSRM 400 error
        try:
            response_ba = requests.get(url_ba, timeout=10) # Add timeout
            response_ba.raise_for_status()
            data_ba = response_ba.json()
            if data_ba and 'routes' in data_ba and len(data_ba['routes']) > 0:
                route_ba = data_ba['routes'][0]
                round_trip_dist_km = round(one_way_dist_km + (route_ba['distance'] / 1000), 2)
                round_trip_duration_min = round(one_way_duration_min + (route_ba['duration'] / 60), 2)
        except requests.exceptions.RequestException as e:
            print(f"[OSRM ERROR] Return route (B->A) failed for ({lat2},{lon2}) to ({lat1},{lon1}): {e}")
        except Exception as e:
            print(f"[OSRM ERROR] Return route (B->A) parsing error: {e}")

    time.sleep(0.1) # Small delay to be polite to the public server
    return one_way_dist_km, one_way_duration_min, round_trip_dist_km, round_trip_duration_min

# === OpenRouteService Routing Function (One-way and Round Trip) ===
def get_ors_route_info(load_lat, load_lon, dump_lat, dump_lon):
    """
    Fetches one-way and round-trip distance and duration from OpenRouteService.
    Requires an initialized ors_client.
    """
    if ors_client is None:
        return None, None, None, None # Return None if client not initialized

    one_way_dist_km, one_way_duration_min = None, None
    round_trip_dist_km, round_trip_duration_min = None, None

    try:
        # One-way trip (A to B)
        coords_ab = ((load_lon, load_lat), (dump_lon, dump_lat))
        route_ab = ors_client.directions(coords_ab)
        if route_ab and 'routes' in route_ab and len(route_ab['routes']) > 0:
            summary_ab = route_ab['routes'][0]['summary']
            one_way_dist_km = round(summary_ab['distance'] / 1000, 2)
            one_way_duration_min = round(summary_ab['duration'] / 60, 2)

        # Round trip (A to B and B to A)
        if one_way_dist_km is not None and one_way_duration_min is not None:
            coords_ba = ((dump_lon, dump_lat), (load_lon, load_lat)) # Reversed coordinates for return trip
            route_ba = ors_client.directions(coords_ba)
            if route_ba and 'routes' in route_ba and len(route_ba['routes']) > 0:
                summary_ba = route_ba['routes'][0]['summary']
                round_trip_dist_km = round(one_way_dist_km + (summary_ba['distance'] / 1000), 2)
                round_trip_duration_min = round(one_way_duration_min + (summary_ba['duration'] / 60), 2)

    except openrouteservice.exceptions.ApiError as e:
        print(f"[ORS ERROR] API error for coordinates ({load_lat},{load_lon}) to ({dump_lat},{dump_lon}): {e}")
    except Exception as e:
        print(f"[ORS ERROR] Unexpected error for coordinates ({load_lat},{load_lon}) to ({dump_lat},{dump_lon}): {e}")
    finally:
        time.sleep(1) # Recommended delay for OpenRouteService to respect rate limits
    return one_way_dist_km, one_way_duration_min, round_trip_dist_km, round_trip_duration_min

# === Run Routing for Each Trip ===
print("\nStarting routing process...")
osrm_distances = []
osrm_times = []
osrm_round_distances = []
osrm_round_times = []

ors_distances = []
ors_times = []
ors_round_distances = []
ors_round_times = []

for idx, row in df.iterrows():
    print(f"Processing row {idx + 1}/{len(df)}")

    # OSRM Routing
    osrm_dist_km, osrm_time_min, osrm_r_dist_km, osrm_r_time_min = get_osrm_route(
        row["Load_Lat"], row["Load_Lon"],
        row["Dump_Lat"], row["Dump_Lon"]
    )
    osrm_distances.append(osrm_dist_km)
    osrm_times.append(osrm_time_min)
    osrm_round_distances.append(osrm_r_dist_km)
    osrm_round_times.append(osrm_r_time_min)

    # OpenRouteService Routing
    ors_dist_km, ors_time_min, ors_r_dist_km, ors_r_time_min = get_ors_route_info(
        row["Load_Lat"], row["Load_Lon"],
        row["Dump_Lat"], row["Dump_Lon"]
    )
    ors_distances.append(ors_dist_km)
    ors_times.append(ors_time_min)
    ors_round_distances.append(ors_r_dist_km)
    ors_round_times.append(ors_r_time_min)

# Add routing results to DataFrame
df["OSRM_Distance_km"] = osrm_distances
df["OSRM_TravelTime_min"] = osrm_times
df["OSRM_RoundTrip_Distance_km"] = osrm_round_distances
df["OSRM_RoundTrip_TravelTime_min"] = osrm_round_times

df["ORS_Distance_km"] = ors_distances
df["ORS_TravelTime_min"] = ors_times
df["ORS_RoundTrip_Distance_km"] = ors_round_distances
df["ORS_RoundTrip_TravelTime_min"] = ors_round_times

# === Apply Loading/Dumping Time and Buffer to Round Trip Times ===
# Convert to numeric, coercing errors to NaN, then fill NaN with 0 for calculations
df["OSRM_RoundTrip_TravelTime_min"] = pd.to_numeric(df["OSRM_RoundTrip_TravelTime_min"], errors='coerce').fillna(0)
df["ORS_RoundTrip_TravelTime_min"] = pd.to_numeric(df["ORS_RoundTrip_TravelTime_min"], errors='coerce').fillna(0)

df["OSRM_Adjusted_RoundTrip_TravelTime_min"] = (
    (df["OSRM_RoundTrip_TravelTime_min"] + LOADING_DUMPING_TIME_MIN) * (1 + BUFFER_PERCENTAGE)
).round(2)

df["ORS_Adjusted_RoundTrip_TravelTime_min"] = (
    (df["ORS_RoundTrip_TravelTime_min"] + LOADING_DUMPING_TIME_MIN) * (1 + BUFFER_PERCENTAGE)
).round(2)

# Calculate Time Efficiency for both services - Using pandas Series methods for robust NaN handling
df["OSRM_Time_Efficiency_%"] = (df["OSRM_TravelTime_min"] / df["Actual_TravelTime_min"] * 100).round(2)
df["ORS_Time_Efficiency_%"] = (df["ORS_TravelTime_min"] / df["Actual_TravelTime_min"] * 100).round(2)

print("\nRouting complete. Starting regression analysis.")

# === Linear Regression Analysis with Statsmodels ===
# Prepare data for regression
# We'll predict Actual_TravelTime_min using OSRM_Adjusted_RoundTrip_TravelTime_min and ORS_Adjusted_RoundTrip_TravelTime_min
# Filter out rows with NaN values in the relevant columns for regression
regression_df = df.dropna(subset=[
    "Actual_TravelTime_min",
    "OSRM_Adjusted_RoundTrip_TravelTime_min", # Changed for adjusted round trip time
    "ORS_Adjusted_RoundTrip_TravelTime_min"   # Changed for adjusted round trip time
]).copy()

# Also filter out zero or negative travel times, as they are not meaningful for regression
regression_df = regression_df[
    (regression_df["Actual_TravelTime_min"] > 0) &
    (regression_df["OSRM_Adjusted_RoundTrip_TravelTime_min"] > 0) & # Changed for adjusted round trip time
    (regression_df["ORS_Adjusted_RoundTrip_TravelTime_min"] > 0)   # Changed for adjusted round trip time
]

# Initialize a list to hold regression summary dataframes
regression_output_dfs = {}

if regression_df.empty:
    print("[WARNING] No valid data points for regression after cleaning. Skipping regression analysis and plot generation.")
else:
    # --- Regression for OSRM Adjusted Round Trip Time vs Actual One-Way Time ---
    # Add a constant to the independent variable for the intercept calculation in statsmodels
    X_osrm_adjusted_round = sm.add_constant(regression_df["OSRM_Adjusted_RoundTrip_TravelTime_min"]) # Changed to adjusted
    y_actual = regression_df["Actual_TravelTime_min"]

    model_osrm_adjusted_round = sm.OLS(y_actual, X_osrm_adjusted_round).fit()
    y_pred_osrm_adjusted_round = model_osrm_adjusted_round.predict(X_osrm_adjusted_round)

    print("\n--- OSRM Adjusted Round Trip Travel Time vs Actual One-Way Travel Time Regression (Statsmodels Summary) ---")
    print(model_osrm_adjusted_round.summary())

    # Extracting data for Excel output from statsmodels summary
    # Regression Statistics
    reg_stats_osrm = pd.DataFrame({
        "Metric": ["Multiple R", "R Square", "Adjusted R Square", "Standard Error", "Observations"],
        "Value": [
            np.sqrt(model_osrm_adjusted_round.rsquared), # Multiple R is sqrt of R-squared
            model_osrm_adjusted_round.rsquared,
            model_osrm_adjusted_round.rsquared_adj,
            model_osrm_adjusted_round.bse[1], # Standard Error of the coefficient
            model_osrm_adjusted_round.nobs
        ]
    })
    reg_stats_osrm.set_index("Metric", inplace=True)
    regression_output_dfs['OSRM_Adjusted_Regression_Stats'] = reg_stats_osrm

    # ANOVA Table
    anova_osrm = pd.DataFrame({
        "df": [model_osrm_adjusted_round.df_model, model_osrm_adjusted_round.df_resid, model_osrm_adjusted_round.df_model + model_osrm_adjusted_round.df_resid],
        "SS": [model_osrm_adjusted_round.ess, model_osrm_adjusted_round.ssr, model_osrm_adjusted_round.centered_tss],
        "MS": [model_osrm_adjusted_round.ess / model_osrm_adjusted_round.df_model, model_osrm_adjusted_round.ssr / model_osrm_adjusted_round.df_resid, np.nan],
        "F": [model_osrm_adjusted_round.fvalue, np.nan, np.nan],
        "Significance F": [model_osrm_adjusted_round.f_pvalue, np.nan, np.nan]
    }, index=["Regression", "Residual", "Total"])
    regression_output_dfs['OSRM_Adjusted_ANOVA'] = anova_osrm

    # Coefficients Table
    coeffs_osrm = model_osrm_adjusted_round.summary2().tables[1] # Access coefficients table
    coeffs_osrm.columns = ['Coefficients', 'Standard Error', 't Stat', 'P-value', 'Lower 95%', 'Upper 95%'] # Rename columns
    regression_output_dfs['OSRM_Adjusted_Coefficients'] = coeffs_osrm

    # --- Regression for OpenRouteService Adjusted Round Trip Time vs Actual One-Way Time ---
    X_ors_adjusted_round = sm.add_constant(regression_df["ORS_Adjusted_RoundTrip_TravelTime_min"]) # Changed to adjusted
    model_ors_adjusted_round = sm.OLS(y_actual, X_ors_adjusted_round).fit()
    y_pred_ors_adjusted_round = model_ors_adjusted_round.predict(X_ors_adjusted_round)

    print("\n--- OpenRouteService Adjusted Round Trip Travel Time vs Actual One-Way Travel Time Regression (Statsmodels Summary) ---")
    print(model_ors_adjusted_round.summary())

    # Extracting data for Excel output from statsmodels summary
    # Regression Statistics
    reg_stats_ors = pd.DataFrame({
        "Metric": ["Multiple R", "R Square", "Adjusted R Square", "Standard Error", "Observations"],
        "Value": [
            np.sqrt(model_ors_adjusted_round.rsquared),
            model_ors_adjusted_round.rsquared,
            model_ors_adjusted_round.rsquared_adj,
            model_ors_adjusted_round.bse[1],
            model_ors_adjusted_round.nobs
        ]
    })
    reg_stats_ors.set_index("Metric", inplace=True)
    regression_output_dfs['ORS_Adjusted_Regression_Stats'] = reg_stats_ors

    # ANOVA Table
    anova_ors = pd.DataFrame({
        "df": [model_ors_adjusted_round.df_model, model_ors_adjusted_round.df_resid, model_ors_adjusted_round.df_model + model_ors_adjusted_round.df_resid],
        "SS": [model_ors_adjusted_round.ess, model_ors_adjusted_round.ssr, model_ors_adjusted_round.centered_tss],
        "MS": [model_ors_adjusted_round.ess / model_ors_adjusted_round.df_model, model_ors_adjusted_round.ssr / model_ors_adjusted_round.df_resid, np.nan],
        "F": [model_ors_adjusted_round.fvalue, np.nan, np.nan],
        "Significance F": [model_ors_adjusted_round.f_pvalue, np.nan, np.nan]
    }, index=["Regression", "Residual", "Total"])
    regression_output_dfs['ORS_Adjusted_ANOVA'] = anova_ors

    # Coefficients Table
    coeffs_ors = model_ors_adjusted_round.summary2().tables[1] # Access coefficients table
    coeffs_ors.columns = ['Coefficients', 'Standard Error', 't Stat', 'P-value', 'Lower 95%', 'Upper 95%'] # Rename columns
    regression_output_dfs['ORS_Adjusted_Coefficients'] = coeffs_ors


    # === Plotting Regression Chart and Saving as Images ===
    output_dir = os.path.dirname(OUTPUT_EXCEL_PATH)
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    osrm_plot_path = os.path.join(output_dir, "osrm_regression_plot_adjusted_round_trip.png") # Changed plot filename
    ors_plot_path = os.path.join(output_dir, "ors_regression_plot_adjusted_round_trip.png")   # Changed plot filename

    plt.figure(figsize=(14, 7))

    # OSRM Plot
    plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
    sns.scatterplot(x="OSRM_Adjusted_RoundTrip_TravelTime_min", y="Actual_TravelTime_min", data=regression_df, alpha=0.6) # Changed x-axis
    plt.plot(regression_df["OSRM_Adjusted_RoundTrip_TravelTime_min"], y_pred_osrm_adjusted_round, color='red', linestyle='--', label=f'Regression Line (R²={model_osrm_adjusted_round.rsquared:.2f})')
    plt.title('OSRM Adjusted Round Trip Predicted vs Actual Travel Time') # Changed title
    plt.xlabel('OSRM Adjusted Round Trip Predicted Travel Time (min)') # Changed x-label
    plt.ylabel('Actual One-Way Travel Time (min)') # Clarified y-label
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # OpenRouteService Plot
    plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
    sns.scatterplot(x="ORS_Adjusted_RoundTrip_TravelTime_min", y="Actual_TravelTime_min", data=regression_df, alpha=0.6, color='green') # Changed x-axis
    plt.plot(regression_df["ORS_Adjusted_RoundTrip_TravelTime_min"], y_pred_ors_adjusted_round, color='blue', linestyle='--', label=f'Regression Line (R²={model_ors_adjusted_round.rsquared:.2f})')
    plt.title('OpenRouteService Adjusted Round Trip Predicted vs Actual Travel Time') # Changed title
    plt.xlabel('OpenRouteService Adjusted Round Trip Predicted Travel Time (min)') # Changed x-label
    plt.ylabel('Actual One-Way Travel Time (min)') # Clarified y-label
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    plt.suptitle('Adjusted Round Trip Travel Time Regression Analysis', fontsize=16, y=1.02) # Changed suptitle
    plt.savefig(os.path.join(output_dir, "combined_regression_plots_adjusted_round_trip.png")) # Save combined plot
    plt.close() # Close the plot to free up memory

    print(f"✅ Combined regression plots saved to: {os.path.join(output_dir, 'combined_regression_plots_adjusted_round_trip.png')}")

# === Save to Excel with Multiple Sheets ===
print(f"\nSaving processed data and regression summary to: {OUTPUT_EXCEL_PATH}")
try:
    with pd.ExcelWriter(OUTPUT_EXCEL_PATH, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Routing_Data', index=False)
        print("Sheet 'Routing_Data' saved.")

        if regression_output_dfs: # Check if regression output DataFrames were created
            # Write OSRM Regression Details
            regression_output_dfs['OSRM_Adjusted_Regression_Stats'].to_excel(writer, sheet_name='Regression_Summary', startrow=0, startcol=0, header=True)
            writer.sheets['Regression_Summary'].cell(row=1, column=1).value = "OSRM Adjusted Regression Statistics"

            regression_output_dfs['OSRM_Adjusted_ANOVA'].to_excel(writer, sheet_name='Regression_Summary', startrow=regression_output_dfs['OSRM_Adjusted_Regression_Stats'].shape[0] + 3, startcol=0, header=True)
            writer.sheets['Regression_Summary'].cell(row=regression_output_dfs['OSRM_Adjusted_Regression_Stats'].shape[0] + 4, column=1).value = "OSRM Adjusted ANOVA"

            regression_output_dfs['OSRM_Adjusted_Coefficients'].to_excel(writer, sheet_name='Regression_Summary', startrow=regression_output_dfs['OSRM_Adjusted_Regression_Stats'].shape[0] + regression_output_dfs['OSRM_Adjusted_ANOVA'].shape[0] + 6, startcol=0, header=True)
            writer.sheets['Regression_Summary'].cell(row=regression_output_dfs['OSRM_Adjusted_Regression_Stats'].shape[0] + regression_output_dfs['OSRM_Adjusted_ANOVA'].shape[0] + 7, column=1).value = "OSRM Adjusted Coefficients"


            # Write ORS Regression Details (offset by enough rows)
            ors_start_row = regression_output_dfs['OSRM_Adjusted_Regression_Stats'].shape[0] + \
                            regression_output_dfs['OSRM_Adjusted_ANOVA'].shape[0] + \
                            regression_output_dfs['OSRM_Adjusted_Coefficients'].shape[0] + 10 # Sufficient offset

            regression_output_dfs['ORS_Adjusted_Regression_Stats'].to_excel(writer, sheet_name='Regression_Summary', startrow=ors_start_row, startcol=0, header=True)
            writer.sheets['Regression_Summary'].cell(row=ors_start_row + 1, column=1).value = "ORS Adjusted Regression Statistics"

            regression_output_dfs['ORS_Adjusted_ANOVA'].to_excel(writer, sheet_name='Regression_Summary', startrow=ors_start_row + regression_output_dfs['ORS_Adjusted_Regression_Stats'].shape[0] + 3, startcol=0, header=True)
            writer.sheets['Regression_Summary'].cell(row=ors_start_row + regression_output_dfs['ORS_Adjusted_Regression_Stats'].shape[0] + 4, column=1).value = "ORS Adjusted ANOVA"

            regression_output_dfs['ORS_Adjusted_Coefficients'].to_excel(writer, sheet_name='Regression_Summary', startrow=ors_start_row + regression_output_dfs['ORS_Adjusted_Regression_Stats'].shape[0] + regression_output_dfs['ORS_Adjusted_ANOVA'].shape[0] + 6, startcol=0, header=True)
            writer.sheets['Regression_Summary'].cell(row=ors_start_row + regression_output_dfs['ORS_Adjusted_Regression_Stats'].shape[0] + regression_output_dfs['ORS_Adjusted_ANOVA'].shape[0] + 7, column=1).value = "ORS Adjusted Coefficients"

            print("Sheet 'Regression_Summary' saved with detailed regression output.")
        else:
            print("[INFO] No regression summary sheet saved as no valid data for regression was found.")
    print(f"✅ Output Excel file saved successfully to: {OUTPUT_EXCEL_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to save output Excel file: {e}")

print("\nScript execution finished.")
