import pandas as pd
import numpy as np
import os

# Define output directory
OUTPUT_DIR = "processed_weather_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_wbi_rsm(df):
    """Computes Water Balance Index (WBI) and Relative Soil Moisture (RSM) while keeping valid_time format intact."""
    
    # Extract year for grouping without altering valid_time format
    df["year"] = df["valid_time"].astype(str).str[:4].astype(int)

    # Compute Water Balance Index (WBI)
    df["wbi"] = df["tp"] - df["ro"] - df["pev"]

    # Compute yearly max(VSWC3, VSWC4) for RSM
    yearly_max = df.groupby("year")[["swvl3", "swvl4"]].max().max(axis=1).to_dict()

    # Calculate RSM
    df["rsm"] = df.apply(
        lambda row: max(row["swvl3"], row["swvl4"]) / yearly_max[row["year"]]
        if yearly_max[row["year"]] != 0 else 0,
        axis=1
    )

    # Drop unnecessary columns
    return df.drop(columns=["ro", "pev", "year"])

def process_weather_data(file_path):
    """Processes a weather dataset while maintaining valid_time format."""

    # Load the dataset
    df = pd.read_csv(file_path, dtype={"valid_time": str})  # Ensure valid_time stays as string
    
    # Convert temperature from Kelvin to Celsius
    df['t2m_C'] = df['t2m'] - 273.15

    # Compute wind speed
    df['windSpeed'] = np.sqrt(df['u10']**2 + df['v10']**2)

    # Compute relative humidity using August-Roche-Magnus approximation
    A, B = 17.625, 243.04  # Constants for the Magnus equation
    alpha = ((A * df['d2m']) / (B + df['d2m'])) - ((A * df['t2m']) / (B + df['t2m']))
    df['relativeHumidity'] = np.exp(alpha) * 100

    # Drop unnecessary columns
    df.drop(columns=['d2m', 'u10', 'v10', 't2m'], inplace=True)

    # Compute WBI & RSM
    df = compute_wbi_rsm(df)

    # Drop 'number' and 'expver' columns if they exist
    df.drop(columns=['number', 'expver'], errors='ignore', inplace=True)

    # Save the processed file with valid_time intact
    output_file = os.path.join(OUTPUT_DIR, os.path.basename(file_path).replace(".csv", "_processed.csv"))
    df.to_csv(output_file, index=False)
    print(f"Processed file saved: {output_file}")

# Process multiple files
def process_multiple_files(file_paths):
    for file_path in file_paths:
        process_weather_data(file_path)

# Example usage with actual file paths
input_files = [
    "ap.csv",
    "assam.csv",
    "bihar.csv",
    "chhattisgarh.csv",
    "gujarat.csv",
    "haryana.csv",
    "jharkhand.csv",
    "karnataka.csv",
    "maharashtra.csv",
    "mp.csv",
    "odhisa.csv",
    "punjab.csv",
    "rajasthan.csv",
    "tamilnadu.csv",
    "up.csv",
    "wb.csv"
]

process_multiple_files(input_files)
