import pandas as pd
import os

# Create output directory
output_dir = "seasonal_outputs"
os.makedirs(output_dir, exist_ok=True)

def process_file(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Define feature groups
    sum_features = ["swvl3", "swvl4", "tp"]
    avg_features = [col for col in df.columns if col not in sum_features + ["valid_time", "latitude", "longitude"]]
    
    # Extract year and month from valid_time
    df["year"] = df["valid_time"].astype(str).str[:4].astype(int)
    df["month"] = df["valid_time"].astype(str).str[4:6].astype(int)
    
    # Compute Kharif season (June - October of each year)
    kharif = df[df["month"].between(6, 10)].groupby("year").agg({
        **{col: "mean" for col in avg_features},
        **{col: "sum" for col in sum_features}
    }).reset_index()
    kharif["season"] = "Kharif"
    kharif["year_span"] = kharif["year"].astype(str) + "-" + (kharif["year"] + 1).astype(str)
    kharif.drop(columns=["year"], inplace=True)
    
    # Compute Rabi season (Oct-Dec of prev year + Jan-Apr of current year)
    rabi_list = []
    for year in df["year"].unique():
        prev_year_data = df[(df["year"] == year - 1) & (df["month"].between(10, 12))]
        curr_year_data = df[(df["year"] == year) & (df["month"].between(1, 4))]
        
        if prev_year_data.empty:
            season_data = curr_year_data  # If prev-year data is missing, take only Jan-Apr
        else:
            season_data = pd.concat([prev_year_data, curr_year_data])
        
        if not season_data.empty:
            rabi_agg = season_data.agg({
                **{col: "mean" for col in avg_features},
                **{col: "sum" for col in sum_features}
            }).to_frame().T
            rabi_agg["year_span"] = str(year) + "-" + str(year + 1)
            rabi_agg["season"] = "Rabi"
            rabi_list.append(rabi_agg)
    
    rabi = pd.concat(rabi_list, ignore_index=True)
    
    # Combine results
    seasonal_data = pd.concat([kharif, rabi], ignore_index=True)
    seasonal_data.drop(columns=["year", "month"], errors="ignore", inplace=True)
    
    # Save output
    output_file = os.path.join(output_dir, os.path.basename(file_path).replace(".csv", "_seasonal.csv"))
    seasonal_data.to_csv(output_file, index=False)
    print(f"Seasonal data saved to {output_file}")

# Process multiple files
input_files = [
    "ap_processed.csv",
    "assam_processed.csv",
    "bihar_processed.csv",
    "chhattisgarh_processed.csv",
    "gujarat_processed.csv",
    "haryana_processed.csv",
    "jharkhand_processed.csv",
    "karnataka_processed.csv",
    "maharashtra_processed.csv",
    "mp_processed.csv",
    "odhisa_processed.csv",
    "punjab_processed.csv",
    "rajasthan_processed.csv",
    "tamilnadu_processed.csv",
    "up_processed.csv",
    "wb_processed.csv"]  # Add more filenames as needed
for file in input_files:
    process_file(file)
