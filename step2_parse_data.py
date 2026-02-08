import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Path to the folder containing the 984 individual files
RAW_DATA_PATH = os.path.join("raw_data", "2nd_test", "2nd_test")

# MODIFICATION: We now save the CSV inside the 'results' folder
RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "merged_bearing_data.csv")

def parse_raw_data():
    print(f"--- STEP 2: DATA PARSING PIPELINE ---")
    
    # 1. Ensure the results folder exists
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"   -> Created folder: '{RESULTS_DIR}'")

    print(f"   -> Reading files from: {RAW_DATA_PATH}")

    # 2. Get all filenames and sort them
    try:
        filenames = sorted([f for f in os.listdir(RAW_DATA_PATH) if f.startswith("2004")])
    except FileNotFoundError:
        print("ERROR: Raw data folder not found. Check your path!")
        return

    total_files = len(filenames)
    print(f"   -> Found {total_files} files. Starting processing... (This may take 1-2 minutes)")
    
    data_list = []
    
    # 3. Loop through every single file
    for i, filename in enumerate(filenames):
        filepath = os.path.join(RAW_DATA_PATH, filename)
        
        try:
            # Read the raw file (Tab-separated, No header)
            df_temp = pd.read_csv(filepath, sep='\t', header=None)
            
            # Calculate Mean Absolute Value (MAV) for ALL 4 Bearings
            data_row = {
                'timestamp': filename,
                'Bearing 1': np.mean(np.abs(df_temp[0])), # The failing one
                'Bearing 2': np.mean(np.abs(df_temp[1])),
                'Bearing 3': np.mean(np.abs(df_temp[2])),
                'Bearing 4': np.mean(np.abs(df_temp[3]))
            }
            data_list.append(data_row)

        except Exception as e:
            print(f"   Warning: Could not read file {filename}. Skipping.")

        # Show progress every 100 files
        if (i + 1) % 100 == 0:
            print(f"   -> Processed {i + 1}/{total_files} files...")

    # 4. Save to CSV inside the 'results' folder
    print("   -> Combining data into a DataFrame...")
    final_df = pd.DataFrame(data_list)
    
    # Format the timestamp
    final_df['timestamp'] = pd.to_datetime(final_df['timestamp'], format='%Y.%m.%d.%H.%M.%S')
    final_df.set_index('timestamp', inplace=True)
    
    final_df.to_csv(OUTPUT_FILE)
    print(f"--- SUCCESS! ---")
    print(f"   -> Merged data saved to: {OUTPUT_FILE}")
    print(f"   -> You can now proceed to Step 3 (Macro-EDA).")

if __name__ == "__main__":
    parse_raw_data()