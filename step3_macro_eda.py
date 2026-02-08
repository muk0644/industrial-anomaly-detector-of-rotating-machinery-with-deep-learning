import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# We load the processed CSV from the 'results' folder
RESULTS_DIR = "results"
INPUT_FILE = os.path.join(RESULTS_DIR, "merged_bearing_data.csv")
IMAGE_NAME = "Macro_EDA_Run_to_Failure.png"

def show_trend():
    print(f"--- STEP 3: MACRO-EDA (7-DAY TREND) ---")
    
    # 1. Load Data
    print(f"   -> Loading data from: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
        
        # Convert the string timestamp back to a Date object
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    except FileNotFoundError:
        print("ERROR: CSV file not found! Did Step 2 finish correctly?")
        return

    print(f"   -> Data loaded. Plotting the life cycle of 4 bearings...")

    # 2. Plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot Bearing 1 (The Failure) in RED
    ax.plot(df.index, df['Bearing 1'], label='Bearing 1 (FAILURE)', color='red', linewidth=1.5)
    
    # Plot the healthy bearings in other colors (thinner lines)
    ax.plot(df.index, df['Bearing 2'], label='Bearing 2', color='green', alpha=0.3, linewidth=1)
    ax.plot(df.index, df['Bearing 3'], label='Bearing 3', color='blue', alpha=0.3, linewidth=1)
    ax.plot(df.index, df['Bearing 4'], label='Bearing 4', color='grey', alpha=0.3, linewidth=1)

    # 3. Formatting the Chart
    ax.set_title("Run-to-Failure Analysis: 7-Day Vibration Trend", fontsize=16, fontweight='bold')
    ax.set_xlabel("Date (Feb 12 - Feb 19, 2004)", fontsize=12)
    ax.set_ylabel("Vibration (Mean Abs Value)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Save the Result
    save_path = os.path.join(RESULTS_DIR, IMAGE_NAME)
    plt.savefig(save_path, dpi=300)
    print(f"   -> ✅ Graph saved to: {save_path}")

    # 5. Show
    plt.show()

if __name__ == "__main__":
    show_trend()