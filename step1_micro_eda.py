import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# We select the very first file to see the "Healthy State"
# Adjust path if your folder structure is different
FILE_PATH = os.path.join("raw_data", "2nd_test", "2nd_test", "2004.02.12.10.32.39")

# Location to save the results/images
RESULTS_DIR = "results"
IMAGE_NAME = "Micro_EDA_All_Bearings_Healthy.png"

def show_raw_signal_all_bearings():
    print(f"--- STEP 1: MICRO-EDA (ALL 4 SENSORS) ---")
    
    # 1. Create results folder if it does not exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"   -> Created folder: '{RESULTS_DIR}'")
    
    # 2. Load Data
    print(f"   -> Loading file: {FILE_PATH}")
    
    # The file is tab-separated with no header. 
    # Columns 0, 1, 2, 3 correspond to Bearings 1, 2, 3, 4.
    try:
        df = pd.read_csv(FILE_PATH, sep='\t', header=None)
    except FileNotFoundError:
        print("ERROR: File not found. Please check the RAW_DATA_PATH!")
        return

    print(f"   -> Data loaded successfully. {len(df)} data points per sensor.")

    # 3. Create the Dashboard (4 subplots stacked vertically)
    # sharex=True means they all share the same time axis (zooming one zooms all)
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f"Micro-View: 1 Second of Vibration (All Bearings) - 'Healthy' State", fontsize=16)

    bearings = ["Bearing 1 (Target - Fails later)", "Bearing 2", "Bearing 3", "Bearing 4"]
    colors = ['blue', 'green', 'orange', 'purple']

    for i in range(4):
        ax = axes[i]
        signal = df[i] # Get column i
        
        ax.plot(signal, color=colors[i], linewidth=0.5)
        ax.set_title(bearings[i], fontsize=12, fontweight='bold')
        ax.set_ylabel("Amplitude (Volts)")
        ax.grid(True, alpha=0.3)
        
        # Fix Y-axis limits to make comparison easier
        # (Vibration is low now, so +/- 0.5 is a good range)
        ax.set_ylim(-0.5, 0.5) 

    # Only add the X-axis label to the bottom plot
    axes[3].set_xlabel("Data Points (Samples 0 - 20,480)")

    plt.tight_layout()

    # 4. Save the Result
    save_path = os.path.join(RESULTS_DIR, IMAGE_NAME)
    plt.savefig(save_path, dpi=300) # dpi=300 ensures high resolution
    print(f"   -> ✅ Dashboard saved to: {save_path}")

    # 5. Show Plot
    plt.show()

if __name__ == "__main__":
    show_raw_signal_all_bearings()