import kagglehub
import shutil
import os

# ---------------------------------------------------------
# STEP 1: DOWNLOAD FROM KAGGLE
# ---------------------------------------------------------
# The data is first downloaded to a hidden system cache folder
print("Starting download from Kaggle... (This may take a moment)")
path = kagglehub.dataset_download("vinayak123tyagi/bearing-dataset")

print(f"Data downloaded to cache at: {path}")

# ---------------------------------------------------------
# STEP 2: DEFINE DESTINATION
# ---------------------------------------------------------
# We want the data inside your current VS Code project folder
current_folder = os.getcwd()
destination_dir = os.path.join(current_folder, "raw_data")

# ---------------------------------------------------------
# STEP 3: COPY TO PROJECT FOLDER
# ---------------------------------------------------------
print(f"Copying data to: {destination_dir} ...")

# shutil.copytree copies the entire folder structure.
# dirs_exist_ok=True ensures it doesn't crash if you run the script twice.
shutil.copytree(path, destination_dir, dirs_exist_ok=True)

print("------------------------------------------------")
print("✅ SUCCESS! You should now see a new 'raw_data' folder in your file explorer.")
print("   Inside, you will find folders like '1st_test', '2nd_test', etc.")