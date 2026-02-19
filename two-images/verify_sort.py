import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

def calculate_average_intensity(image_path):
    """Calculates the average pixel intensity of an image."""
    with Image.open(image_path) as img:
        # Convert to grayscale to compute a single average intensity value
        img_gray = img.convert('L')
        return np.mean(img_gray)

def verify_partition_csv(csv_path, folder_path):
    # Read the CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    # Verify that the necessary columns exist
    if 'Image1' not in df.columns or 'Image2' not in df.columns:
        print("Error: The CSV must contain 'Image1' and 'Image2' columns.")
        return

    print(f"Verifying {len(df)} pairs in '{csv_path}'...")
    
    correct_count = 0
    incorrect_count = 0
    equal_count = 0
    missing_files_count = 0
    
    # Dictionary to cache intensities and avoid redundant processing
    intensity_cache = {}

    for index, row in tqdm(df.iterrows(), total=len(df)):
        img1_name = row['Image1']
        img2_name = row['Image2']
        
        img1_path = os.path.join(folder_path, img1_name)
        img2_path = os.path.join(folder_path, img2_name)
        
        # Check if both image files actually exist
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            missing_files_count += 1
            continue
            
        # Calculate or get cached intensity for Image 1 (Added image -> should be darker)
        if img1_name not in intensity_cache:
            intensity_cache[img1_name] = calculate_average_intensity(img1_path)
        int1 = intensity_cache[img1_name]
        
        # Calculate or get cached intensity for Image 2 (Original image -> should be brighter)
        if img2_name not in intensity_cache:
            intensity_cache[img2_name] = calculate_average_intensity(img2_path)
        int2 = intensity_cache[img2_name]
        
        # Verify the sorting logic (Image2 > Image1)
        if int2 > int1:
            correct_count += 1
        elif int2 < int1:
            incorrect_count += 1
        else:
            equal_count += 1

    # Print the final report
    total_checked = correct_count + incorrect_count + equal_count
    
    print("\n" + "="*40)
    print("           VERIFICATION REPORT")
    print("="*40)
    print(f"Total rows in CSV:           {len(df)}")
    if missing_files_count > 0:
        print(f"Skipped (missing images):    {missing_files_count}")
    print(f"Total pairs verified:        {total_checked}")
    print("-" * 40)
    print(f"✅ Correctly sorted:          {correct_count} (Image2 > Image1)")
    print(f"❌ Incorrectly sorted:        {incorrect_count} (Image1 > Image2)")
    print(f"⚠️  Equal intensity:           {equal_count} (Image1 == Image2)")
    print("="*40)
    
    if incorrect_count == 0 and equal_count == 0 and total_checked > 0:
        print("\nSUCCESS: All pairs are perfectly sorted according to the paper's criteria!")
    elif total_checked > 0:
        print("\nWARNING: The CSV contains improperly sorted pairs or pairs with identical intensities.")

if __name__ == "__main__":
    # Specify your paths here
    CSV_FILE_PATH = "partition.csv"
    IMAGE_FOLDER_PATH = "Oxford102Flowers/jpg"
    
    verify_partition_csv(CSV_FILE_PATH, IMAGE_FOLDER_PATH)