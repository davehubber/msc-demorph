import os
import random
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

def generate_partition_csv(folder_path, output_csv="partition.csv"):
    # 1. Get all jpg images in the directory
    all_images = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    
    if len(all_images) != 8189:
        print(f"Warning: Expected 8189 images, found {len(all_images)}.")

    # 2. Calculate pixel intensities for all images beforehand
    print("Calculating average pixel intensities...")
    intensities = {}
    for img_name in tqdm(all_images):
        img_path = os.path.join(folder_path, img_name)
        intensities[img_name] = calculate_average_intensity(img_path)

    # 3. Shuffle and split into Train and Test sets
    random.seed(42) # For reproducibility 
    random.shuffle(all_images)
    
    test_images = all_images[:1000]   # 1,000 for testing (as per the paper)
    train_images = all_images[1000:]  # 7,189 for training
    
    data = []
    
    # 4. Function to generate pairs and sort them by intensity
    def create_pairs(image_list, partition_name):
        for img1 in image_list:
            # Pair each image with another randomly selected image from the SAME set
            img2 = random.choice(image_list)
            while img1 == img2:
                img2 = random.choice(image_list)
            
            # Sort based on pixel intensity:
            # The paper defines the "original" image as the one with the highest intensity.
            # The dataloader unpacks Image2 as the original image.
            if intensities[img1] > intensities[img2]:
                image_brighter = img1
                image_darker = img2
            else:
                image_brighter = img2
                image_darker = img1
                
            data.append({
                'partition': partition_name,
                'Image1': image_darker,   # Added image (darker)
                'Image2': image_brighter  # Original image (brighter)
            })

    print("Generating train and test pairs...")
    create_pairs(train_images, 'train')
    create_pairs(test_images, 'test')
    
    # 5. Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Successfully saved {output_csv} with {len(df)} total pairs.")

if __name__ == "__main__":
    # Replace with the actual path to your folder containing the 8189 images
    IMAGE_FOLDER_PATH = "Oxford102Flowers/jpg"
    generate_partition_csv(IMAGE_FOLDER_PATH)