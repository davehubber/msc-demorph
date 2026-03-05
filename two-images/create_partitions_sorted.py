import os
import random
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

def calculate_average_intensity(image_path):
    with Image.open(image_path) as img:
        img_gray = img.convert('L')
        return np.mean(img_gray)

def generate_partition_csv(folder_path, output_csv="partition.csv"):
    all_images = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    
    if len(all_images) != 8189:
        print(f"Warning: Expected 8189 images, found {len(all_images)}.")

    print("Calculating average pixel intensities...")
    intensities = {}
    for img_name in tqdm(all_images):
        img_path = os.path.join(folder_path, img_name)
        intensities[img_name] = calculate_average_intensity(img_path)

    random.seed(42)
    random.shuffle(all_images)
    
    test_images = all_images[:1000]
    train_images = all_images[1000:]
    
    data = []
    
    def create_pairs(image_list, partition_name):
        for img1 in image_list:
            img2 = random.choice(image_list)
            while img1 == img2:
                img2 = random.choice(image_list)
            
            if intensities[img1] > intensities[img2]:
                image_brighter = img1
                image_darker = img2
            else:
                image_brighter = img2
                image_darker = img1
                
            data.append({
                'partition': partition_name,
                'Image1': image_brighter,
                'Image2': image_darker
            })

    print("Generating train and test pairs...")
    create_pairs(train_images, 'train')
    create_pairs(test_images, 'test')
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Successfully saved {output_csv} with {len(df)} total pairs.")

if __name__ == "__main__":
    IMAGE_FOLDER_PATH = "/nas-ctm01/datasets/public/Oxford102Flowers/jpg"
    generate_partition_csv(IMAGE_FOLDER_PATH)