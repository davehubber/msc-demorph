import os
import random
import pandas as pd

def generate_partition_csv(folder_path, output_csv="partition.csv"):
    all_images = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    
    if len(all_images) != 8189:
        print(f"Warning: Expected 8189 images, found {len(all_images)}.")

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
            
            data.append({
                'partition': partition_name,
                'Image1': img1,
                'Image2': img2
            })

    print("Generating train and test pairs...")
    create_pairs(train_images, 'train')
    create_pairs(test_images, 'test')
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Successfully saved {output_csv} with {len(df)} total pairs.")

if __name__ == "__main__":
    IMAGE_FOLDER_PATH = "Oxford102Flowers/jpg"
    generate_partition_csv(IMAGE_FOLDER_PATH)