import os, numpy as np, pandas as pd
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, obtained_images, original_images, added_images, path, **kwargs):
    input_images = (original_images*0.5 + added_images*0.5).type(torch.uint8)
    all_images = torch.cat((original_images, added_images, input_images, images, obtained_images), axis=0)
    grid = torchvision.utils.make_grid(all_images, nrow=len(images), **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def save_image_sampling(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, nrow=8, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

class TheDataset(Dataset):
    def __init__(self, dataset_path, partition, csv_file, transform=None):
        self.dataset_path = dataset_path
        self.partition = partition
        
        csv_file = pd.read_csv(csv_file, sep=",")
        self.image_paths_1 = np.asarray(csv_file['Image1'].values)
        self.image_paths_2 = np.asarray(csv_file['Image2'].values)

        self.transform = transform
        print('Size of ' + partition + ': ' + str(len(self.image_paths_1)))
        
    def __getitem__(self, index):
        x1 = Image.open(os.path.join(self.dataset_path, self.image_paths_1[index]))
        x2 = Image.open(os.path.join(self.dataset_path, self.image_paths_2[index]))
        if self.transform is not None:
            x1 = self.transform(x1)
            x2 = self.transform(x2)
        return x2, x1
    
    def __len__(self):
        return len(self.image_paths_1)

def get_data(args, partition):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = TheDataset(args.dataset_path, partition, args.partition_file, transform=transforms)
    batch_size = args.batch_size
    shuffle = True
    if partition == 'test':
        shuffle = False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("samples", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    os.makedirs(os.path.join("samples", run_name), exist_ok=True)
