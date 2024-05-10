import os, torch, numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from lpips_pytorch import lpips

class Diffusion:
    def __init__(self, max_timesteps=1000, beta_start=0., beta_end=0.8, img_size=256, device="cuda"):
        self.max_timesteps = max_timesteps
        self.img_size = img_size
        self.device = device
        self.alteration_per_t = (beta_end - beta_start) / max_timesteps      # size of alteration in each timestep
        self.init_sampling_timestep = int(0.5 / self.alteration_per_t)       # initial timestep for sampling

    def noise_images(self, original_image, added_image, t):
        return original_image * (1. - self.alteration_per_t * t)[:, None, None, None] + added_image * (self.alteration_per_t * t)[:, None, None, None]

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.max_timesteps, size=(n,))

    def sample(self, model, images, added_images, sampling_method="unaveraging", epoch=0, run_name='flowers_exp2', save_image=True):
        n = len(images)
        model.eval()
        with torch.no_grad():
            x = (images * 0.5 + added_images * 0.5).to(self.device)
            for i in reversed(range(1, self.init_sampling_timestep + 1)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_image = model(x, t).to(self.device) 
                if sampling_method == "cold_diffusion":
                    other_image = 2.0 * (images * 0.5 + added_images * 0.5) - predicted_image
                    x_t = self.noise_images(other_image, predicted_image, t-1) + x - self.noise_images(other_image, predicted_image, t)
                elif sampling_method == "cold_diffusion_original":
                    other_image = 2.0 * (images * 0.5 + added_images * 0.5) - predicted_image
                    x_t = self.noise_images(predicted_image, other_image, t-1) + x - self.noise_images(predicted_image, other_image, t)
                elif sampling_method == 'one_step':
                    x = predicted_image
                    break
                elif sampling_method == 'added':
                    delta = (1 - self.alteration_per_t * (t - 1)) / ((1 - self.alteration_per_t * t))
                    x_t = delta[:, None, None, None] * x - (delta - 1)[:, None, None, None] * predicted_image
                elif sampling_method == "added_error":
                    error = 0
                    other_image = (x - (self.alteration_per_t * t)[:, None, None, None] * predicted_image) / (1-self.alteration_per_t * t)[:, None, None, None]
                    if i != self.init_sampling_timestep:
                        error = (0.5 * predicted_image + 0.5 * other_image) - (images * 0.5 + added_images * 0.5)
                        error = error * ((self.alteration_per_t * (t-1) - self.alteration_per_t * t) / (0.5 - self.alteration_per_t * t))[:, None, None, None]
                    
                    delta = (1 - self.alteration_per_t * (t - 1)) / ((1 - self.alteration_per_t * t))
                    x_t = delta[:, None, None, None] * x - (delta - 1)[:, None, None, None] * predicted_image - error
                elif sampling_method == 'original':
                    other_image = (x - (1-self.alteration_per_t * t)[:, None, None, None] * predicted_image) / (self.alteration_per_t * t)[:, None, None, None]
                    delta = (1 - self.alteration_per_t * (t - 1)) / ((1 - self.alteration_per_t * t))
                    x_t = delta[:, None, None, None] * x - (delta - 1)[:, None, None, None] * other_image
                elif sampling_method == 'original_error':
                    other_image = 2.0 * (images * 0.5 + added_images * 0.5) - predicted_image
                    #other_image = (x - (1-self.alteration_per_t * t)[:, None, None, None] * predicted_image) / (self.alteration_per_t * t)[:, None, None, None]
                    
                    error = 0
                    if i != self.init_sampling_timestep:
                        error = (0.5 * predicted_image + 0.5 * other_image) - (images * 0.5 + added_images * 0.5)
                        error = error * ((self.alteration_per_t * (t-1) - self.alteration_per_t * t) / (0.5 - self.alteration_per_t * t))[:, None, None, None]
                    
                    delta = (1 - self.alteration_per_t * (t - 1)) / ((1 - self.alteration_per_t * t))
                    x_t = delta[:, None, None, None] * x - (delta - 1)[:, None, None, None] * other_image - error
                if i == self.init_sampling_timestep - 1:
                    images_to_save = torch.cat((x, x_t), axis=0)
                x = x_t
                if i < self.init_sampling_timestep - 1 and i % 60 == 0:
                    images_to_save = torch.cat((images_to_save, x), axis=0)
        model.train()
        other_image = (images + added_images - x).to(self.device)
        x = (x.clamp(-1, 1) + 1) / 2
        other_image = (other_image.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        other_image = (other_image * 255).type(torch.uint8)
        if save_image:
            images_to_save = (images_to_save.clamp(-1, 1) + 1) / 2
            images_to_save = (images_to_save * 255).type(torch.uint8)
            save_image_sampling(images_to_save, os.path.join("results", run_name, f"sample_{epoch+1}.jpg"))
        return x, other_image

def train(args):
    setup_logging(args.run_name)
    device = args.device
    train_dataloader = get_data(args, 'train')
    test_dataloader = get_data(args, 'test')
    model = UNet(device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    for epoch in range(args.epochs):
        for i, (images, images_add) in enumerate(train_dataloader):
            images = images.to(device)
            images_add = images_add.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t = diffusion.noise_images(images, images_add, t)
            predicted_image = model(x_t, t)
            loss = mse(images_add, predicted_image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for i, (images, images_add) in enumerate(test_dataloader):
            images = images.to(device)
            images_add = images_add.to(device)
            sampled_images, other_images = diffusion.sample(model, images, images_add, args.sampling_method, epoch, args.run_name)
            images = (images.clamp(-1, 1) + 1) / 2
            images = (images * 255).type(torch.uint8)
            images_add = (images_add.clamp(-1, 1) + 1) / 2
            images_add = (images_add * 255).type(torch.uint8)
            save_images(sampled_images, other_images, images, images_add, os.path.join("results", args.run_name, f"{epoch+1}.jpg"))
            break
        
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

def calculate_metrics(image, add_image, result_ori_image, result_add_image):
    ssim_original = structural_similarity(image, result_ori_image, data_range=255, channel_axis=-1)
    ssim_added = structural_similarity(add_image, result_add_image, data_range=255, channel_axis=-1)
    psnr_original = peak_signal_noise_ratio(image, result_ori_image, data_range=255)
    psnr_added = peak_signal_noise_ratio(add_image, result_add_image, data_range=255)
    return ssim_original, ssim_added, psnr_original, psnr_added

def eval(args):
    device = args.device
    test_dataloader = get_data(args, 'test')
    model = UNet().to(device)
    model.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ckpt.pt"), map_location=torch.device(device)))
    model.to(device)
    model.eval()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    ssim_o = []
    ssim_a = []
    lpips_o = []
    lpips_a = []
    psnr_o = []
    psnr_a = []
    ssim_o2 = []
    ssim_a2 = []
    lpips_o2 = []
    lpips_a2 = []
    psnr_o2 = []
    psnr_a2 = []
    for i, (images, images_add) in enumerate(test_dataloader):
        images = images.to(device)
        images_add = images_add.to(device)
        sampled_images, sampled_other_image = diffusion.sample(model, images, images_add, args.sampling_method, i, args.run_name, save_image=False)
        images = (images.clamp(-1, 1) + 1) / 2
        images = (images * 255).type(torch.uint8)
        images_add = (images_add.clamp(-1, 1) + 1) / 2
        images_add = (images_add * 255).type(torch.uint8)
        sampled_images.to(device)
        save_images(sampled_images, sampled_other_image, images, images_add, os.path.join("samples", args.sampling_name, f"{i}.jpg"))

        images_np = images.to('cpu').permute(0, 2, 3, 1).numpy()
        sampled_images_np = sampled_images.to('cpu').permute(0, 2, 3, 1).numpy()
        images_add_np = images_add.to('cpu').permute(0, 2, 3, 1).numpy()
        sampled_other_image_np = sampled_other_image.to('cpu').permute(0, 2, 3, 1).numpy()
        
        for k in range(len(images_np)):
            ssim_1 = structural_similarity(images_np[k], sampled_images_np[k], data_range=255, channel_axis=-1)
            ssim_2 = structural_similarity(images_add_np[k], sampled_images_np[k], data_range=255, channel_axis=-1)
            if ssim_1 > ssim_2:
                so, sa, po, pa = calculate_metrics(images_np[k], images_add_np[k], sampled_images_np[k], sampled_other_image_np[k])
                lo = lpips((images[k] - 127.5) / 127.5, (sampled_images[k] - 127.5) / 127.5, net_type='alex', version='0.1')
                la = lpips((images_add[k] - 127.5) / 127.5, (sampled_other_image[k] - 127.5) / 127.5, net_type='alex', version='0.1')
                so2, sa2, po2, pa2, lo2, la2 = so, sa, po, pa, lo, la
            else:
                so, sa, po, pa = calculate_metrics(images_np[k], images_add_np[k], sampled_other_image_np[k], sampled_images_np[k])
                lo = lpips((images[k] - 127.5) / 127.5, (sampled_other_image[k] - 127.5) / 127.5, net_type='alex', version='0.1')
                la = lpips((images_add[k] - 127.5) / 127.5, (sampled_images[k] - 127.5) / 127.5, net_type='alex', version='0.1')
                so2, sa2, po2, pa2, lo2, la2 = sa, so, pa, po, la, lo
            
            ssim_o.append(so)
            ssim_a.append(sa)
            psnr_o.append(po)
            psnr_a.append(pa)
            lpips_o.append(lo.to('cpu').numpy())
            lpips_a.append(la.to('cpu').numpy())
            ssim_o2.append(so2)
            ssim_a2.append(sa2)
            psnr_o2.append(po2)
            psnr_a2.append(pa2)
            lpips_o2.append(lo2.to('cpu').numpy())
            lpips_a2.append(la2.to('cpu').numpy())
            
    print('\nMetrics organized by original images:')
    print('SSIM Original: ' + str(np.average(ssim_o)))
    print('SSIM Added: ' + str(np.average(ssim_a)))
    print('PSNR Original: ' + str(np.average(psnr_o)))
    print('PSNR Added: ' + str(np.average(psnr_a)))
    print('LPIPS Original: ' + str(np.average(lpips_o)))
    print('LPIPS Added: ' + str(np.average(lpips_a)))
    
    print('\nMetrics organized by sampled images:')
    print('SSIM Original: ' + str(np.average(ssim_o2)))
    print('SSIM Added: ' + str(np.average(ssim_a2)))
    print('PSNR Original: ' + str(np.average(psnr_o2)))
    print('PSNR Added: ' + str(np.average(psnr_a2)))
    print('LPIPS Original: ' + str(np.average(lpips_o2)))
    print('LPIPS Added: ' + str(np.average(lpips_a2)))

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "flowers_run1"
    args.epochs = 1000
    args.batch_size = 16
    args.image_size = (64, 64)
    args.dataset_path = r"/Oxford102Flowers/jpg"
    args.partition_file = r"csvs/flower_pairs.csv"
    # cold_diffusion, cold_diffusion_original, one_step, added, added_error, original, original_error
    args.sampling_method = r"added"
    args.sampling_name = r"added"
    print(args.sampling_method, flush=True)
    args.device = "cuda"
    args.lr = 3e-4
    train(args)
    eval(args)

if __name__ == '__main__':
    launch()
