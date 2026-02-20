import os, torch, numpy as np, math
import torch.nn as nn
import wandb
from torch import optim
from utils import *
from modules import UNet
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from lpips_pytorch import lpips

class Diffusion:
    def __init__(self, max_timesteps=1000, alpha_start=0., alpha_max=0.8, img_size=256, device="cuda"):
        self.max_timesteps = max_timesteps
        self.img_size = img_size
        self.device = device
        self.alteration_per_t = (alpha_max - alpha_start) / max_timesteps      # size of alteration in each timestep

    def noise_images(self, original_image, added_image, t):
        return original_image * (1. - self.alteration_per_t * t)[:, None, None, None] + added_image * (self.alteration_per_t * t)[:, None, None, None]

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.max_timesteps, size=(n,), device=self.device)

    def sample(self, model, superimposed_image, alpha_init = 0.5, prediction="original"):
        n = len(superimposed_image)
        init_timestep = math.ceil(alpha_init / self.alteration_per_t)
        model.eval()
        with torch.no_grad():
            x = superimposed_image.to(self.device)
            for i in reversed(range(1, init_timestep + 1)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_image = model(x, t).to(self.device)

                current_alpha_raw = self.alteration_per_t * t
                current_alpha = current_alpha_raw[:, None, None, None]
                if prediction == "added":
                    other_image = (x - current_alpha * predicted_image) / (1 - current_alpha)
                    x_t = self.noise_images(other_image, predicted_image, t-1) + x - self.noise_images(other_image, predicted_image, t)
                elif prediction == "original":
                    other_image = (x - (1 - current_alpha) * predicted_image) / current_alpha
                    x_t = self.noise_images(predicted_image, other_image, t-1) + x - self.noise_images(predicted_image, other_image, t)
                else:
                    print("Invalid prediction.")
                    exit(-1)
                
                x = x_t
        
        model.train()
        other_image = ((superimposed_image - (1 - alpha_init) * x) / alpha_init).to(self.device)
        other_image = (other_image.clamp(-1, 1) + 1) / 2
        other_image = (other_image * 255).type(torch.uint8)
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
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

    wandb.init(
        project="demorph",
        name=args.run_name,
        config=vars(args)
    )
    
    global_step = 0
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(args.epochs):
        for _, (images, images_add) in enumerate(train_dataloader):
            images = images.to(device)
            images_add = images_add.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)

            with torch.amp.autocast("cuda"):
                x_t = diffusion.noise_images(images, images_add, t)
                predicted_image = model(x_t, t)
                if args.prediction == "added":
                    loss = mse(images_add, predicted_image)
                elif args.prediction == "original":
                    loss = mse(images, predicted_image)
                elif args.prediction == "differences":
                    x_diff = diffusion.noise_images(images, images_add, t-1) - x_t
                    loss = mse(x_diff, predicted_image)
                else:
                    print("Invalid model prediction.")
                    exit(-1)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if global_step % 1000 == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "epoch": epoch
                })

        # Sample and save images every 10 epochs (No metric calculation)
        if (epoch + 1) % 10 == 0:
            for _, (images, images_add) in enumerate(test_dataloader):
                images = images.to(device)
                images_add = images_add.to(device)
                sampled_images, other_images = diffusion.sample(model, (images + images_add) / 2., prediction=args.prediction)
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
    for i, (images, images_add) in enumerate(test_dataloader):
        images = images.to(device)
        images_add = images_add.to(device)
        sampled_images, sampled_other_image = diffusion.sample(model, images * (1-args.alpha_init) + images_add * args.alpha_init, args.alpha_init, prediction=args.prediction)
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
            else:
                so, sa, po, pa = calculate_metrics(images_np[k], images_add_np[k], sampled_other_image_np[k], sampled_images_np[k])
                lo = lpips((images[k] - 127.5) / 127.5, (sampled_other_image[k] - 127.5) / 127.5, net_type='alex', version='0.1')
                la = lpips((images_add[k] - 127.5) / 127.5, (sampled_images[k] - 127.5) / 127.5, net_type='alex', version='0.1')
            
            ssim_o.append(so)
            ssim_a.append(sa)
            psnr_o.append(po)
            psnr_a.append(pa)
            lpips_o.append(lo.to('cpu').numpy())
            lpips_a.append(la.to('cpu').numpy())

    avg_ssim_o = np.average(ssim_o)
    avg_ssim_a = np.average(ssim_a)
    avg_psnr_o = np.average(psnr_o)
    avg_psnr_a = np.average(psnr_a)
    avg_lpips_o = np.average(lpips_o)
    avg_lpips_a = np.average(lpips_a)

    print('\nMetrics organized by original images:')

    print('SSIM Original: ' + str(np.average(avg_ssim_o)))
    print('SSIM Added: ' + str(np.average(avg_ssim_a)))
    print('PSNR Original: ' + str(np.average(avg_psnr_o)))
    print('PSNR Added: ' + str(np.average(avg_psnr_a)))
    print('LPIPS Original: ' + str(np.average(avg_lpips_o)))
    print('LPIPS Added: ' + str(np.average(avg_lpips_a)))

    # Save metrics to txt file
    metrics_report = (
        f"Metrics organized by original images:\n"
        f"SSIM Original: {avg_ssim_o}\n"
        f"SSIM Added: {avg_ssim_a}\n"
        f"PSNR Original: {avg_psnr_o}\n"
        f"PSNR Added: {avg_psnr_a}\n"
        f"LPIPS Original: {avg_lpips_o}\n"
        f"LPIPS Added: {avg_lpips_a}\n"
    )
    
    results_dir = os.path.join("results", args.run_name)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "final_metrics.txt"), "w") as f:
        f.write(metrics_report)

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help='Path to dataset', required=True)
    parser.add_argument('--run_name', help='Name of the experiment for saving models and results', required=True)
    parser.add_argument('--partition_file', help='CSV file with test indexes', required=True)
    parser.add_argument('--prediction', default='original', help='The prediction of the model, choose between [added, original, differences]', required=False)
    parser.add_argument('--alpha_max', default=0.8, type=float, help='Maximum weight of the added image at the last time step of the forward diffusion process: alpha_max', required=False)
    parser.add_argument('--alpha_init', default=0.5, type=float, help='Weight of the added image: alpha_init', required=False)
    parser.add_argument('--image_size', default=64, type=int, help='Dimension of the images', required=False)
    parser.add_argument('--batch_size', default=16, help='Batch size', type=int, required=False)
    parser.add_argument('--epochs', default=1000, help='Number of epochs', type=int, required=False)
    parser.add_argument('--lr', default=3e-4, help='Learning rate', type=float, required=False)
    parser.add_argument('--device', default='cuda', help='Device, choose between [cuda, cpu]', required=False)
    args = parser.parse_args()
    args.image_size = (args.image_size, args.image_size)
    args.sampling_name = args.run_name
    train(args)
    eval(args)

if __name__ == '__main__':
    launch()