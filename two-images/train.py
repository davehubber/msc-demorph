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

    def noise_images(self, image_1, image_2, t):
        return image_1 * (1. - self.alteration_per_t * t)[:, None, None, None] + image_2 * (self.alteration_per_t * t)[:, None, None, None]

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.max_timesteps, size=(n,), device=self.device)

    def sample(self, model, superimposed_image, alpha_init = 0.5):
        n = len(superimposed_image)
        init_timestep = math.ceil(alpha_init / self.alteration_per_t)
        model.eval()

        final_pred_original = None
        final_pred_added = None

        with torch.no_grad():
            x_1 = superimposed_image.to(self.device)
            x_2 = superimposed_image.to(self.device)

            for i in reversed(range(1, init_timestep + 1)):
                t = (torch.ones(n) * i).long().to(self.device)

                x_combined = torch.cat([x_1, x_2], dim=1)

                predicted_both = model(x_combined, t).to(self.device)
                pred_original, pred_added = torch.chunk(predicted_both, 2, dim=1)

                x_1_t = self.noise_images(pred_original, pred_added, t-1) + x_1 - self.noise_images(pred_original, pred_added, t)
                x_1 = x_1_t

                x_2_t = self.noise_images(pred_added, pred_original, t-1) + x_2 - self.noise_images(pred_added, pred_original, t)
                x_2 = x_2_t

                if i == 1:
                    final_pred_original = pred_original
                    final_pred_added = pred_added
        
        model.train()

        final_pred_added = (final_pred_added.clamp(-1, 1) + 1) / 2
        final_pred_added = (final_pred_added * 255).type(torch.uint8)
        
        final_pred_original = (final_pred_original.clamp(-1, 1) + 1) / 2
        final_pred_original = (final_pred_original * 255).type(torch.uint8)

        return final_pred_original, final_pred_added

def train(args):
    setup_logging(args.run_name)
    device = args.device
    train_dataloader = get_data(args, 'train')
    test_dataloader = get_data(args, 'test')
    model = UNet(c_in=6, c_out=6, device=device).to(device)
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
                x_t_1 = diffusion.noise_images(images, images_add, t)

                x_t_2 = diffusion.noise_images(images_add, images, t)

                x_combined = torch.cat([x_t_1, x_t_2], dim=1)

                predicted_both = model(x_combined, t)
                pred_original, pred_added = torch.chunk(predicted_both, 2, dim=1)

                loss_original = mse(images, pred_original)
                loss_added = mse(images_add, pred_added)
                loss = loss_original + loss_added

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

                sampled_original, sampled_added = diffusion.sample(model, (images + images_add) / 2., alpha_init=args.alpha_init)

                images = (images.clamp(-1, 1) + 1) / 2
                images = (images * 255).type(torch.uint8)
                images_add = (images_add.clamp(-1, 1) + 1) / 2
                images_add = (images_add * 255).type(torch.uint8)

                save_images(sampled_original, sampled_added, images, images_add, os.path.join("results", args.run_name, f"{epoch+1}.jpg"))
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

    model = UNet(c_in=6, c_out=6).to(device)
    model.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ckpt.pt"), map_location=torch.device(device)))
    model.to(device)
    model.eval()
    diffusion = Diffusion(img_size=args.image_size, device=device)

    ssim_o, ssim_a = [], []
    lpips_o, lpips_a = [], []
    psnr_o, psnr_a = [], []

    for i, (images, images_add) in enumerate(test_dataloader):
        images = images.to(device)
        images_add = images_add.to(device)

        sampled_original, sampled_added = diffusion.sample(model, images * (1-args.alpha_init) + images_add * args.alpha_init, args.alpha_init)
        
        images_eval = (images.clamp(-1, 1) + 1) / 2
        images_eval = (images_eval * 255).type(torch.uint8)
        images_add_eval = (images_add.clamp(-1, 1) + 1) / 2
        images_add_eval = (images_add_eval * 255).type(torch.uint8)

        save_images(sampled_original, sampled_added, images_eval, images_add_eval, 
                          os.path.join("samples", args.sampling_name, f"{i}.jpg"))

        images_np = images_eval.cpu().permute(0, 2, 3, 1).numpy()
        images_add_np = images_add_eval.cpu().permute(0, 2, 3, 1).numpy()
        sampled_original_np = sampled_original.cpu().permute(0, 2, 3, 1).numpy()
        sampled_added_np = sampled_added.cpu().permute(0, 2, 3, 1).numpy()
        
        for k in range(len(images_np)):
            so, sa, po, pa = calculate_metrics(images_np[k], images_add_np[k], sampled_original_np[k], sampled_added_np[k])
            lo = lpips((images[k] - 127.5) / 127.5, (sampled_original[k] - 127.5) / 127.5, net_type='alex', version='0.1')
            la = lpips((images_add[k] - 127.5) / 127.5, (sampled_added[k] - 127.5) / 127.5, net_type='alex', version='0.1')
            
            ssim_o.append(so)
            ssim_a.append(sa)
            psnr_o.append(po)
            psnr_a.append(pa)
            lpips_o.append(lo.cpu().numpy())
            lpips_a.append(la.cpu().numpy())

    avg_ssim_o = np.average(ssim_o)
    avg_ssim_a = np.average(ssim_a)
    avg_psnr_o = np.average(psnr_o)
    avg_psnr_a = np.average(psnr_a)
    avg_lpips_o = np.average(lpips_o)
    avg_lpips_a = np.average(lpips_a)

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

    print(metrics_report)
    
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