import os, torch, numpy as np, math
import torch.nn as nn
import wandb
from torch import optim
from utils import *
from modules import UNet
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from lpips_pytorch import lpips

class Diffusion:
    def __init__(self, max_timesteps=300, alpha_start=0., alpha_max=0.5, img_size=256, device="cuda"):
        self.max_timesteps = max_timesteps
        self.img_size = img_size
        self.device = device
        self.alteration_per_t = (alpha_max - alpha_start) / max_timesteps

    def noise_images(self, image_1, image_2, t):
        return image_1 * (1. - self.alteration_per_t * t)[:, None, None, None] + image_2 * (self.alteration_per_t * t)[:, None, None, None]

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.max_timesteps, size=(n,), device=self.device)

    def sample(self, model, superimposed_image, alpha_init = 0.5):
        n = len(superimposed_image)
        init_timestep = math.ceil(alpha_init / self.alteration_per_t)
        model.eval()

        with torch.no_grad():
            x_A = superimposed_image.clone().to(self.device)
            x_B = superimposed_image.clone().to(self.device)

            t_init_tensor = (torch.ones(n) * init_timestep).long().to(self.device)
            pred_both = model(x_A, t_init_tensor)
            anchor_A, anchor_B = torch.chunk(pred_both, 2, dim=1)

            for i in reversed(range(1, init_timestep)):
                t = (torch.ones(n) * i).long().to(self.device)

                pA_1, pA_2 = torch.chunk(model(x_A, t), 2, dim=1)
                pB_1, pB_2 = torch.chunk(model(x_B, t), 2, dim=1)

                mse_A_straight = nn.F.mse_loss(pA_1, anchor_A, reduction='none').view(n, -1).mean(dim=1) + \
                                 nn.F.mse_loss(pA_2, anchor_B, reduction='none').view(n, -1).mean(dim=1)
                mse_A_crossed  = nn.F.mse_loss(pA_1, anchor_B, reduction='none').view(n, -1).mean(dim=1) + \
                                 nn.F.mse_loss(pA_2, anchor_A, reduction='none').view(n, -1).mean(dim=1)
                
                swap_mask_A = (mse_A_crossed < mse_A_straight).view(-1, 1, 1, 1)
                pA_1_aligned = torch.where(swap_mask_A, pA_2, pA_1)
                pA_2_aligned = torch.where(swap_mask_A, pA_1, pA_2)

                mse_B_straight = nn.F.mse_loss(pB_1, anchor_A, reduction='none').view(n, -1).mean(dim=1) + \
                                 nn.F.mse_loss(pB_2, anchor_B, reduction='none').view(n, -1).mean(dim=1)
                mse_B_crossed  = nn.F.mse_loss(pB_1, anchor_B, reduction='none').view(n, -1).mean(dim=1) + \
                                 nn.F.mse_loss(pB_2, anchor_A, reduction='none').view(n, -1).mean(dim=1)
                
                swap_mask_B = (mse_B_crossed < mse_B_straight).view(-1, 1, 1, 1)
                pB_1_aligned = torch.where(swap_mask_B, pB_2, pB_1)
                pB_2_aligned = torch.where(swap_mask_B, pB_1, pB_2)

                pred_A_strong = pA_1_aligned.clamp(-1.0, 1.0)
                pred_B_strong = pB_2_aligned.clamp(-1.0, 1.0)

                anchor_A = pred_A_strong.clone()
                anchor_B = pred_B_strong.clone()

                x_A = x_A - self.noise_images(pred_A_strong, pred_B_strong, t) + self.noise_images(pred_A_strong, pred_B_strong, t-1)
                x_B = x_B - self.noise_images(pred_B_strong, pred_A_strong, t) + self.noise_images(pred_B_strong, pred_A_strong, t-1)
        
        model.train()

        final_A = (pred_A_strong.clamp(-1, 1) + 1) / 2
        final_A = (final_A * 255).type(torch.uint8)
        
        final_B = (pred_B_strong.clamp(-1, 1) + 1) / 2
        final_B = (final_B * 255).type(torch.uint8)

        return final_A, final_B

def train(args):
    setup_logging(args.run_name)
    device = args.device
    train_dataloader = get_data(args, 'train')
    test_dataloader = get_data(args, 'test')

    model = UNet(c_in=3, c_out=6, device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    diffusion = Diffusion(img_size=args.image_size, device=device, alpha_max=args.alpha_max)

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

            if torch.rand(1).item() > 0.5:
                images, images_add = images_add, images

            t = diffusion.sample_timesteps(images.shape[0]).to(device)

            with torch.amp.autocast("cuda"):
                x_t = diffusion.noise_images(images, images_add, t)

                predicted_both = model(x_t, t)
                pred_1, pred_2 = torch.chunk(predicted_both, 2, dim=1)

                mse_straight = nn.F.mse_loss(pred_1, images, reduction='none').view(images.shape[0], -1).mean(dim=1) + \
                               nn.F.mse_loss(pred_2, images_add, reduction='none').view(images.shape[0], -1).mean(dim=1)
                
                mse_crossed = nn.F.mse_loss(pred_1, images_add, reduction='none').view(images.shape[0], -1).mean(dim=1) + \
                              nn.F.mse_loss(pred_2, images, reduction='none').view(images.shape[0], -1).mean(dim=1)
                
                loss = torch.min(mse_straight, mse_crossed).mean()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if global_step % 100 == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "epoch": epoch
                })

        # Sample and save images every 10 epochs (No metric calculation)
        if (epoch + 1) % 10 == 0:
            for _, (images, images_add) in enumerate(test_dataloader):
                images = images.to(device)
                images_add = images_add.to(device)

                superimposed = (images + images_add) / 2.
                sampled_A, sampled_B = diffusion.sample(model, superimposed, alpha_init=args.alpha_init)

                images = (images.clamp(-1, 1) + 1) / 2
                images = (images * 255).type(torch.uint8)
                images_add = (images_add.clamp(-1, 1) + 1) / 2
                images_add = (images_add * 255).type(torch.uint8)

                save_images(sampled_A, sampled_B, images, images_add, os.path.join("results", args.run_name, f"{epoch+1}.jpg"))
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

    model = UNet(c_in=3, c_out=6).to(device)
    model.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ckpt.pt"), map_location=torch.device(device)))
    model.to(device)
    model.eval()

    diffusion = Diffusion(img_size=args.image_size, device=device)

    results = {"ssim_o": [], "ssim_a": [], "psnr_o": [], "psnr_a": [], "lpips_o": [], "lpips_a": []}

    for i, (images, images_add) in enumerate(test_dataloader):
        images = images.to(device)
        images_add = images_add.to(device)

        superimposed = (images + images_add) / 2.
        sampled_A, sampled_B = diffusion.sample(model, superimposed, args.alpha_init)
        
        gt_A_eval = ((images.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
        gt_B_eval = ((images_add.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)

        for k in range(len(images)):
            cur_gt_A = gt_A_eval[k].cpu().permute(1, 2, 0).numpy()
            cur_gt_B = gt_B_eval[k].cpu().permute(1, 2, 0).numpy()
            cur_s_A = sampled_A[k].cpu().permute(1, 2, 0).numpy()
            cur_s_B = sampled_B[k].cpu().permute(1, 2, 0).numpy()

            mse_straight = np.mean((cur_s_A - cur_gt_A)**2) + np.mean((cur_s_B - cur_gt_B)**2)
            mse_crossed  = np.mean((cur_s_A - cur_gt_B)**2) + np.mean((cur_s_B - cur_gt_A)**2)

            if mse_crossed < mse_straight:
                cur_s_A, cur_s_B = cur_s_B, cur_s_A
                tensor_s_A, tensor_s_B = sampled_B[k], sampled_A[k]
            else:
                tensor_s_A, tensor_s_B = sampled_A[k], sampled_B[k]

            so, sa, po, pa = calculate_metrics(cur_gt_A, cur_gt_B, cur_s_A, cur_s_B)
            
            lo = lpips((images[k]), (tensor_s_A.float() / 127.5) - 1.0, net_type='alex')
            la = lpips((images_add[k]), (tensor_s_B.float() / 127.5) - 1.0, net_type='alex')
            
            results["ssim_o"].append(so); results["ssim_a"].append(sa)
            results["psnr_o"].append(po); results["psnr_a"].append(pa)
            results["lpips_o"].append(lo.item()); results["lpips_a"].append(la.item())

        if i == 0:
            save_images(sampled_A, sampled_B, gt_A_eval, gt_B_eval, 
                        os.path.join("samples", args.sampling_name, "eval_grid.jpg"))

    metrics_report = "\n".join([f"Avg {k}: {np.mean(v)}" for k, v in results.items()])
    print(metrics_report)
    with open(os.path.join("results", args.run_name, "final_metrics.txt"), "w") as f:
        f.write(metrics_report)

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help='Path to dataset', required=True)
    parser.add_argument('--run_name', help='Name of the experiment for saving models and results', required=True)
    parser.add_argument('--partition_file', help='CSV file with test indexes', required=True)
    parser.add_argument('--alpha_max', default=0.5, type=float, help='Maximum weight of the added image at the last time step of the forward diffusion process: alpha_max', required=False)
    parser.add_argument('--alpha_init', default=0.5, type=float, help='Weight of the added image: alpha_init', required=False)
    parser.add_argument('--image_size', default=64, type=int, help='Dimension of the images', required=False)
    parser.add_argument('--batch_size', default=16, help='Batch size', type=int, required=False)
    parser.add_argument('--epochs', default=800, help='Number of epochs', type=int, required=False)
    parser.add_argument('--lr', default=3e-4, help='Learning rate', type=float, required=False)
    parser.add_argument('--device', default='cuda', help='Device, choose between [cuda, cpu]', required=False)
    args = parser.parse_args()
    args.image_size = (args.image_size, args.image_size)
    args.sampling_name = args.run_name
    train(args)
    eval(args)

if __name__ == '__main__':
    launch()