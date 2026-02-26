import os, torch, numpy as np, math
import torch.nn.functional as F
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

            for i in reversed(range(1, init_timestep + 1)):
                t = (torch.ones(n) * i).long().to(self.device)

                if i == init_timestep:
                    p_1, p_2 = torch.chunk(model(x_A, t), 2, dim=1)
                    
                    best_pred_1 = p_1.clamp(-1.0, 1.0)
                    best_pred_2 = p_2.clamp(-1.0, 1.0)
                    
                    anchor_A = best_pred_1.clamp(-1.0, 1.0).clone()
                    anchor_B = best_pred_2.clamp(-1.0, 1.0).clone()
                    
                else:
                    pA_1, pA_2 = torch.chunk(model(x_A, t), 2, dim=1)
                    pB_1, pB_2 = torch.chunk(model(x_B, t), 2, dim=1)

                    mse_A_straight = F.mse_loss(pA_1, anchor_A, reduction='none').view(n, -1).mean(dim=1) + \
                                     F.mse_loss(pA_2, anchor_B, reduction='none').view(n, -1).mean(dim=1)
                    mse_A_crossed  = F.mse_loss(pA_1, anchor_B, reduction='none').view(n, -1).mean(dim=1) + \
                                     F.mse_loss(pA_2, anchor_A, reduction='none').view(n, -1).mean(dim=1)
                    
                    swap_mask_A = (mse_A_crossed < mse_A_straight).view(-1, 1, 1, 1)
                    pA_1_aligned = torch.where(swap_mask_A, pA_2, pA_1)
                    pA_2_aligned = torch.where(swap_mask_A, pA_1, pA_2)

                    mse_B_straight = F.mse_loss(pB_1, anchor_B, reduction='none').view(n, -1).mean(dim=1) + \
                                     F.mse_loss(pB_2, anchor_A, reduction='none').view(n, -1).mean(dim=1)
                    mse_B_crossed  = F.mse_loss(pB_1, anchor_A, reduction='none').view(n, -1).mean(dim=1) + \
                                     F.mse_loss(pB_2, anchor_B, reduction='none').view(n, -1).mean(dim=1)
                    
                    swap_mask_B = (mse_B_crossed < mse_B_straight).view(-1, 1, 1, 1)
                    pB_1_aligned = torch.where(swap_mask_B, pB_2, pB_1)
                    pB_2_aligned = torch.where(swap_mask_B, pB_1, pB_2)

                    best_pred_1 = pA_1_aligned.clamp(-1.0, 1.0)
                    best_pred_2 = pB_1_aligned.clamp(-1.0, 1.0)
                    
                    anchor_A = best_pred_1.clone()
                    anchor_B = best_pred_2.clone()

                x_A = x_A - self.noise_images(best_pred_1, best_pred_2, t) + self.noise_images(best_pred_1, best_pred_2, t-1)
                
                x_B = x_B - self.noise_images(best_pred_2, best_pred_1, t) + self.noise_images(best_pred_2, best_pred_1, t-1)
        
        model.train()

        final_A = (pA_1_aligned.clamp(-1.0, 1.0) + 1) / 2
        final_A = (final_A * 255).type(torch.uint8)
        
        final_B = (pB_1_aligned.clamp(-1.0, 1.0) + 1) / 2
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

                mse_straight = F.mse_loss(pred_1, images, reduction='none').view(images.shape[0], -1).mean(dim=1) + \
                               F.mse_loss(pred_2, images_add, reduction='none').view(images.shape[0], -1).mean(dim=1)
                
                mse_crossed = F.mse_loss(pred_1, images_add, reduction='none').view(images.shape[0], -1).mean(dim=1) + \
                              F.mse_loss(pred_2, images, reduction='none').view(images.shape[0], -1).mean(dim=1)
                
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

    model = UNet(c_in=3, c_out=6, device=device).to(device)
    model.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ckpt.pt"), map_location=torch.device(device)))
    model.eval()

    diffusion = Diffusion(img_size=args.image_size, device=device)

    results = {
        "ssim": [], "psnr": [], "lpips": [], 
        "success_count": 0, "total_images": 0
    }
    
    saved_grid = False

    for i, (images, images_add) in enumerate(test_dataloader):
        images = images.to(device)
        images_add = images_add.to(device)

        superimposed = (images + images_add) / 2.
        superimposed_np = ((superimposed.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()

        sampled_A, sampled_B = diffusion.sample(model, superimposed, args.alpha_init)
        
        gt_A_eval = ((images.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
        gt_B_eval = ((images_add.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)

        batch_s_A = []
        batch_s_B = []

        for k in range(len(images)):
            cur_gt_A = gt_A_eval[k].cpu().permute(1, 2, 0).numpy()
            cur_gt_B = gt_B_eval[k].cpu().permute(1, 2, 0).numpy()
            cur_s_A = sampled_A[k].cpu().permute(1, 2, 0).numpy()
            cur_s_B = sampled_B[k].cpu().permute(1, 2, 0).numpy()
            cur_super = superimposed_np[k]

            mse_straight = np.mean((cur_s_A - cur_gt_A)**2) + np.mean((cur_s_B - cur_gt_B)**2)
            mse_crossed  = np.mean((cur_s_A - cur_gt_B)**2) + np.mean((cur_s_B - cur_gt_A)**2)

            if mse_crossed < mse_straight:
                cur_s_A, cur_s_B = cur_s_B, cur_s_A
                tensor_s_A, tensor_s_B = sampled_B[k], sampled_A[k]
            else:
                tensor_s_A, tensor_s_B = sampled_A[k], sampled_B[k]

            if not saved_grid:
                batch_s_A.append(tensor_s_A)
                batch_s_B.append(tensor_s_B)

            s_A = structural_similarity(cur_gt_A, cur_s_A, data_range=255, channel_axis=-1)
            s_B = structural_similarity(cur_gt_B, cur_s_B, data_range=255, channel_axis=-1)
            
            p_A = peak_signal_noise_ratio(cur_gt_A, cur_s_A, data_range=255)
            p_B = peak_signal_noise_ratio(cur_gt_B, cur_s_B, data_range=255)

            l_A = lpips((images[k]), (tensor_s_A.float() / 127.5) - 1.0, net_type='alex').item()
            l_B = lpips((images_add[k]), (tensor_s_B.float() / 127.5) - 1.0, net_type='alex').item()

            results["ssim"].append((s_A + s_B) / 2)
            results["psnr"].append((p_A + p_B) / 2)
            results["lpips"].append((l_A + l_B) / 2)

            ssim_avg_A = structural_similarity(cur_gt_A, cur_super, data_range=255, channel_axis=-1)
            ssim_avg_B = structural_similarity(cur_gt_B, cur_super, data_range=255, channel_axis=-1)

            if s_A > ssim_avg_A:
                results["success_count"] += 1
            
            if s_B > ssim_avg_B:
                results["success_count"] += 1
            
            results["total_images"] += 2

        if i == 0 and not saved_grid:
            aligned_A_stack = torch.stack(batch_s_A)
            aligned_B_stack = torch.stack(batch_s_B)
            
            save_images(aligned_A_stack, aligned_B_stack, gt_A_eval, gt_B_eval, 
                        os.path.join("samples", args.sampling_name, "eval_grid.jpg"))
            saved_grid = True

    avg_ssim = np.mean(results["ssim"])
    avg_psnr = np.mean(results["psnr"])
    avg_lpips = np.mean(results["lpips"])
    success_rate = (results["success_count"] / results["total_images"]) * 100

    metrics_report = (
        f"SSIM: {avg_ssim:.4f}\n"
        f"PSNR: {avg_psnr:.4f}\n"
        f"LPIPS: {avg_lpips:.4f}\n"
        f"Success Rate (%S): {success_rate:.2f}%"
    )
    
    print(metrics_report)
    with open(os.path.join("results", args.run_name, "final_metrics.txt"), "w") as f:
        f.write(metrics_report)

def one_shot_eval(args):
    device = args.device
    test_dataloader = get_data(args, 'test')

    model = UNet(c_in=3, c_out=6, device=device).to(device)
    model.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ckpt.pt"), map_location=torch.device(device)))
    model.eval()

    diffusion = Diffusion(img_size=args.image_size, device=device, alpha_max=args.alpha_max)
    
    init_timestep = math.ceil(args.alpha_init / diffusion.alteration_per_t)
    
    os.makedirs(os.path.join("samples", args.run_name, "one_shot"), exist_ok=True)

    with torch.no_grad():
        for i, (images, images_add) in enumerate(test_dataloader):
            images = images.to(device)
            images_add = images_add.to(device)

            superimposed = images * 0.9 + images_add * 0.1
            n = len(superimposed)

            t_init_tensor = (torch.ones(n) * init_timestep).long().to(device)
            
            pred_both = model(superimposed, t_init_tensor)
            pred_A, pred_B = torch.chunk(pred_both, 2, dim=1)
            
            mse_straight = F.mse_loss(pred_A, images, reduction='none').view(n, -1).mean(dim=1) + \
                           F.mse_loss(pred_B, images_add, reduction='none').view(n, -1).mean(dim=1)
            mse_crossed  = F.mse_loss(pred_A, images_add, reduction='none').view(n, -1).mean(dim=1) + \
                           F.mse_loss(pred_B, images, reduction='none').view(n, -1).mean(dim=1)
            
            swap_mask = (mse_crossed < mse_straight).view(-1, 1, 1, 1)
            pred_A_aligned = torch.where(swap_mask, pred_B, pred_A)
            pred_B_aligned = torch.where(swap_mask, pred_A, pred_B)

            final_A = ((pred_A_aligned.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
            final_B = ((pred_B_aligned.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
            
            gt_A = ((images.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
            gt_B = ((images_add.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)

            save_path = os.path.join("samples", args.run_name, "one_shot", f"batch_{i}.jpg")
            save_images(final_A, final_B, gt_A, gt_B, save_path)
            
            print(f"Saved one-shot batch {i} to {save_path}")

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
    parser.add_argument('--mode', default='train', choices=['train', 'eval', 'one_shot'], help='Mode to run')
    args = parser.parse_args()
    args.image_size = (args.image_size, args.image_size)
    args.sampling_name = args.run_name

    if args.mode == 'train':
        train(args)
        eval(args)
    elif args.mode == 'eval':
        eval(args)
    elif args.mode == 'one_shot':
        one_shot_eval(args)

if __name__ == '__main__':
    launch()