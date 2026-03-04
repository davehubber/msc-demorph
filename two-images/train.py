import os, torch, numpy as np, math
import torch.nn.functional as F
import wandb
import lpips
from torch import optim
from utils import *
from modules import UNet
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

class Diffusion:
    def __init__(self, max_timesteps=300, img_size=256, device="cuda"):
        self.max_timesteps = max_timesteps
        self.img_size = img_size
        self.device = device
        
        # Initialize LPIPS once here so it doesn't reload on every sample call
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(self.device)
        self.loss_fn_alex.eval()

    def noise_images(self, image, superimposed, t):
        # Linear interpolation: t=0 gives the single image, t=max_timesteps gives the 50/50 superimposed image
        t_frac = (t / self.max_timesteps)[:, None, None, None]
        return image * (1. - t_frac) + superimposed * t_frac

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.max_timesteps, size=(n,), device=self.device)

    def sample(self, model, superimposed_image):
        n = len(superimposed_image)
        init_timestep = self.max_timesteps
        model.eval()

        with torch.no_grad():
            x_A = superimposed_image.clone().to(self.device)
            x_B = superimposed_image.clone().to(self.device)

            for i in reversed(range(1, init_timestep + 1)):
                t = (torch.ones(n) * i).long().to(self.device)
                t_prev = (torch.ones(n) * (i - 1)).long().to(self.device)

                if i == init_timestep:
                    # Initial Prediction establishes the goals (Anchors)
                    p_1, p_2 = torch.chunk(model(x_A, t), 2, dim=1)
                    
                    anchor_A = p_1.clamp(-1.0, 1.0)
                    anchor_B = p_2.clamp(-1.0, 1.0)
                    
                    best_pred_A = anchor_A.clone()
                    best_pred_B = anchor_B.clone()
                    
                else:
                    pA_1, pA_2 = torch.chunk(model(x_A, t), 2, dim=1)
                    pB_1, pB_2 = torch.chunk(model(x_B, t), 2, dim=1)

                    # --- ALIGN PATH A USING LPIPS ---
                    lpips_A_straight = self.loss_fn_alex(pA_1, anchor_A) + self.loss_fn_alex(pA_2, anchor_B)
                    lpips_A_crossed  = self.loss_fn_alex(pA_1, anchor_B) + self.loss_fn_alex(pA_2, anchor_A)
                    
                    swap_mask_A = (lpips_A_crossed < lpips_A_straight).view(-1, 1, 1, 1)
                    pA_1_aligned = torch.where(swap_mask_A, pA_2, pA_1)

                    # --- ALIGN PATH B USING LPIPS ---
                    lpips_B_straight = self.loss_fn_alex(pB_1, anchor_A) + self.loss_fn_alex(pB_2, anchor_B)
                    lpips_B_crossed  = self.loss_fn_alex(pB_1, anchor_B) + self.loss_fn_alex(pB_2, anchor_A)
                    
                    swap_mask_B = (lpips_B_crossed < lpips_B_straight).view(-1, 1, 1, 1)
                    pB_2_aligned = torch.where(swap_mask_B, pB_1, pB_2)

                    # Goals extracted from aligned paths
                    best_pred_A = pA_1_aligned.clamp(-1.0, 1.0)
                    best_pred_B = pB_2_aligned.clamp(-1.0, 1.0)
                    
                    # Update Anchors ONLY if they swapped (shifted concepts)
                    anchor_A = torch.where(swap_mask_A, best_pred_A.clone(), anchor_A)
                    anchor_B = torch.where(swap_mask_B, best_pred_B.clone(), anchor_B)

                # --- RENOISE (COLD DIFFUSION STEP) ---
                # Applying TACOS update using directly the known superimposed image as the M concept
                x_A = x_A - self.noise_images(best_pred_A, superimposed_image, t) + self.noise_images(best_pred_A, superimposed_image, t_prev)
                x_B = x_B - self.noise_images(best_pred_B, superimposed_image, t) + self.noise_images(best_pred_B, superimposed_image, t_prev)
        
        model.train()

        final_A = (best_pred_A + 1) / 2
        final_A = (final_A * 255).type(torch.uint8)
        
        final_B = (best_pred_B + 1) / 2
        final_B = (final_B * 255).type(torch.uint8)

        return final_A, final_B


def train(args):
    setup_logging(args.run_name)
    device = args.device
    train_dataloader = get_data(args, 'train')
    test_dataloader = get_data(args, 'test')

    # Get fixed 4 images from testing data for regular visual tracking
    fixed_test_batch = next(iter(test_dataloader))
    fixed_images = fixed_test_batch[0][:4].to(device)
    fixed_images_add = fixed_test_batch[1][:4].to(device)

    model = UNet(c_in=3, c_out=6, device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

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
            
            # The perfect 50/50 superimposed average (constant target for max timestep)
            superimposed = (images + images_add) / 2.

            # Permutation invariant start (selects one of the components as the start)
            if torch.rand(1).item() > 0.5:
                images, images_add = images_add, images

            t = diffusion.sample_timesteps(images.shape[0]).to(device)

            with torch.amp.autocast("cuda"):
                # Forward noise interpolating single image and perfect average
                x_t = diffusion.noise_images(images, superimposed, t)

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

        # Sample and save the SAME 4 images every 10 epochs
        if (epoch + 1) % 10 == 0:
            superimposed = (fixed_images + fixed_images_add) / 2.
            
            sampled_A, sampled_B = diffusion.sample(model, superimposed)

            fixed_images_vis = (fixed_images.clamp(-1, 1) + 1) / 2
            fixed_images_vis = (fixed_images_vis * 255).type(torch.uint8)
            fixed_images_add_vis = (fixed_images_add.clamp(-1, 1) + 1) / 2
            fixed_images_add_vis = (fixed_images_add_vis * 255).type(torch.uint8)

            save_images(sampled_A, sampled_B, fixed_images_vis, fixed_images_add_vis, os.path.join("results", args.run_name, f"{epoch+1}.jpg"))
        
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


def eval(args):
    device = args.device
    test_dataloader = get_data(args, 'test')

    model = UNet(c_in=3, c_out=6, device=device).to(device)
    model.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ckpt.pt"), map_location=torch.device(device)))
    model.eval()

    diffusion = Diffusion(img_size=args.image_size, device=device)

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    results = {
        "ssim_1": [], "ssim_2": [], 
        "psnr_1": [], "psnr_2": [], 
        "lpips_1": [], "lpips_2": [], 
        "success_count_1": 0, "success_count_2": 0, 
        "total_pairs": 0
    }
    
    saved_grid = False

    os.makedirs(os.path.join("samples", args.sampling_name), exist_ok=True)
    os.makedirs(os.path.join("results", args.run_name), exist_ok=True)
    
    ind_dir = os.path.join("samples", args.sampling_name, "individual")
    os.makedirs(ind_dir, exist_ok=True)

    for i, (images, images_add) in enumerate(test_dataloader):
        images = images.to(device)
        images_add = images_add.to(device)

        superimposed = (images + images_add) / 2.
        superimposed_np = ((superimposed.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()

        sampled_A, sampled_B = diffusion.sample(model, superimposed)
        
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

            l_A = loss_fn_alex(images[k], (tensor_s_A.float() / 127.5) - 1.0).item()
            l_B = loss_fn_alex(images_add[k], (tensor_s_B.float() / 127.5) - 1.0).item()

            results["ssim_1"].append(s_A)
            results["ssim_2"].append(s_B)
            results["psnr_1"].append(p_A)
            results["psnr_2"].append(p_B)
            results["lpips_1"].append(l_A)
            results["lpips_2"].append(l_B)

            ssim_avg_A = structural_similarity(cur_gt_A, cur_super, data_range=255, channel_axis=-1)
            ssim_avg_B = structural_similarity(cur_gt_B, cur_super, data_range=255, channel_axis=-1)

            if s_A > ssim_avg_A:
                results["success_count_1"] += 1
            if s_B > ssim_avg_B:
                results["success_count_2"] += 1
            
            results["total_pairs"] += 1

        if i == 0 and not saved_grid:
            from PIL import Image
            
            aligned_A_stack = torch.stack(batch_s_A)
            aligned_B_stack = torch.stack(batch_s_B)
            
            # Save the evaluation grid containing the whole first batch showing GT, Avg, and Separation perfectly aligned
            save_images(aligned_A_stack, aligned_B_stack, gt_A_eval, gt_B_eval, 
                        os.path.join("samples", args.sampling_name, "eval_grid.jpg"))
            
            for b_idx in range(len(batch_s_A)):
                img_A_np = batch_s_A[b_idx].cpu().permute(1, 2, 0).numpy()
                img_B_np = batch_s_B[b_idx].cpu().permute(1, 2, 0).numpy()
                
                Image.fromarray(img_A_np).save(os.path.join(ind_dir, f"sample_{b_idx}_img1.jpg"))
                Image.fromarray(img_B_np).save(os.path.join(ind_dir, f"sample_{b_idx}_img2.jpg"))

            saved_grid = True

    avg_ssim_1, avg_ssim_2 = np.mean(results["ssim_1"]), np.mean(results["ssim_2"])
    avg_psnr_1, avg_psnr_2 = np.mean(results["psnr_1"]), np.mean(results["psnr_2"])
    avg_lpips_1, avg_lpips_2 = np.mean(results["lpips_1"]), np.mean(results["lpips_2"])
    
    success_rate_1 = (results["success_count_1"] / results["total_pairs"]) * 100
    success_rate_2 = (results["success_count_2"] / results["total_pairs"]) * 100

    metrics_report = (
        f"--- Image 1 Metrics ---\n"
        f"SSIM: {avg_ssim_1:.4f}\n"
        f"PSNR: {avg_psnr_1:.4f}\n"
        f"LPIPS: {avg_lpips_1:.4f}\n"
        f"Success Rate: {success_rate_1:.2f}%\n\n"
        f"--- Image 2 Metrics ---\n"
        f"SSIM: {avg_ssim_2:.4f}\n"
        f"PSNR: {avg_psnr_2:.4f}\n"
        f"LPIPS: {avg_lpips_2:.4f}\n"
        f"Success Rate: {success_rate_2:.2f}%"
    )
    
    print(metrics_report)
    with open(os.path.join("results", args.run_name, "final_metrics.txt"), "w") as f:
        f.write(metrics_report)

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help='Path to dataset', required=True)
    parser.add_argument('--run_name', help='Name of the experiment for saving models and results', required=True)
    parser.add_argument('--partition_file', help='CSV file with test indexes', required=True)
    parser.add_argument('--image_size', default=64, type=int, help='Dimension of the images', required=False)
    parser.add_argument('--batch_size', default=16, help='Batch size', type=int, required=False)
    parser.add_argument('--epochs', default=800, help='Number of epochs', type=int, required=False)
    parser.add_argument('--lr', default=3e-4, help='Learning rate', type=float, required=False)
    parser.add_argument('--device', default='cuda', help='Device, choose between [cuda, cpu]', required=False)
    parser.add_argument('--mode', default='train', choices=['train', 'eval'], help='Mode to run')

    args = parser.parse_args()
    args.image_size = (args.image_size, args.image_size)
    args.sampling_name = args.run_name

    if args.mode == 'train':
        train(args)
        eval(args)
    elif args.mode == 'eval':
        eval(args)

if __name__ == '__main__':
    launch()