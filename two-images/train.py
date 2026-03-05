import os, torch, numpy as np, math
import torch.nn.functional as F
import wandb
import lpips
from torch import optim
from utils import *
from modules import UNet
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

class Diffusion:
    def __init__(self, max_timesteps=300, alpha_start=0., alpha_max=0.5, img_size=256, device="cuda"):
        self.max_timesteps = max_timesteps
        self.img_size = img_size
        self.device = device
        self.alteration_per_t = (alpha_max - alpha_start) / max_timesteps
        
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(self.device)
        self.loss_fn_alex.eval()

    def noise_images(self, image_1, image_2, t):
        return image_1 * (1. - self.alteration_per_t * t)[:, None, None, None] + image_2 * (self.alteration_per_t * t)[:, None, None, None]

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.max_timesteps, size=(n,), device=self.device)

    def sample(self, model, superimposed_image, alpha_init=0.5):
        n = len(superimposed_image)
        init_timestep = math.ceil(alpha_init / self.alteration_per_t)
        model.eval()

        with torch.no_grad():
            x_t = superimposed_image.clone().to(self.device)

            for i in reversed(range(1, init_timestep + 1)):
                t = (torch.ones(n) * i).long().to(self.device)
                
                pred_v = model(x_t, t)

                if i == init_timestep:
                    anchor_v = pred_v.clone()
                    pred_v_aligned = pred_v
                else:
                    dist_straight = F.mse_loss(pred_v, anchor_v, reduction='none').view(n, -1).mean(dim=1)
                    dist_crossed  = F.mse_loss(pred_v, -anchor_v, reduction='none').view(n, -1).mean(dim=1)
                    
                    swap_mask = (dist_crossed < dist_straight).view(-1, 1, 1, 1)
                    
                    pred_v_aligned = torch.where(swap_mask, -pred_v, pred_v)
                    
                    anchor_v = torch.where(swap_mask, pred_v_aligned.clone(), anchor_v)

                x_t = x_t - self.alteration_per_t * pred_v_aligned
                
            model.train()

            best_pred_A = x_t.clamp(-1.0, 1.0)
            
            best_pred_B = (x_t + pred_v_aligned).clamp(-1.0, 1.0)

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

    model = UNet(c_in=3, c_out=3, device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    diffusion = Diffusion(img_size=args.image_size, device=device, alpha_max=args.alpha_max)

    wandb.init(
        project="demorph",
        name=args.run_name,
        config=vars(args)
    )

    fixed_images, fixed_images_add = next(iter(test_dataloader))
    fixed_images = fixed_images[:4].to(device)
    fixed_images_add = fixed_images_add[:4].to(device)
    fixed_superimposed = (fixed_images + fixed_images_add) / 2.
    
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

                v_target = images_add - images

                pred_v = model(x_t, t)

                mse_straight = F.mse_loss(pred_v, v_target, reduction='none').view(images.shape[0], -1).mean(dim=1)
                mse_crossed  = F.mse_loss(pred_v, -v_target, reduction='none').view(images.shape[0], -1).mean(dim=1)
                
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

        if (epoch + 1) % 10 == 0:
            sampled_A, sampled_B = diffusion.sample(model, fixed_superimposed, alpha_init=args.alpha_init)

            disp_images = (fixed_images.clamp(-1, 1) + 1) / 2
            disp_images = (disp_images * 255).type(torch.uint8)
            
            disp_images_add = (fixed_images_add.clamp(-1, 1) + 1) / 2
            disp_images_add = (disp_images_add * 255).type(torch.uint8)

            save_images(sampled_A, sampled_B, disp_images, disp_images_add, os.path.join("results", args.run_name, f"{epoch+1}.jpg"))
        
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

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    # 1. Track metrics separately for Image 1 and Image 2
    results = {
        "ssim_1": [], "ssim_2": [], 
        "psnr_1": [], "psnr_2": [], 
        "lpips_1": [], "lpips_2": [], 
        "success_count_1": 0, "success_count_2": 0, 
        "total_pairs": 0
    }
    
    saved_grid = False

    # Ensure output directories exist before writing files
    os.makedirs(os.path.join("samples", args.sampling_name), exist_ok=True)
    os.makedirs(os.path.join("results", args.run_name), exist_ok=True)
    
    # Create a directory specifically for the individual separated images
    ind_dir = os.path.join("samples", args.sampling_name, "individual")
    os.makedirs(ind_dir, exist_ok=True)

    for i, (images, images_add) in enumerate(test_dataloader):
        images = images.to(device)
        images_add = images_add.to(device)

        superimposed = (images + images_add) / 2.
        superimposed_np = ((superimposed.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()

        # Passed ground truth images here
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

            l_A = loss_fn_alex(images[k], (tensor_s_A.float() / 127.5) - 1.0).item()
            l_B = loss_fn_alex(images_add[k], (tensor_s_B.float() / 127.5) - 1.0).item()

            # Append metrics independently
            results["ssim_1"].append(s_A)
            results["ssim_2"].append(s_B)
            results["psnr_1"].append(p_A)
            results["psnr_2"].append(p_B)
            results["lpips_1"].append(l_A)
            results["lpips_2"].append(l_B)

            ssim_avg_A = structural_similarity(cur_gt_A, cur_super, data_range=255, channel_axis=-1)
            ssim_avg_B = structural_similarity(cur_gt_B, cur_super, data_range=255, channel_axis=-1)

            # Track success independently
            if s_A > ssim_avg_A:
                results["success_count_1"] += 1
            if s_B > ssim_avg_B:
                results["success_count_2"] += 1
            
            results["total_pairs"] += 1

        if i == 0 and not saved_grid:
            from PIL import Image
            
            aligned_A_stack = torch.stack(batch_s_A)
            aligned_B_stack = torch.stack(batch_s_B)
            
            # Still save the grid for a quick overview
            save_images(aligned_A_stack, aligned_B_stack, gt_A_eval, gt_B_eval, 
                        os.path.join("samples", args.sampling_name, "eval_grid.jpg"))
            
            # 2. Save individual image files
            for b_idx in range(len(batch_s_A)):
                img_A_np = batch_s_A[b_idx].cpu().permute(1, 2, 0).numpy()
                img_B_np = batch_s_B[b_idx].cpu().permute(1, 2, 0).numpy()
                
                Image.fromarray(img_A_np).save(os.path.join(ind_dir, f"sample_{b_idx}_img1.jpg"))
                Image.fromarray(img_B_np).save(os.path.join(ind_dir, f"sample_{b_idx}_img2.jpg"))

            saved_grid = True

    # Calculate final averages
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
    parser.add_argument('--alpha_max', default=0.5, type=float, help='Maximum weight of the added image at the last time step of the forward diffusion process: alpha_max', required=False)
    parser.add_argument('--alpha_init', default=0.5, type=float, help='Weight of the added image: alpha_init', required=False)
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
        # eval(args)
    elif args.mode == 'eval':
        eval(args)

if __name__ == '__main__':
    launch()