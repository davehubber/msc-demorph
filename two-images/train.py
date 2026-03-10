import os, torch, numpy as np, math
import torch.nn.functional as F
import wandb
import lpips
import argparse
from torch import optim
from utils import *
from modules import UNet
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

class Diffusion:
    def __init__(self, max_timesteps=250, arch="6-channel", img_size=64, device="cuda"):
        self.max_timesteps = max_timesteps
        self.img_size = img_size
        self.device = device
        self.arch = arch
        
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(self.device)
        self.loss_fn_alex.eval()

    def noise_images(self, image_A, image_B, t):
        alpha_t = (t / self.max_timesteps).view(-1, 1, 1, 1)
        
        if self.arch == "6-channel":
            # Linear cross-fade into each other
            x_top = (1. - alpha_t / 2.) * image_A + (alpha_t / 2.) * image_B
            x_bot = (alpha_t / 2.) * image_A + (1. - alpha_t / 2.) * image_B
            return torch.cat([x_top, x_bot], dim=1)
            
        elif self.arch == "9-channel":
            # Square root scheduling for variance stability as they fade into the middle
            x_top = torch.sqrt(1. - alpha_t) * image_A
            x_mid = torch.sqrt(alpha_t) * (image_A + image_B) / 2.0
            x_bot = torch.sqrt(1. - alpha_t) * image_B
            return torch.cat([x_top, x_mid, x_bot], dim=1)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.max_timesteps + 1, size=(n,), device=self.device)

    def sample(self, model, superimposed_image):
        n = len(superimposed_image)
        model.eval()

        with torch.no_grad():
            # 1. Construct initial fully degraded state x_T
            if self.arch == "6-channel":
                x_s = torch.cat([superimposed_image, superimposed_image], dim=1)
            else: # 9-channel
                zeros = torch.zeros_like(superimposed_image)
                x_s = torch.cat([zeros, superimposed_image, zeros], dim=1)

            # 2. Break Symmetry: Inject tiny noise to give the UNet distinct local features
            x_s = x_s + torch.randn_like(x_s) * 0.002
            x_s = x_s.clamp(-1.0, 1.0)

            for i in reversed(range(1, self.max_timesteps + 1)):
                t_s = (torch.ones(n) * i).long().to(self.device)
                t_prev = (torch.ones(n) * (i - 1)).long().to(self.device)

                # Predict clean state
                pred_x0 = model(x_s, t_s)
                
                if self.arch == "6-channel":
                    p_A, p_B = torch.chunk(pred_x0, 2, dim=1)
                else:
                    p_A, p_0, p_B = torch.chunk(pred_x0, 3, dim=1)
                    
                # 3. Handle Permutation Invariance Alignment
                if i == self.max_timesteps:
                    anchor_A = p_A.clamp(-1.0, 1.0)
                    anchor_B = p_B.clamp(-1.0, 1.0)
                    best_pred_A = anchor_A.clone()
                    best_pred_B = anchor_B.clone()
                else:
                    lpips_straight = self.loss_fn_alex(p_A, anchor_A) + self.loss_fn_alex(p_B, anchor_B)
                    lpips_crossed  = self.loss_fn_alex(p_A, anchor_B) + self.loss_fn_alex(p_B, anchor_A)
                    
                    swap_mask = (lpips_crossed < lpips_straight).view(-1, 1, 1, 1)
                    
                    best_pred_A = torch.where(swap_mask, p_B, p_A).clamp(-1.0, 1.0)
                    best_pred_B = torch.where(swap_mask, p_A, p_B).clamp(-1.0, 1.0)
                    
                    anchor_A = torch.where(swap_mask, best_pred_A.clone(), anchor_A)
                    anchor_B = torch.where(swap_mask, best_pred_B.clone(), anchor_B)

                # 4. Pure TACOS Algorithm: x_{s-1} = x_s - D(x_0, s) + D(x_0, s-1)
                D_x0_s = self.noise_images(best_pred_A, best_pred_B, t_s)
                D_x0_s_prev = self.noise_images(best_pred_A, best_pred_B, t_prev)
                
                x_s = x_s - D_x0_s + D_x0_s_prev
        
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

    c_in = 6 if args.architecture == "6-channel" else 9
    model = UNet(c_in=c_in, c_out=c_in, device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    diffusion = Diffusion(img_size=args.image_size, device=device, arch=args.architecture, max_timesteps=args.max_timesteps)

    wandb.init(
        project="demorph",
        name=args.run_name,
        config=vars(args)
    )
    
    global_step = 0
    scaler = torch.amp.GradScaler("cuda")

    fixed_test_images, fixed_test_images_add = next(iter(test_dataloader))
    fixed_test_images = fixed_test_images[:4].to(device)
    fixed_test_images_add = fixed_test_images_add[:4].to(device)

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

                # Calculate permutation invariant loss based on architecture
                if args.architecture == "6-channel":
                    pred_1, pred_2 = torch.chunk(predicted_both, 2, dim=1)
                    
                    mse_straight = F.mse_loss(pred_1, images, reduction='none').mean(dim=(1,2,3)) + \
                                   F.mse_loss(pred_2, images_add, reduction='none').mean(dim=(1,2,3))
                    
                    mse_crossed = F.mse_loss(pred_1, images_add, reduction='none').mean(dim=(1,2,3)) + \
                                  F.mse_loss(pred_2, images, reduction='none').mean(dim=(1,2,3))
                                  
                    loss = torch.min(mse_straight, mse_crossed).mean()
                    
                else: # 9-channel
                    pred_1, pred_0, pred_2 = torch.chunk(predicted_both, 3, dim=1)
                    
                    # Force the middle channel to route information outwards (target 0)
                    mse_0 = F.mse_loss(pred_0, torch.zeros_like(pred_0), reduction='none').mean(dim=(1,2,3))
                    
                    mse_straight = F.mse_loss(pred_1, images, reduction='none').mean(dim=(1,2,3)) + \
                                   F.mse_loss(pred_2, images_add, reduction='none').mean(dim=(1,2,3)) + mse_0
                                   
                    mse_crossed = F.mse_loss(pred_1, images_add, reduction='none').mean(dim=(1,2,3)) + \
                                  F.mse_loss(pred_2, images, reduction='none').mean(dim=(1,2,3)) + mse_0
                                  
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

            if (epoch + 1) % 50 == 0:
                superimposed = (fixed_test_images + fixed_test_images_add) / 2.
                sampled_A, sampled_B = diffusion.sample(model, superimposed)

                imgs_to_save = (fixed_test_images.clamp(-1, 1) + 1) / 2
                imgs_to_save = (imgs_to_save * 255).type(torch.uint8)
                
                imgs_add_to_save = (fixed_test_images_add.clamp(-1, 1) + 1) / 2
                imgs_add_to_save = (imgs_add_to_save * 255).type(torch.uint8)

                save_images(sampled_A, sampled_B, imgs_to_save, imgs_add_to_save, os.path.join("results", args.run_name, f"{epoch+1}.jpg"))
        
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

def eval(args):
    device = args.device
    test_dataloader = get_data(args, 'test')

    c_in = 6 if args.architecture == "6-channel" else 9
    model = UNet(c_in=c_in, c_out=c_in, device=device).to(device)
    model.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ckpt.pt"), map_location=torch.device(device)))
    model.eval()

    diffusion = Diffusion(img_size=args.image_size, device=device, arch=args.architecture, max_timesteps=args.max_timesteps)
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

            mse_straight = np.mean((cur_s_A.astype(np.float32) - cur_gt_A.astype(np.float32))**2) + np.mean((cur_s_B.astype(np.float32) - cur_gt_B.astype(np.float32))**2)
            mse_crossed  = np.mean((cur_s_A.astype(np.float32) - cur_gt_B.astype(np.float32))**2) + np.mean((cur_s_B.astype(np.float32) - cur_gt_A.astype(np.float32))**2)

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

            l_A = loss_fn_alex(images[k].unsqueeze(0), ((tensor_s_A.float() / 127.5) - 1.0).unsqueeze(0)).item()
            l_B = loss_fn_alex(images_add[k].unsqueeze(0), ((tensor_s_B.float() / 127.5) - 1.0).unsqueeze(0)).item()

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
            aligned_A_stack = torch.stack(batch_s_A)
            aligned_B_stack = torch.stack(batch_s_B)
            grid_limit = min(8, len(aligned_A_stack))
            
            save_images(aligned_A_stack[:grid_limit], 
                        aligned_B_stack[:grid_limit], 
                        gt_A_eval[:grid_limit], 
                        gt_B_eval[:grid_limit], 
                        os.path.join("samples", args.sampling_name, "eval_grid.jpg"))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help='Path to dataset', required=True)
    parser.add_argument('--run_name', help='Name of the experiment for saving models and results', required=True)
    parser.add_argument('--partition_file', help='CSV file with test indexes', required=True)
    parser.add_argument('--architecture', default='6-channel', choices=['6-channel', '9-channel'], help='Architecture to use')
    parser.add_argument('--max_timesteps', default=250, type=int, help='Total timesteps for diffusion')
    parser.add_argument('--image_size', default=64, type=int, help='Dimension of the images', required=False)
    parser.add_argument('--batch_size', default=16, help='Batch size', type=int, required=False)
    parser.add_argument('--epochs', default=1000, help='Number of epochs', type=int, required=False)
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