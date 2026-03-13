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
    def __init__(self, max_timesteps=250, alpha_start=0., alpha_max=0.5, img_size=256, device="cuda"):
        self.max_timesteps = max_timesteps
        self.img_size = img_size
        self.device = device
        self.alteration_per_t = (alpha_max - alpha_start) / max_timesteps
        
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(self.device)
        self.loss_fn_alex.eval()

    def noise_images(self, image_1, image_2, t):
        return image_1 * (1. - self.alteration_per_t * t)[:, None, None, None] + image_2 * (self.alteration_per_t * t)[:, None, None, None]

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.max_timesteps + 1, size=(n,), device=self.device)

    @staticmethod
    def _append_clamp_stats(stats_dict, name, x, low=-1.0, high=1.0):
        below = (x < low).float().mean().item()
        above = (x > high).float().mean().item()
        stats_dict[f"{name}_below"].append(below)
        stats_dict[f"{name}_above"].append(above)
        stats_dict[f"{name}_oor"].append(below + above)  # out-of-range fraction
        stats_dict[f"{name}_min"].append(x.min().item())
        stats_dict[f"{name}_max"].append(x.max().item())

    @staticmethod
    def _summarize_clamp_stats(stats_dict):
        summary = {}
        per_step = {}
        for k, v in stats_dict.items():
            per_step[k] = v
            if len(v) > 0:
                summary[f"{k}_mean"] = float(np.mean(v))
                summary[f"{k}_max"] = float(np.max(v))
            else:
                summary[f"{k}_mean"] = 0.0
                summary[f"{k}_max"] = 0.0
        return {"summary": summary, "per_step": per_step}

    def sample(self, model, superimposed_image, alpha_init=0.5, return_clamp_stats=False):
        n = len(superimposed_image)
        init_timestep = math.ceil(alpha_init / self.alteration_per_t)
        model.eval()

        actual_alpha_init = self.alteration_per_t * init_timestep

        clamp_stats = None
        if return_clamp_stats:
            clamp_stats = {
                "anchor_A_below": [], "anchor_A_above": [], "anchor_A_oor": [], "anchor_A_min": [], "anchor_A_max": [],
                "anchor_B_below": [], "anchor_B_above": [], "anchor_B_oor": [], "anchor_B_min": [], "anchor_B_max": [],
                "pA_1_below": [], "pA_1_above": [], "pA_1_oor": [], "pA_1_min": [], "pA_1_max": [],
                "pA_2_below": [], "pA_2_above": [], "pA_2_oor": [], "pA_2_min": [], "pA_2_max": [],
                "pB_1_below": [], "pB_1_above": [], "pB_1_oor": [], "pB_1_min": [], "pB_1_max": [],
                "pB_2_below": [], "pB_2_above": [], "pB_2_oor": [], "pB_2_min": [], "pB_2_max": [],
                "best_pred_A_below": [], "best_pred_A_above": [], "best_pred_A_oor": [], "best_pred_A_min": [], "best_pred_A_max": [],
                "best_pred_B_below": [], "best_pred_B_above": [], "best_pred_B_oor": [], "best_pred_B_min": [], "best_pred_B_max": [],
                "extracted_B_from_A_below": [], "extracted_B_from_A_above": [], "extracted_B_from_A_oor": [], "extracted_B_from_A_min": [], "extracted_B_from_A_max": [],
                "extracted_A_from_B_below": [], "extracted_A_from_B_above": [], "extracted_A_from_B_oor": [], "extracted_A_from_B_min": [], "extracted_A_from_B_max": [],
            }

        with torch.no_grad():
            x_A = superimposed_image.clone().to(self.device)
            x_B = superimposed_image.clone().to(self.device)

            for i in reversed(range(1, init_timestep + 1)):
                t = (torch.ones(n) * i).long().to(self.device)

                if i == init_timestep:
                    p_1, p_2 = torch.chunk(model(x_A, t), 2, dim=1)

                    if return_clamp_stats:
                        self._append_clamp_stats(clamp_stats, "anchor_A", p_1)
                        self._append_clamp_stats(clamp_stats, "anchor_B", p_2)

                    anchor_A = p_1.clamp(-1.0, 1.0)
                    anchor_B = p_2.clamp(-1.0, 1.0)

                    best_pred_A = anchor_A.clone()
                    best_pred_B = anchor_B.clone()

                    # Optional: if you want to log init best_pred too
                    if return_clamp_stats:
                        self._append_clamp_stats(clamp_stats, "best_pred_A", p_1)
                        self._append_clamp_stats(clamp_stats, "best_pred_B", p_2)

                else:
                    raw_pA_1, raw_pA_2 = torch.chunk(model(x_A, t), 2, dim=1)
                    raw_pB_1, raw_pB_2 = torch.chunk(model(x_B, t), 2, dim=1)

                    if return_clamp_stats:
                        self._append_clamp_stats(clamp_stats, "pA_1", raw_pA_1)
                        self._append_clamp_stats(clamp_stats, "pA_2", raw_pA_2)
                        self._append_clamp_stats(clamp_stats, "pB_1", raw_pB_1)
                        self._append_clamp_stats(clamp_stats, "pB_2", raw_pB_2)

                    pA_1, pA_2 = raw_pA_1.clamp(-1.0, 1.0), raw_pA_2.clamp(-1.0, 1.0)
                    pB_1, pB_2 = raw_pB_1.clamp(-1.0, 1.0), raw_pB_2.clamp(-1.0, 1.0)

                    lpips_A_straight = self.loss_fn_alex(pA_1, anchor_A) + self.loss_fn_alex(pA_2, anchor_B)
                    lpips_A_crossed  = self.loss_fn_alex(pA_1, anchor_B) + self.loss_fn_alex(pA_2, anchor_A)

                    swap_mask_A = (lpips_A_crossed < lpips_A_straight).view(-1, 1, 1, 1)
                    raw_best_pred_A = torch.where(swap_mask_A, pA_2, pA_1)

                    lpips_B_straight = self.loss_fn_alex(pB_1, anchor_A) + self.loss_fn_alex(pB_2, anchor_B)
                    lpips_B_crossed  = self.loss_fn_alex(pB_1, anchor_B) + self.loss_fn_alex(pB_2, anchor_A)

                    swap_mask_B = (lpips_B_crossed < lpips_B_straight).view(-1, 1, 1, 1)
                    raw_best_pred_B = torch.where(swap_mask_B, pB_1, pB_2)

                    if return_clamp_stats:
                        self._append_clamp_stats(clamp_stats, "best_pred_A", raw_best_pred_A)
                        self._append_clamp_stats(clamp_stats, "best_pred_B", raw_best_pred_B)

                    best_pred_A = raw_best_pred_A.clamp(-1.0, 1.0)
                    best_pred_B = raw_best_pred_B.clamp(-1.0, 1.0)

                    anchor_A = torch.where(swap_mask_A, best_pred_A.clone(), anchor_A)
                    anchor_B = torch.where(swap_mask_B, best_pred_B.clone(), anchor_B)

                raw_extracted_B_from_A = (superimposed_image - best_pred_A * (1. - actual_alpha_init)) / actual_alpha_init
                raw_extracted_A_from_B = (superimposed_image - best_pred_B * (1. - actual_alpha_init)) / actual_alpha_init

                if return_clamp_stats:
                    self._append_clamp_stats(clamp_stats, "extracted_B_from_A", raw_extracted_B_from_A)
                    self._append_clamp_stats(clamp_stats, "extracted_A_from_B", raw_extracted_A_from_B)

                extracted_B_from_A = raw_extracted_B_from_A.clamp(-1.0, 1.0)
                extracted_A_from_B = raw_extracted_A_from_B.clamp(-1.0, 1.0)

                x_A = x_A - self.noise_images(best_pred_A, extracted_B_from_A, t) + self.noise_images(best_pred_A, extracted_B_from_A, t-1)
                x_B = x_B - self.noise_images(best_pred_B, extracted_A_from_B, t) + self.noise_images(best_pred_B, extracted_A_from_B, t-1)

        model.train()

        final_A = (best_pred_A + 1) / 2
        final_A = (final_A * 255).type(torch.uint8)

        final_B = (best_pred_B + 1) / 2
        final_B = (final_B * 255).type(torch.uint8)

        if return_clamp_stats:
            return final_A, final_B, self._summarize_clamp_stats(clamp_stats)

        return final_A, final_B

def train(args):
    setup_logging(args.run_name)
    device = args.device
    train_dataloader = get_data(args, 'train')
    test_dataloader = get_data(args, 'test')

    model = UNet(c_in=3, c_out=6, device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    diffusion = Diffusion(img_size=args.image_size, device=device, alpha_max=args.alpha_max)

    if args.use_lpips_loss:
        loss_fn_alex_train = lpips.LPIPS(net='alex').to(device)
        loss_fn_alex_train.eval()

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
                pred_1, pred_2 = torch.chunk(predicted_both, 2, dim=1)

                mse_straight = F.mse_loss(pred_1, images, reduction='none').view(images.shape[0], -1).mean(dim=1) + \
                               F.mse_loss(pred_2, images_add, reduction='none').view(images.shape[0], -1).mean(dim=1)
                
                mse_crossed = F.mse_loss(pred_1, images_add, reduction='none').view(images.shape[0], -1).mean(dim=1) + \
                              F.mse_loss(pred_2, images, reduction='none').view(images.shape[0], -1).mean(dim=1)
                
                if args.use_lpips_loss:
                    pred_1_c = pred_1.clamp(-1.0, 1.0)
                    pred_2_c = pred_2.clamp(-1.0, 1.0)

                    lpips_straight = (loss_fn_alex_train(pred_1_c, images) + loss_fn_alex_train(pred_2_c, images_add)).view(-1)
                    lpips_crossed = (loss_fn_alex_train(pred_1_c, images_add) + loss_fn_alex_train(pred_2_c, images)).view(-1)
                    
                    loss_straight = mse_straight + args.lambda_lpips * lpips_straight
                    loss_crossed = mse_crossed + args.lambda_lpips * lpips_crossed
                    
                    loss = torch.min(loss_straight, loss_crossed).mean()
                else:
                    loss = torch.min(mse_straight, mse_crossed).mean()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                sampled_A, sampled_B = diffusion.sample(model, superimposed, alpha_init=args.alpha_init)

                imgs_to_save = (fixed_test_images.clamp(-1, 1) + 1) / 2
                imgs_to_save = (imgs_to_save * 255).type(torch.uint8)
                
                imgs_add_to_save = (fixed_test_images_add.clamp(-1, 1) + 1) / 2
                imgs_add_to_save = (imgs_add_to_save * 255).type(torch.uint8)

                save_images(sampled_A, sampled_B, imgs_to_save, imgs_add_to_save, os.path.join("results", args.run_name, f"{epoch+1}.jpg"))
        
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

        sampled_A, sampled_B, clamp_stats = diffusion.sample(
            model,
            superimposed,
            args.alpha_init,
            return_clamp_stats=True
        )

        print(
            f"[eval batch {i}] "
            f"best_pred_A_oor={clamp_stats['summary']['best_pred_A_oor_mean']:.4f}, "
            f"best_pred_B_oor={clamp_stats['summary']['best_pred_B_oor_mean']:.4f}, "
            f"extracted_B_from_A_oor={clamp_stats['summary']['extracted_B_from_A_oor_mean']:.4f}, "
            f"extracted_A_from_B_oor={clamp_stats['summary']['extracted_A_from_B_oor_mean']:.4f}"
        )
        
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
    parser.add_argument('--alpha_max', default=0.5, type=float, help='Maximum weight of the added image at the last time step of the forward diffusion process: alpha_max', required=False)
    parser.add_argument('--alpha_init', default=0.5, type=float, help='Weight of the added image: alpha_init', required=False)
    parser.add_argument('--image_size', default=64, type=int, help='Dimension of the images', required=False)
    parser.add_argument('--batch_size', default=16, help='Batch size', type=int, required=False)
    parser.add_argument('--epochs', default=1000, help='Number of epochs', type=int, required=False)
    parser.add_argument('--lr', default=3e-4, help='Learning rate', type=float, required=False)
    parser.add_argument('--device', default='cuda', help='Device, choose between [cuda, cpu]', required=False)
    parser.add_argument('--mode', default='train', choices=['train', 'eval'], help='Mode to run')
    
    parser.add_argument('--use_lpips_loss', action='store_true', help='If passed, adds LPIPS perceptual loss to the training objective')
    parser.add_argument('--lambda_lpips', default=1.0, type=float, help='Multiplier for the LPIPS loss component', required=False)

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