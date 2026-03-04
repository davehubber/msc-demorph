import os, torch, numpy as np, math
import torch.nn.functional as F
import wandb
import lpips
from torch import optim
from utils import *
from modules import UNet
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchvision.utils import save_image

class Diffusion:
    def __init__(self, max_timesteps=300, alpha_start=0., alpha_max=0.5, img_size=256, device="cuda"):
        self.max_timesteps = max_timesteps
        self.img_size = img_size
        self.device = device
        self.alteration_per_t = (alpha_max - alpha_start) / max_timesteps
        
        # Initialize LPIPS once here so it doesn't reload on every sample call
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(self.device)
        self.loss_fn_alex.eval()

    def noise_images(self, image_1, image_2, t):
        return image_1 * (1. - self.alteration_per_t * t)[:, None, None, None] + image_2 * (self.alteration_per_t * t)[:, None, None, None]
    
    def noise_images_superimposed(self, image, superimposed, t):
        return image * (1. - self.alteration_per_t * t)[:, None, None, None] + superimposed * (self.alteration_per_t * t)[:, None, None, None]

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.max_timesteps, size=(n,), device=self.device)

    def sample(self, model, superimposed_image, alpha_init=0.5):
        n = len(superimposed_image)
        init_timestep = math.ceil(alpha_init / self.alteration_per_t)
        model.eval()

        # Calculate the exact float alpha used at the initial timestep
        actual_alpha_init = self.alteration_per_t * init_timestep

        with torch.no_grad():
            x_A = superimposed_image.clone().to(self.device)
            x_B = superimposed_image.clone().to(self.device)

            for i in reversed(range(1, init_timestep + 1)):
                t = (torch.ones(n) * i).long().to(self.device)

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

                # --- ALGEBRAIC EXTRACTION (From Initial State) ---
                # Using the original superimposed_image and the scalar actual_alpha_init
                extracted_B_from_A = (superimposed_image - best_pred_A * (1. - actual_alpha_init)) / actual_alpha_init
                extracted_A_from_B = (superimposed_image - best_pred_B * (1. - actual_alpha_init)) / actual_alpha_init
                
                # Clamp the extracted replicas to prevent numerical overshoots
                extracted_B_from_A = extracted_B_from_A.clamp(-1.0, 1.0)
                extracted_A_from_B = extracted_A_from_B.clamp(-1.0, 1.0)

                # --- RENOISE (COLD DIFFUSION STEP) ---
                x_A = x_A - self.noise_images(best_pred_A, extracted_B_from_A, t) + self.noise_images(best_pred_A, extracted_B_from_A, t-1)
                x_B = x_B - self.noise_images(best_pred_B, extracted_A_from_B, t) + self.noise_images(best_pred_B, extracted_A_from_B, t-1)
        
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
                # Passed ground truth images here
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

def eval_with_recovery(args):
    device = args.device
    test_dataloader = get_data(args, 'test')

    model = UNet(c_in=3, c_out=6, device=device).to(device)
    model.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ckpt.pt"), map_location=torch.device(device)))
    model.eval()

    diffusion = Diffusion(img_size=args.image_size, device=device)

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    # 1. Track metrics and our new recovery counter
    results = {
        "ssim_1": [], "ssim_2": [], 
        "psnr_1": [], "psnr_2": [], 
        "lpips_1": [], "lpips_2": [], 
        "success_count_1": 0, "success_count_2": 0, 
        "recovered_count": 0,
        "total_pairs": 0
    }
    
    saved_grid = False

    os.makedirs(os.path.join("samples", args.sampling_name), exist_ok=True)
    os.makedirs(os.path.join("results", args.run_name), exist_ok=True)
    
    ind_dir = os.path.join("samples", args.sampling_name, "individual_recovery")
    os.makedirs(ind_dir, exist_ok=True)

    # Define a Laplacian kernel for blur detection
    laplacian_kernel = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]]).to(device)

    for i, (images, images_add) in enumerate(test_dataloader):
        images = images.to(device)
        images_add = images_add.to(device)

        # Ground truth superimposed average
        superimposed = (images + images_add) / 2.
        superimposed_np = ((superimposed.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()

        # Normal sampling process
        sampled_A, sampled_B = diffusion.sample(model, superimposed, args.alpha_init)
        
        # --- BLUR DETECTION & RECOVERY LOGIC ---
        # Convert to [0, 1] float space for mathematical operations
        float_A = sampled_A.float() / 255.0
        float_B = sampled_B.float() / 255.0
        float_super = (superimposed.clamp(-1, 1) + 1) / 2.0

        # Convert to grayscale to evaluate structural sharpness
        gray_A = float_A.mean(dim=1, keepdim=True)
        gray_B = float_B.mean(dim=1, keepdim=True)
        
        # Compute Variance of the Laplacian (Sharpness Score)
        lap_A = F.conv2d(gray_A, laplacian_kernel, padding=1).view(float_A.shape[0], -1).var(dim=1)
        lap_B = F.conv2d(gray_B, laplacian_kernel, padding=1).view(float_B.shape[0], -1).var(dim=1)

        corrected_sampled_A = sampled_A.clone()
        corrected_sampled_B = sampled_B.clone()

        for k in range(len(images)):
            # Ratio threshold: If variance is less than 20% of the other, we consider it blurred
            if lap_A[k] < lap_B[k] * 0.2:
                # Image A is blurry -> Recover A using Superimposed and B
                recovered_A = (2.0 * float_super[k] - float_B[k]).clamp(0, 1)
                corrected_sampled_A[k] = (recovered_A * 255).type(torch.uint8)
                results["recovered_count"] += 1
                
            elif lap_B[k] < lap_A[k] * 0.2:
                # Image B is blurry -> Recover B using Superimposed and A
                recovered_B = (2.0 * float_super[k] - float_A[k]).clamp(0, 1)
                corrected_sampled_B[k] = (recovered_B * 255).type(torch.uint8)
                results["recovered_count"] += 1
        # ---------------------------------------

        gt_A_eval = ((images.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
        gt_B_eval = ((images_add.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)

        batch_s_A = []
        batch_s_B = []

        for k in range(len(images)):
            cur_gt_A = gt_A_eval[k].cpu().permute(1, 2, 0).numpy()
            cur_gt_B = gt_B_eval[k].cpu().permute(1, 2, 0).numpy()
            
            # Use our dynamically corrected samples moving forward
            cur_s_A = corrected_sampled_A[k].cpu().permute(1, 2, 0).numpy()
            cur_s_B = corrected_sampled_B[k].cpu().permute(1, 2, 0).numpy()
            cur_super = superimposed_np[k]

            mse_straight = np.mean((cur_s_A - cur_gt_A)**2) + np.mean((cur_s_B - cur_gt_B)**2)
            mse_crossed  = np.mean((cur_s_A - cur_gt_B)**2) + np.mean((cur_s_B - cur_gt_A)**2)

            if mse_crossed < mse_straight:
                cur_s_A, cur_s_B = cur_s_B, cur_s_A
                tensor_s_A, tensor_s_B = corrected_sampled_B[k], corrected_sampled_A[k]
            else:
                tensor_s_A, tensor_s_B = corrected_sampled_A[k], corrected_sampled_B[k]

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
            
            save_images(aligned_A_stack, aligned_B_stack, gt_A_eval, gt_B_eval, 
                        os.path.join("samples", args.sampling_name, "eval_recovery_grid.jpg"))
            
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
        f"--- Image 1 Metrics (Post-Recovery) ---\n"
        f"SSIM: {avg_ssim_1:.4f}\n"
        f"PSNR: {avg_psnr_1:.4f}\n"
        f"LPIPS: {avg_lpips_1:.4f}\n"
        f"Success Rate: {success_rate_1:.2f}%\n\n"
        f"--- Image 2 Metrics (Post-Recovery) ---\n"
        f"SSIM: {avg_ssim_2:.4f}\n"
        f"PSNR: {avg_psnr_2:.4f}\n"
        f"LPIPS: {avg_lpips_2:.4f}\n"
        f"Success Rate: {success_rate_2:.2f}%\n\n"
        f"Total Fallback Recoveries Executed: {results['recovered_count']} out of {results['total_pairs']} pairs"
    )
    
    print(metrics_report)
    with open(os.path.join("results", args.run_name, "recovery_metrics.txt"), "w") as f:
        f.write(metrics_report)

def one_shot_eval(args):
    device = args.device
    test_dataloader = get_data(args, 'test')

    model = UNet(c_in=3, c_out=6, device=device).to(device)
    model.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ckpt.pt"), map_location=torch.device(device)))
    model.eval()

    diffusion = Diffusion(img_size=args.image_size, device=device, alpha_max=args.alpha_max)

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    
    init_timestep = math.ceil(args.alpha_init / diffusion.alteration_per_t)
    
    os.makedirs(os.path.join("samples", args.run_name, "one_shot"), exist_ok=True)
    os.makedirs(os.path.join("results", args.run_name), exist_ok=True)

    # 1. Initialize metrics dictionary
    results = {
        "ssim_1": [], "ssim_2": [], 
        "psnr_1": [], "psnr_2": [], 
        "lpips_1": [], "lpips_2": [], 
        "success_count_1": 0, "success_count_2": 0, 
        "total_pairs": 0
    }

    with torch.no_grad():
        for i, (images, images_add) in enumerate(test_dataloader):
            images = images.to(device)
            images_add = images_add.to(device)

            average = images * 0.5 + images_add * 0.5

            superimposed = images * 0.5 + average * 0.5
            n = len(superimposed)
            
            # Prepare baseline for success metric calculation
            superimposed_np = ((superimposed.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()

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

            # 2. Iterate through batch to calculate metrics
            for k in range(n):
                cur_gt_A = gt_A[k].cpu().permute(1, 2, 0).numpy()
                cur_gt_B = gt_B[k].cpu().permute(1, 2, 0).numpy()
                cur_s_A = final_A[k].cpu().permute(1, 2, 0).numpy()
                cur_s_B = final_B[k].cpu().permute(1, 2, 0).numpy()
                cur_super = superimposed_np[k]

                s_A = structural_similarity(cur_gt_A, cur_s_A, data_range=255, channel_axis=-1)
                s_B = structural_similarity(cur_gt_B, cur_s_B, data_range=255, channel_axis=-1)
                
                p_A = peak_signal_noise_ratio(cur_gt_A, cur_s_A, data_range=255)
                p_B = peak_signal_noise_ratio(cur_gt_B, cur_s_B, data_range=255)

                l_A = loss_fn_alex(images[k], (final_A[k].float() / 127.5) - 1.0).item()
                l_B = loss_fn_alex(images_add[k], (final_B[k].float() / 127.5) - 1.0).item()

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

    # 3. Calculate final averages and write report
    avg_ssim_1, avg_ssim_2 = np.mean(results["ssim_1"]), np.mean(results["ssim_2"])
    avg_psnr_1, avg_psnr_2 = np.mean(results["psnr_1"]), np.mean(results["psnr_2"])
    avg_lpips_1, avg_lpips_2 = np.mean(results["lpips_1"]), np.mean(results["lpips_2"])
    
    success_rate_1 = (results["success_count_1"] / results["total_pairs"]) * 100
    success_rate_2 = (results["success_count_2"] / results["total_pairs"]) * 100

    metrics_report = (
        f"--- One-Shot Image 1 Metrics ---\n"
        f"SSIM: {avg_ssim_1:.4f}\n"
        f"PSNR: {avg_psnr_1:.4f}\n"
        f"LPIPS: {avg_lpips_1:.4f}\n"
        f"Success Rate: {success_rate_1:.2f}%\n\n"
        f"--- One-Shot Image 2 Metrics ---\n"
        f"SSIM: {avg_ssim_2:.4f}\n"
        f"PSNR: {avg_psnr_2:.4f}\n"
        f"LPIPS: {avg_lpips_2:.4f}\n"
        f"Success Rate: {success_rate_2:.2f}%"
    )
    
    print("\n" + metrics_report)
    with open(os.path.join("results", args.run_name, "one_shot_metrics.txt"), "w") as f:
        f.write(metrics_report)

def save_transitions(args):
    device = args.device
    test_dataloader = get_data(args, 'test')

    # 1. Setup Model & Diffusion
    model = UNet(c_in=3, c_out=6, device=device).to(device)
    model.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ckpt.pt"), map_location=torch.device(device)))
    model.eval()

    diffusion = Diffusion(img_size=args.image_size, device=device, alpha_max=args.alpha_max)

    # 2. Extract strictly the first batch and slice the 1st and 3rd pairs
    images, images_add = next(iter(test_dataloader))
    images = images[[0, 2]].to(device)
    images_add = images_add[[0, 2]].to(device)

    superimposed = images * 0.5 + images_add * 0.5
    n = len(superimposed)
    init_timestep = math.ceil(args.alpha_init / diffusion.alteration_per_t)
    
    sampling_steps = getattr(args, 'sampling_steps', init_timestep)
    if sampling_steps is None:
        sampling_steps = init_timestep

    times = torch.linspace(init_timestep, 1, sampling_steps).round().long().tolist()
    times_next = times[1:] + [0]
    
    # Trackers for the 4 columns
    pred_A_1 = {0: [], 1: []}
    pred_A_2 = {0: [], 1: []}
    pred_B_1 = {0: [], 1: []}
    pred_B_2 = {0: [], 1: []}

    # 3. Custom Loop: Reverse Path Diffusion
    with torch.no_grad():
        x_A = superimposed.clone()
        x_B = superimposed.clone()
        
        gt_1 = images.clamp(-1.0, 1.0)
        gt_2 = images_add.clamp(-1.0, 1.0)
        
        for i, next_i in zip(times, times_next):
            t = (torch.ones(n) * i).long().to(device)
            t_next = (torch.ones(n) * next_i).long().to(device)
            
            if i == times[0]:
                p_1, p_2 = torch.chunk(model(x_A, t), 2, dim=1)
                
                anchor_A = p_1.clamp(-1.0, 1.0)
                anchor_B = p_2.clamp(-1.0, 1.0)
                
                # --- ALIGN GROUND TRUTHS USING LPIPS ---
                lpips_gt_straight = diffusion.loss_fn_alex(anchor_A, gt_1) + diffusion.loss_fn_alex(anchor_B, gt_2)
                lpips_gt_crossed  = diffusion.loss_fn_alex(anchor_A, gt_2) + diffusion.loss_fn_alex(anchor_B, gt_1)
                
                swap_mask_gt = (lpips_gt_crossed < lpips_gt_straight).view(-1, 1, 1, 1)
                aligned_gt_1 = torch.where(swap_mask_gt, gt_2, gt_1)
                aligned_gt_2 = torch.where(swap_mask_gt, gt_1, gt_2)
                
                pA_1_aligned = anchor_A
                pA_2_aligned = anchor_B
                pB_1_aligned = anchor_A
                pB_2_aligned = anchor_B
                
                best_pred_A = anchor_A.clone()
                best_pred_B = anchor_B.clone()
                
            else:
                pA_1, pA_2 = torch.chunk(model(x_A, t), 2, dim=1)
                pB_1, pB_2 = torch.chunk(model(x_B, t), 2, dim=1)

                # --- ALIGN PATH A USING LPIPS ---
                lpips_A_straight = diffusion.loss_fn_alex(pA_1, anchor_A) + diffusion.loss_fn_alex(pA_2, anchor_B)
                lpips_A_crossed  = diffusion.loss_fn_alex(pA_1, anchor_B) + diffusion.loss_fn_alex(pA_2, anchor_A)
                
                swap_mask_A = (lpips_A_crossed < lpips_A_straight).view(-1, 1, 1, 1)
                pA_1_aligned = torch.where(swap_mask_A, pA_2, pA_1)
                pA_2_aligned = torch.where(swap_mask_A, pA_1, pA_2)

                # --- ALIGN PATH B USING LPIPS ---
                lpips_B_straight = diffusion.loss_fn_alex(pB_1, anchor_A) + diffusion.loss_fn_alex(pB_2, anchor_B)
                lpips_B_crossed  = diffusion.loss_fn_alex(pB_1, anchor_B) + diffusion.loss_fn_alex(pB_2, anchor_A)
                
                swap_mask_B = (lpips_B_crossed < lpips_B_straight).view(-1, 1, 1, 1)
                pB_1_aligned = torch.where(swap_mask_B, pB_2, pB_1)
                pB_2_aligned = torch.where(swap_mask_B, pB_1, pB_2)

                best_pred_A = pA_1_aligned.clamp(-1.0, 1.0)
                best_pred_B = pB_2_aligned.clamp(-1.0, 1.0)
                
                anchor_A = torch.where(swap_mask_A, best_pred_A.clone(), anchor_A)
                anchor_B = torch.where(swap_mask_B, best_pred_B.clone(), anchor_B)

            # --- TACOS correction ---
            corr_A = x_A - diffusion.noise_images(best_pred_A, aligned_gt_2, t)
            corr_B = x_B - diffusion.noise_images(best_pred_B, aligned_gt_1, t)

            # 4. Save current timestep visuals
            for b_idx in range(n):
                show_pA1 = (pA_1_aligned[b_idx].clone().clamp(-1.0, 1.0) + 1) / 2
                show_pA2 = (pA_2_aligned[b_idx].clone().clamp(-1.0, 1.0) + 1) / 2
                show_pB1 = (pB_1_aligned[b_idx].clone().clamp(-1.0, 1.0) + 1) / 2
                show_pB2 = (pB_2_aligned[b_idx].clone().clamp(-1.0, 1.0) + 1) / 2

                pred_A_1[b_idx].append(show_pA1)
                pred_A_2[b_idx].append(show_pA2)
                pred_B_1[b_idx].append(show_pB1)
                pred_B_2[b_idx].append(show_pB2)

            # Renoising jumps directly to t_next
            x_A = corr_A + diffusion.noise_images(best_pred_A, aligned_gt_2, t_next)
            x_B = corr_B + diffusion.noise_images(best_pred_B, aligned_gt_1, t_next)
            
    # 5. Output Image Grids
    save_dir = os.path.join("results", args.run_name, "transitions")
    os.makedirs(save_dir, exist_ok=True)
    original_indices = [1, 3]
    
    for b_idx, real_idx in enumerate(original_indices):
        pair_sequence = []
        for step_idx in range(len(pred_A_1[b_idx])):
            pair_sequence.append(pred_A_1[b_idx][step_idx])
            pair_sequence.append(pred_A_2[b_idx][step_idx])
            pair_sequence.append(pred_B_1[b_idx][step_idx])
            pair_sequence.append(pred_B_2[b_idx][step_idx])
            
        grid_tensor = torch.stack(pair_sequence)
        save_path = os.path.join(save_dir, f"pair_{real_idx}_full_transition.jpg")
        
        save_image(grid_tensor, save_path, nrow=4)
        print(f"Saved 4-column transition grid for Pair {real_idx} to {save_path}")

def verify_tacos_step(args):
    device = args.device
    test_dataloader = get_data(args, 'test')

    # 1. Setup Model & Diffusion
    model = UNet(c_in=3, c_out=6, device=device).to(device)
    model.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ckpt.pt"), map_location=torch.device(device)))
    model.eval()

    diffusion = Diffusion(img_size=args.image_size, device=device, alpha_max=args.alpha_max)

    # 2. Extract strictly the first batch
    images, images_add = next(iter(test_dataloader))
    images = images.to(device)
    images_add = images_add.to(device)

    superimposed = (images + images_add) / 2.
    n = len(superimposed)
    init_timestep = math.ceil(args.alpha_init / diffusion.alteration_per_t)
    
    os.makedirs(os.path.join("results", args.run_name, "verification"), exist_ok=True)
    log_file_path = os.path.join("results", args.run_name, "verification", "tacos_mse_log.txt")

    with torch.no_grad():
        x_A = superimposed.clone()
        x_B = superimposed.clone()
        
        gt_1 = images.clamp(-1.0, 1.0)
        gt_2 = images_add.clamp(-1.0, 1.0)
        
        # Define current step (t) and the target step (t-1)
        t_val = init_timestep
        t = (torch.ones(n) * t_val).long().to(device)
        t_minus_1 = (torch.ones(n) * (t_val - 1)).long().to(device)

        # --- Forward Pass at t ---
        p_1, p_2 = torch.chunk(model(x_A, t), 2, dim=1)
        
        best_pred_1 = p_1.clamp(-1.0, 1.0)
        best_pred_2 = p_2.clamp(-1.0, 1.0)

        anchor_A = best_pred_1.clone()
        anchor_B = best_pred_2.clone()
        
        # Align Ground Truths to match model predictions
        mse_gt_straight = F.mse_loss(anchor_A, gt_1, reduction='none').view(n, -1).mean(dim=1) + \
                          F.mse_loss(anchor_B, gt_2, reduction='none').view(n, -1).mean(dim=1)
        mse_gt_crossed  = F.mse_loss(anchor_A, gt_2, reduction='none').view(n, -1).mean(dim=1) + \
                          F.mse_loss(anchor_B, gt_1, reduction='none').view(n, -1).mean(dim=1)
        
        swap_mask_gt = (mse_gt_crossed < mse_gt_straight).view(-1, 1, 1, 1)
        aligned_gt_1 = torch.where(swap_mask_gt, gt_2, gt_1)
        aligned_gt_2 = torch.where(swap_mask_gt, gt_1, gt_2)

        # --- 1 Step of TACOS Sampling to t-1 ---
        # Renoising: Model prediction is noised by the corresponding ground truth
        x_A_t_minus_1 = diffusion.noise_images(best_pred_1, aligned_gt_2, t_minus_1)
        x_B_t_minus_1 = diffusion.noise_images(best_pred_2, aligned_gt_1, t_minus_1)

        # --- Ground Truth & Averages at t-1 ---
        # The average of the predicted paths at t-1
        sampled_avg_t_minus_1 = (x_A_t_minus_1 + x_B_t_minus_1) / 2.0

        # The true paths at t-1 based on the forward process
        gt_path_A_t_minus_1 = diffusion.noise_images(aligned_gt_1, aligned_gt_2, t_minus_1)
        gt_path_B_t_minus_1 = diffusion.noise_images(aligned_gt_2, aligned_gt_1, t_minus_1)
        
        # The true superimposed average at t-1
        gt_avg_t_minus_1 = (gt_path_A_t_minus_1 + gt_path_B_t_minus_1) / 2.0

        # --- Calculate MSE Metrics ---
        # Check how well the sampled average matches the GT average
        mse_avg_per_pair = F.mse_loss(sampled_avg_t_minus_1, gt_avg_t_minus_1, reduction='none').view(n, -1).mean(dim=1)
        
        # Check how well the individual sampled states match the individual GT states
        mse_A_per_pair = F.mse_loss(x_A_t_minus_1, gt_path_A_t_minus_1, reduction='none').view(n, -1).mean(dim=1)
        mse_B_per_pair = F.mse_loss(x_B_t_minus_1, gt_path_B_t_minus_1, reduction='none').view(n, -1).mean(dim=1)

        # --- Save Logs ---
        with open(log_file_path, "w") as f:
            f.write(f"Verification of TACOS Step (t={t_val} -> t={t_val-1})\n")
            f.write("="*50 + "\n")
            for i in range(n):
                log_str = (f"Pair {i}: \n"
                           f"  MSE (Sampled Avg vs GT Avg): {mse_avg_per_pair[i].item():.16f}\n"
                           f"  MSE (Sampled x_A vs GT x_A): {mse_A_per_pair[i].item():.16f}\n"
                           f"  MSE (Sampled x_B vs GT x_B): {mse_B_per_pair[i].item():.16f}\n\n")
                f.write(log_str)
                print(log_str.strip())

        # --- Save Visual Verification Grid ---
        def norm(tensor):
            return (tensor.clamp(-1.0, 1.0) + 1) / 2.0

        # We'll save the first 4 pairs to a grid (to avoid overly large output images)
        num_show = min(4, n)
        visual_list = []
        for i in range(num_show):
            visual_list.extend([
                norm(superimposed[i]),          # 1. Start Average (t)
                norm(sampled_avg_t_minus_1[i]), # 2. Sampled Average (t-1)
                norm(gt_avg_t_minus_1[i]),      # 3. Ground Truth Average (t-1)
                norm(x_A_t_minus_1[i]),         # 4. Sampled Path A (t-1)
                norm(gt_path_A_t_minus_1[i]),   # 5. Ground Truth Path A (t-1)
                norm(x_B_t_minus_1[i]),         # 6. Sampled Path B (t-1)
                norm(gt_path_B_t_minus_1[i])    # 7. Ground Truth Path B (t-1)
            ])

        grid_tensor = torch.stack(visual_list)
        grid_save_path = os.path.join("results", args.run_name, "verification", "tacos_step_verification.jpg")
        save_image(grid_tensor, grid_save_path, nrow=7)
        
        print(f"\nSaved MSE log to: {log_file_path}")
        print(f"Saved Image Grid (7 columns) to: {grid_save_path}")

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
    parser.add_argument('--mode', default='train', choices=['train', 'eval', 'one_shot', 'transition'], help='Mode to run')
    parser.add_argument('--sampling_steps', default=300, type=int, help='Number of strided steps for sampling')

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
    elif args.mode == 'transition':
        save_transitions(args)

if __name__ == '__main__':
    launch()