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

def test_renoising_configurations(model, diffusion, dataloader, device="cuda"):
    """
    Tests 4 different cold diffusion re-noising update steps to analyze 
    prediction error propagation using 1 pair of images over 10 steps.
    """
    model.eval()
    
    # 1. Fetch exactly 1 pair of images from the dataloader
    for images, images_add in dataloader:
        gt_A = images[:1].to(device)
        gt_B = images_add[:1].to(device)
        break
        
    superimposed_image = (gt_A + gt_B) / 2.0
    
    # Setup for exactly 10 sampling steps
    init_timestep = 10
    actual_alpha_init = diffusion.alteration_per_t * init_timestep
    
    results = {1: [], 2: [], 3: [], 4: []}
    
    # Run the full 10 steps independently for each configuration
    for config in [1, 2, 3, 4]:
        x_A = superimposed_image.clone()
        x_B = superimposed_image.clone()
        
        anchor_A, anchor_B = None, None
        
        with torch.no_grad():
            for i in reversed(range(1, init_timestep + 1)):
                t = (torch.ones(1) * i).long().to(device)
                t_minus_1 = (torch.ones(1) * (i - 1)).long().to(device)
                
                # --- PREDICTION & ALIGNMENT ---
                if i == init_timestep:
                    p_1, p_2 = torch.chunk(model(x_A, t), 2, dim=1)
                    
                    anchor_A = p_1.clamp(-1.0, 1.0)
                    anchor_B = p_2.clamp(-1.0, 1.0)
                    
                    best_pred_A = anchor_A.clone()
                    best_pred_B = anchor_B.clone()
                    
                    pA_2_aligned = anchor_B.clone()
                    pB_1_aligned = anchor_A.clone()
                    
                else:
                    pA_1, pA_2 = torch.chunk(model(x_A, t), 2, dim=1)
                    pB_1, pB_2 = torch.chunk(model(x_B, t), 2, dim=1)

                    # Align Path A
                    lpips_A_straight = diffusion.loss_fn_alex(pA_1, anchor_A) + diffusion.loss_fn_alex(pA_2, anchor_B)
                    lpips_A_crossed  = diffusion.loss_fn_alex(pA_1, anchor_B) + diffusion.loss_fn_alex(pA_2, anchor_A)
                    swap_mask_A = (lpips_A_crossed < lpips_A_straight).view(-1, 1, 1, 1)
                    
                    pA_1_aligned = torch.where(swap_mask_A, pA_2, pA_1)
                    pA_2_aligned = torch.where(swap_mask_A, pA_1, pA_2) 

                    # Align Path B
                    lpips_B_straight = diffusion.loss_fn_alex(pB_1, anchor_A) + diffusion.loss_fn_alex(pB_2, anchor_B)
                    lpips_B_crossed  = diffusion.loss_fn_alex(pB_1, anchor_B) + diffusion.loss_fn_alex(pB_2, anchor_A)
                    swap_mask_B = (lpips_B_crossed < lpips_B_straight).view(-1, 1, 1, 1)
                    
                    pB_2_aligned = torch.where(swap_mask_B, pB_1, pB_2)
                    pB_1_aligned = torch.where(swap_mask_B, pB_2, pB_1) 

                    best_pred_A = pA_1_aligned.clamp(-1.0, 1.0)
                    best_pred_B = pB_2_aligned.clamp(-1.0, 1.0)
                    
                    anchor_A = torch.where(swap_mask_A, best_pred_A.clone(), anchor_A)
                    anchor_B = torch.where(swap_mask_B, best_pred_B.clone(), anchor_B)

                # Dynamically determine the "other" Ground Truth for Config 1
                lpips_A_gtA = diffusion.loss_fn_alex(best_pred_A, gt_A)
                lpips_A_gtB = diffusion.loss_fn_alex(best_pred_A, gt_B)
                gt_other_for_A = gt_B if lpips_A_gtA < lpips_A_gtB else gt_A
                
                lpips_B_gtA = diffusion.loss_fn_alex(best_pred_B, gt_A)
                lpips_B_gtB = diffusion.loss_fn_alex(best_pred_B, gt_B)
                gt_other_for_B = gt_B if lpips_B_gtA < lpips_B_gtB else gt_A

                # --- CONFIGURATION LOGIC & RENOISING ---
                if config == 1:
                    # Baseline: Image 2 is the actual ground truth of the other image
                    img2_A = gt_other_for_A
                    img2_B = gt_other_for_B
                    x_A_next = x_A - diffusion.noise_images(best_pred_A, img2_A, t) + diffusion.noise_images(best_pred_A, img2_A, t_minus_1)
                    x_B_next = x_B - diffusion.noise_images(best_pred_B, img2_B, t) + diffusion.noise_images(best_pred_B, img2_B, t_minus_1)
                    
                elif config == 2:
                    # Mathematical algebraic extraction
                    extracted_B_from_A = (superimposed_image - best_pred_A * (1. - actual_alpha_init)) / actual_alpha_init
                    extracted_A_from_B = (superimposed_image - best_pred_B * (1. - actual_alpha_init)) / actual_alpha_init
                    img2_A = extracted_B_from_A.clamp(-1.0, 1.0)
                    img2_B = extracted_A_from_B.clamp(-1.0, 1.0)
                    x_A_next = x_A - diffusion.noise_images(best_pred_A, img2_A, t) + diffusion.noise_images(best_pred_A, img2_A, t_minus_1)
                    x_B_next = x_B - diffusion.noise_images(best_pred_B, img2_B, t) + diffusion.noise_images(best_pred_B, img2_B, t_minus_1)
                    
                elif config == 3:
                    # Cross-pollination using best_preds from the OTHER branch
                    img2_A = best_pred_B
                    img2_B = best_pred_A
                    x_A_next = x_A - diffusion.noise_images(best_pred_A, img2_A, t) + diffusion.noise_images(best_pred_A, img2_A, t_minus_1)
                    x_B_next = x_B - diffusion.noise_images(best_pred_B, img2_B, t) + diffusion.noise_images(best_pred_B, img2_B, t_minus_1)

                elif config == 4:
                    # Helena Montenegro's 'with_error_removal' sampling method
                    delta = (1 - diffusion.alteration_per_t * (t - 1)) / (1 - diffusion.alteration_per_t * t)
                    delta = delta.view(-1, 1, 1, 1)
                    
                    # Path A
                    other_image_A = pA_2_aligned.clamp(-1.0, 1.0)
                    error_A = 0
                    if i != init_timestep:
                        error_A = (actual_alpha_init * best_pred_A + (1. - actual_alpha_init) * other_image_A) - superimposed_image
                        error_A = error_A * ((diffusion.alteration_per_t * (t-1) - diffusion.alteration_per_t * t) / (actual_alpha_init - diffusion.alteration_per_t * t)).view(-1, 1, 1, 1)
                    
                    x_A_next = delta * x_A - (delta - 1) * other_image_A - error_A
                    
                    # Path B
                    other_image_B = pB_1_aligned.clamp(-1.0, 1.0)
                    error_B = 0
                    if i != init_timestep:
                        error_B = (actual_alpha_init * best_pred_B + (1. - actual_alpha_init) * other_image_B) - superimposed_image
                        error_B = error_B * ((diffusion.alteration_per_t * (t-1) - diffusion.alteration_per_t * t) / (actual_alpha_init - diffusion.alteration_per_t * t)).view(-1, 1, 1, 1)
                    
                    x_B_next = delta * x_B - (delta - 1) * other_image_B - error_B

                # --- ERROR CALCULATION ---
                # Calculate the TWO possible ground truth states at t-1
                gt_state_1 = diffusion.noise_images(gt_A, gt_B, t_minus_1)
                gt_state_2 = diffusion.noise_images(gt_B, gt_A, t_minus_1)
                
                # Check for permutation/swaps to assign the correct GT state to Path A and Path B
                mse_straight = F.mse_loss(x_A_next, gt_state_1).item() + F.mse_loss(x_B_next, gt_state_2).item()
                mse_crossed = F.mse_loss(x_A_next, gt_state_2).item() + F.mse_loss(x_B_next, gt_state_1).item()
                
                if mse_straight < mse_crossed:
                    mse_A = F.mse_loss(x_A_next, gt_state_1).item()
                    mse_B = F.mse_loss(x_B_next, gt_state_2).item()
                else:
                    mse_A = F.mse_loss(x_A_next, gt_state_2).item()
                    mse_B = F.mse_loss(x_B_next, gt_state_1).item()
                
                results[config].append((i - 1, mse_A, mse_B))
                
                # Prepare states for next iteration
                x_A = x_A_next
                x_B = x_B_next

    # --- SAVE FINDINGS TO TXT ---
    output_filename = "renoise_configurations_analysis.txt"
    with open(output_filename, "w") as f:
        f.write("Renoising Configurations - Error Propagation Analysis\n")
        f.write("=========================================================\n\n")
        
        config_descriptions = {
            1: "Configuration 1: Baseline - Image 2 = Actual Ground Truth of the other image",
            2: "Configuration 2: Image 2 = Mathematical algebraic extraction from superimposed (Current standard)",
            3: "Configuration 3: Image 2 = The best prediction of the opposing branch (Cross-pollination)",
            4: "Configuration 4: Helena Montenegro's 'with_error_removal' sampling method"
        }
        
        for config in [1, 2, 3, 4]:
            f.write(f"{config_descriptions[config]}\n")
            f.write("-" * 60 + "\n")
            for step_idx, (t_val, mse_A, mse_B) in enumerate(results[config]):
                f.write(f"Step {step_idx + 1} (target t={t_val}):\n")
                f.write(f"   Branch A MSE: {mse_A:.8f}\n")
                f.write(f"   Branch B MSE: {mse_B:.8f}\n")
                f.write(f"   Avg Step MSE: {(mse_A + mse_B)/2:.8f}\n")
            f.write("\n")
            
    print(f"Testing complete. Findings saved successfully to '{output_filename}'.")

def run_renoise_test(args):
    device = args.device
    test_dataloader = get_data(args, 'test')

    # Initialize model and load checkpoint
    model = UNet(c_in=3, c_out=6, device=device).to(device)
    model.load_state_dict(torch.load(os.path.join("models", args.run_name, "ckpt.pt"), map_location=torch.device(device)))
    
    # Initialize Diffusion
    diffusion = Diffusion(img_size=args.image_size, device=device, alpha_max=args.alpha_max)

    print("Starting Renoising Configurations Test...")
    test_renoising_configurations(model, diffusion, test_dataloader, device)

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
        eval(args)
    elif args.mode == 'eval':
        run_renoise_test(args)

if __name__ == '__main__':
    launch()