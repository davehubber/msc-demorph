import os, torch, numpy as np, math
import torch.nn as nn
import wandb
import lpips
from torch import optim
from utils import *
from modules import UNet
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

class Diffusion:
    def __init__(self, max_timesteps=250, alpha_start=0., alpha_max=0.8, img_size=256, device="cuda"):
        self.max_timesteps = max_timesteps
        self.img_size = img_size
        self.device = device
        self.alteration_per_t = (alpha_max - alpha_start) / max_timesteps      # size of alteration in each timestep

    def noise_images(self, original_image, added_image, t):
        return original_image * (1. - self.alteration_per_t * t)[:, None, None, None] + added_image * (self.alteration_per_t * t)[:, None, None, None]

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.max_timesteps + 1, size=(n,), device=self.device)

    def sample(self, model, superimposed_image, alpha_init = 0.5, prediction="original", sampling_method="with_error"):
        n = len(superimposed_image)
        init_timestep = math.ceil(alpha_init / self.alteration_per_t)
        model.eval()
        with torch.no_grad():
            x = superimposed_image.to(self.device)
            for i in reversed(range(1, init_timestep + 1)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_image = model(x, t).to(self.device)
                
                if (prediction == "original" or prediction == "added") and sampling_method == "one_step":
                    x = predicted_image
                    break

                elif prediction == "differences" and sampling_method == "one_step":
                    x = x + predicted_image * t[:, None, None, None]
                    break
            
                elif prediction == "differences":
                    x = x + predicted_image

                elif prediction == "original" and sampling_method == "with_error_removal":
                    alpha_t = (self.alteration_per_t * t)[:, None, None, None]
                    
                    other_image = (x - (1 - alpha_t) * predicted_image) / alpha_t 
                    
                    error = 0
                    if i != init_timestep:
                        error = ((1. - alpha_init) * predicted_image + alpha_init * other_image) - superimposed_image
                        error = error * ((self.alteration_per_t * (t-1) - self.alteration_per_t * t) / (alpha_init - self.alteration_per_t * t))[:, None, None, None]
                    
                    delta = (1 - self.alteration_per_t * (t - 1)) / ((1 - self.alteration_per_t * t))
                    x_t = delta[:, None, None, None] * x - (delta - 1)[:, None, None, None] * other_image - error
                
                elif prediction == "added" and sampling_method == "with_error_removal":
                    other_image = (x - (self.alteration_per_t * t)[:, None, None, None] * predicted_image) / (1-self.alteration_per_t * t)[:, None, None, None] 
                
                    error = 0
                    if i != init_timestep:
                        error = (alpha_init * predicted_image + (1.-alpha_init) * other_image) - superimposed_image
                        error = error * ((self.alteration_per_t * (t-1) - self.alteration_per_t * t) / (alpha_init - self.alteration_per_t * t))[:, None, None, None]
                    
                    delta = (1 - self.alteration_per_t * (t - 1)) / ((1 - self.alteration_per_t * t))
                    x_t = delta[:, None, None, None] * x - (delta - 1)[:, None, None, None] * predicted_image - error
                
                elif sampling_method == "cold_diffusion":
                    alpha_t = (self.alteration_per_t * t)[:, None, None, None]
                    other_image_i = (superimposed_image - (alpha_init) * predicted_image) / (1.-alpha_init)
                    
                    if prediction == "added":
                        other_image = (x - alpha_t * predicted_image) / (1. - alpha_t)
                        
                        x_t = self.noise_images(other_image_i, predicted_image, t-1) + x - self.noise_images(other_image_i, predicted_image, t)
                        
                    elif prediction == "original":
                        other_image = (x - (1. - alpha_t) * predicted_image) / alpha_t
                        
                        x_t = self.noise_images(predicted_image, other_image_i, t-1) + x - self.noise_images(predicted_image, other_image_i, t)
                        
                    else:
                        print("Invalid prediction/sampling_method combination.")
                        exit(-1)

                elif prediction == 'added' and sampling_method == "without_error_removal":
                    delta = (1 - self.alteration_per_t * (t - 1)) / ((1 - self.alteration_per_t * t))
                    x_t = delta[:, None, None, None] * x - (delta - 1)[:, None, None, None] * predicted_image
                elif prediction == 'original' and sampling_method == "without_error_removal":
                    other_image = (x - (1-self.alteration_per_t * t)[:, None, None, None] * predicted_image) / (self.alteration_per_t * t)[:, None, None, None]
                    delta = (1 - self.alteration_per_t * (t - 1)) / ((1 - self.alteration_per_t * t))
                    x_t = delta[:, None, None, None] * x - (delta - 1)[:, None, None, None] * other_image
                else:
                    print(f"Invalid prediction - {prediction} - and sampling_method - {sampling_method} - combination.")
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
                sampled_images, other_images = diffusion.sample(model, (images + images_add) / 2., prediction=args.prediction, sampling_method=args.sampling_method)
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
    
    lpips_model = lpips.LPIPS(net='alex').to(device)

    sample_dir = os.path.join("samples", args.sampling_name)
    os.makedirs(sample_dir, exist_ok=True)

    ssim_o, ssim_a, lpips_o, lpips_a, psnr_o, psnr_a = [], [], [], [], [], []
    success_count = 0
    total_count = 0

    for i, (images, images_add) in enumerate(test_dataloader):
        images = images.to(device)
        images_add = images_add.to(device)
        
        # Explicitly define the superimposed image so we can use it for the %S metric
        superimposed = images * (1 - args.alpha_init) + images_add * args.alpha_init
        
        sampled_images, sampled_other_image = diffusion.sample(model, superimposed, args.alpha_init, prediction=args.prediction, sampling_method=args.sampling_method)
        
        # Scale all images to 0-255 uint8
        images = (images.clamp(-1, 1) + 1) / 2
        images = (images * 255).type(torch.uint8)
        
        images_add = (images_add.clamp(-1, 1) + 1) / 2
        images_add = (images_add * 255).type(torch.uint8)
        
        superimposed = (superimposed.clamp(-1, 1) + 1) / 2
        superimposed = (superimposed * 255).type(torch.uint8)
        
        sampled_images.to(device)
        save_images(sampled_images, sampled_other_image, images, images_add, os.path.join("samples", args.sampling_name, f"{i}.jpg"))

        # Convert to numpy for skimage metrics
        images_np = images.to('cpu').permute(0, 2, 3, 1).numpy()
        sampled_images_np = sampled_images.to('cpu').permute(0, 2, 3, 1).numpy()
        images_add_np = images_add.to('cpu').permute(0, 2, 3, 1).numpy()
        sampled_other_image_np = sampled_other_image.to('cpu').permute(0, 2, 3, 1).numpy()
        superimposed_np = superimposed.to('cpu').permute(0, 2, 3, 1).numpy()
        
        with torch.no_grad():
            for k in range(len(images_np)):
                ssim_1 = structural_similarity(images_np[k], sampled_images_np[k], data_range=255, channel_axis=-1)
                ssim_2 = structural_similarity(images_add_np[k], sampled_images_np[k], data_range=255, channel_axis=-1)
                
                # Identify which prediction belongs to which ground truth
                if ssim_1 > ssim_2:
                    so, sa, po, pa = calculate_metrics(images_np[k], images_add_np[k], sampled_images_np[k], sampled_other_image_np[k])
                    lo = lpips_model((images[k].unsqueeze(0).float() - 127.5) / 127.5, (sampled_images[k].unsqueeze(0).float() - 127.5) / 127.5)
                    la = lpips_model((images_add[k].unsqueeze(0).float() - 127.5) / 127.5, (sampled_other_image[k].unsqueeze(0).float() - 127.5) / 127.5)
                else:
                    so, sa, po, pa = calculate_metrics(images_np[k], images_add_np[k], sampled_other_image_np[k], sampled_images_np[k])
                    lo = lpips_model((images[k].unsqueeze(0).float() - 127.5) / 127.5, (sampled_other_image[k].unsqueeze(0).float() - 127.5) / 127.5)
                    la = lpips_model((images_add[k].unsqueeze(0).float() - 127.5) / 127.5, (sampled_images[k].unsqueeze(0).float() - 127.5) / 127.5)
                
                # --- NEW: Success Rate of Reversal (%S) ---
                # Calculate SSIM between the input superimposed image and both ground truths
                ssim_s_o = structural_similarity(images_np[k], superimposed_np[k], data_range=255, channel_axis=-1)
                ssim_s_a = structural_similarity(images_add_np[k], superimposed_np[k], data_range=255, channel_axis=-1)
                
                if so > ssim_s_o:
                    success_count += 1
                if sa > ssim_s_a:
                    success_count += 1
                total_count += 2
                # ------------------------------------------

                ssim_o.append(so)
                ssim_a.append(sa)
                psnr_o.append(po)
                psnr_a.append(pa)
                lpips_o.append(lo.detach().cpu().numpy())
                lpips_a.append(la.detach().cpu().numpy())

    avg_ssim_o = np.average(ssim_o)
    avg_ssim_a = np.average(ssim_a)
    avg_psnr_o = np.average(psnr_o)
    avg_psnr_a = np.average(psnr_a)
    avg_lpips_o = np.average(lpips_o)
    avg_lpips_a = np.average(lpips_a)
    success_rate = (success_count / total_count) * 100

    print('\nMetrics organized by original images:')
    print(f'SSIM Original: {avg_ssim_o}')
    print(f'SSIM Added: {avg_ssim_a}')
    print(f'PSNR Original: {avg_psnr_o}')
    print(f'PSNR Added: {avg_psnr_a}')
    print(f'LPIPS Original: {avg_lpips_o}')
    print(f'LPIPS Added: {avg_lpips_a}')
    print(f'Success Rate of Reversal (%S): {success_rate:.2f}%')

    metrics_report = (
        f"Metrics organized by original images:\n"
        f"SSIM Original: {avg_ssim_o}\n"
        f"SSIM Added: {avg_ssim_a}\n"
        f"PSNR Original: {avg_psnr_o}\n"
        f"PSNR Added: {avg_psnr_a}\n"
        f"LPIPS Original: {avg_lpips_o}\n"
        f"LPIPS Added: {avg_lpips_a}\n"
        f"Success Rate of Reversal (%S): {success_rate:.2f}%\n"
    )
    
    results_dir = os.path.join("results", args.run_name)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "final_metrics.txt"), "w") as f:
        f.write(metrics_report)

def one_shot_eval(args):
    device = args.device
    test_dataloader = get_data(args, 'test')
    images, images_add = next(iter(test_dataloader))
    images = images.to(device)
    images_add = images_add.to(device)
    n = len(images)
    
    model = UNet().to(device)
    model.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ckpt.pt"), map_location=torch.device(device)))
    model.eval()
    
    diffusion = Diffusion(img_size=args.image_size, device=device)
    lpips_model = lpips.LPIPS(net='alex').to(device)

    sample_dir = os.path.join("samples", args.sampling_name)
    os.makedirs(sample_dir, exist_ok=True)
    
    S = images * (1 - args.alpha_init) + images_add * args.alpha_init
    init_timestep = math.ceil(args.alpha_init / diffusion.alteration_per_t)
    t = (torch.ones(n) * init_timestep).long().to(device)
    
    with torch.no_grad():
        predicted_image = model(S, t)
        
        if args.prediction == "original":
            sampled_images = predicted_image
            sampled_other_image = (S - (1 - args.alpha_init) * sampled_images) / args.alpha_init
        elif args.prediction == "added":
            sampled_other_image = predicted_image
            sampled_images = (S - args.alpha_init * sampled_other_image) / (1 - args.alpha_init)
        elif args.prediction == "differences":
            sampled_images = S + predicted_image * t[:, None, None, None]
            sampled_other_image = (S - (1 - args.alpha_init) * sampled_images) / args.alpha_init
        else:
            print("Invalid model prediction argument.")
            return

    images = (images.clamp(-1, 1) + 1) / 2
    images = (images * 255).type(torch.uint8)
    images_add = (images_add.clamp(-1, 1) + 1) / 2
    images_add = (images_add * 255).type(torch.uint8)
    sampled_images = (sampled_images.clamp(-1, 1) + 1) / 2
    sampled_images = (sampled_images * 255).type(torch.uint8)
    sampled_other_image = (sampled_other_image.clamp(-1, 1) + 1) / 2
    sampled_other_image = (sampled_other_image * 255).type(torch.uint8)
    
    # Format initial superimposed image
    S_formatted = (S.clamp(-1, 1) + 1) / 2
    S_formatted = (S_formatted * 255).type(torch.uint8)

    save_dir = os.path.join("samples", args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    save_images(sampled_images, sampled_other_image, images, images_add, os.path.join(save_dir, "one_shot_batch_0.jpg"))

    images_np = images.cpu().permute(0, 2, 3, 1).numpy()
    sampled_images_np = sampled_images.cpu().permute(0, 2, 3, 1).numpy()
    images_add_np = images_add.cpu().permute(0, 2, 3, 1).numpy()
    sampled_other_image_np = sampled_other_image.cpu().permute(0, 2, 3, 1).numpy()
    S_np = S_formatted.cpu().permute(0, 2, 3, 1).numpy()
    
    ssim_o, ssim_a, psnr_o, psnr_a, lpips_o, lpips_a = [], [], [], [], [], []
    success_count = 0
    total_count = 0
    
    with torch.no_grad():
        for k in range(n):
            ssim_1 = structural_similarity(images_np[k], sampled_images_np[k], data_range=255, channel_axis=-1)
            ssim_2 = structural_similarity(images_add_np[k], sampled_images_np[k], data_range=255, channel_axis=-1)
            
            if ssim_1 > ssim_2:
                so, sa, po, pa = calculate_metrics(images_np[k], images_add_np[k], sampled_images_np[k], sampled_other_image_np[k])
                lo = lpips_model((images[k].unsqueeze(0).float() - 127.5) / 127.5, (sampled_images[k].unsqueeze(0).float() - 127.5) / 127.5)
                la = lpips_model((images_add[k].unsqueeze(0).float() - 127.5) / 127.5, (sampled_other_image[k].unsqueeze(0).float() - 127.5) / 127.5)
            else:
                so, sa, po, pa = calculate_metrics(images_np[k], images_add_np[k], sampled_other_image_np[k], sampled_images_np[k])
                lo = lpips_model((images[k].unsqueeze(0).float() - 127.5) / 127.5, (sampled_other_image[k].unsqueeze(0).float() - 127.5) / 127.5)
                la = lpips_model((images_add[k].unsqueeze(0).float() - 127.5) / 127.5, (sampled_images[k].unsqueeze(0).float() - 127.5) / 127.5)
            
            # --- Success Rate of Reversal (%S) ---
            ssim_s_o = structural_similarity(images_np[k], S_np[k], data_range=255, channel_axis=-1)
            ssim_s_a = structural_similarity(images_add_np[k], S_np[k], data_range=255, channel_axis=-1)
            
            if so > ssim_s_o:
                success_count += 1
            if sa > ssim_s_a:
                success_count += 1
            total_count += 2
            
            ssim_o.append(so)
            ssim_a.append(sa)
            psnr_o.append(po)
            psnr_a.append(pa)
            lpips_o.append(lo.detach().cpu().numpy())
            lpips_a.append(la.detach().cpu().numpy())

    # Calculate averages
    avg_ssim_o, avg_ssim_a = np.average(ssim_o), np.average(ssim_a)
    avg_psnr_o, avg_psnr_a = np.average(psnr_o), np.average(psnr_a)
    avg_lpips_o, avg_lpips_a = np.average(lpips_o), np.average(lpips_a)
    success_rate = (success_count / total_count) * 100

    metrics_report = (
        f"--- One-Shot Evaluation Metrics (First Batch) ---\n"
        f"SSIM Original: {avg_ssim_o:.4f}\n"
        f"SSIM Added: {avg_ssim_a:.4f}\n"
        f"PSNR Original: {avg_psnr_o:.4f}\n"
        f"PSNR Added: {avg_psnr_a:.4f}\n"
        f"LPIPS Original: {avg_lpips_o:.4f}\n"
        f"LPIPS Added: {avg_lpips_a:.4f}\n"
        f"Success Rate of Reversal (%S): {success_rate:.2f}%\n"
    )
    
    print(f"\n{metrics_report}")
    
    # Save the text file
    results_dir = os.path.join("results", args.run_name)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "one_shot_metrics.txt"), "w") as f:
        f.write(metrics_report)

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help='Path to dataset', required=True)
    parser.add_argument('--run_name', help='Name of the experiment for saving models and results', required=True)
    parser.add_argument('--partition_file', help='CSV file with test indexes', required=True)
    parser.add_argument('--prediction', default='original', help='The prediction of the model, choose between [added, original, differences]', required=False)
    parser.add_argument('--sampling_method', default='with_error_removal', help='Choose between [cold_diffusion, with_error_removal, one_step, without_error_removal, differences]', required=False)
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
    one_shot_eval(args)

if __name__ == '__main__':
    launch()