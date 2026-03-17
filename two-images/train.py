import math
import os
import random
from contextlib import nullcontext

import lpips
import torchvision
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import wandb
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import optim

from modules import UNet
from utils import get_data, save_images, setup_logging, tensor_to_uint8


class Diffusion:
    def __init__(self, max_timesteps=250, alpha_start=0.0, alpha_max=0.5, img_size=256, device="cuda"):
        if alpha_max <= alpha_start:
            raise ValueError("alpha_max must be greater than alpha_start.")
        if alpha_max > 0.5:
            raise ValueError("alpha_max should stay <= 0.5 so Image1 remains the dominant image.")

        self.max_timesteps = max_timesteps
        self.img_size = img_size
        self.device = device
        self.alpha_start = alpha_start
        self.alpha_max = alpha_max
        self.alteration_per_t = (alpha_max - alpha_start) / max_timesteps

    def alpha_from_t(self, t: torch.Tensor | int) -> torch.Tensor:
        if isinstance(t, int):
            t = torch.tensor([t], device=self.device, dtype=torch.float32)
        elif not torch.is_tensor(t):
            t = torch.as_tensor(t, device=self.device, dtype=torch.float32)
        else:
            t = t.to(self.device, dtype=torch.float32)
        return self.alpha_start + self.alteration_per_t * t

    def build_mixture(self, image_1: torch.Tensor, image_2: torch.Tensor, alpha: float) -> torch.Tensor:
        return image_1 * (1.0 - alpha) + image_2 * alpha

    def noise_images(self, image_1: torch.Tensor, image_2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_t = self.alpha_from_t(t).view(-1, 1, 1, 1)
        return image_1 * (1.0 - alpha_t) + image_2 * alpha_t

    def sample_timesteps(self, n: int) -> torch.Tensor:
        return torch.randint(low=1, high=self.max_timesteps + 1, size=(n,), device=self.device)

    def alpha_to_timestep(self, alpha: float) -> int:
        if alpha < self.alpha_start or alpha > self.alpha_max:
            raise ValueError(f"alpha must be in [{self.alpha_start}, {self.alpha_max}], got {alpha}.")
        timestep = int(round((alpha - self.alpha_start) / self.alteration_per_t))
        return max(1, min(timestep, self.max_timesteps))

    def sample(self, model, superimposed_image, alpha_init=0.5, step_scale=1.0):
        """
        Cold Diffusion Tacos sampler.
        The model predicts the clean Image1.
        The other image is extracted using the ground truth average, and x_t is refined.
        """
        n = len(superimposed_image)
        init_timestep = self.alpha_to_timestep(alpha_init)

        model.eval()
        with torch.no_grad():
            mixed_init = superimposed_image.to(self.device)
            x_t = mixed_init.clone()

            for i in reversed(range(1, init_timestep + 1)):
                t = torch.full((n,), i, device=self.device, dtype=torch.long)
                t_minus_1 = torch.full((n,), i - 1, device=self.device, dtype=torch.long)
                
                # Predict the clean image (Image1)
                predicted_image = model(x_t, t)
                predicted_image = predicted_image.clamp(-1, 1)
                
                # Extract the other image mathematically using the initial superimposed image
                other_image = (mixed_init - (1.0 - alpha_init) * predicted_image) / alpha_init
                other_image = other_image.clamp(-1, 1)

                # Update x_t with cold diffusion tacos sampling
                x_t = x_t - self.noise_images(predicted_image, other_image, t) + \
                      self.noise_images(predicted_image, other_image, t_minus_1)

            primary_image = x_t.clamp(-1, 1)
            other_image = (mixed_init - (1.0 - alpha_init) * primary_image) / alpha_init
            other_image = other_image.clamp(-1, 1)

        model.train()
        return primary_image, other_image

    def one_shot(self, model, superimposed_image, alpha_init=0.5):
        init_timestep = self.alpha_to_timestep(alpha_init)
        n = len(superimposed_image)
        t = torch.full((n,), init_timestep, device=self.device, dtype=torch.long)

        model.eval()
        with torch.no_grad():
            mixed_init = superimposed_image.to(self.device)
            
            # Predict the clean image
            predicted_image = model(mixed_init, t)
            primary_image = predicted_image.clamp(-1, 1)
            
            # Extract the other image mathematically
            other_image = (mixed_init - (1.0 - alpha_init) * primary_image) / alpha_init
            other_image = other_image.clamp(-1, 1)

        model.train()
        return primary_image, other_image


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def use_amp(device: str) -> bool:
    return str(device).startswith("cuda") and torch.cuda.is_available()


def calculate_metrics(image, add_image, result_ori_image, result_add_image):
    ssim_original = structural_similarity(image, result_ori_image, data_range=255, channel_axis=-1)
    ssim_added = structural_similarity(add_image, result_add_image, data_range=255, channel_axis=-1)
    psnr_original = peak_signal_noise_ratio(image, result_ori_image, data_range=255)
    psnr_added = peak_signal_noise_ratio(add_image, result_add_image, data_range=255)
    return ssim_original, ssim_added, psnr_original, psnr_added


def build_model(device: str):
    return UNet(device=device).to(device)


def save_preview(diffusion, model, images, images_add, args, output_path):
    superimposed = diffusion.build_mixture(images, images_add, args.alpha_init)
    sampled_images, other_images = diffusion.sample(
        model,
        superimposed,
        alpha_init=args.alpha_init,
        step_scale=args.step_scale,
    )

    save_images(
        tensor_to_uint8(sampled_images),
        tensor_to_uint8(other_images),
        tensor_to_uint8(images),
        tensor_to_uint8(images_add),
        output_path,
        input_images=tensor_to_uint8(superimposed),
    )


def train(args):
    if args.alpha_init > args.alpha_max:
        raise ValueError("alpha_init must be <= alpha_max.")

    setup_logging(args.run_name)
    seed_everything(args.seed)
    device = args.device
    train_dataloader = get_data(args, "train")
    test_dataloader = get_data(args, "test")

    model = build_model(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(
        max_timesteps=args.max_timesteps,
        alpha_max=args.alpha_max,
        img_size=args.image_size,
        device=device,
    )

    wandb_mode = "disabled" if args.disable_wandb else "online"
    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args), mode=wandb_mode)

    amp_enabled = use_amp(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        for images, images_add in train_dataloader:
            images = images.to(device, non_blocking=True)
            images_add = images_add.to(device, non_blocking=True)
            t = diffusion.sample_timesteps(images.shape[0])

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled) if amp_enabled else nullcontext():
                x_t = diffusion.noise_images(images, images_add, t)
                predicted_img = model(x_t, t)
                target_img = images
                loss = mse(predicted_img, target_img)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if global_step % args.log_every == 0:
                wandb.log({"train_loss": loss.item(), "epoch": epoch + 1, "step": global_step})

        if (epoch + 1) % args.sample_every == 0:
            images, images_add = next(iter(test_dataloader))
            images = images.to(device, non_blocking=True)
            images_add = images_add.to(device, non_blocking=True)
            save_preview(
                diffusion,
                model,
                images,
                images_add,
                args,
                os.path.join("results", args.run_name, f"{epoch + 1}.jpg"),
            )

        torch.save(model.state_dict(), os.path.join("models", args.run_name, "ckpt.pt"))

    wandb.finish()


def evaluate_model(args, one_shot: bool = False):
    if args.alpha_init > args.alpha_max:
        raise ValueError("alpha_init must be <= alpha_max.")

    device = args.device
    test_dataloader = get_data(args, "test")
    model = build_model(device)
    model.load_state_dict(torch.load(os.path.join("models", args.run_name, "ckpt.pt"), map_location=torch.device(device)))
    model.eval()

    diffusion = Diffusion(
        max_timesteps=args.max_timesteps,
        alpha_max=args.alpha_max,
        img_size=args.image_size,
        device=device,
    )
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    sampling_name = args.sampling_name if not one_shot else f"{args.sampling_name}_one_shot"
    sample_dir = os.path.join("samples", sampling_name)
    os.makedirs(sample_dir, exist_ok=True)

    ssim_o, ssim_a, lpips_o, lpips_a, psnr_o, psnr_a = [], [], [], [], [], []
    success_count = 0
    total_count = 0

    for i, (images, images_add) in enumerate(test_dataloader):
        images = images.to(device, non_blocking=True)
        images_add = images_add.to(device, non_blocking=True)
        superimposed = diffusion.build_mixture(images, images_add, args.alpha_init)

        if one_shot:
            sampled_images, sampled_other_image = diffusion.one_shot(model, superimposed, args.alpha_init)
        else:
            sampled_images, sampled_other_image = diffusion.sample(
                model,
                superimposed,
                args.alpha_init,
                step_scale=args.step_scale,
            )

        images_u8 = tensor_to_uint8(images)
        images_add_u8 = tensor_to_uint8(images_add)
        superimposed_u8 = tensor_to_uint8(superimposed)
        sampled_images_u8 = tensor_to_uint8(sampled_images)
        sampled_other_u8 = tensor_to_uint8(sampled_other_image)

        images_np = images_u8.cpu().permute(0, 2, 3, 1).numpy()
        sampled_images_np = sampled_images_u8.cpu().permute(0, 2, 3, 1).numpy()
        images_add_np = images_add_u8.cpu().permute(0, 2, 3, 1).numpy()
        sampled_other_np = sampled_other_u8.cpu().permute(0, 2, 3, 1).numpy()
        superimposed_np = superimposed_u8.cpu().permute(0, 2, 3, 1).numpy()

        with torch.no_grad():
            for k in range(len(images_np)):
                so, sa, po, pa = calculate_metrics(
                    images_np[k],
                    images_add_np[k],
                    sampled_images_np[k],
                    sampled_other_np[k],
                )
                lo = lpips_model(images[k:k + 1], sampled_images[k:k + 1]).item()
                la = lpips_model(images_add[k:k + 1], sampled_other_image[k:k + 1]).item()

                ssim_s_o = structural_similarity(images_np[k], superimposed_np[k], data_range=255, channel_axis=-1)
                ssim_s_a = structural_similarity(images_add_np[k], superimposed_np[k], data_range=255, channel_axis=-1)

                if so > ssim_s_o:
                    success_count += 1
                if sa > ssim_s_a:
                    success_count += 1
                total_count += 2

                ssim_o.append(so)
                ssim_a.append(sa)
                psnr_o.append(po)
                psnr_a.append(pa)
                lpips_o.append(lo)
                lpips_a.append(la)

        save_images(
            sampled_images_u8,
            sampled_other_u8,
            images_u8,
            images_add_u8,
            os.path.join(sample_dir, f"{i}.jpg"),
            input_images=superimposed_u8,
        )

    avg_ssim_o = float(np.mean(ssim_o))
    avg_ssim_a = float(np.mean(ssim_a))
    avg_psnr_o = float(np.mean(psnr_o))
    avg_psnr_a = float(np.mean(psnr_a))
    avg_lpips_o = float(np.mean(lpips_o))
    avg_lpips_a = float(np.mean(lpips_a))
    success_rate = (success_count / total_count) * 100.0

    title = "One-Shot Evaluation Metrics" if one_shot else "Metrics organized by canonical image order"
    metrics_report = (
        f"{title}:\n"
        f"SSIM Image1: {avg_ssim_o}\n"
        f"SSIM Image2: {avg_ssim_a}\n"
        f"PSNR Image1: {avg_psnr_o}\n"
        f"PSNR Image2: {avg_psnr_a}\n"
        f"LPIPS Image1: {avg_lpips_o}\n"
        f"LPIPS Image2: {avg_lpips_a}\n"
        f"Success Rate of Reversal (%S): {success_rate:.2f}%\n"
    )

    print(f"\n{metrics_report}")

    results_dir = os.path.join("results", args.run_name)
    os.makedirs(results_dir, exist_ok=True)
    output_name = "one_shot_metrics.txt" if one_shot else "final_metrics.txt"
    with open(os.path.join(results_dir, output_name), "w") as f:
        f.write(metrics_report)


def eval(args):
    evaluate_model(args, one_shot=False)


def one_shot_eval(args):
    evaluate_model(args, one_shot=True)


def visualize_sampling_transition(args):
    device = args.device
    test_dataloader = get_data(args, "test")
    images, images_add = next(iter(test_dataloader))
    img = images[3:4].to(device)
    img_add = images_add[3:4].to(device)

    model = build_model(device)
    model.load_state_dict(torch.load(os.path.join("models", args.run_name, "ckpt.pt"), map_location=torch.device(device)))
    model.eval()

    diffusion = Diffusion(
        max_timesteps=args.max_timesteps,
        alpha_max=args.alpha_max,
        img_size=args.image_size,
        device=device,
    )

    save_dir = os.path.join("samples", args.run_name, "transition")
    os.makedirs(save_dir, exist_ok=True)

    S = diffusion.build_mixture(img, img_add, args.alpha_init)
    init_timestep = diffusion.alpha_to_timestep(args.alpha_init)
    x_t = S.clone()

    with torch.no_grad():
        for i in reversed(range(1, init_timestep + 1)):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            alpha_t = diffusion.alpha_from_t(t).view(-1, 1, 1, 1)
            predicted_d = model(x_t, t)

            predicted_image = (x_t + alpha_t * predicted_d).clamp(-1, 1)
            other_image = (x_t - (1.0 - alpha_t) * predicted_d).clamp(-1, 1)

            grid_tensor = torch.cat([
                (img.clamp(-1, 1) + 1) / 2,
                (img_add.clamp(-1, 1) + 1) / 2,
                (S.clamp(-1, 1) + 1) / 2,
                (x_t.clamp(-1, 1) + 1) / 2,
                (predicted_image.clamp(-1, 1) + 1) / 2,
                (other_image.clamp(-1, 1) + 1) / 2,
            ], dim=0)
            grid = torchvision.utils.make_grid(grid_tensor, nrow=6, padding=2)
            ndarr = grid.permute(1, 2, 0).mul(255).clamp(0, 255).byte().cpu().numpy()
            Image.fromarray(ndarr).save(os.path.join(save_dir, f"step_{i:03d}.jpg"))

            x_t = (x_t + args.step_scale * diffusion.alteration_per_t * predicted_d).clamp(-1, 1)

    print(f"Saved {init_timestep} transition images to {save_dir}")


def launch():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", help="Path to dataset", required=True)
    parser.add_argument("--run_name", help="Name of the experiment for saving models and results", required=True)
    parser.add_argument("--partition_file", help="CSV file with canonical ordered pairs", required=True)
    parser.add_argument("--mode", choices=["train", "eval", "one_shot", "transition", "all"], default="all")
    parser.add_argument("--sampling_name", type=str, default=None)
    parser.add_argument("--alpha_max", default=0.5, type=float, help="Maximum Image2 weight used during training")
    parser.add_argument("--alpha_init", default=0.5, type=float, help="Image2 weight used to build the starting mixture")
    parser.add_argument("--image_size", default=64, type=int, help="Square resize dimension")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_timesteps", default=250, type=int)
    parser.add_argument("--step_scale", default=1.0, type=float)
    parser.add_argument("--sample_every", default=50, type=int)
    parser.add_argument("--log_every", default=100, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_workers", default=min(8, os.cpu_count() or 1), type=int)
    parser.add_argument("--wandb_project", default="demorph", type=str)
    parser.add_argument("--disable_wandb", action="store_true")

    args = parser.parse_args()
    args.image_size = (args.image_size, args.image_size)
    args.sampling_name = args.sampling_name or args.run_name

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        eval(args)
    elif args.mode == "one_shot":
        one_shot_eval(args)
    elif args.mode == "transition":
        visualize_sampling_transition(args)
    elif args.mode == "all":
        train(args)
        eval(args)
        one_shot_eval(args)


if __name__ == "__main__":
    launch()
