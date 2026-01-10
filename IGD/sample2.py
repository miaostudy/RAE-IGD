import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import importlib
from tqdm import tqdm
from torchvision.utils import save_image
import argparse
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from collections import OrderedDict, defaultdict
from PIL import Image
import numpy as np
import gc
import time

# --- Project Imports ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "RAE", "src"))

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
from IGD.data import load_data
import IGD.train_models.resnet as RN

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.environ["http_proxy"] = "127.0.0.1:7890"
os.environ["https_proxy"] = "127.0.0.1:7890"


# ==========================================
# 1. Latent Proxy Definition & Helpers
# ==========================================
class LatentProxy(nn.Module):
    def __init__(self, input_dim, num_classes=1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.net(x)


@torch.no_grad()
def extract_latents_and_normalize(model, data_loader, device, max_batches=200):
    print(f"Extracting latents (max_batches={max_batches})...")
    model.eval()
    latents_list = []
    labels_list = []

    for i, (x, y) in enumerate(tqdm(data_loader, desc="Extracting")):
        if i >= max_batches: break
        x = x.to(device)
        z = model.encode(x)
        if isinstance(z, tuple): z = z[0]
        if hasattr(z, 'sample'): z = z.mode()

        latents_list.append(z.cpu())
        labels_list.append(y)

    latents = torch.cat(latents_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    # Normalize to N(0, 1) to match Diffusion space
    mean = latents.mean()
    std = latents.std()
    print(f"Original Latents Stats -> Mean: {mean:.4f}, Std: {std:.4f}")

    latents = (latents - mean) / (std + 1e-6)
    print(f"Normalized Latents -> Mean: {latents.mean():.4f}, Std: {latents.std():.4f}")

    return latents, labels, {'mean': mean, 'std': std}


def train_proxy(latents, labels, device, num_classes=1000, epochs=5):
    print(f"Training Proxy on {len(labels)} samples (Normalized)...")
    input_dim = np.prod(latents.shape[1:])

    proxy = LatentProxy(input_dim, num_classes).to(device)
    optimizer = optim.Adam(proxy.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(latents, labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    proxy.train()
    for epoch in range(epochs):
        correct = 0
        total = 0
        total_loss = 0
        for z_batch, y_batch in loader:
            z_batch, y_batch = z_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = proxy(z_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)
            total_loss += loss.item()

        print(f"  Proxy Epoch {epoch + 1}/{epochs} | Acc: {correct / total:.4f} | Loss: {total_loss / len(loader):.4f}")

    proxy.eval()
    return proxy


# ==========================================
# 2. Config Helpers
# ==========================================
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class DecoderWrapper:
    def __init__(self, decode_fn):
        self.decode_fn = decode_fn

    def decode(self, z): return self.decode_fn(z)


# ==========================================
# 3. IGD Sampling Logic (Adaptive Scaling Eq.9)
# ==========================================
@torch.no_grad()
def igd_ode_sample(model, z, steps, model_kwargs, device):
    x = z
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

    # --- IGD Parameters ---
    latent_proxy = model_kwargs.get('latent_proxy', None)

    # k: Influence Scale (Controls the adaptive term rho)
    k_scale = model_kwargs.get('k_scale', 1.0)

    # gamma: Deviation/Diversity Scale (Controls threshold or deviation term)
    gamma_scale = model_kwargs.get('gamma_scale', 0.0)

    # Practical Technique: Diversity Threshold (Stop guidance if confident)
    # Using gamma_scale essentially to control "how strict" we are if we don't have Memory Bank.
    # Here strictly adhering to user request: diversity_threshold helps retains intra-class variance.
    diversity_thresh = model_kwargs.get('diversity_threshold', 0.9)

    low, high = model_kwargs.get('low', 0), model_kwargs.get('high', 1000)
    target_y_global = model_kwargs['y']
    target_y_proxy = model_kwargs.get('y_proxy', target_y_global)

    is_cfg = (x.shape[0] == 2 * target_y_proxy.shape[0])

    for i in tqdm(range(steps), desc="ODE Sampling"):
        t_curr = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_curr  # Negative

        t_in = torch.full((x.shape[0],), t_curr, device=device, dtype=torch.float)

        # 1. Forward DiT (Get epsilon_phi / Velocity)
        if 'cfg_scale' in model_kwargs and model_kwargs['cfg_scale'] > 1.0:
            v_pred = model(x, t_in, model_kwargs['y'])
        else:
            v_pred = model(x, t_in, model_kwargs['y'])

        # 2. IGD Guidance Calculation
        # Check active range
        t_check = t_curr.item() * 1000
        should_guide = (t_check >= low) and (t_check <= high) and (latent_proxy is not None) and (k_scale > 0)

        guidance_term = torch.zeros_like(v_pred)
        rho_t = 0.0
        conf = -1.0

        if should_guide:
            # Estimate x_0 (z_hat_0|t)
            x_0_est = x - t_curr * v_pred

            if is_cfg:
                x_0_cond, _ = x_0_est.chunk(2)
            else:
                x_0_cond = x_0_est

            with torch.enable_grad():
                x_in = x_0_cond.detach().requires_grad_(True)
                logits = latent_proxy(x_in)
                probs = torch.softmax(logits, dim=1)

                # Check Confidence
                if target_y_proxy.max() < probs.shape[1]:
                    current_conf = probs.gather(1, target_y_proxy.view(-1, 1)).mean().item()
                else:
                    current_conf = 0.0
                conf = current_conf

                # --- Practical Technique: Confidence Threshold ---
                # "Enhancing Efficiency": If sample is good enough, stop guiding to preserve diversity.
                if current_conf > diversity_thresh:
                    # Pass (No guidance)
                    pass
                else:
                    # Calculate Gradient of Influence (GI)
                    target_logits = logits.gather(1, target_y_proxy.view(-1, 1))
                    # We want to maximize logits => minimize (-logits)
                    loss = -target_logits.sum()

                    grads_cond = torch.autograd.grad(loss, x_in)[0]

                    # Formulate full gradient
                    if is_cfg:
                        grads_uncond = torch.zeros_like(grads_cond)
                        grads = torch.cat([grads_cond, grads_uncond], dim=0)
                    else:
                        grads = grads_cond

                    # --- Adaptive Scaling (Eq. 9) ---
                    # rho_t = k * sqrt(1-alpha) * (||epsilon|| / ||grad||)
                    # In Flow Matching: sqrt(1-alpha) ~ t_curr

                    v_norm = (v_pred.detach() ** 2).mean().sqrt()
                    g_norm = (grads ** 2).mean().sqrt() + 1e-8

                    # Adaptive factor
                    rho_t = k_scale * t_curr * (v_norm / g_norm)

                    # Guidance Term: rho_t * grad
                    # Note: grads points to HIGHER loss (bad). We want to move OPPOSITE.
                    # But wait, grad(Loss) points to higher loss.
                    # Update rule: v_new = v_old - rho * grad(Loss)
                    guidance_term = grads * rho_t

        # 3. Update Velocity (Reweighting)
        # z_{t-1} = s(z_t) - rho * grad_I - gamma * grad_D
        # Note: We implement the subtraction in the velocity update
        v_final = v_pred - guidance_term

        # Logging
        if should_guide and i % 10 == 0:
            status = "PASS" if conf > diversity_thresh else "PUSH"
            print(f"  Step {i} | t={t_curr:.3f} | Conf: {conf:.4f} | [{status}] Rho: {rho_t:.4f}")

        # ODE Step
        x = x + v_final * dt

    return x


# ==========================================
# 4. Main
# ==========================================
def main(args):
    torch.manual_seed(args.seed)
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Config Loading ---
    config = None
    if args.config and os.path.exists(args.config):
        print(f"Loading config from {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        if 'stage_2' in config and 'target' in config['stage_2']:
            target_str = config['stage_2']['target']
            if 'stage2.dmvae_models' in target_str:
                config['stage_2']['target'] = target_str.replace('stage2.dmvae_models', 'stage2.models')

    # --- Class Selection Logic ---
    if args.spec == 'cifar10':
        sel_classes = [str(i) for i in range(10)]
        class_labels = [int(x) for x in sel_classes]
    else:
        if args.spec == 'woof':
            list_file = 'IGD/misc/class_woof.txt'
        elif args.spec == 'nette':
            list_file = 'IGD/misc/class_nette.txt'
        elif args.spec == '1k':
            list_file = 'IGD/misc/class_indices.txt'
        else:
            list_file = 'IGD/misc/class100.txt'

        if os.path.exists(list_file):
            with open(list_file, 'r') as fp:
                sel_classes_raw = [s.strip() for s in fp.readlines()]
        else:
            sel_classes_raw = []

        phase = max(0, args.phase)
        cls_from = args.nclass * phase
        cls_to = args.nclass * (phase + 1)
        if len(sel_classes_raw) > 0:
            sel_classes = sel_classes_raw[cls_from:cls_to]
            master_file = 'IGD/misc/class_indices.txt'
            if os.path.exists(master_file):
                with open(master_file, 'r') as f:
                    all_classes = [l.strip() for l in f.readlines()]
                class_labels = [all_classes.index(c) for c in sel_classes if c in all_classes]
            else:
                class_labels = list(range(cls_from, cls_to))
        else:
            sel_classes = []
            class_labels = []

    # --- Model Initialization ---
    if args.model_type == 'rae' and config:
        print("Initializing RAE from config...")
        rae = instantiate_from_config(config['stage_1']).to(device).eval()
        if args.rae_ckpt:
            print(f"Loading RAE checkpoint: {args.rae_ckpt}")
            st = torch.load(args.rae_ckpt, map_location='cpu')
            if 'model' in st: st = st['model']
            rae.load_state_dict(st, strict=False)

        model = instantiate_from_config(config['stage_2']).to(device).eval()
        dit_ckpt_path = args.dit_ckpt if args.dit_ckpt else config['stage_2'].get('ckpt')
        if dit_ckpt_path:
            print(f"Loading DiT checkpoint: {dit_ckpt_path}")
            ckpt = torch.load(dit_ckpt_path, map_location='cpu')
            if 'model' in ckpt:
                model.load_state_dict(ckpt['model'])
            elif 'ema' in ckpt:
                model.load_state_dict(ckpt['ema'])
            else:
                model.load_state_dict(ckpt)

        def decode_fn(z):
            img = rae.decode(z)
            return img * 2.0 - 1.0

        if 'misc' in config and 'latent_size' in config['misc']:
            c, h, w = config['misc']['latent_size']
            latent_size = h * 16
        else:
            latent_size = args.image_size // 16
    else:
        raise NotImplementedError

    # --- Train Latent Proxy ---
    print("\n[Proxy Setup] Loading training data...")
    latent_proxy = None
    proxy_class_map = {}

    try:
        dataset_obj, train_loader, _, num_classes_data = load_data(args, tsne=False)

        if hasattr(dataset_obj, 'class_to_idx'):
            proxy_class_map = dataset_obj.class_to_idx
        elif hasattr(train_loader.dataset, 'class_to_idx'):
            proxy_class_map = train_loader.dataset.class_to_idx
        elif hasattr(train_loader.dataset, 'dataset') and hasattr(train_loader.dataset.dataset, 'class_to_idx'):
            proxy_class_map = train_loader.dataset.dataset.class_to_idx
        else:
            for idx, name in enumerate(sorted(sel_classes)):
                proxy_class_map[name] = idx

        # Extract & Normalize
        latents, labels, stats = extract_latents_and_normalize(rae, train_loader, device, max_batches=100)
        latent_proxy = train_proxy(latents, labels, device, num_classes=num_classes_data, epochs=5)

    except Exception as e:
        print(f"Warning: Failed to train proxy. Error: {e}")
        import traceback
        traceback.print_exc()

    # --- Sampling Loop ---
    print(f"\n[Sampling] Start generating {args.num_samples} samples per class...")
    os.makedirs(args.save_dir, exist_ok=True)
    batch_size = 1

    for i, class_label in enumerate(class_labels):
        sel_class_name = sel_classes[i] if i < len(sel_classes) else str(class_label)

        if sel_class_name in proxy_class_map:
            proxy_label_idx = proxy_class_map[sel_class_name]
        else:
            proxy_label_idx = class_label
            if latent_proxy is not None and proxy_label_idx >= 10:
                proxy_label_idx = 0

        save_class_dir = os.path.join(args.save_dir, sel_class_name)
        os.makedirs(save_class_dir, exist_ok=True)

        print(f"Generating class: {sel_class_name} | DiT Idx: {class_label} | Proxy Idx: {proxy_label_idx}")

        for shift in tqdm(range(0, args.num_samples, batch_size)):
            if config and 'misc' in config:
                C, H, W = config['misc']['latent_size']
            else:
                C, H, W = 32, latent_size, latent_size

            z = torch.randn(batch_size, C, H, W, device=device)
            y = torch.tensor([class_label] * batch_size, device=device)
            y_p = torch.tensor([proxy_label_idx] * batch_size, device=device)

            if args.cfg_scale > 1.0:
                z_in = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * batch_size, device=device)
                y_in = torch.cat([y, y_null], 0)
            else:
                z_in = z
                y_in = y

            model_kwargs = dict(
                y=y_in,
                cfg_scale=args.cfg_scale,
                latent_proxy=latent_proxy,
                # IGD Parameters
                k_scale=args.k_scale,  # Influence Scale (k)
                gamma_scale=args.gamma_scale,  # Deviation Scale (gamma)
                diversity_threshold=args.diversity_threshold,
                low=args.low,
                high=args.high,
                y_proxy=y_p
            )

            samples = igd_ode_sample(model, z_in, args.num_sampling_steps, model_kwargs, device)

            if args.cfg_scale > 1.0:
                samples, _ = samples.chunk(2, dim=0)

            imgs = decode_fn(samples)

            for j, img in enumerate(imgs):
                idx = shift + j + args.total_shift
                save_path = os.path.join(save_class_dir, f"{idx}.png")
                save_image(img, save_path, normalize=True, value_range=(-1, 1))

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--model_type", type=str, default='rae')

    parser.add_argument("--rae_ckpt", type=str, default=None)
    parser.add_argument("--dit_ckpt", type=str, default=None)

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default='./results/generated')
    parser.add_argument("--total-shift", type=int, default=0)

    # IGD 专属参数
    parser.add_argument("--k-scale", type=float, default=1.0, help="Influence Scale (k) in Eq.9")
    parser.add_argument("--gamma-scale", type=float, default=0.0,
                        help="Deviation Scale (gamma) in Eq.9 (Not actively used without memory bank)")
    parser.add_argument("--diversity-threshold", type=float, default=0.9,
                        help="Stop guidance if confidence > threshold")

    # 遗留兼容处理
    parser.add_argument("--gm-scale", type=float, default=0.0, help="Deprecated. Maps to k-scale")

    parser.add_argument("--low", type=int, default=0)
    parser.add_argument("--high", type=int, default=1000)

    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--data_dir", type=str, default='/data')
    parser.add_argument('--imagenet_dir', nargs='+', default=['/data/imagenet'])
    parser.add_argument("--nclass", type=int, default=1000)
    parser.add_argument("--spec", type=str, default='nette')
    parser.add_argument("--phase", type=int, default=0)

    parser.add_argument("--augment", action='store_true', default=False)
    parser.add_argument("--slct_type", type=str, default='random')
    parser.add_argument("--ipc", type=int, default=-1)
    parser.add_argument("--load_memory", action='store_true', default=False)
    parser.add_argument("--dseed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_real", type=int, default=32)
    parser.add_argument("--device", type=str, default='cuda')

    args = parser.parse_args()

    if args.gm_scale > 0 and args.k_scale == 1.0:
        print(f"Note: Using --gm-scale ({args.gm_scale}) as --k-scale.")
        args.k_scale = args.gm_scale

    main(args)