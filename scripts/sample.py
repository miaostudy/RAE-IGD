import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import math
import gc
import time
from tqdm import tqdm
from collections import defaultdict
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms


from data import ImageFolder_mp
from download import find_model
import train_models.resnet as RN
import train_models.resnet_ap as RNAP
import train_models.convnet as CN
import train_models.densenet_cifar as DN
from reparam_module import ReparamModule
sys.path.append(os.path.abspath("."))


from src.utils.model_utils import instantiate_from_config
from src.stage2.transport import create_transport, Sampler
from src.utils.train_utils import parse_configs
from src.stage1 import RAE
from src.stage2.models import Stage2ModelProtocol

# 环境设置
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



def define_surrogate(args, nclass, size=None):
    """定义代理模型 (ResNet/ConvNet) 用于计算梯度"""
    args.size = 256
    args.width = 1.0
    args.norm_type = 'instance'
    args.nch = 3
    if size is None: size = args.size

    if args.net_type == 'resnet':
        model = RN.ResNet(args.spec, args.depth, nclass, norm_type=args.norm_type, size=size, nch=args.nch)
    elif args.net_type == 'resnet_ap':
        model = RNAP.ResNetAP(args.spec, args.depth, nclass, width=args.width, norm_type=args.norm_type, size=size,
                              nch=args.nch)
    elif args.net_type == 'convnet':
        model = CN.ConvNet(channel=4, num_classes=nclass, net_width=128, net_depth=3, net_act='relu',
                           net_norm='instancenorm', net_pooling='avgpooling', im_size=(args.size, args.size))
        model.classifier = nn.Linear(2048, nclass)
    elif args.net_type == 'convnet6':
        model = CN.ConvNet(channel=args.nch, num_classes=nclass, net_width=128, net_depth=6, net_act='relu',
                           net_norm='instancenorm', net_pooling='avgpooling', im_size=(args.size, args.size))
    elif args.net_type == 'convnet4':
        model = CN.ConvNet(channel=args.nch, num_classes=nclass, net_width=128, net_depth=4, net_act='relu',
                           net_norm='instancenorm', net_pooling='avgpooling', im_size=(args.size, args.size))
    else:
        raise Exception(f'Unknown network architecture: {args.net_type}')
    return model


def rand_ckpts(args):
    """加载代理模型的随机 Checkpoints"""
    expert_path = os.path.join(args.ckpt_path, args.spec, args.net_type)
    if not os.path.exists(expert_path):
        print(f"Warning: Checkpoint path {expert_path} does not exist.")
        return []

    expert_files = os.listdir(expert_path)
    # 简单的加载逻辑，这里固定加载第一个文件作为示例
    rand_id1 = 0
    ckpt_file = os.path.join(expert_path, expert_files[rand_id1])
    print(f'Loading surrogate ckpts from: {ckpt_file}')

    # 兼容 PyTorch 新版本的安全性检查
    try:
        state = torch.load(ckpt_file, map_location='cpu', weights_only=False)
    except TypeError:
        state = torch.load(ckpt_file, map_location='cpu')

    if isinstance(state, list):
        ckpts = state[0]  # 假设 state 是一个列表的列表
    else:
        ckpts = state

    # 选择特定的索引 (复用你原代码的逻辑)
    idxs = [0, len(ckpts) // 2, len(ckpts) - 1]  # 默认值
    if args.spec == '1k' and args.net_type == 'convnet6':
        idxs = [0, 5, 18, 50]
    elif args.spec == 'woof' and args.net_type == 'convnet6':
        idxs = [0, 5, 16, 40]

    print('Selected ckpt idxs:', idxs)
    return [ckpts[ii] for ii in idxs]


def get_grads(args, sel_class, ckpts, surrogate, target_label, device='cuda'):
    """从真实图像计算梯度 (Real Gradient Memory)"""
    criterion_ce = nn.CrossEntropyLoss().to(device)
    grads_memory = []

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    # 使用 ImageFolder_mp (支持单类别)
    dataset = ImageFolder_mp(args.data_path, transform=transform, nclass=args.nclass,
                             ipc=args.real_ipc, spec=args.spec, phase=args.phase,
                             seed=0, return_origin=True, sel_class=sel_class)

    loader = DataLoader(dataset, batch_size=200, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    print(f'Loading real gradients for class {sel_class} (Global Label: {target_label})...')

    # 1. 收集真实图像
    for x, _, _ in loader:
        grads_memory.extend(x.detach().split(1))
        if len(grads_memory) >= args.grad_ipc:
            break

    grads_memory = grads_memory[:args.grad_ipc]
    print(f'Collected {len(grads_memory)} real images for gradient matching.')

    # 2. 计算梯度
    real_gradients = []
    gap = 50  # Batch size for gradient calculation
    gap_idxs = np.arange(0, len(grads_memory), gap).tolist()

    # 遍历每一个 Checkpoint 计算平均梯度
    for ckpt in ckpts:
        acc_grad = None
        for gi in gap_idxs:
            batch_imgs = torch.stack(grads_memory[gi:gi + gap]).squeeze(1).to(device).requires_grad_(True)
            batch_targets = torch.full((len(batch_imgs),), target_label, device=device, dtype=torch.long)

            cur_params = torch.cat([p.data.to(device).reshape(-1) for p in ckpt], 0).requires_grad_(True)

            preds = surrogate(batch_imgs, flat_param=cur_params)
            loss = criterion_ce(preds, batch_targets)

            grad = torch.autograd.grad(loss, cur_params)[0]

            if acc_grad is None:
                acc_grad = torch.zeros_like(grad)
            acc_grad += grad.detach()

        real_gradients.append(acc_grad / len(gap_idxs))

    del grads_memory
    gc.collect()
    torch.cuda.empty_cache()

    return real_gradients


def igd_guidance_wrapper(model_out, z, t, rae, surrogate, ckpts, real_grads, target_label, scale):
    # 只有在引导比例 > 0 时计算
    if scale <= 0:
        return model_out

    with torch.enable_grad():
        # 确保 z 可以求导
        z = z.detach().requires_grad_(True)

        # 1. 估计 Clean Latent (z0/z1)
        # 假设 Rectified Flow: z_t = t * z_1 + (1-t) * z_0 (z0=Noise, z1=Data)
        # v = z_1 - z_0
        # z_1 = z_t + (1-t) * v
        t_view = t.view(-1, 1, 1, 1)
        z_data_est = z + (1 - t_view) * model_out.detach()  # 使用 detach 的 model_out 作为基准

        # 2. RAE 解码
        # 注意：RAE 解码过程显存消耗较大，需注意 Batch Size
        x_pixel_est = rae.decode(z_data_est)

        # 3. 计算 IGD Loss (梯度匹配)
        total_obj = 0.0
        criterion = nn.CrossEntropyLoss()

        for i, (ckpt_params, real_g) in enumerate(zip(ckpts, real_grads)):
            # 准备 Surrogate 参数
            flat_param = torch.cat([p.data.to(z.device).reshape(-1) for p in ckpt_params], 0).detach().requires_grad_(
                True)

            # Surrogate 前向
            pred = surrogate(x_pixel_est, flat_param=flat_param)
            target = torch.full((len(pred),), target_label, device=z.device, dtype=torch.long)
            loss_surr = criterion(pred, target)

            # 计算 d(Loss_surr)/d(Theta)
            grad_theta = torch.autograd.grad(loss_surr, flat_param, create_graph=True)[0]

            # L2 距离 (IGD 目标)
            obj = (grad_theta - real_g.detach()).pow(2).sum()
            total_obj += obj


        grad_z = torch.autograd.grad(total_obj, z)[0]

        guided_out = model_out - scale * grad_z.detach()

    return guided_out


def main(args):
    # Setup
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rae_config, model_config, transport_config, sampler_config, guidance_config, misc, _ = parse_configs(args.config)

    print("=> Loading RAE (Stage 1)...")
    rae: RAE = instantiate_from_config(rae_config).to(device).eval()

    print("=> Loading DiT-DH (Stage 2)...")
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device).eval()

    print("=> Loading Surrogate (IGD)...")
    surrogate = define_surrogate(args, args.target_nclass).to(device)
    surrogate = ReparamModule(surrogate).eval()
    ckpts = rand_ckpts(args)

    shift_dim = misc.get("time_dist_shift_dim", 768 * 16 * 16)
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    transport = create_transport(**transport_config['params'], time_dist_shift=time_dist_shift)
    sampler = Sampler(transport)

    # 确定采样模式
    mode, sampler_params = sampler_config['mode'], sampler_config['params']
    if args.num_sampling_steps > 0:
        sampler_params['num_steps'] = args.num_sampling_steps

    if mode == "ODE":
        base_sample_fn = sampler.sample_ode(**sampler_params)
    elif mode == "SDE":
        base_sample_fn = sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Unknown sampling mode: {mode}")

    # === 准备类别 ===
    try:
        with open('IGD/misc/class_indices.txt', 'r') as f:
            all_classes = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("Warning: class_indices.txt not found, using dummy classes.")
        all_classes = [f"class_{i}" for i in range(1000)]

    # 读取要采样的子集 (spec)
    if args.spec == '1k':
        file_list = 'IGD/misc/class_indices.txt'
    else:
        file_list = 'IGD/misc/class100.txt'  # 示例 Fallback

    try:
        with open(file_list, 'r') as f:
            sel_classes_names = [line.strip() for line in f.readlines()]
    except:
        sel_classes_names = all_classes[:args.nclass]

    # 切分 Phase
    phase = max(0, args.phase)
    cls_from = args.nclass * phase
    cls_to = args.nclass * (phase + 1)
    sel_classes_names = sel_classes_names[cls_from:cls_to]

    target_indices = [all_classes.index(c) for c in sel_classes_names]

    # === 采样循环 ===
    latent_size = misc.get("latent_size", (768, 16, 16))
    if hasattr(model, 'latent_size') and model.latent_size:
        latent_size = model.latent_size

    output_dir = os.path.join(args.save_dir, f"{args.spec}_ipc{args.num_samples}_gm{args.gm_scale}")
    os.makedirs(output_dir, exist_ok=True)

    for i, class_idx in enumerate(target_indices):
        class_name = sel_classes_names[i]
        print(f"\nProcessing Class {class_idx}: {class_name}")
        save_path = os.path.join(output_dir, class_name)
        os.makedirs(save_path, exist_ok=True)

        # 1. 计算该类别的真实梯度
        real_grads = get_grads(args, class_name, ckpts, surrogate, class_idx, device=device)

        # 2. 批量生成
        # IGD 计算开销大，强制 batch_size=1
        batch_size = 1
        num_batches = math.ceil(args.num_samples / batch_size)

        for b in tqdm(range(num_batches), desc="Sampling"):
            current_batch_size = min(batch_size, args.num_samples - b * batch_size)

            # 初始化 Latent (Noise)
            z = torch.randn(current_batch_size, *latent_size, device=device)
            y = torch.tensor([class_idx] * current_batch_size, device=device)
            y_null = torch.tensor([1000] * current_batch_size, device=device)

            # 定义带 IGD 引导的模型闭包
            def model_fn_igd(x, t, **kwargs):
                # x: (B, C, H, W)
                # t: (B,)

                # 扩展输入进行 CFG
                x_in = torch.cat([x, x], 0)
                t_in = torch.cat([t, t], 0)
                y_in = torch.cat([y, y_null], 0)

                # 1. DiT Forward (带 CFG)
                # DiT-DH 的 forward_with_cfg 会返回 CFG 后的结果
                # cfg_scale 控制 unconditional 和 conditional 的混合
                out = model.forward_with_cfg(x, t, y=y_in, cfg_scale=args.cfg_scale)

                # 将 CFG 后的速度场、当前的 x 和 t 传入 wrapper
                out_guided = igd_guidance_wrapper(
                    out, x, t,
                    rae, surrogate, ckpts, real_grads, class_idx, args.gm_scale
                )

                return out_guided

            # 执行采样
            # sample_fn 返回 shape (num_steps, B, C, H, W)，取 [-1] 为最终结果
            samples = base_sample_fn(z, model_fn_igd)[-1]

            decoded_images = rae.decode(samples)

            for k in range(current_batch_size):
                global_idx = b * batch_size + k + args.total_shift
                save_image(decoded_images[k], os.path.join(save_path, f"{global_idx}.png"),
                           normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path to RAE sampling config yaml")

    parser.add_argument("--data-path", type=str, required=True, help="Path to ImageNet train data")
    parser.add_argument("--ckpt_path", type=str, default="ckpts", help="Path to Surrogate ckpts")
    parser.add_argument("--save-dir", type=str, default="../results/rae-igd")

    parser.add_argument("--num-samples", type=int, default=10, help="Images per class (IPC)")
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=4.0)

    parser.add_argument("--gm-scale", type=float, default=0.02, help="IGD gradient guidance scale")
    parser.add_argument("--spec", type=str, default='1k', choices=['1k', '100', 'woof', 'nette'])
    parser.add_argument("--net-type", type=str, default='convnet6', help="Surrogate architecture")
    parser.add_argument("--target-nclass", type=int, default=1000)
    parser.add_argument("--nclass", type=int, default=10, help="Number of classes to sample")
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--real-ipc", type=int, default=1000, help="Images to load for gradient calculation")
    parser.add_argument("--grad-ipc", type=int, default=80, help="Images used for gradient matching")
    parser.add_argument("--total-shift", type=int, default=0)

    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--norm_type", type=str, default='instance')
    parser.add_argument("--nch", type=int, default=3)

    args = parser.parse_args()

    print(args)
    main(args)