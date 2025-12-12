# RAE+IGD
使用IGD改进RAE

# 环境
下包
``` shell
conda create -n raeigd python=3.12 -y
conda activate raeigd
pip install uv
uv pip install torch==2.2.0 torchvision==0.17.0 torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install timm==0.9.16 accelerate==0.23.0 torchdiffeq==0.2.5 wandb
uv pip install "numpy<2" transformers einops omegaconf
```
下载模型（21G）, 需要设置代理
```shell
cd RAE
pip install huggingface_hub
hf download nyu-visionx/RAE-collections --local-dir models 
```
# 实验结果

# 改进位置