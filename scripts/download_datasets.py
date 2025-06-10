import os
from huggingface_hub import login, snapshot_download,HfFolder
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
# from IPython.display import YouTubeVideo # 如果不在此脚本中播放视频，可以注释掉
# from diffusers import StableDiffusionPipeline # 如果不在此脚本中使用pipeline，可以注释掉
from tqdm import tqdm # 引入tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def download_dataset(repo_id, save_dir, token=None): # 添加token参数以便更灵活处理
    print(f"\nStarting download for: {repo_id}")
    try:
        snapshot_download(repo_id=repo_id,
                          repo_type="dataset",
                          cache_dir=save_dir, # cache_dir 指定缓存位置，下载的文件会存放在 cache_dir/models--<repo_owner>--<repo_name> 类似的路径下
                          local_dir=os.path.join(save_dir, repo_id.replace("/", "_")), # 使用 local_dir 将其明确下载到指定子目录
                        #   local_dir_use_symlinks=False,
                        #   resume_download=True,
                          token=token, # 传递token
                          # `snapshot_download` 默认会使用 tqdm 显示每个文件的下载进度 (如果 tqdm 已安装)
                          )
        print(f"Successfully downloaded or updated: {repo_id}")
    except Exception as e:
        print(f"Error downloading {repo_id}: {e}")


def NetTurbo():
    import subprocess
    import os

    result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
    output = result.stdout
    for line in output.splitlines():
        if '=' in line:
            var, value = line.split('=', 1)
            os.environ[var] = value


if __name__ == "__main__":
    NetTurbo()

    # "https://hf-mirror.com"
    # 'hf-cdn.sufy.com'
    os.environ['HF_ENDPOINT'] = "hf-cdn.sufy.com"

    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    os.environ['HF_TOKEN'] = hf_token
    HfFolder.save_token(hf_token)
    
    # 尝试登录，如果失败或token无效，后续操作可能会受影响
    try:
        login(token=hf_token)
        print("Hugging Face login successful.")
    except Exception as e:
        print(f"Hugging Face login failed: {e}. Downloads might rely on cached credentials or public access.")

    # 为了更清晰地将每个数据集下载到其独立的文件夹，并利用缓存，我调整了 local_dir 的设置
    base_save_dir = "/root/autodl-tmp/HuggingFace_Datasets" # 修改为一个更明确的父目录
    os.makedirs(base_save_dir, exist_ok=True) # 确保目录存在


    repo_ids = ["ItzLoghotXD/Ghibli"]
    
    # 去重 repo_ids 列表，避免重复下载
    unique_repo_ids = sorted(list(set(repo_ids)))

    print(f"Found {len(unique_repo_ids)} unique datasets to download.")

    # 使用tqdm包装迭代器，以显示总体下载进度
    # desc 参数为进度条提供描述信息
    for repo_id in tqdm(unique_repo_ids, desc="Downloading datasets"):
        # 为每个数据集创建一个单独的目录
        specific_dataset_dir = os.path.join(base_save_dir, repo_id.replace("/", "--"))
        # 注意：snapshot_download 的 cache_dir 是用于huggingface缓存系统，
        # local_dir 是实际文件存放的位置。
        # 如果你想让所有文件都通过huggingface的缓存管理，那么只用cache_dir。
        # 如果你想把特定snapshot下载到特定文件夹，使用local_dir。
        # 这里，我们将让 snapshot_download 把文件下载到特定文件夹
        download_dataset(repo_id, save_dir=specific_dataset_dir, token=hf_token)
        # 如果希望所有下载都进入统一的huggingface缓存（如~/.cache/huggingface/hub），然后由你手动管理，
        # 可以不设置 local_dir，只设置 cache_dir 为一个通用的缓存位置，
        # 但这样你需要知道文件在缓存中的确切路径。
        # 为了方便访问，下载到指定目录 (local_dir) 通常更直观。

    print("\nAll dataset download tasks complete.")