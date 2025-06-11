import os
from huggingface_hub import login, snapshot_download, HfFolder
import torch
from tqdm import tqdm

# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def download_model(repo_id, save_dir, token=None):
    print(f"\nStarting download for model: {repo_id}")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            cache_dir=save_dir,
            local_dir=os.path.join(save_dir, repo_id.replace("/", "--")),
            token=token,
        )
        print(f"Successfully downloaded or updated model: {repo_id}")
    except Exception as e:
        print(f"Error downloading model {repo_id}: {e}")


def NetTurbo():
    import subprocess
    import os

    result = subprocess.run(
        'bash -c "source /etc/network_turbo && env | grep proxy"',
        shell=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout
    for line in output.splitlines():
        if "=" in line:
            var, value = line.split("=", 1)
            os.environ[var] = value


if __name__ == "__main__":
    NetTurbo()

    os.environ["HF_ENDPOINT"] = "hf-cdn.sufy.com"
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    HfFolder.save_token(hf_token)

    base_save_dir = "/root/autodl-tmp/models/"
    model_repo_ids = ["black-forest-labs/FLUX.1-dev"]

    os.makedirs(base_save_dir, exist_ok=True)
    try:
        login(token=hf_token)
        print("Hugging Face login successful.")
    except Exception as e:
        print(
            f"Hugging Face login failed: {e}. Downloads might rely on cached credentials or public access."
        )

    unique_model_repo_ids = sorted(list(set(model_repo_ids)))
    print(f"Found {len(unique_model_repo_ids)} unique models to download.")

    for repo_id in tqdm(unique_model_repo_ids, desc="Downloading models"):
        specific_model_dir = os.path.join(base_save_dir, repo_id.replace("/", "--"))
        download_model(repo_id, save_dir=specific_model_dir, token=hf_token)

    print("\nAll model download tasks complete.")
