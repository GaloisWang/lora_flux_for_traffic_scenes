import os
import kagglehub

def NetTurbo():
    import subprocess
    import os

    result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
    output = result.stdout
    for line in output.splitlines():
        if '=' in line:
            var, value = line.split('=', 1)
            os.environ[var] = value


def ensure_kaggle_token():
    """确保 ~/.kaggle/kaggle.json 存在"""
    if not os.path.exists(KAGGLE_JSON_PATH):
        os.makedirs(os.path.dirname(KAGGLE_JSON_PATH), exist_ok=True)
        os.system(f"cp {YOUR_LOCAL_KAGGLE_JSON} {KAGGLE_JSON_PATH}")
        os.chmod(KAGGLE_JSON_PATH, 0o600)
        print(f"✅ Copied Kaggle token to {KAGGLE_JSON_PATH}")
    else:
        print(f"✅ Found Kaggle token at {KAGGLE_JSON_PATH}")


def download_dataset(dataset_id):
    """使用 kagglehub 下载并缓存数据集"""
    print(f">>> Downloading dataset '{dataset_id}' using kagglehub...")

    try:
        path = kagglehub.dataset_download(dataset_id)
        print(f"✅ Dataset '{dataset_id}' downloaded successfully.")
        print("📁 Local path to dataset files:", path)
        return path
    except Exception as e:
        print(f"❌ Error downloading dataset '{dataset_id}' with kagglehub: {e}")
        print("⚠️  Please check your internet connection, Kaggle API key, and dataset ID.")
        exit(1)


if __name__ == "__main__":
    # === 配置区 ===
    KAGGLE_JSON_PATH = os.path.expanduser("~/.kaggle/kaggle.json")
    YOUR_LOCAL_KAGGLE_JSON = "/root/autodl-tmp/kaggle_token/kaggle.json"
    KAGGLE_DATASET_ID = "solesensei/solesensei_bdd100k"
    os.environ['KAGGLEHUB_CACHE'] = '/root/autodl-tmp/HuggingFace_Datasets/'

    NetTurbo()
    ensure_kaggle_token()
    download_dataset(KAGGLE_DATASET_ID)
