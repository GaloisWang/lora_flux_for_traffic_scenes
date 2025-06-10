import os
import shutil

# 原始根目录和目标目录
src_root = "/root/autodl-tmp/HuggingFace_Datasets/bdd100k_fulldata_train/validation_filtered/"
dst_root = "/root/autodl-tmp/HuggingFace_Datasets/bdd100k_fulldata_train/validation_filtered_2"

# 创建目标目录（如果不存在）
os.makedirs(dst_root, exist_ok=True)

# 支持的图片后缀
image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# 计数器用于避免文件名冲突（可选）
counter = 0

# 遍历所有子文件夹
for root, dirs, files in os.walk(src_root):
    for file in files:
        if file.lower().endswith(image_exts):
            src_path = os.path.join(root, file)
            # 保证目标文件唯一性（加上编号前缀）
            dst_filename = f"{counter:06d}_" + file
            dst_path = os.path.join(dst_root, dst_filename)
            shutil.copyfile(src_path, dst_path)
            counter += 1

print(f"✅ 总共复制了 {counter} 张图片到 {dst_root}")
