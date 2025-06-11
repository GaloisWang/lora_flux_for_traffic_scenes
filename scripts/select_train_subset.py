import random
import shutil
import os


def create_train_subset(
    source_img_dir,
    source_caption_dir,
    target_img_dir,
    target_caption_dir,
    subset_size=100,
):
    """创建测试子集"""
    # 获取所有图片文件
    img_files = [f for f in os.listdir(source_img_dir) if f.endswith((".jpg", ".png"))]

    # 随机选择
    selected_files = random.sample(img_files, min(subset_size, len(img_files)))

    # 清空目标目录
    def clear_directory(dir_path):
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                shutil.rmtree(os.path.join(root, dir))

    if os.path.exists(target_img_dir):
        clear_directory(target_img_dir)
    if os.path.exists(target_caption_dir):
        clear_directory(target_caption_dir)

    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_caption_dir, exist_ok=True)

    for img_file in selected_files:
        # 复制图片
        shutil.copy2(
            os.path.join(source_img_dir, img_file),
            os.path.join(target_img_dir, img_file),
        )

        # 复制对应的caption文件
        caption_file = img_file.rsplit(".", 1)[0] + ".txt"
        shutil.copy2(
            os.path.join(source_caption_dir, caption_file),
            os.path.join(target_caption_dir, caption_file),
        )

    print(f"创建了包含{len(selected_files)}张图片的测试集")


if __name__ == "__main__":
    source_img_dir = (
        "/root/autodl-tmp/HuggingFace_Datasets/Nechintosh--ghibli/Nechintosh_ghibli/"
    )
    source_caption_dir = (
        "/root/autodl-tmp/HuggingFace_Datasets/Nechintosh--ghibli/captions_gstyle/"
    )
    target_img_dir = "/root/autodl-tmp/HuggingFace_Datasets/Ghibli_demo_dataset/images/"
    target_caption_dir = (
        "/root/autodl-tmp/HuggingFace_Datasets/Ghibli_demo_dataset/captions/"
    )
    subset_size = 100
    create_train_subset(
        source_img_dir,
        source_caption_dir,
        target_img_dir,
        target_caption_dir,
        subset_size,
    )
