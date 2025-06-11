import os
from collections import defaultdict
from PIL import Image

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F


def is_front_camera(sample):
    """判断是否为前置摄像头图像"""
    if hasattr(sample.metadata, "camera_position"):
        return sample.metadata.camera_position.lower() == "front"
    if "front" in os.path.basename(sample.filepath).lower():
        return True
    return True  # 默认视为前向摄像头


def safe_label(label):
    """清理非法字符，作为文件夹名"""
    return label.replace("/", "_").replace(" ", "_").lower()


def get_combined_label(sample):
    """返回如 rainy-night 的联合标签，若不满足条件返回 None"""
    weather = getattr(sample.weather, "label", None)
    timeofday = getattr(sample.timeofday, "label", None)

    weather_ok = weather in CONFIG["ATTRIBUTES_TO_FILTER"]["weather"]
    time_ok = timeofday in CONFIG["ATTRIBUTES_TO_FILTER"]["timeofday"]

    if weather_ok or time_ok:
        parts = []
        if weather_ok:
            parts.append(safe_label(weather))
        if time_ok:
            parts.append(safe_label(timeofday))
        return "-".join(parts)
    return None


def export_filtered_images(view, export_dir):
    """按标签导出图像"""
    groups = defaultdict(list)

    for sample in view:
        if not is_front_camera(sample):
            continue
        label = get_combined_label(sample)
        if label:
            groups[label].append(sample.filepath)

    total_exported = 0

    for label, filepaths in groups.items():
        label_dir = os.path.join(export_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        for filepath in filepaths:
            try:
                img = Image.open(filepath).convert("RGB")
                save_path = os.path.join(label_dir, os.path.basename(filepath))
                img.save(save_path)
                total_exported += 1
            except Exception as e:
                print(f"图像处理失败: {filepath}，错误: {e}")

        print(f"已导出 {len(filepaths)} 张 '{label}' 图像至: {label_dir}")

    return total_exported


def main():
    print("--- 基于 FiftyOne 的 BDD100K 场景筛选器 ---")

    for split in CONFIG["SPLITS"]:
        dataset_name = f"{CONFIG['DATASET_NAME_PREFIX']}-{split}"
        print(f"\n--- 正在处理 {split.upper()} 数据集 ---")

        if fo.dataset_exists(dataset_name):
            fo.delete_dataset(dataset_name)

        try:
            print(f"加载 {dataset_name} 数据集...")
            dataset = foz.load_zoo_dataset(
                "bdd100k",
                split=split,
                dataset_name=dataset_name,
                source_dir=CONFIG["BDD100K_SOURCE_DIR"],
                max_samples=1000,
            )
        except Exception as e:
            print(f"加载失败: {e}")
            continue

        print(f"数据集加载成功，共 {len(dataset)} 张图像")

        # 属性筛选
        weather_filter = F("weather.label").is_in(
            CONFIG["ATTRIBUTES_TO_FILTER"]["weather"]
        )
        time_filter = F("timeofday.label").is_in(
            CONFIG["ATTRIBUTES_TO_FILTER"]["timeofday"]
        )
        view = dataset.match(F.any([weather_filter, time_filter]))

        print(f"属性筛选后剩余 {len(view)} 张图像")

        if len(view) == 0:
            continue

        # 导出处理
        split_export_dir = os.path.join(CONFIG["EXPORT_DIR"], split)
        os.makedirs(split_export_dir, exist_ok=True)

        exported_count = export_filtered_images(view, split_export_dir)
        print(f"共导出 {exported_count} 张图像到 {split_export_dir}")

    print(f"全部完成！图像已保存在：'{CONFIG['EXPORT_DIR']}'")


if __name__ == "__main__":
    # --- 配置区 ---
    CONFIG = {
        "ATTRIBUTES_TO_FILTER": {
            "weather": ["rainy", "snowy", "foggy"],
            "timeofday": ["night", "dawn/dusk"],
        },
        "EXPORT_DIR": "/Users/cooper/Data/PublicDatasets/bdd100k_filtered_images3",
        "DATASET_NAME_PREFIX": "bdd100k_filtered",
        "SPLITS": ["train", "validation"],
        "BDD100K_SOURCE_DIR": "/Users/cooper/Data/PublicDatasets/bdd100k",
    }
    main()
