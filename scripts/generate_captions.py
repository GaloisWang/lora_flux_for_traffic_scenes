import os
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import json

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")


def load_model(model_path):
    print("正在加载模型和Tokenizer...")
    try:
        model = (
            AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            .eval()
            .cuda()
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        print("模型和Tokenizer加载完成。")
        return model, tokenizer
    except Exception as e:
        print(f"加载模型或Tokenizer时发生错误: {e}")
        raise


def generate_caption(model, tokenizer, image_path, question):
    try:
        image = Image.open(image_path).convert("RGB")
        msgs = [{"role": "user", "content": [image, question]}]
        res = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
        return str(res).strip()
    except Exception as e:
        print(f"图片 {image_path} 处理失败: {e}")
        return None


def process_all_images(model, tokenizer, image_dir, caption_output_dir, question):
    os.makedirs(caption_output_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(image_dir) if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    total_images = len(image_files)
    print(f"共找到 {total_images} 张图片待处理。")

    all_captions_metadata = []

    for i, image_filename in enumerate(image_files):
        print(f"处理第 {i+1}/{total_images} 张: {image_filename}")
        image_path = os.path.join(image_dir, image_filename)
        base_filename, _ = os.path.splitext(image_filename)
        caption_txt_path = os.path.join(caption_output_dir, f"{base_filename}.txt")

        # 跳过已存在的caption文件
        # if os.path.exists(caption_txt_path):
        #     print(f"Caption 已存在: {caption_txt_path}，跳过。")
        #     continue

        caption = generate_caption(model, tokenizer, image_path, question)

        if not caption:
            caption = "ghibli style"
            print(f"空caption使用默认文本: {caption}")

        # 保存caption
        try:
            with open(caption_txt_path, "w", encoding="utf-8") as f:
                f.write(caption)
            print(f"Caption 保存到: {caption_txt_path}")
        except Exception as e:
            print(f"写入caption文件失败: {e}")

        all_captions_metadata.append(
            {
                "image_filename": image_filename,
                "caption_file": os.path.basename(caption_txt_path),
                "caption": caption,
            }
        )

    return all_captions_metadata


def save_metadata(metadata, output_dir):
    if not metadata:
        return
    metadata_path = os.path.join(output_dir, "_metadata_captions.jsonl")
    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            for item in metadata:
                f.write(json.dumps(item) + "\n")
        print(f"Caption数据保存至: {metadata_path}")
    except Exception as e:
        print(f"保存数据失败: {e}")


if __name__ == "__main__":
    model_path = "/root/autodl-tmp/models/MiniCPM-V-2_6"
    image_dir = "/root/autodl-tmp/HuggingFace_Datasets/bdd100k/fulldata_train/images"
    caption_output_dir = os.path.join(os.path.dirname(image_dir), "captions")
    question = (
        "This is a picture of a real road scene. Please give descriptive synthetic captions for the image for LoRA fine-tuning."
        " Please only output the prompt, do not output any thought process."
    )

    try:
        model, tokenizer = load_model(model_path)
        metadata = process_all_images(
            model, tokenizer, image_dir, caption_output_dir, question
        )
        save_metadata(metadata, caption_output_dir)
        print("所有图片处理完成！")
    except Exception as main_error:
        print(f"主流程中发生错误: {main_error}")
