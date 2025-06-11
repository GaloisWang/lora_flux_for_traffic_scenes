import torch
from diffusers import FluxPipeline
import os
import logging
import gc
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Config:
    PROMPTS_DIR = (
        "/root/autodl-tmp/HuggingFace_Datasets/bdd100k/fulldata_test/captions/"
    )
    OFFICIAL_MODEL_PATH = "/root/autodl-tmp/models/black-forest-labs--FLUX.1-dev/"
    LORA_MODEL_PATH = "/home/lora_flux/train_logs_fulldata_060816e5/lora_epoch_30/"
    BASE_OUTPUT_DIR = "/home/lora_flux/model_compare/06082357"

    SEED = 1641421826
    WIDTH = 512
    HEIGHT = 512
    STEPS = 50
    GUIDANCE = 7.5


def load_pipeline(model_path, lora_path=None):
    """加载基础模型并可选地应用 LoRA。"""
    logger.info(f"Loading base model from: {model_path}")
    pipe = FluxPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="balanced"
    )

    if lora_path:
        logger.info(f"Applying LoRA from: {lora_path}")
        pipe.load_lora_weights(lora_path)
    else:
        logger.info("No LoRA applied. Using the base model.")

    return pipe


def clear_model(pipe):
    """完全清除模型释放显存。"""
    logger.info("Clearing model from memory...")
    del pipe

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(
                f"GPU {i} Memory after cleanup - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
            )
    gc.collect()
    logger.info("Memory cleanup completed.")


def generate_image(pipe, prompt, seed, width, height, steps, guidance):
    """使用给定的参数生成单张图片。"""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        generator=generator,
        width=width,
        height=height,
        guidance_scale=guidance,
    ).images[0]
    return image


def save_image(image, output_path):
    """保存图片到指定路径。"""
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    image.save(output_path, format="PNG")
    logger.info(f"Image saved to {output_path}")


def read_prompts_from_directory(directory_path):
    """
    遍历指定目录，读取所有 .txt 文件内容。
    返回一个字典，key 是不带后缀的文件名，value 是 prompt 内容。
    """
    prompts = {}
    if not os.path.isdir(directory_path):
        logger.error(f"Prompts directory not found: {directory_path}")
        return prompts

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            try:
                prompt_name = os.path.splitext(filename)[0]
                with open(file_path, "r", encoding="utf-8") as f:
                    prompts[prompt_name] = f.read().strip()
            except Exception as e:
                logger.error(f"Failed to read or process {file_path}: {e}")

    logger.info(f"Found {len(prompts)} prompts to process.")
    return prompts


def process_model_generation(
    model_name, model_path, lora_path=None, prompts=None, config=None, output_dir=None
):
    """处理模型生成过程的通用函数"""
    logger.info("=" * 50)
    logger.info(f"STARTING: {model_name} Generation")
    logger.info("=" * 50)

    # 加载模型
    pipe = load_pipeline(model_path, lora_path)

    # 融合LoRA权重
    if lora_path:
        logger.info("Fusing LoRA weights for optimized performance...")
        # 由于flux.1-dev占用显存过大,需要多gpu加载.使用fuse逻辑可以提升推理速度.降低GPU数据通信带来的延时
        pipe.fuse_lora()
        logger.info("LoRA weights fused.")

    # 生成图片
    for name, prompt_text in prompts.items():
        output_file = os.path.join(output_dir, f"{name}.png")
        if os.path.exists(output_file):
            logger.info(f"Skipping '{output_file}' as it already exists.")
            continue

        logger.info(f"Generating for prompt: '{name}.txt'")
        image = generate_image(
            pipe,
            prompt_text,
            config.SEED,
            config.WIDTH,
            config.HEIGHT,
            config.STEPS,
            config.GUIDANCE,
        )
        save_image(image, output_file)

    # 清理LoRA权重
    if lora_path:
        logger.info("Unfusing LoRA weights...")
        pipe.unfuse_lora()

    # 释放内存
    clear_model(pipe)


if __name__ == "__main__":
    cfg = Config()
    prompts_to_process = read_prompts_from_directory(cfg.PROMPTS_DIR)

    if not prompts_to_process:
        logger.warning("No prompts found. Exiting.")
        exit()

    # 创建一个唯一的顶层输出目录，用于区分每次运行
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = os.path.join(cfg.BASE_OUTPUT_DIR, timestamp)
    logger.info(f"All outputs for this run will be saved in: {main_output_dir}")

    # 处理基础模型生成
    base_output_path = os.path.join(main_output_dir, "base_model")
    process_model_generation(
        "Base Model",
        cfg.OFFICIAL_MODEL_PATH,
        lora_path=None,
        prompts=prompts_to_process,
        config=cfg,
        output_dir=base_output_path,
    )

    # 处理LoRA模型生成
    if cfg.LORA_MODEL_PATH:
        lora_output_path = os.path.join(main_output_dir, "lora_model")
        process_model_generation(
            "LoRA Model",
            cfg.OFFICIAL_MODEL_PATH,
            lora_path=cfg.LORA_MODEL_PATH,
            prompts=prompts_to_process,
            config=cfg,
            output_dir=lora_output_path,
        )

    logger.info("All tasks completed!")
    logger.info(f"Check your images in: {main_output_dir}")
