import torch
from diffusers import FluxPipeline
from pathlib import Path
from datetime import datetime
import os
import logging
import gc

# ==== 日志配置 ====
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==== 配置类 ====
# 将所有可配置项集中管理，方便修改
class Config:
    # --- 路径配置 ---
    PROMPTS_DIR = "/root/autodl-tmp/HuggingFace_Datasets/bdd100k/fulldata_test/captions/"
    OFFICIAL_MODEL_PATH = "/root/autodl-tmp/models/black-forest-labs--FLUX.1-dev/"
    LORA_MODEL_PATH = "/home/lora_flux/train_logs_fulldata_060816e5/lora_epoch_30/"
    BASE_OUTPUT_DIR = "/home/lora_flux/model_compare/06082357"

    # --- 生成参数 ---
    SEED = 1641421826
    WIDTH = 512
    HEIGHT = 512
    STEPS = 50
    GUIDANCE = 7.5

def load_pipeline(model_path, lora_path=None):
    """加载基础模型并可选地应用 LoRA。"""
    logger.info(f"🚀 Loading base model from: {model_path}")
    pipe = FluxPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="balanced"
    )

    if lora_path:
        logger.info(f"✨ Applying LoRA from: {lora_path}")
        pipe.load_lora_weights(lora_path)
    else:
        logger.info("✅ No LoRA applied. Using the base model.")

    return pipe

def clear_model(pipe: FluxPipeline):
    """完全清除模型释放显存。"""
    logger.info("🧹 Clearing model from memory...")
    del pipe

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(
                f"📊 GPU {i} Memory after cleanup - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
            )
    gc.collect()
    logger.info("✅ Memory cleanup completed.")

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
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    logger.info(f"Image saved to {output_path}")

def save_image_lora(image, output_path):
    """保存图片到指定路径。"""
    output_path_dir = os.path.dirname(output_path)
    os.makedirs(output_path_dir,exist_ok=True)
    image.save(output_path, format="PNG")
    logger.info(f"Image saved to {output_path}")

def read_prompts_from_directory(directory_path):
    """
    遍历指定目录，读取所有 .txt 文件内容。
    返回一个字典，key 是不带后缀的文件名，value 是 prompt 内容。
    """
    prompts = {}
    path = Path(directory_path)
    if not path.is_dir():
        logger.error(f"Prompts directory not found: {directory_path}")
        return prompts

    for file_path in path.glob("*.txt"):
        try:
            prompt_name = file_path.stem
            with open(file_path, "r", encoding="utf-8") as f:
                prompts[prompt_name] = f.read().strip()
        except Exception as e:
            logger.error(f"Failed to read or process {file_path}: {e}")
    
    logger.info(f"Found {len(prompts)} prompts to process.")
    return prompts

if __name__ == "__main__":
    cfg = Config()
    prompts_to_process = read_prompts_from_directory(cfg.PROMPTS_DIR)

    if not prompts_to_process:
        logger.warning("No prompts found. Exiting.")
        exit()

    # 创建一个唯一的顶层输出目录，用于区分每次运行
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # main_output_dir = Path(cfg.BASE_OUTPUT_DIR) / timestamp
    main_output_dir = "/home/lora_flux/model_compare/06082357/20250608_235827/"
    logger.info(f"All outputs for this run will be saved in: {main_output_dir}")
    
    # # --- 任务一：使用官方基础模型生成图片 ---
    # logger.info("=" * 50)
    # logger.info("▶️ STARTING: Base Model Generation")
    # logger.info("=" * 50)
    
    # base_model_pipe = load_pipeline(cfg.OFFICIAL_MODEL_PATH, lora_path=None)
    # base_output_path = main_output_dir / "base_model"
    
    # for name, prompt_text in prompts_to_process.items():
    #     logger.info(f"Generating for prompt: '{name}.txt'")
    #     image = generate_image(
    #         base_model_pipe, prompt_text, cfg.SEED, cfg.WIDTH, cfg.HEIGHT, cfg.STEPS, cfg.GUIDANCE
    #     )
    #     # 图片名与 prompt 文件名一致
    #     save_image(image, base_output_path / f"{name}.png")
    
    # clear_model(base_model_pipe)
    
    # --- 任务二：使用 LoRA 微调模型生成图片 (如果配置了 LoRA 路径) ---
    if cfg.LORA_MODEL_PATH:
        logger.info("=" * 50)
        logger.info("▶️ STARTING: LoRA Model Generation")
        logger.info("=" * 50)

        lora_model_pipe = load_pipeline(cfg.OFFICIAL_MODEL_PATH, lora_path=cfg.LORA_MODEL_PATH)

        logger.info("Fusing LoRA weights for optimized performance...")
        lora_model_pipe.fuse_lora()
        logger.info("✅ LoRA weights fused.")

        lora_output_path = os.path.join(main_output_dir,"lora_model")

        for name, prompt_text in prompts_to_process.items():
            output_file = os.path.join(lora_output_path , f"{name}.png")
            if os.path.exists(output_file):
                logger.info(f"Skipping '{output_file}' as it already exists.")
                continue  # 跳过当前循环，处理下一个 prompt
            logger.info(f"Generating for prompt: '{name}.txt'")
            image = generate_image(
                lora_model_pipe, prompt_text, cfg.SEED, cfg.WIDTH, cfg.HEIGHT, cfg.STEPS, cfg.GUIDANCE
            )
            # 图片名与 prompt 文件名一致
            save_image_lora(image, output_file)


        logger.info("Unfusing LoRA weights...")
        lora_model_pipe.unfuse_lora()
        clear_model(lora_model_pipe)

    logger.info("🎉 All tasks completed!")
    logger.info(f"Check your images in: {main_output_dir}")