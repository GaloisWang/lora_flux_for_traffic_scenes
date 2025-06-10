import os
import gc
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from diffusers import FluxPipeline
from contextlib import contextmanager
from typing import Optional, List, Tuple

import os
import json
from safetensors import safe_open

# ==== 日志配置 ====
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==== 配置路径 ====
PATHS = {
    "real_style_folder": "/root/autodl-tmp/HuggingFace_Datasets/bdd100k/fulldata_test/images/",
    "captions_folder": "/root/autodl-tmp/HuggingFace_Datasets/bdd100k/fulldata_test/captions/",
    "official_model_path": "/root/autodl-tmp/models/black-forest-labs--FLUX.1-dev/",
    "lora_model_dir": "/home/lora_flux/train_logs_fulldata_060816e5/lora_epoch_30/",
    "output_official": "/home/lora_flux/model_compare/official_output/",
    "output_lora": "/home/lora_flux/model_compare/fulldata_060816e5_output/",
}

# ==== 生成配置 ====
CONFIG = {
    "guidance_scale": 7.5,
    "num_inference_steps": 50,
    "use_deterministic": True,
    "base_seed": 42,
    "enable_memory_efficient_attention": True,
    "enable_cpu_offload": True,
    "batch_size": 1,
}


class FluxBatchGenerator:
    def __init__(self):
        self.device = self._get_device()
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        # 创建输出目录
        for output_path in [PATHS["output_official"], PATHS["output_lora"]]:
            Path(output_path).mkdir(parents=True, exist_ok=True)

        # 显存信息
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU Memory: {total_memory:.1f} GB")

    def _get_device(self) -> str:
        """智能设备检测"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @contextmanager
    def memory_cleanup(self):
        """上下文管理器，用于内存清理"""
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def _load_pipeline(
        self, model_path: str, enable_lora: bool = False
    ) -> FluxPipeline:
        """使用自动检测的LoRA加载pipeline"""
        model_type = "LoRA" if enable_lora else "Official"
        logger.info(f"🔄 Loading {model_type} Flux model...")

        pipe = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map="balanced",
        )

        # 如果是LoRA模型，使用改进的加载方法
        if enable_lora:
            logger.info("🔄 Loading LoRA weights with auto-detection...")
            pipe.load_lora_weights(PATHS["lora_model_dir"])

        pipe.set_progress_bar_config(disable=True)

        return pipe

    def _clear_model(self, pipe: FluxPipeline):
        """完全清除模型释放显存"""
        logger.info("🧹 Clearing model from memory...")
        del pipe

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 获取清理后的显存信息
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            logger.info(
                f"📊 GPU Memory after cleanup - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
            )

        gc.collect()
        logger.info("✅ Memory cleanup completed")

    def _validate_prompt(self, prompt: str) -> str:
        """验证和清理prompt"""
        prompt = " ".join(prompt.split())
        if len(prompt) > 512:
            logger.debug(f"Processing long prompt with {len(prompt)} characters")
        return prompt

    def generate_batch_images(
        self, pipe: FluxPipeline, prompts: List[str], seeds: List[int]
    ) -> List[Optional[Image.Image]]:
        """
        生成一个批次的图片。
        接受 prompts 列表和 seeds 列表。
        """
        if not prompts:
            return []

        # 验证和清理 prompts
        processed_prompts = [self._validate_prompt(p) for p in prompts]

        # 生成器列表
        generators = [torch.Generator(device=self.device).manual_seed(s) for s in seeds]

        try:
            results = pipe(
                prompt=processed_prompts, # <- 传入 prompts 列表
                guidance_scale=CONFIG["guidance_scale"],
                num_inference_steps=CONFIG["num_inference_steps"],
                generator=generators, # <- 传入 generators 列表
            ).images # 返回的是图片列表

            return results # 返回 Image.Image 对象的列表

        except torch.cuda.OutOfMemoryError:
            logger.error(f"❌ CUDA out of memory during batch generation for {len(prompts)} images. Consider reducing batch_size.")
            torch.cuda.empty_cache()
            return [None] * len(prompts) # 返回相应数量的 None
        except Exception as e:
            logger.error(f"❌ Batch generation failed for {len(prompts)} images: {e}")
            return [None] * len(prompts) # 返回相应数量的 None


    def get_tasks(self) -> List[Tuple[str, str, int, Path, Path]]:
        """获取需要处理的任务列表"""
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        tasks = []

        # 遍历图片文件
        for file_path in Path(PATHS["real_style_folder"]).iterdir():
            if file_path.suffix.lower() not in image_extensions:
                continue

            base_name = file_path.stem

            # 检查caption是否存在
            caption_path = Path(PATHS["captions_folder"]) / f"{base_name}.txt"
            if not caption_path.exists():
                logger.warning(f"⚠️ Caption not found for {base_name}")
                continue

            # 读取caption
            try:
                with open(caption_path, "r", encoding="utf-8") as f:
                    prompt = f.read().strip()
            except Exception as e:
                logger.error(f"❌ Failed to read caption for {base_name}: {e}")
                continue

            # 生成种子
            if CONFIG["use_deterministic"]:
                seed = CONFIG["base_seed"] + hash(base_name) % 1000000
            else:
                seed = CONFIG["base_seed"]

            # 输出路径
            official_path = Path(PATHS["output_official"]) / f"{base_name}.png"
            lora_path = Path(PATHS["output_lora"]) / f"{base_name}.png"

            tasks.append((base_name, prompt, seed, official_path, lora_path))

        return sorted(tasks)

    def save_image(self, image: Image.Image, output_path: Path) -> bool:
        """保存图片"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, optimize=True, quality=95)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save image to {output_path}: {e}")
            return False

    def generate_batch(
        self, tasks: List[Tuple[str, str, int, Path, Path]], model_type: str
    ):
        """批量生成图片"""
        if model_type == "official":
            logger.info("🎨 Starting Official Model Generation Phase")
            pipe = self._load_pipeline(PATHS["official_model_path"], enable_lora=False)
            output_index = 3  # official_path
        else:
            logger.info("🎨 Starting LoRA Model Generation Phase")
            pipe = self._load_pipeline(PATHS["official_model_path"], enable_lora=True)
            output_index = 4  # lora_path

        successful = 0
        failed = 0
        skipped = 0

        tasks_to_process = []
        for task in tasks:
            output_path = task[output_index]
            if output_path.exists():
                skipped += 1
            else:
                tasks_to_process.append(task)

        logger.info(
            f"📋 {model_type.upper()} - Total: {len(tasks)}, To process: {len(tasks_to_process)}, Skipped: {skipped}"
        )

        if not tasks_to_process:
            logger.info(f"✅ All {model_type} images already exist, skipping...")
            self._clear_model(pipe)
            return successful, failed, skipped

        # 使用批次处理
        total_tasks_to_process = len(tasks_to_process)
        # progress_bar = tqdm(tasks_to_process, desc=f"Generating {model_type} images") # 不再直接迭代任务，而是迭代批次

        # 进度条应该基于批次数量
        num_batches = (total_tasks_to_process + CONFIG["batch_size"] - 1) // CONFIG["batch_size"]
        progress_bar = tqdm(range(num_batches), desc=f"Generating {model_type} batches")


        for i in progress_bar:
            start_idx = i * CONFIG["batch_size"]
            end_idx = min((i + 1) * CONFIG["batch_size"], total_tasks_to_process)
            
            current_batch_tasks = tasks_to_process[start_idx:end_idx]
            
            batch_prompts = [task[1] for task in current_batch_tasks]
            batch_seeds = [task[2] for task in current_batch_tasks]
            batch_output_paths = [task[output_index] for task in current_batch_tasks]
            batch_base_names = [task[0] for task in current_batch_tasks] # for logging

            # 生成图片批次
            batch_images = self.generate_batch_images(pipe, batch_prompts, batch_seeds)

            # 处理生成的图片
            for j, image in enumerate(batch_images):
                if image is not None:
                    if self.save_image(image, batch_output_paths[j]):
                        successful += 1
                    else:
                        failed += 1
                else:
                    # 如果 generate_batch_images 返回 None，表示该图片生成失败
                    failed += 1

            progress_bar.set_postfix(
                {"Success": successful, "Failed": failed, "Skipped": skipped, "BatchSize": len(current_batch_tasks)}
            )

            # 每次处理完一个批次后清理内存
            torch.cuda.empty_cache()
            gc.collect()

        # 清理模型
        self._clear_model(pipe)

        logger.info(
            f"✅ {model_type.upper()} Generation Complete - Success: {successful}, Failed: {failed}, Skipped: {skipped}"
        )
        return successful, failed, skipped


    def generate_comparison_images(self):
        """主要的图片生成函数 - 批处理模式"""
        # 获取所有任务
        tasks = self.get_tasks()
        logger.info(f"📋 Total tasks found: {len(tasks)}")

        if not tasks:
            logger.warning("⚠️ No valid tasks found!")
            return

        total_stats = {"successful": 0, "failed": 0, "skipped": 0}

        # 第一阶段：生成官方模型图片
        logger.info("=" * 60)
        logger.info("🚀 PHASE 1: Official Model Generation")
        logger.info("=" * 60)

        success, failed, skipped = self.generate_batch(tasks, "official")
        total_stats['successful'] += success
        total_stats['failed'] += failed
        total_stats['skipped'] += skipped

        # 等待一下确保清理完成
        logger.info("⏳ Waiting for memory cleanup...")
        torch.cuda.empty_cache()
        gc.collect()

        # 第二阶段：生成LoRA模型图片
        logger.info("=" * 60)
        logger.info("🚀 PHASE 2: LoRA Model Generation")
        logger.info("=" * 60)

        success, failed, skipped = self.generate_batch(tasks, "lora")
        total_stats["successful"] += success
        total_stats["failed"] += failed
        total_stats["skipped"] += skipped

        # 最终统计
        logger.info("=" * 60)
        logger.info("📊 FINAL SUMMARY")
        logger.info("=" * 60)
        logger.info(
            f"""
                    Total Tasks: {len(tasks)}
                    Total Successful: {total_stats['successful']}
                    Total Failed: {total_stats['failed']}
                    Total Skipped: {total_stats['skipped']}
                    Seed Strategy: {'Deterministic per image' if CONFIG['use_deterministic'] else 'Fixed seed'}
                    """
        )


def main():
    """主函数"""
    generator = FluxBatchGenerator()

    try:
        logger.info("🎯 Starting Flux Batch Generation")
        logger.info("📝 Strategy: Sequential model loading (Official → LoRA)")

        # 生成对比图片
        generator.generate_comparison_images()

        logger.info("🎉 All generations completed successfully!")

    except KeyboardInterrupt:
        logger.info("⏸️ Generation interrupted by user")
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        raise
    finally:
        # 最终清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("🧹 Final cleanup completed")


if __name__ == "__main__":
    main()
