import torch
from diffusers import FluxPipeline
from pathlib import Path
from datetime import datetime
import os
import logging
import gc


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_pipeline(model_path, lora_path=None):
    print(f"Loading base model from: {model_path}")
    pipe = FluxPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="balanced"
    )

    if lora_path:
        print(f"Applying LoRA from: {lora_path}")
        pipe.load_lora_weights(lora_path)
    else:
        print("No LoRA applied.")

    return pipe


def clear_model(pipe: FluxPipeline):
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


def generate_image(pipe, prompt, seed=42, width=512, height=512, steps=50, guidance=4):
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
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    print(f"Image saved to {output_path}")


def run_generation(model_path, prompt, save_path, seed=42, lora_path=None):
    pipe = load_pipeline(model_path, lora_path)
    image = generate_image(pipe, prompt, seed=seed)
    save_image(image, save_path)
    clear_model(pipe)


if __name__ == "__main__":
    prompt = "A silver sedan is parked on a suburban street with overcast skies. The houses are two-story, featuring a mix of beige and brown exteriors, with some having red accents and others displaying white trim. A lamp post stands near the curb, and several cars are visible in the background, suggesting a quiet residential area."

    seed = 1641421826
    output_dir = "/root/Codes/lora_flux/compare/output_images/"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    guidance = 7.5
    official_model_path = "/root/autodl-tmp/models/black-forest-labs--FLUX.1-dev/"
    lora_model_path = "/home/lora_flux/train_logs_fulldata_060816e5/lora_epoch_30/"
    segments = lora_model_path.strip(os.sep).split(os.sep)
    target_segment = next((s for s in segments if "train_logs_" in s), None)

    os.makedirs(output_dir, exist_ok=True)

    # generate images by base model.
    run_generation(
        official_model_path,
        prompt,
        f"{output_dir}/origin_{target_segment}_cfg{guidance}_{timestamp}.png",
        seed=seed,
        lora_path=None,
    )

    # generate images by lora fintuned model.
    run_generation(
        official_model_path,
        prompt,
        f"{output_dir}/copax_lora_{target_segment}_cfg{guidance}_{timestamp}.png",
        seed=seed,
        lora_path=lora_model_path,
    )
