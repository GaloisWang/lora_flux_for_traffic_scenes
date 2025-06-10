import torch
import torchvision.transforms as T
import os
from PIL import Image
from tqdm import tqdm
import lpips
from pytorch_fid import fid_score
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import re
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型权重文件的完整本地路径
local_hf_model_dir = "/home/models/openai--clip-vit-base-patch32/openai--clip-vit-base-patch32/" 
if not os.path.exists(local_hf_model_dir):
    raise FileNotFoundError(f"CLIP模型权重文件未找到: {local_hf_model_dir}。请检查路径或文件名。")

try:
    # 使用正确的CLIP模型和处理器
    clip_model = CLIPModel.from_pretrained(local_hf_model_dir)
    clip_processor = CLIPProcessor.from_pretrained(local_hf_model_dir)
    
    print("✅ CLIP模型加载成功")
    
except Exception as e:
    print(f"❌ 加载CLIP模型时出错: {e}")
    print("请确保目录包含所有必需文件 (config.json, pytorch_model.bin, tokenizer等)")
    raise

# 将模型移动到设备
clip_model = clip_model.to(device)
clip_model.eval()  # 设置为评估模式

# 加载 LPIPS 模型
lpips_model = lpips.LPIPS(net='alex').to(device)

def extract_key_phrases(text, max_phrases=10):
    """从长文本中提取关键短语"""
    # 移除常见的停用词和连接词
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    
    # 提取形容词+名词的组合
    important_patterns = [
        r'\b(?:beautiful|stunning|gorgeous|elegant|serene|mystical|magical|ancient|traditional|modern|colorful|vibrant|peaceful|dramatic)\s+\w+',
        r'\b(?:ghibli|studio|anime|manga|japanese|traditional|style|art|painting|illustration)',
        r'\b(?:forest|mountain|castle|village|garden|temple|house|building|landscape|nature|sky|cloud|water|tree|flower)',
        r'\b(?:character|girl|boy|woman|man|person|people|figure)',
        r'\b(?:green|blue|red|yellow|purple|pink|orange|white|black|golden|silver)\s+\w+',
    ]
    
    key_phrases = []
    text_lower = text.lower()
    
    # 提取匹配的关键短语
    for pattern in important_patterns:
        matches = re.findall(pattern, text_lower)
        key_phrases.extend(matches)
    
    # 移除重复并限制数量
    unique_phrases = list(dict.fromkeys(key_phrases))[:max_phrases]
    
    if unique_phrases:
        return ', '.join(unique_phrases)
    else:
        # 如果没有找到关键短语，返回前几个重要词汇
        words = text_lower.split()
        important_words = [w for w in words if w not in stop_words and len(w) > 2]
        return ' '.join(important_words[:15])

def split_and_average_embeddings(text, images, max_length=75):
    """将长文本分割成多个片段，分别计算embeddings然后平均"""
    # 按句子分割文本
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        sentences = [text]
    
    # 如果单个句子仍然太长，按逗号分割
    final_segments = []
    for sentence in sentences:
        if len(clip_processor.tokenizer.encode(sentence)) <= max_length:
            final_segments.append(sentence)
        else:
            # 按逗号分割长句子
            parts = sentence.split(',')
            current_part = ""
            for part in parts:
                test_part = current_part + ("," if current_part else "") + part.strip()
                if len(clip_processor.tokenizer.encode(test_part)) <= max_length:
                    current_part = test_part
                else:
                    if current_part:
                        final_segments.append(current_part)
                    current_part = part.strip()
            if current_part:
                final_segments.append(current_part)
    
    # 如果还是有过长的片段，使用关键短语提取
    processed_segments = []
    for segment in final_segments:
        if len(clip_processor.tokenizer.encode(segment)) > max_length:
            segment = extract_key_phrases(segment)
        processed_segments.append(segment)
    
    # 计算所有片段的文本embeddings
    text_embeddings = []
    
    for segment in processed_segments:
        try:
            inputs = clip_processor(
                text=[segment], 
                images=images, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=77
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = clip_model(**inputs)
                text_emb = outputs.text_embeds  # 使用text_embeds而不是logits
                text_embeddings.append(text_emb)
                
        except Exception as e:
            print(f"⚠️  处理文本片段时出错: {e}")
            continue
    
    if not text_embeddings:
        raise RuntimeError("无法处理任何文本片段")
    
    # 平均所有文本embeddings
    avg_text_embedding = torch.stack(text_embeddings).mean(dim=0)
    
    # 计算图像embeddings
    image_inputs = clip_processor(images=images, return_tensors="pt")
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    
    with torch.no_grad():
        image_outputs = clip_model(**image_inputs)
        image_embeddings = image_outputs.image_embeds
    
    # 计算相似度
    similarity = F.cosine_similarity(avg_text_embedding, image_embeddings, dim=-1)
    
    return similarity

def compute_clip_score_with_long_text(image_folder, prompt_folder, method="key_phrases"):
    """
    支持长文本的CLIP分数计算
    method: "key_phrases" | "split_average" | "sliding_window"
    """
    images, prompts, names = load_images_and_prompts(image_folder, prompt_folder)
    
    if len(images) == 0:
        raise RuntimeError("❌ 未找到有效的图像用于CLIP评分")
    
    similarities = []
    long_text_count = 0
    
    print(f"🔄 使用 {method} 方法计算CLIP分数...")
    
    for i, (image, prompt, name) in enumerate(tqdm(zip(images, prompts, names), total=len(images))):
        try:
            token_length = len(clip_processor.tokenizer.encode(prompt))
            
            if token_length <= 75:
                # 短文本直接处理
                inputs = clip_processor(
                    text=[prompt], 
                    images=[image], 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True,
                    max_length=77
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = clip_model(**inputs)
                    similarity = outputs.logits_per_image[0, 0]
                    similarities.append(similarity.cpu().item())
            else:
                # 长文本处理
                long_text_count += 1
                
                if method == "key_phrases":
                    # 方法1：提取关键短语
                    short_prompt = extract_key_phrases(prompt)
                    inputs = clip_processor(
                        text=[short_prompt], 
                        images=[image], 
                        return_tensors="pt", 
                        padding=True,
                        truncation=True,
                        max_length=77
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = clip_model(**inputs)
                        similarity = outputs.logits_per_image[0, 0]
                        similarities.append(similarity.cpu().item())
                
                elif method == "split_average":
                    # 方法2：分割文本并平均embeddings
                    similarity = split_and_average_embeddings(prompt, [image])
                    similarities.append(similarity.cpu().item())
                
                elif method == "sliding_window":
                    # 方法3：滑动窗口，取最高相似度
                    words = prompt.split()
                    max_similarity = -1
                    window_size = 50  # 大约对应75个tokens
                    
                    for start in range(0, len(words), window_size//2):  # 50%重叠
                        window_text = ' '.join(words[start:start+window_size])
                        if len(clip_processor.tokenizer.encode(window_text)) > 77:
                            continue
                            
                        inputs = clip_processor(
                            text=[window_text], 
                            images=[image], 
                            return_tensors="pt", 
                            padding=True,
                            truncation=True,
                            max_length=77
                        )
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            outputs = clip_model(**inputs)
                            sim = outputs.logits_per_image[0, 0].cpu().item()
                            max_similarity = max(max_similarity, sim)
                    
                    if max_similarity > -1:
                        similarities.append(max_similarity)
                    else:
                        # 回退到关键短语方法
                        short_prompt = extract_key_phrases(prompt)
                        inputs = clip_processor(
                            text=[short_prompt], 
                            images=[image], 
                            return_tensors="pt", 
                            padding=True,
                            truncation=True,
                            max_length=77
                        )
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            outputs = clip_model(**inputs)
                            similarity = outputs.logits_per_image[0, 0]
                            similarities.append(similarity.cpu().item())
                
        except Exception as e:
            print(f"⚠️  处理 {name} 时出错: {e}")
            continue
    
    if not similarities:
        raise RuntimeError("❌ 无法计算任何有效的CLIP分数")
    
    avg_similarity = sum(similarities) / len(similarities)
    print(f"✅ CLIP平均相似度: {avg_similarity:.4f} (基于 {len(similarities)} 个样本)")
    print(f"📊 其中 {long_text_count} 个长文本使用了 {method} 方法处理")
    
    return avg_similarity

def analyze_prompt_lengths(prompt_folder):
    """分析提示文本的长度分布"""
    print("🔍 分析提示文本长度...")
    
    lengths = []
    long_prompts = []
    
    prompt_files = [f for f in os.listdir(prompt_folder) if f.endswith('.txt')]
    
    for fname in prompt_files:
        try:
            with open(os.path.join(prompt_folder, fname), 'r', encoding='utf-8') as f:
                text = f.read().strip()
                token_length = len(clip_processor.tokenizer.encode(text))
                lengths.append(token_length)
                
                if token_length > 77:
                    long_prompts.append((fname, token_length, text[:100] + "..."))
                    
        except Exception as e:
            print(f"⚠️  读取文件 {fname} 时出错: {e}")
            continue
    
    if lengths:
        avg_length = sum(lengths) / len(lengths)
        max_length = max(lengths)
        min_length = min(lengths)
        over_limit = len([l for l in lengths if l > 77])
        
        print(f"📊 提示文本统计:")
        print(f"   • 总数: {len(lengths)}")
        print(f"   • 平均长度: {avg_length:.1f} tokens")
        print(f"   • 最大长度: {max_length} tokens")
        print(f"   • 最小长度: {min_length} tokens")
        print(f"   • 超过77 tokens: {over_limit} 个 ({over_limit/len(lengths)*100:.1f}%)")
        
        if long_prompts:
            print(f"\n📝 部分过长的提示文本示例:")
            for fname, length, preview in long_prompts[:3]:  # 只显示前3个
                print(f"   • {fname}: {length} tokens")
                print(f"     '{preview}'")
    
    return lengths

def load_images_and_prompts(image_folder, prompt_folder):
    """加载图像和对应的提示文本"""
    images, prompts, names = [], [], []

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for fname in sorted(image_files):
        base_name = os.path.splitext(fname)[0]
        prompt_path = os.path.join(prompt_folder, f"{base_name}.txt")
        
        if not os.path.exists(prompt_path):
            print(f"⚠️  提示文件未找到: {fname}, 跳过该文件")
            continue
            
        try:
            # 加载图像
            img_path = os.path.join(image_folder, fname)
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            
            # 加载对应 prompt
            with open(prompt_path, 'r', encoding='utf-8') as pf:
                prompt_text = pf.read().strip()
                prompts.append(prompt_text)
                
            names.append(fname)
            
        except Exception as e:
            print(f"⚠️  处理文件 {fname} 时出错: {e}")
            continue
    
    print(f"📊 成功加载 {len(images)} 个图像-文本对")
    return images, prompts, names

def load_images_for_lpips(image_folder):
    """为LPIPS加载图像并转换为张量"""
    images = []
    transform = T.Compose([
        T.Resize((256, 256)),  # LPIPS通常使用256x256
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化到[-1,1]
    ])
    
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    for fname in image_files:
        try:
            img_path = os.path.join(image_folder, fname)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img)
            images.append(img_tensor)
        except Exception as e:
            print(f"⚠️  处理LPIPS图像 {fname} 时出错: {e}")
            continue
    
    return torch.stack(images) if images else torch.empty(0)

def compute_lpips(folder1, folder2):
    """计算两个文件夹之间图像的LPIPS距离"""
    print("🔄 加载图像用于LPIPS计算...")
    imgs1 = load_images_for_lpips(folder1)
    imgs2 = load_images_for_lpips(folder2)
    
    if len(imgs1) == 0 or len(imgs2) == 0:
        raise RuntimeError("❌ LPIPS计算时未找到有效图像")
    
    if len(imgs1) != len(imgs2):
        min_len = min(len(imgs1), len(imgs2))
        imgs1 = imgs1[:min_len]
        imgs2 = imgs2[:min_len]
        print(f"⚠️  图像数量不匹配，使用前 {min_len} 个图像")
    
    imgs1 = imgs1.to(device)
    imgs2 = imgs2.to(device)
    
    distances = []
    print("🔄 计算LPIPS距离...")
    
    for i in tqdm(range(len(imgs1))):
        with torch.no_grad():
            d = lpips_model(imgs1[i:i+1], imgs2[i:i+1])
            distances.append(d.item())
    
    avg_distance = sum(distances) / len(distances)
    print(f"✅ LPIPS平均距离: {avg_distance:.4f}")
    
    return avg_distance

def compute_fid(folder1, folder2, batch_size=50):
    """计算两个文件夹之间的FID分数"""
    print("🔄 计算FID分数...")
    try:
        fid_value = fid_score.calculate_fid_given_paths(
            [folder1, folder2], 
            batch_size, 
            device, 
            2048
        )
        print(f"✅ FID分数: {fid_value:.2f}")
        return fid_value
    except Exception as e:
        print(f"❌ FID计算出错: {e}")
        raise

# 兼容性函数，保持原有接口
def compute_clip_score(image_folder, prompt_folder):
    """默认使用关键短语方法的CLIP分数计算"""
    return compute_clip_score_with_long_text(image_folder, prompt_folder, method="key_phrases")