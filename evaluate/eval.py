import os
from utils import (
    compute_clip_score_with_long_text,
    compute_lpips,
    compute_fid,
    analyze_prompt_lengths,
)


def main():
    base_folder = "/home/lora_flux/model_compare/06082357/20250608_235827/base_model/"
    lora_folder = "/home/lora_flux/model_compare/06082357/20250608_235827/lora_model/"
    prompts_dir = (
        "/root/autodl-tmp/HuggingFace_Datasets/bdd100k/fulldata_test/captions/"
    )
    real_style_folder = (
        "/root/autodl-tmp/HuggingFace_Datasets/bdd100k/fulldata_test/images/"
    )

    paths_to_check = {
        "基础模型输出": base_folder,
        "LoRA模型输出": lora_folder,
        "提示文件": prompts_dir,
        "真实风格图像": real_style_folder,
    }

    for name, path in paths_to_check.items():
        if not os.path.exists(path):
            print(f"{name}路径不存在: {path}")
            return
        else:
            print(f"{name}: {path}")

    print("\n" + "=" * 60)

    print("预处理: 分析提示文本")
    print("-" * 30)
    lengths = analyze_prompt_lengths(prompts_dir)

    if lengths:
        over_limit_ratio = len([l for l in lengths if l > 77]) / len(lengths)
        avg_length = sum(lengths) / len(lengths)

        if over_limit_ratio < 0.1:
            recommended_method = "key_phrases"
            print(f"建议使用关键短语方法 (只有 {over_limit_ratio*100:.1f}% 的文本过长)")
        elif avg_length < 150:
            recommended_method = "split_average"
            print(f"建议使用文本分割平均方法 (平均长度: {avg_length:.1f} tokens)")
        else:
            recommended_method = "sliding_window"
            print(f"建议使用滑动窗口方法 (文本较长，平均: {avg_length:.1f} tokens)")
    else:
        recommended_method = "key_phrases"

    print(f"将使用 {recommended_method} 方法处理长文本")

    print("\n" + "=" * 60)

    try:
        # 1. CLIP Score评估 - 比较不同方法
        print("1. CLIP Score 评估 (多种方法对比)")
        print("-" * 40)

        # methods = ["key_phrases", "split_average", "sliding_window"]
        methods = ["sliding_window"]
        method_names = {
            "key_phrases": "关键短语提取",
            "split_average": "文本分割平均",
            "sliding_window": "滑动窗口最大值",
        }

        results = {}

        for method in methods:
            print(f"使用 {method_names[method]} 方法:")
            try:
                clip_base = compute_clip_score_with_long_text(
                    base_folder, prompts_dir, method=method
                )
                clip_lora = compute_clip_score_with_long_text(
                    lora_folder, prompts_dir, method=method
                )

                results[method] = {
                    "base": clip_base,
                    "lora": clip_lora,
                    "diff": clip_lora - clip_base,
                }

                print(f"基础模型: {clip_base:.4f}")
                print(f"LoRA模型: {clip_lora:.4f}")
                print(f"差异: {clip_lora - clip_base:+.4f}")

            except Exception as e:
                print(f"{method_names[method]} 方法失败: {e}")
                continue

        # 显示方法比较
        if results:
            print(f"\n方法对比总结:")
            print(
                f"{'方法':<12} {'基础模型':<10} {'LoRA模型':<10} {'差异':<10} {'LoRA更好'}"
            )
            print("-" * 55)

            for method, data in results.items():
                better = "更好" if data["diff"] > 0 else "更差"
                print(
                    f"{method_names[method]:<12} {data['base']:<10.4f} {data['lora']:<10.4f} {data['diff']:<+10.4f} {better}"
                )

        # 使用推荐方法的结果进行后续分析
        if recommended_method in results:
            clip_base = results[recommended_method]["base"]
            clip_lora = results[recommended_method]["lora"]
            print(f"\n后续分析将使用 {method_names[recommended_method]} 的结果")
        else:
            print("推荐方法失败，使用关键短语方法")
            clip_base = compute_clip_score_with_long_text(
                base_folder, prompts_dir, method="key_phrases"
            )
            clip_lora = compute_clip_score_with_long_text(
                lora_folder, prompts_dir, method="key_phrases"
            )

        print("\n" + "=" * 60)

        # 2. LPIPS评估
        print("2. LPIPS 感知差异评估")
        print("-" * 30)
        lpips_val = compute_lpips(base_folder, lora_folder)
        print(f"🔹 LPIPS分数: {lpips_val:.4f}")

        if lpips_val < 0.3:
            print("两个模型生成的图像感知上相似")
        elif lpips_val < 0.6:
            print("两个模型生成的图像有中等程度的感知差异")
        else:
            print("两个模型生成的图像感知差异较大")

        print("\n" + "=" * 60)

        # 3. FID评估
        print("3. FID 图像质量评估")
        print("-" * 30)
        fid_base = compute_fid(base_folder, real_style_folder)
        fid_lora = compute_fid(lora_folder, real_style_folder)

        print(f"基础模型FID: {fid_base:.2f}")
        print(f"LoRA模型FID: {fid_lora:.2f}")

        if fid_lora < fid_base:
            print(f"LoRA模型更接近真实风格 (FID降低 {fid_base-fid_lora:.2f})")
        else:
            print(f"基础模型更接近真实风格 (FID降低 {fid_lora-fid_base:.2f})")

        print("\n" + "=" * 60)

        # 总结
        print("最终评估总结")
        print("-" * 30)
        print(f"使用方法: {method_names.get(recommended_method, '关键短语提取')}")
        print(f"CLIP分数对比: 基础({clip_base:.4f}) vs LoRA({clip_lora:.4f})")
        print(f"LPIPS差异: {lpips_val:.4f}")
        print(f"FID分数对比: 基础({fid_base:.2f}) vs LoRA({fid_lora:.2f})")

        # 综合评估
        clip_better = "LoRA" if clip_lora > clip_base else "基础"
        fid_better = "LoRA" if fid_lora < fid_base else "基础"

        print(f"\n结论:")
        print(f"   • 文本对齐: {clip_better}模型更好")
        print(f"   • 风格相似: {fid_better}模型更好")
        print(f"   • 感知差异: {lpips_val:.4f} (越小越相似)")

        # 方法建议
        if results and len(results) > 1:
            print(f"\n方法选择建议:")
            best_method = max(results.keys(), key=lambda x: results[x]["diff"])
            print(
                f"   • 对于您的数据，{method_names[best_method]}方法显示LoRA改进最明显"
            )

            consistent_better = all(
                results[method]["diff"] > 0 for method in results.keys()
            )
            if consistent_better:
                print(f"   • 所有方法都显示LoRA模型更好，结果一致性高")
            else:
                print(f"   • 不同方法得出了不同结论，建议综合考虑")

    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
