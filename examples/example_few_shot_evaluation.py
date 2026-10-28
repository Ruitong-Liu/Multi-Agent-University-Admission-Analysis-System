"""
Few-shot Learning评估系统使用示例
演示如何使用FewShotEvaluator进行zero-shot和few-shot评估
"""

import os

from experiments.few_shot_evaluation import FewShotEvaluator, load_transcript_json, load_requirements_json
import google.generativeai as genai
import json


def example_basic_evaluation():
    """基本评估示例 - 不使用transcript和requirements"""
    print("=" * 60)
    print("示例1: 基本评估（仅使用acceptance.json中的标签）")
    print("=" * 60)
    
    # 配置API密钥
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    genai.configure(api_key=api_key)
    
    # 创建Gemini提供者
    class SimpleGeminiProvider:
        def __init__(self):
            self.genai = genai
        
        def generate(self, prompt: str, model: str = "gemini-3-pro-preview") -> str:
            model_instance = self.genai.GenerativeModel(model)
            response = model_instance.generate_content(prompt)
            return response.text
    
    llm_provider = SimpleGeminiProvider()
    
    # 初始化评估器
    evaluator = FewShotEvaluator("acceptance.json")
    
    # 加载数据
    evaluator.load_and_convert_data()
    
    # 选择few-shot示例
    few_shot_examples, test_set = evaluator.select_few_shot_examples(k=5)
    
    # 运行评估（不使用transcript和requirements）
    print("\n注意: 此示例不使用真实的transcript和requirements数据")
    print("仅基于acceptance.json中的标签进行评估\n")
    
    zero_shot_metrics, few_shot_metrics = evaluator.evaluate(
        llm_provider=llm_provider,
        test_set=test_set[:5],  # 仅评估前5个样本以节省时间
        few_shot_examples=few_shot_examples,
        model_name="gemini-3-pro-preview"
    )
    
    # 打印结果
    evaluator.print_evaluation_summary(zero_shot_metrics, few_shot_metrics)
    
    # 保存结果
    evaluator.save_evaluation_results(zero_shot_metrics, few_shot_metrics, 
                                     "evaluation_results_basic.json")


def example_with_transcript_and_requirements():
    """使用真实transcript和requirements的评估示例"""
    print("\n" + "=" * 60)
    print("示例2: 使用真实transcript和requirements数据")
    print("=" * 60)
    
    # 配置API密钥
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    genai.configure(api_key=api_key)
    
    # 创建Gemini提供者
    class SimpleGeminiProvider:
        def __init__(self):
            self.genai = genai
        
        def generate(self, prompt: str, model: str = "gemini-3-pro-preview") -> str:
            model_instance = self.genai.GenerativeModel(model)
            response = model_instance.generate_content(prompt)
            return response.text
    
    llm_provider = SimpleGeminiProvider()
    
    # 初始化评估器
    evaluator = FewShotEvaluator("acceptance.json")
    
    # 加载数据
    evaluator.load_and_convert_data()
    
    # 选择few-shot示例
    few_shot_examples, test_set = evaluator.select_few_shot_examples(k=5)
    
    # 运行评估（使用transcript和requirements加载器）
    print("\n注意: 此示例使用真实的transcript和requirements数据")
    print("评估可能需要较长时间...\n")
    
    zero_shot_metrics, few_shot_metrics = evaluator.evaluate(
        llm_provider=llm_provider,
        test_set=test_set[:3],  # 仅评估前3个样本以节省时间
        few_shot_examples=few_shot_examples,
        model_name="gemini-3-pro-preview",
        transcript_loader=lambda sid: load_transcript_json(sid),
        requirements_loader=lambda school, program: load_requirements_json(
            school, program, use_daad=False  # 不使用DAAD爬虫以加快速度
        )
    )
    
    # 打印结果
    evaluator.print_evaluation_summary(zero_shot_metrics, few_shot_metrics)
    
    # 保存结果
    evaluator.save_evaluation_results(zero_shot_metrics, few_shot_metrics,
                                     "evaluation_results_with_data.json")


def example_custom_few_shot_selection():
    """自定义few-shot示例选择"""
    print("\n" + "=" * 60)
    print("示例3: 自定义few-shot示例选择")
    print("=" * 60)
    
    evaluator = FewShotEvaluator("acceptance.json")
    evaluator.load_and_convert_data()
    
    # 选择不同数量的few-shot示例
    for k in [3, 5, 7]:
        print(f"\n选择 {k} 个few-shot示例:")
        few_shot_examples, test_set = evaluator.select_few_shot_examples(k=k)
        
        print(f"  选定的示例:")
        for i, example in enumerate(few_shot_examples, 1):
            print(f"    {i}. Student {example.student_id}: {example.school} - {example.program} ({example.label})")


def example_prompt_generation():
    """演示提示词生成"""
    print("\n" + "=" * 60)
    print("示例4: 提示词生成演示")
    print("=" * 60)
    
    evaluator = FewShotEvaluator("acceptance.json")
    evaluator.load_and_convert_data()
    
    # 选择few-shot示例
    few_shot_examples, _ = evaluator.select_few_shot_examples(k=3)
    
    # 选择一个测试样本
    test_sample = evaluator.test_set[0] if evaluator.test_set else evaluator.all_samples[0]
    
    # 生成zero-shot提示词
    print("\n[Zero-shot提示词]")
    zero_shot_prompt = evaluator.build_prompt(test_sample, few_shot_examples=None)
    print(zero_shot_prompt[:500] + "...\n")
    
    # 生成few-shot提示词
    print("\n[Few-shot提示词]")
    few_shot_prompt = evaluator.build_prompt(test_sample, few_shot_examples=few_shot_examples)
    print(few_shot_prompt[:800] + "...\n")
    
    # 生成带transcript和requirements的提示词
    transcript = load_transcript_json("01")
    if transcript:
        print("\n[带transcript和requirements的提示词]")
        full_prompt = evaluator.build_prompt(
            test_sample,
            few_shot_examples=few_shot_examples,
            transcript_json=transcript,
            requirements_json={
                "University": "Test University",
                "Program": "Test Program",
                "requirements": {"Description/content": "Test requirements"}
            }
        )
        print(full_prompt[:1000] + "...\n")


if __name__ == "__main__":
    # 运行示例
    print("Few-shot Learning评估系统使用示例\n")
    
    # 示例1: 基本评估
    example_basic_evaluation()
    
    # 示例2: 使用真实数据（可选，需要较长时间）
    # example_with_transcript_and_requirements()
    
    # 示例3: 自定义few-shot选择
    example_custom_few_shot_selection()
    
    # 示例4: 提示词生成
    example_prompt_generation()





