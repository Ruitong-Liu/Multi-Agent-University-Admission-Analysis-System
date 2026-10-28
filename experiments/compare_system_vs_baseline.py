"""
对比脚本：系统评估 vs Baseline评估
同时运行两种评估方法并对比结果
- Gemini：系统使用 Agent JSON 输入，Baseline 直接读取 PDF（Vision API）
- OpenAI：系统使用 Agent JSON 输入，Baseline 直接读取 PDF（GPT Vision API）
唯一区别是是否使用 Agent 系统预处理，确保公平对比
"""

import json
import os
import tempfile
import importlib.util
import random
from typing import Dict, Any, Optional, List, Tuple

from experiments.few_shot_evaluation import FewShotEvaluator, load_transcript_json, load_requirements_json
from experiments.gemini_baseline_evaluation import GeminiBaselineEvaluator
from experiments.openai_baseline_evaluation import OpenAIBaselineEvaluator
from experiments.text_baseline_evaluation import TextBaselineEvaluator
from agent3 import OpenAIProvider
import google.generativeai as genai  # type: ignore


def _build_filtered_acceptance(acceptance_path: str, student_ids: Optional[List[str]]) -> str:
    if not student_ids:
        return acceptance_path

    with open(acceptance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = {sid: data[sid] for sid in student_ids if sid in data}
    if not filtered:
        raise ValueError(f"筛选后为空，请检查 student_ids: {student_ids}")

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    with open(tmp_file.name, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    return tmp_file.name


def run_system_evaluation(
    test_size: int = None,
    acceptance_path: str = "acceptance.json",
    provider: str = "gemini",
    model_name: str = "gemini-2.5-flash",
    openai_api_key: Optional[str] = None,
    few_shot_examples: Optional[List[Any]] = None,
    test_set: Optional[List[Any]] = None,
):
    """运行系统评估（使用Agent系统 + Gemini）"""
    print("\n" + "="*80)
    print("系统评估（Agent系统 + Gemini）")
    print("="*80)
    
    llm_provider = None
    model_to_use = model_name

    if provider == "gemini":
        # 配置API密钥
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        genai.configure(api_key=api_key)

        # 创建Gemini提供者
        class SimpleGeminiProvider:
            def __init__(self):
                self.genai = genai
                self.models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]

            def generate(self, prompt: str, model: str = "gemini-2.5-flash") -> str:
                import time

                models_to_try = [model] + [m for m in self.models if m != model]

                last_error = None
                for model_name in models_to_try:
                    try:
                        model_instance = self.genai.GenerativeModel(model_name)
                        response = model_instance.generate_content(prompt)
                        return response.text
                    except Exception as e:
                        last_error = e
                        error_str = str(e)
                        if "429" in error_str or "quota" in error_str.lower():
                            print(f"    ⚠ {model_name} 配额不足，尝试下一个模型...")
                            time.sleep(1)
                            continue
                        print(f"    ⚠ {model_name} 错误: {str(e)[:100]}，尝试下一个模型...")
                        time.sleep(1)
                        continue

                print(f"    ✗ 所有模型都失败，最后错误: {str(last_error)[:200]}")
                return '{"admission_decision": "Unknown", "matching_score": 50, "reasoning": "API error"}'

        llm_provider = SimpleGeminiProvider()
        model_to_use = model_name
    elif provider == "openai":
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("缺少 OpenAI API 密钥，请设置 OPENAI_API_KEY 或传入 --openai-api-key")
        llm_provider = OpenAIProvider(openai_api_key)
        model_to_use = model_name or "gpt-4o"
    else:
        raise ValueError(f"未知 provider: {provider}")
    
    # 初始化评估器
    evaluator = FewShotEvaluator(acceptance_path)
    evaluator.load_and_convert_data()
    if few_shot_examples is None or test_set is None:
        few_shot_examples, test_set = evaluator.select_few_shot_examples(k=5)
        if test_size is not None:
            test_set = test_set[:test_size]
    
    # 运行评估
    zero_shot_metrics, few_shot_metrics = evaluator.evaluate(
        llm_provider=llm_provider,
        test_set=test_set,
        few_shot_examples=few_shot_examples,
        model_name=model_to_use,
        transcript_loader=lambda sid: load_transcript_json(sid),
        requirements_loader=lambda school, program: load_requirements_json(school, program, use_daad=False)
    )
    
    return zero_shot_metrics, few_shot_metrics, few_shot_examples, test_set


def run_baseline_evaluation(
    test_size: int = None,
    model_name: str = "gemini-2.5-flash",
    acceptance_path: str = "acceptance.json",
    provider: str = "gemini",
    openai_api_key: Optional[str] = None,
    few_shot_examples: Optional[List[Any]] = None,
    test_set: Optional[List[Any]] = None,
):
    """运行 Baseline 评估（不使用 Agent 系统，直接读取 PDF）
    
    Gemini Baseline: 用 Gemini Vision API 直接读 PDF
    OpenAI Baseline: 用 GPT Vision API 直接读 PDF
    两者都不经过 Agent1/Agent2 预处理，实现公平对比
    """
    print("\n" + "="*80)
    if provider == "gemini":
        print(f"Gemini Baseline 评估（直接读取 PDF，不使用 Agent 系统）— {model_name}")
    else:
        print(f"OpenAI Baseline 评估（直接读取 PDF，不使用 Agent 系统）— {model_name}")
    print("="*80)

    if provider == "gemini":
        evaluator = GeminiBaselineEvaluator(
            acceptance_json_path=acceptance_path,
            model_name=model_name,
        )
    elif provider == "openai":
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("缺少 OpenAI API 密钥，请设置 OPENAI_API_KEY 或传入 --openai-api-key")
        evaluator = OpenAIBaselineEvaluator(
            acceptance_json_path=acceptance_path,
            openai_api_key=openai_api_key,
            model_name=model_name,
        )
    else:
        raise ValueError(f"未知 provider: {provider}")

    evaluator.base_evaluator.load_and_convert_data()
    if few_shot_examples is None or test_set is None:
        few_shot_examples, test_set = evaluator.base_evaluator.select_few_shot_examples(k=5)
        if test_size is not None:
            test_set = test_set[:test_size]

    # 两种 Baseline 都直接读取 PDF，无需 JSON loader
    zero_shot_metrics, few_shot_metrics = evaluator.evaluate(
        test_set=test_set,
        few_shot_examples=few_shot_examples,
    )

    return zero_shot_metrics, few_shot_metrics, few_shot_examples, test_set


def run_text_baseline_evaluation(
    model_name: str = "gemini-2.5-flash",
    acceptance_path: str = "acceptance.json",
    provider: str = "gemini",
    openai_api_key: Optional[str] = None,
    few_shot_examples: Optional[List[Any]] = None,
    test_set: Optional[List[Any]] = None,
):
    """运行 Text Baseline 评估（纯文本输入，不使用 Agent 系统，不使用 Vision API 决策）

    输入：pdfplumber 提取的原始文本（扫描 PDF 用 Vision LLM 预先转录）
    决策：纯文本 LLM（同骨干模型）
    对比位置：Vision Baseline 与 System 之间的中间层
    """
    evaluator = TextBaselineEvaluator(
        acceptance_json_path=acceptance_path,
        provider=provider,
        model_name=model_name,
        openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
    )
    evaluator.base_evaluator.load_and_convert_data()
    if few_shot_examples is None or test_set is None:
        few_shot_examples, test_set = evaluator.base_evaluator.select_few_shot_examples(k=5)

    zero_shot_metrics, few_shot_metrics = evaluator.evaluate(
        test_set=test_set,
        few_shot_examples=few_shot_examples,
    )
    return zero_shot_metrics, few_shot_metrics, few_shot_examples, test_set


def print_comparison(system_z, system_f, baseline_z, baseline_f, test_size: int):
    """打印对比结果"""
    print("\n" + "="*80)
    print("对比结果汇总")
    print("="*80)
    
    print("\n[系统评估 - Zero-shot]")
    print(f"  准确率: {system_z.accuracy:.2%}")
    print(f"  混淆矩阵: TP={system_z.confusion_matrix['TP']}, FP={system_z.confusion_matrix['FP']}, "
          f"TN={system_z.confusion_matrix['TN']}, FN={system_z.confusion_matrix['FN']}")
    print(f"  错误数量: {len(system_z.errors)}")
    
    print("\n[系统评估 - Few-shot]")
    print(f"  准确率: {system_f.accuracy:.2%}")
    print(f"  混淆矩阵: TP={system_f.confusion_matrix['TP']}, FP={system_f.confusion_matrix['FP']}, "
          f"TN={system_f.confusion_matrix['TN']}, FN={system_f.confusion_matrix['FN']}")
    print(f"  错误数量: {len(system_f.errors)}")
    
    print("\n[Gemini Baseline - Zero-shot]")
    print(f"  准确率: {baseline_z.accuracy:.2%}")
    print(f"  混淆矩阵: TP={baseline_z.confusion_matrix['TP']}, FP={baseline_z.confusion_matrix['FP']}, "
          f"TN={baseline_z.confusion_matrix['TN']}, FN={baseline_z.confusion_matrix['FN']}")
    print(f"  错误数量: {len(baseline_z.errors)}")
    
    print("\n[Gemini Baseline - Few-shot]")
    print(f"  准确率: {baseline_f.accuracy:.2%}")
    print(f"  混淆矩阵: TP={baseline_f.confusion_matrix['TP']}, FP={baseline_f.confusion_matrix['FP']}, "
          f"TN={baseline_f.confusion_matrix['TN']}, FN={baseline_f.confusion_matrix['FN']}")
    print(f"  错误数量: {len(baseline_f.errors)}")
    
    print("\n[对比分析]")
    print(f"  Zero-shot: 系统 vs Baseline = {system_z.accuracy:.2%} vs {baseline_z.accuracy:.2%} "
          f"({(system_z.accuracy - baseline_z.accuracy):+.2%})")
    print(f"  Few-shot: 系统 vs Baseline = {system_f.accuracy:.2%} vs {baseline_f.accuracy:.2%} "
          f"({(system_f.accuracy - baseline_f.accuracy):+.2%})")
    print(f"  系统改进 (Few-shot vs Zero-shot): {(system_f.accuracy - system_z.accuracy):+.2%}")
    print(f"  Baseline改进 (Few-shot vs Zero-shot): {(baseline_f.accuracy - baseline_z.accuracy):+.2%}")
    
    print("="*80)


def save_comparison_results(system_z, system_f, baseline_z, baseline_f, 
                           system_few_shot, baseline_few_shot,
                           system_test_set, baseline_test_set,
                           output_path: str = "comparison_system_vs_baseline.json"):
    """保存对比结果"""
    results = {
        "test_set_size": len(system_test_set),
        "system_evaluation": {
            "description": "Agent系统 + Gemini (gemini-2.5-flash)",
            "zero_shot": {
                "accuracy": system_z.accuracy,
                "confusion_matrix": system_z.confusion_matrix,
                "error_count": len(system_z.errors),
                "errors": system_z.errors
            },
            "few_shot": {
                "accuracy": system_f.accuracy,
                "confusion_matrix": system_f.confusion_matrix,
                "error_count": len(system_f.errors),
                "errors": system_f.errors
            },
            "few_shot_examples": [
                {
                    "student_id": s.student_id,
                    "school": s.school,
                    "program": s.program,
                    "label": s.label
                }
                for s in system_few_shot
            ]
        },
        "baseline_evaluation": {
            "description": "Gemini Direct (No Agent System)",
            "zero_shot": {
                "accuracy": baseline_z.accuracy,
                "confusion_matrix": baseline_z.confusion_matrix,
                "error_count": len(baseline_z.errors),
                "errors": baseline_z.errors
            },
            "few_shot": {
                "accuracy": baseline_f.accuracy,
                "confusion_matrix": baseline_f.confusion_matrix,
                "error_count": len(baseline_f.errors),
                "errors": baseline_f.errors
            },
            "few_shot_examples": [
                {
                    "student_id": s.student_id,
                    "school": s.school,
                    "program": s.program,
                    "label": s.label
                }
                for s in baseline_few_shot
            ]
        },
        "comparison": {
            "zero_shot_difference": system_z.accuracy - baseline_z.accuracy,
            "few_shot_difference": system_f.accuracy - baseline_f.accuracy,
            "system_improvement": system_f.accuracy - system_z.accuracy,
            "baseline_improvement": baseline_f.accuracy - baseline_z.accuracy
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 对比结果已保存到: {output_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='对比系统评估 vs Gemini Baseline评估')
    parser.add_argument('--test-size', type=int, default=None,
                       help='测试样本数量（默认：全部）')
    parser.add_argument('--model', type=str, default="gemini-2.5-flash",
                       help='使用的模型（Gemini 或 OpenAI）')
    parser.add_argument('--system-provider', type=str, default="gemini",
                       choices=["gemini", "openai"],
                       help='系统评估使用的提供方（默认：gemini）')
    parser.add_argument('--baseline-provider', type=str, default="gemini",
                       choices=["gemini", "openai"],
                       help='baseline评估使用的提供方（默认：gemini）')
    parser.add_argument('--openai-api-key', type=str, default=None,
                       help='OpenAI API密钥（可选，默认从环境变量OPENAI_API_KEY读取）')
    parser.add_argument('--seed', type=int, default=42,
                       help='固定随机种子，确保few-shot与得分可复现（默认：42）')
    parser.add_argument(
        '--student-ids',
        type=str,
        default=None,
        help='仅评估指定学生ID，逗号分隔。例如: 01,03,04,05',
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("系统评估 vs Gemini Baseline评估对比")
    print(f"使用模型: {args.model}")
    print(f"系统提供方: {args.system_provider}")
    print(f"Baseline提供方: {args.baseline_provider}")
    print("="*80)
    
    student_ids = None
    filtered_acceptance = "acceptance.json"
    if args.student_ids:
        student_ids = [sid.strip() for sid in args.student_ids.split(",") if sid.strip()]
        filtered_acceptance = _build_filtered_acceptance("acceptance.json", student_ids)
        print(f"仅使用学生ID: {student_ids}")

    # 先固定随机种子并选择统一的 few-shot/test 集合
    random.seed(args.seed)
    shared_evaluator = FewShotEvaluator(filtered_acceptance)
    shared_evaluator.load_and_convert_data()
    shared_few_shot, shared_test_set = shared_evaluator.select_few_shot_examples(k=5)
    if args.test_size is not None:
        shared_test_set = shared_test_set[:args.test_size]

    # 运行系统评估（固定种子，确保few-shot构造一致）
    random.seed(args.seed)
    system_z, system_f, system_few_shot, system_test_set = run_system_evaluation(
        args.test_size,
        acceptance_path=filtered_acceptance,
        provider=args.system_provider,
        model_name=args.model if args.system_provider == "gemini" else (args.model or "gpt-4o"),
        openai_api_key=args.openai_api_key,
        few_shot_examples=shared_few_shot,
        test_set=shared_test_set,
    )
    
    # 运行Baseline评估（使用相同的模型）
    # 运行Baseline评估（固定种子，确保few-shot构造一致）
    random.seed(args.seed)
    baseline_z, baseline_f, baseline_few_shot, baseline_test_set = run_baseline_evaluation(
        args.test_size,
        args.model if args.baseline_provider == "gemini" else (args.model or "gpt-4o"),
        acceptance_path=filtered_acceptance,
        provider=args.baseline_provider,
        openai_api_key=args.openai_api_key,
        few_shot_examples=shared_few_shot,
        test_set=shared_test_set,
    )
    
    # 打印对比结果
    print_comparison(system_z, system_f, baseline_z, baseline_f, 
                    len(system_test_set) if args.test_size is None else args.test_size)
    
    # 保存对比结果
    suffix = f"_{'-'.join(student_ids)}" if student_ids else ""
    output_file = f"comparison_system_vs_baseline_{len(system_test_set)}samples{suffix}.json"
    save_comparison_results(system_z, system_f, baseline_z, baseline_f,
                           system_few_shot, baseline_few_shot,
                           system_test_set, baseline_test_set,
                           output_file)


if __name__ == "__main__":
    main()

