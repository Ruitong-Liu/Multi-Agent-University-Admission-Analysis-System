"""
批量模型对比脚本
比较 4 个模型在 System（Agent系统）vs Baseline（直接读取 PDF）两种模式下的性能

模型列表：
  1. gpt-4o              (OpenAI)
  2. gpt-5.2-2025-12-11  (OpenAI, 最新)
  3. gemini-2.5-flash    (Google Gemini 2.5)
  4. gemini-3-flash-preview (Google Gemini 3.0)

关键设计：
  - 固定 seed=42，所有模型共享同一个 few-shot/test 划分，保证公平对比
  - System 模式：使用 Agent1/Agent2 预处理后的 JSON 输入（结构化 Prompt）
  - Baseline 模式：直接读取 PDF（OpenAI GPT Vision / Gemini Vision），不经过 Agent 预处理
  - 两种 provider 的 Baseline 架构对称，确保对比的公平性
"""

import json
import os
import random
import argparse
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from experiments.few_shot_evaluation import FewShotEvaluator, load_transcript_json, load_requirements_json
from experiments.gemini_baseline_evaluation import GeminiBaselineEvaluator
from experiments.compare_system_vs_baseline import run_system_evaluation, run_baseline_evaluation, run_text_baseline_evaluation
import google.generativeai as genai  # type: ignore


OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    os.environ.get("OPENAI_API_KEY", "")
)

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "")


# ── 4 个模型的配置 ─────────────────────────────────────────────────────────────
MODEL_CONFIGS = [
    {
        "label": "GPT-4o",
        "provider": "openai",
        "model": "gpt-4o",
    },
    {
        "label": "GPT-5.2 (2025-12-11)",
        "provider": "openai",
        "model": "gpt-5.2-2025-12-11",
    },
    {
        "label": "Gemini 2.5 Flash",
        "provider": "gemini",
        "model": "gemini-2.5-flash",
    },
    {
        "label": "Gemini 3.0 Flash",
        "provider": "gemini",
        "model": "gemini-3-flash-preview",
    },
]


@dataclass
class ModelResult:
    label: str
    model: str
    system_zero_acc: float = 0.0
    system_few_acc: float = 0.0
    baseline_zero_acc: float = 0.0
    baseline_few_acc: float = 0.0
    text_zero_acc: float = 0.0
    text_few_acc: float = 0.0
    system_zero_cm: Dict = field(default_factory=dict)
    system_few_cm: Dict = field(default_factory=dict)
    baseline_zero_cm: Dict = field(default_factory=dict)
    baseline_few_cm: Dict = field(default_factory=dict)
    text_zero_cm: Dict = field(default_factory=dict)
    text_few_cm: Dict = field(default_factory=dict)
    error: Optional[str] = None


def _cm_str(cm: Dict) -> str:
    return f"TP={cm.get('TP',0)} FP={cm.get('FP',0)} TN={cm.get('TN',0)} FN={cm.get('FN',0)}"


def print_summary_table(results: List[ModelResult]):
    """打印横向对比总表（三层：System / Text Baseline / Vision Baseline）"""
    print("\n" + "=" * 120)
    print("【四模型三层对比总表】  seed=42，共享 few-shot / test 集")
    print("  System = Agent JSON 输入  |  Text = 原始文本输入  |  Vision = PDF 图像直读")
    print("=" * 120)
    header = (
        f"{'模型':<28}"
        f" {'Sys-Z':>7} {'Sys-F':>7}"
        f" {'Txt-Z':>7} {'Txt-F':>7}"
        f" {'Vis-Z':>7} {'Vis-F':>7}"
        f" {'Sys>Txt(Z)':>10} {'Sys>Txt(F)':>10}"
        f" {'Txt>Vis(Z)':>10} {'Txt>Vis(F)':>10}"
    )
    print(header)
    print("-" * 120)
    for r in results:
        if r.error:
            print(f"{r.label:<28}  [ERROR] {r.error}")
            continue
        sys_txt_z = r.system_zero_acc - r.text_zero_acc
        sys_txt_f = r.system_few_acc  - r.text_few_acc
        txt_vis_z = r.text_zero_acc   - r.baseline_zero_acc
        txt_vis_f = r.text_few_acc    - r.baseline_few_acc
        row = (
            f"{r.label:<28}"
            f" {r.system_zero_acc:>6.1%} {r.system_few_acc:>6.1%}"
            f" {r.text_zero_acc:>6.1%} {r.text_few_acc:>6.1%}"
            f" {r.baseline_zero_acc:>6.1%} {r.baseline_few_acc:>6.1%}"
            f" {sys_txt_z:>+9.1%} {sys_txt_f:>+9.1%}"
            f" {txt_vis_z:>+9.1%} {txt_vis_f:>+9.1%}"
        )
        print(row)
    print("=" * 120)

    # 详细混淆矩阵
    print("\n【混淆矩阵详情】")
    for r in results:
        if r.error:
            continue
        print(f"\n  {r.label}  ({r.model})")
        print(f"    System  Zero-shot      : {_cm_str(r.system_zero_cm)}")
        print(f"    System  Few-shot       : {_cm_str(r.system_few_cm)}")
        print(f"    Text    Zero-shot      : {_cm_str(r.text_zero_cm)}")
        print(f"    Text    Few-shot       : {_cm_str(r.text_few_cm)}")
        print(f"    Vision  Zero-shot      : {_cm_str(r.baseline_zero_cm)}")
        print(f"    Vision  Few-shot       : {_cm_str(r.baseline_few_cm)}")


def save_all_results(results: List[ModelResult], output_path: str):
    data = []
    for r in results:
        data.append({
            "label": r.label,
            "model": r.model,
            "error": r.error,
            "system": {
                "zero_shot": {"accuracy": r.system_zero_acc, "confusion_matrix": r.system_zero_cm},
                "few_shot":  {"accuracy": r.system_few_acc,  "confusion_matrix": r.system_few_cm},
            },
            "text_baseline": {
                "zero_shot": {"accuracy": r.text_zero_acc, "confusion_matrix": r.text_zero_cm},
                "few_shot":  {"accuracy": r.text_few_acc,  "confusion_matrix": r.text_few_cm},
            },
            "baseline": {
                "zero_shot": {"accuracy": r.baseline_zero_acc, "confusion_matrix": r.baseline_zero_cm},
                "few_shot":  {"accuracy": r.baseline_few_acc,  "confusion_matrix": r.baseline_few_cm},
            },
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 完整结果已保存到: {output_path}")


def run_one_model(
    cfg: Dict,
    shared_few_shot: List,
    shared_test_set: List,
    acceptance_path: str,
    seed: int,
) -> ModelResult:
    label = cfg["label"]
    provider = cfg["provider"]
    model = cfg["model"]

    print(f"\n{'#' * 80}")
    print(f"# 正在评估: {label}  ({model})")
    print(f"{'#' * 80}")

    result = ModelResult(label=label, model=model)

    try:
        # ── System 评估 ────────────────────────────────────────────
        random.seed(seed)
        sys_z, sys_f, _, _ = run_system_evaluation(
            acceptance_path=acceptance_path,
            provider=provider,
            model_name=model,
            openai_api_key=OPENAI_API_KEY if provider == "openai" else None,
            few_shot_examples=shared_few_shot,
            test_set=shared_test_set,
        )
        result.system_zero_acc = sys_z.accuracy
        result.system_few_acc  = sys_f.accuracy
        result.system_zero_cm  = sys_z.confusion_matrix
        result.system_few_cm   = sys_f.confusion_matrix

        # ── Text Baseline 评估 ────────────────────────────────────
        random.seed(seed)
        txt_z, txt_f, _, _ = run_text_baseline_evaluation(
            model_name=model,
            acceptance_path=acceptance_path,
            provider=provider,
            openai_api_key=OPENAI_API_KEY if provider == "openai" else None,
            few_shot_examples=shared_few_shot,
            test_set=shared_test_set,
        )
        result.text_zero_acc = txt_z.accuracy
        result.text_few_acc  = txt_f.accuracy
        result.text_zero_cm  = txt_z.confusion_matrix
        result.text_few_cm   = txt_f.confusion_matrix

        # ── Vision Baseline 评估 ──────────────────────────────────
        random.seed(seed)
        bl_z, bl_f, _, _ = run_baseline_evaluation(
            model_name=model,
            acceptance_path=acceptance_path,
            provider=provider,
            openai_api_key=OPENAI_API_KEY if provider == "openai" else None,
            few_shot_examples=shared_few_shot,
            test_set=shared_test_set,
        )
        result.baseline_zero_acc = bl_z.accuracy
        result.baseline_few_acc  = bl_f.accuracy
        result.baseline_zero_cm  = bl_z.confusion_matrix
        result.baseline_few_cm   = bl_f.confusion_matrix

    except Exception as e:
        import traceback
        err_msg = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"\n  ✗ 评估失败: {err_msg}")
        traceback.print_exc()
        result.error = err_msg

    return result


def main():
    parser = argparse.ArgumentParser(description="4模型批量对比：System vs Baseline")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认42）")
    parser.add_argument("--acceptance-json", type=str, default="acceptance.json",
                        help="ground truth 文件路径")
    parser.add_argument("--models", type=str, default=None,
                        help="只运行部分模型标签（逗号分隔），例如: 'GPT-4o,Gemini 2.5 Flash'")
    parser.add_argument("--output", type=str, default="results/batch_model_comparison.json",
                        help="结果保存路径")
    args = parser.parse_args()

    print("=" * 120)
    print("批量模型对比实验（三层对比：System / Text Baseline / Vision Baseline）")
    print(f"  seed        = {args.seed}")
    print(f"  acceptance  = {args.acceptance_json}")
    print("=" * 120)

    # ── 配置 Gemini API ────────────────────────────────────────────────────────
    genai.configure(api_key=GEMINI_API_KEY)

    # ── 固定 seed，生成唯一共享 few-shot / test 集 ─────────────────────────────
    random.seed(args.seed)
    shared_evaluator = FewShotEvaluator(args.acceptance_json)
    shared_evaluator.load_and_convert_data()
    shared_few_shot, shared_test_set = shared_evaluator.select_few_shot_examples(k=5)

    print(f"\n共享 few-shot 数: {len(shared_few_shot)}  |  测试集大小: {len(shared_test_set)}")
    print("Few-shot 样本:")
    for s in shared_few_shot:
        print(f"  [{s.label}] student={s.student_id}  {s.school} / {s.program}")
    print("测试集样本 (前5):")
    for s in shared_test_set[:5]:
        print(f"  [{s.label}] student={s.student_id}  {s.school} / {s.program}")

    # ── 过滤模型（可选）──────────────────────────────────────────────────────
    configs = MODEL_CONFIGS
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        configs = [c for c in MODEL_CONFIGS if c["label"] in wanted]
        if not configs:
            print(f"⚠ 找不到指定模型，可选: {[c['label'] for c in MODEL_CONFIGS]}")
            return

    # ── 逐模型运行 ─────────────────────────────────────────────────────────────
    all_results: List[ModelResult] = []
    for cfg in configs:
        r = run_one_model(cfg, shared_few_shot, shared_test_set, args.acceptance_json, args.seed)
        all_results.append(r)

    # ── 汇总输出 ───────────────────────────────────────────────────────────────
    print_summary_table(all_results)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    save_all_results(all_results, args.output)


if __name__ == "__main__":
    main()
