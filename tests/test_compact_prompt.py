"""
测试紧凑型 few-shot prompt（不调用真实 LLM，只构建并打印 prompt 信息）
"""

import json
from typing import Any

from agent3 import MatchingModel, LLMProvider
from experiments.few_shot_evaluation import load_transcript_json, load_requirements_json
from agent3 import GeminiProvider  # 只是为了类型完整性，测试中不会真正调用到远程 API


class DummyProvider:
    """假的 LLM Provider，只为满足 MatchingModel 的接口，不实际调用 API。"""

    def generate(self, prompt: str, model: str | None = None) -> str:
        # 打印 prompt 长度，返回一个固定的合法 JSON
        print(f"\n[DummyProvider] Prompt length: {len(prompt)} characters")
        # 简单返回一个固定结果，方便后续调用不报错
        return json.dumps(
            {
                "admission_decision": "Admit",
                "matching_score": 80,
                "reasoning": "Dummy provider返回的固定结果，仅用于测试流程。"
            },
            ensure_ascii=False
        )


def main():
    # 选择一个简单的样本：学生01 + Tübingen machine learning
    student_id = "01"
    school = "Universität Tübingen"
    program = "machine learning"

    # 加载成绩单和项目要求
    transcript = load_transcript_json(student_id)
    if not transcript:
        print(f"✗ 成绩单JSON不存在: transcript{student_id}.json")
        return

    requirements = load_requirements_json(school, program, use_daad=False)
    if not requirements:
        print(f"✗ 无法加载项目要求: {school} - {program}")
        return

    # 使用 DummyProvider + 紧凑型 prompt
    provider = DummyProvider()
    matcher = MatchingModel(provider, model_name="gemini-3-pro-preview", use_compact_prompt=True)

    # 直接构建紧凑型 prompt（不会真正发请求，因为 DummyProvider 截获）
    prompt = matcher._build_prompt(transcript, requirements)

    print("\n========== COMPACT PROMPT PREVIEW (first 1000 chars) ==========")
    print(prompt[:1000])
    if len(prompt) > 1000:
        print("\n... [truncated] ...")

    print(f"\nTotal prompt length: {len(prompt)} characters")


if __name__ == "__main__":
    main()






