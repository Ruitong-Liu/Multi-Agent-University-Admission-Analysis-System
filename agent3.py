"""
Agent 3 - Matching Model (LLM Prompting)
使用 Gemini / OpenAI API 进行成绩单与入学要求的匹配评估，
并实现基于 Agent2 + Web 搜索的混合入学要求获取与严格录取决策。
"""

import os

import json
from typing import Dict, List, Any, Optional, Protocol
from abc import abstractmethod


class LLMProvider(Protocol):
    """LLM API 提供者抽象接口"""

    @abstractmethod
    def generate(self, prompt: str, model: Optional[str] = None) -> str:
        """
        生成文本响应

        Args:
            prompt: 提示词
            model: 模型名称（可选）

        Returns:
            LLM 生成的文本响应（字符串）
        """
        raise NotImplementedError


class GeminiProvider:
    """Google Gemini API 提供者（推荐：gemini-1.5-pro / gemini-1.5-flash）"""

    def __init__(self, api_key: str):
        """
        初始化 Gemini 提供者

        Args:
            api_key: Gemini API 密钥
        """
        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=api_key)
            self.genai = genai
        except ImportError:
            raise ImportError("请安装 google-generativeai: pip install google-generativeai")

    def generate(self, prompt: str, model: str = "gemini-1.5-flash") -> str:
        """
        使用 Gemini API 生成响应

        Args:
            prompt: 提示词
            model: 模型名称，默认为 gemini-1.5-flash（安全稳定）

        Returns:
            LLM 生成的文本响应
        """
        model_instance = self.genai.GenerativeModel(model)
        response = model_instance.generate_content(prompt)
        return response.text or ""


class OpenAIProvider:
    """OpenAI API 提供者（可选，用于对比）"""

    def __init__(self, api_key: str):
        """
        初始化 OpenAI 提供者

        Args:
            api_key: OpenAI API 密钥
        """
        try:
            from openai import OpenAI  # type: ignore

            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

    def generate(self, prompt: str, model: str = "gpt-5.2-2025-12-11") -> str:
        """
        使用 OpenAI API 生成响应（JSON 对象）

        Args:
            prompt: 提示词
            model: 模型名称，默认为 gpt-5.2-2025-12-11

        Returns:
            LLM 生成的文本响应
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in evaluating student applications for university admission.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""


class MatchingModel:
    """匹配模型 - 评估成绩单与入学要求的匹配度"""

    def __init__(
        self,
        provider: LLMProvider,
        model_name: str = "gemini-1.5-flash",
        use_compact_prompt: bool = False,
    ):
        """
        初始化匹配模型

        Args:
            provider: LLM API 提供者（Gemini 或 OpenAI）
            model_name: 使用的模型名称，推荐 gemini-1.5-pro 或 gemini-1.5-flash
            use_compact_prompt: 是否使用紧凑型prompt（减少token使用）
        """
        self.provider = provider
        self.model_name = model_name
        self.use_compact_prompt = use_compact_prompt
        # 内置 few-shot 示例，作为默认演示用；评估管线会基于真实数据构造自己的 few-shot
        self.few_shot_examples = self._load_few_shot_examples()

    def _load_few_shot_examples(self) -> List[Dict[str, Any]]:
        """
        加载内置 few-shot 示例（包含完整 Transcript / Requirements / 输出）

        Returns:
            few-shot 示例列表
        """
        return [
            {
                "transcript": {
                    "student_information": {
                        "name": "张明",
                        "department": "计算机科学与技术学院",
                        "major": "计算机科学与技术"
                    },
                    "transcript_totals": {
                        "earned_credits": 180,
                        "gpa": 3.8
                    },
                    "courses": [
                        {"course_title": "数据结构与算法", "credit": 6},
                        {"course_title": "操作系统", "credit": 6},
                        {"course_title": "计算机网络", "credit": 4},
                        {"course_title": "数据库系统", "credit": 4},
                        {"course_title": "软件工程", "credit": 4}
                    ]
                },
                "requirement": {
                    "University": "Technical University of Munich",
                    "Program": "Computer Science",
                    "Degree": "Master of Science",
                    "Teaching language": "English",
                    "Application deadline": "Winter semester: 31 May",
                    "requirements": {
                        "Description/content": "Bachelor's degree in Computer Science or related field. Strong background in algorithms, data structures, and software engineering. Minimum GPA of 3.5/4.0."
                    }
                },
                "output": {
                    "admission_decision": "Admit",
                    "matching_score": 92,
                    "reasoning": "该学生具有计算机科学本科学位，GPA 3.8 超过最低要求 3.5。课程背景与项目要求高度匹配，包括数据结构与算法、操作系统、计算机网络等核心课程。学分充足（180 ECTS），完全符合入学要求。"
                }
            },
            {
                "transcript": {
                    "student_information": {
                        "name": "李华",
                        "department": "数学学院",
                        "major": "数学与应用数学"
                    },
                    "transcript_totals": {
                        "earned_credits": 165,
                        "gpa": 3.2
                    },
                    "courses": [
                        {"course_title": "高等数学", "credit": 8},
                        {"course_title": "线性代数", "credit": 6},
                        {"course_title": "概率论与数理统计", "credit": 6},
                        {"course_title": "数学分析", "credit": 8}
                    ]
                },
                "requirement": {
                    "University": "Technical University of Munich",
                    "Program": "Computer Science",
                    "Degree": "Master of Science",
                    "Teaching language": "English",
                    "Application deadline": "Winter semester: 31 May",
                    "requirements": {
                        "Description/content": "Bachelor's degree in Computer Science or related field. Strong background in algorithms, data structures, and software engineering. Minimum GPA of 3.5/4.0."
                    }
                },
                "output": {
                    "admission_decision": "Reject",
                    "matching_score": 45,
                    "reasoning": "该学生虽然具有数学背景，但专业不匹配。缺少计算机科学核心课程（数据结构、算法、软件工程等）。GPA 3.2 低于最低要求 3.5。数学背景虽然有用，但不足以弥补专业不匹配的问题。"
                }
            },
            {
                "transcript": {
                    "student_information": {
                        "name": "王芳",
                        "department": "英文学院",
                        "major": "英语"
                    },
                    "transcript_totals": {
                        "earned_credits": 150,
                        "gpa": 3.9
                    },
                    "courses": [
                        {"course_title": "高级英语", "credit": 6},
                        {"course_title": "英语文学", "credit": 6},
                        {"course_title": "翻译理论与实践", "credit": 4},
                        {"course_title": "跨文化交流", "credit": 3}
                    ]
                },
                "requirement": {
                    "University": "Heidelberg University",
                    "Program": "English Literature",
                    "Degree": "Master of Arts",
                    "Teaching language": "English",
                    "Application deadline": "Winter semester: 15 July",
                    "requirements": {
                        "Description/content": "Bachelor's degree in English, Literature, or related field. Strong academic record. Proficiency in English required."
                    }
                },
                "output": {
                    "admission_decision": "Admit",
                    "matching_score": 88,
                    "reasoning": "该学生具有英语专业本科学位，GPA 3.9 非常优秀。课程背景与英语文学项目高度匹配，包括高级英语、英语文学、翻译理论等核心课程。英语能力毋庸置疑。学分充足，完全符合入学要求。"
                }
            },
            {
                "transcript": {
                    "student_information": {
                        "name": "刘强",
                        "department": "物理学院",
                        "major": "物理学"
                    },
                    "transcript_totals": {
                        "earned_credits": 175,
                        "gpa": 3.4
                    },
                    "courses": [
                        {"course_title": "理论物理", "credit": 8},
                        {"course_title": "量子力学", "credit": 6},
                        {"course_title": "电动力学", "credit": 6},
                        {"course_title": "统计物理", "credit": 6},
                        {"course_title": "计算物理", "credit": 4}
                    ]
                },
                "requirement": {
                    "University": "Heidelberg University",
                    "Program": "Physics",
                    "Degree": "Master of Science",
                    "Teaching language": "English",
                    "Application deadline": "Winter semester: 15 July",
                    "requirements": {
                        "Description/content": "Bachelor's degree in Physics or related field. Strong background in theoretical physics, quantum mechanics, and mathematical methods. Minimum GPA of 3.3/4.0."
                    }
                },
                "output": {
                    "admission_decision": "Admit",
                    "matching_score": 85,
                    "reasoning": "该学生具有物理学本科学位，GPA 3.4 超过最低要求 3.3。课程背景与项目要求高度匹配，包括理论物理、量子力学、电动力学、统计物理等核心课程。计算物理课程显示了良好的数学和计算能力。学分充足，符合入学要求。"
                }
            },
            {
                "transcript": {
                    "student_information": {
                        "name": "陈静",
                        "department": "信息科学与技术学院",
                        "major": "信息工程"
                    },
                    "transcript_totals": {
                        "earned_credits": 160,
                        "gpa": 3.1
                    },
                    "courses": [
                        {"course_title": "信号与系统", "credit": 6},
                        {"course_title": "数字信号处理", "credit": 4},
                        {"course_title": "通信原理", "credit": 4},
                        {"course_title": "电路分析", "credit": 6}
                    ]
                },
                "requirement": {
                    "University": "Technical University of Munich",
                    "Program": "Computer Science",
                    "Degree": "Master of Science",
                    "Teaching language": "English",
                    "Application deadline": "Winter semester: 31 May",
                    "requirements": {
                        "Description/content": "Bachelor's degree in Computer Science or related field. Strong background in algorithms, data structures, and software engineering. Minimum GPA of 3.5/4.0."
                    }
                },
                "output": {
                    "admission_decision": "Reject",
                    "matching_score": 38,
                    "reasoning": "该学生专业为信息工程，虽然相关但不够匹配。缺少计算机科学核心课程（数据结构、算法、软件工程等）。GPA 3.1 明显低于最低要求 3.5。课程更偏向信号处理和通信，而非计算机科学核心内容。"
                }
            },
            {
                "transcript": {
                    "student_information": {
                        "name": "赵磊",
                        "department": "计算机科学与技术学院",
                        "major": "软件工程"
                    },
                    "transcript_totals": {
                        "earned_credits": 185,
                        "gpa": 3.6
                    },
                    "courses": [
                        {"course_title": "软件工程", "credit": 6},
                        {"course_title": "数据结构与算法", "credit": 6},
                        {"course_title": "操作系统", "credit": 6},
                        {"course_title": "数据库系统", "credit": 4},
                        {"course_title": "计算机网络", "credit": 4},
                        {"course_title": "编译原理", "credit": 4}
                    ]
                },
                "requirement": {
                    "University": "Technical University of Munich",
                    "Program": "Computer Science",
                    "Degree": "Master of Science",
                    "Teaching language": "English",
                    "Application deadline": "Winter semester: 31 May",
                    "requirements": {
                        "Description/content": "Bachelor's degree in Computer Science or related field. Strong background in algorithms, data structures, and software engineering. Minimum GPA of 3.5/4.0."
                    }
                },
                "output": {
                    "admission_decision": "Admit",
                    "matching_score": 95,
                    "reasoning": "该学生具有软件工程本科学位，与计算机科学高度相关。GPA 3.6 超过最低要求 3.5。课程背景与项目要求完美匹配，包括所有核心课程：软件工程、数据结构与算法、操作系统、数据库系统、计算机网络等。学分充足（185 ECTS），完全符合入学要求。"
                }
            }
        ]

    def _build_prompt(
        self,
        transcript_json: Dict[str, Any],
        requirement_json: Dict[str, Any],
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        构建提示词，包含严格决策规则与 few-shot 示例。

        Args:
            transcript_json: 成绩单 JSON
            requirement_json: 入学要求 JSON（可来自 Agent2 或 Web Fallback）
            few_shot_examples: 可选的 few-shot 示例列表（每个元素包含
                               transcript / requirement / output 字段）。

        Returns:
            完整的提示词
        """
        if self.use_compact_prompt:
            return self._build_compact_prompt(transcript_json, requirement_json)

        examples = few_shot_examples if few_shot_examples is not None else self.few_shot_examples

        # 构建 few-shot 示例部分，使用统一模板
        few_shot_text = ""
        for i, example in enumerate(examples, 1):
            transcript = example.get("transcript", {})
            requirement = example.get("requirement", {})
            output = example.get("output", {})

            student_id = output.get("student_id", f"example-{i}")
            school = requirement.get("University") or requirement.get("university", "Unknown University")
            program = requirement.get("Program") or requirement.get("program", "Unknown Program")

            few_shot_text += f"""
### FEW-SHOT EXAMPLE {i}

Student ID: {student_id}
School: {school}
Program: {program}

Student Transcript (JSON):
{json.dumps(transcript, ensure_ascii=False, indent=2)}

Program Requirements (JSON):
{json.dumps(requirement, ensure_ascii=False, indent=2)}

Expected output:
{json.dumps(output, ensure_ascii=False, indent=2)}
"""

        # 构建完整提示词，加入严格决策规则
        prompt = f"""You are a strict admission officer for German Master's programmes.

You will receive:

- A student's transcript in JSON format.

- Program admission requirements in JSON format (when available).

IMPORTANT - CRITICAL DISTINCTION:

- The requirement JSON may be incomplete, generic, outdated, or noisy.

- You must treat it only as a hint, NOT as a perfectly reliable source.

- **CRITICAL**: There is a fundamental difference between:
  
  a) "Cannot fetch/obtain requirements" (信息缺失) - This means the requirement data is missing or unavailable.
     → In this case, you MUST base your decision on:
        - The student's transcript quality (GPA, major, core courses)
        - Your knowledge of typical admission criteria for similar programmes at German universities
        - General expectations for this type of programme
     → DO NOT automatically reject due to missing requirements.
     → If the student's academic profile is strong and matches typical expectations, you should consider "Admit".
  
  b) "Does not meet requirements" (条件不达标) - This means the student clearly lacks prerequisites or has insufficient qualifications.
     → In this case, you MUST output "Reject" with a clear explanation of what is missing.

- Your primary signals are:

  1) The student's transcript (GPA, major, core courses and their relevance),

  2) Your knowledge (and web search) about typical admission criteria for similar programmes

     at German universities (e.g. expected background, key prerequisite subjects, usual GPA range),

  3) The requirement JSON, but only when it contains concrete and reasonable conditions.

Decision rules:

- If, based on the transcript AND typical expectations for this kind of programme,

  the student clearly lacks core prerequisites or has a clearly too low GPA,

  you MUST output "Reject".

- If the student clearly matches the usual background and level for the programme

  (e.g. right major, strong relevant courses, reasonable GPA), you may output "Admit"

  even if the scraped requirements JSON is incomplete or missing.

- **If you cannot get any useful key requirements from the JSON:**

  → This is "信息缺失" (missing information), NOT "条件不达标" (does not meet requirements).
  
  → You MUST base your decision on:
     - The transcript quality
     - Typical admission standards for similar programmes
     - Your knowledge of German university admission practices
  
  → DO NOT automatically reject. If the student's profile is strong, consider "Admit".
  
  → Only reject if the transcript itself clearly shows insufficient qualifications.

- If after checking the transcript and typical criteria you are still unsure,

  you must prefer "Reject".

- matching_score < 50 always corresponds to "Reject".

- **Competition intensity note**: Some rejections may be due to high competition rather than academic mismatch.
  Focus on academic match quality, not competition factors. If academically qualified, use "Admit".

You will see few-shot examples showing how to:

- Down-weight noisy or generic requirements,

- Rely on the transcript and typical admission expectations,

- Use explicit reasoning to justify Admit vs Reject.

Follow the patterns from the few-shot examples below.

{few_shot_text}

### NEW APPLICATION TO EVALUATE

Student Transcript (JSON):
{json.dumps(transcript_json, ensure_ascii=False, indent=2)}

Program Requirements (JSON):
{json.dumps(requirement_json, ensure_ascii=False, indent=2)}

Respond STRICTLY in the following JSON format:
{{
  "admission_decision": "Admit" or "Reject",
  "matching_score": integer between 0 and 100,
  "reasoning": "detailed reasoning referencing actual GPA, prerequisite courses, language level, and any missing information"
}}"""

        return prompt

    def _build_compact_prompt(self, transcript_json: Dict[str, Any], requirement_json: Dict[str, Any]) -> str:
        """
        构建紧凑型提示词（减少token使用，但仍遵守严格决策规则）
        """
        # 构建 few-shot 示例部分，包含完整的transcript和requirements（简化版）
        few_shot_text = ""
        for i, example in enumerate(self.few_shot_examples, 1):
            transcript = example.get("transcript", {})
            requirement = example.get("requirement", {})
            output = example.get("output", {})

            # 提取关键信息（简化版）
            student_info = transcript.get("student_information", {})
            totals = transcript.get("transcript_totals", {})
            courses = transcript.get("courses", [])[:5]  # 只取前5门课程
            
            major = student_info.get("major", "")
            gpa = totals.get("gpa", 0)
            
            univ = requirement.get("University") or requirement.get("university", "")
            prog = requirement.get("Program") or requirement.get("program", "")
            req_content = requirement.get("requirements", {}).get("Description/content", "")
            if len(req_content) > 200:
                req_content = req_content[:200] + "..."
            
            label = output.get("admission_decision", "")
            score = output.get("matching_score", 0)
            reasoning = output.get("reasoning", "")

            few_shot_text += f"\n\nExample {i}:\n"
            few_shot_text += f"Transcript: Major={major}, GPA={gpa}, Courses={[c.get('course_title', '') for c in courses[:3]]}\n"
            few_shot_text += f"Requirement: {univ} - {prog}\n"
            few_shot_text += f"Key Requirements: {req_content[:150]}...\n"
            few_shot_text += f"Output: {{\"admission_decision\": \"{label}\", \"matching_score\": {score}, \"reasoning\": \"{reasoning}\"}}\n"

        # 使用紧凑但完整的 prompt 模板，加入严格规则
        prompt = f"""You are a strict admission officer.

CRITICAL DISTINCTION:
- "Cannot fetch requirements" (信息缺失) ≠ "Does not meet requirements" (条件不达标)
- If requirements are missing/unavailable:
  → Base decision on transcript quality + typical standards
  → DO NOT automatically reject
  → If student profile is strong, consider "Admit"
- Only reject if transcript clearly shows insufficient qualifications

Decision rules:
- Output "Reject" if student clearly lacks prerequisites or has too low GPA.
- Output "Admit" if student matches typical background/level, even if requirements JSON is incomplete.
- If requirements missing: use transcript + typical standards, do NOT auto-reject.
- matching_score < 50 always corresponds to "Reject".

Use the pattern from the few-shot examples below.

### FEW-SHOT EXAMPLES{few_shot_text}

### NEW SAMPLE

Transcript JSON:
{json.dumps(transcript_json, ensure_ascii=False, indent=2)}

Requirement JSON:
{json.dumps(requirement_json, ensure_ascii=False, indent=2)}

Return strictly in JSON format:
{{
  "admission_decision": "Admit" or "Reject",
  "matching_score": 0-100,
  "reasoning": "detailed reasoning referencing actual GPA, prerequisites and language level"
}}"""

        return prompt

    def evaluate(
        self,
        transcript_json: Dict[str, Any],
        requirement_json: Dict[str, Any],
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        评估成绩单与入学要求的匹配度
        
        Args:
            transcript_json: 成绩单 JSON
            requirement_json: 入学要求 JSON（已包含 Agent2 / Web fallback 结果）
            few_shot_examples: 可选 few-shot 示例（用于覆盖默认示例）

        Returns:
            评估结果字典，包含：
            - admission_decision: "Admit" | "Reject"
            - matching_score: int (0-100)
            - reasoning: string
        """
        # 构建提示词（集成严格决策规则与 fallback 说明）
        prompt = self._build_prompt(
            transcript_json=transcript_json,
            requirement_json=requirement_json,
            few_shot_examples=few_shot_examples,
        )

        # 调用 LLM
        try:
            response_text = self.provider.generate(prompt, model=self.model_name)

            # 尝试解析 JSON 响应
            # 如果响应包含 markdown 代码块，先提取 JSON 部分
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            # 解析 JSON
            result = json.loads(response_text)

            # 验证和标准化输出格式
            if "admission_decision" not in result:
                raise ValueError("响应中缺少 admission_decision 字段")
            if "matching_score" not in result:
                raise ValueError("响应中缺少 matching_score 字段")
            if "reasoning" not in result:
                raise ValueError("响应中缺少 reasoning 字段")

            # 确保 matching_score 是 0-100 的整数
            score = result["matching_score"]
            if isinstance(score, str):
                score = int(float(score))
            else:
                score = int(score)
            score = max(0, min(100, score))
            result["matching_score"] = score

            # 标准化决策并应用严格规则
            decision_raw = str(result["admission_decision"])
            decision = "Admit" if "admit" in decision_raw.lower() or "录取" in decision_raw else "Reject"

            # 严格规则：matching_score < 50 一律 Reject
            if score < 50:
                decision = "Reject"

            result["admission_decision"] = decision

            return result

        except json.JSONDecodeError as e:
            raise ValueError(f"无法解析 LLM 响应为 JSON: {str(e)}\n响应内容: {response_text}")
        except Exception as e:
            raise RuntimeError(f"评估过程中出错: {str(e)}")
    
    def save_result(self, result: Dict[str, Any], output_path: str):
        """
        保存评估结果到 JSON 文件
        
        Args:
            result: 评估结果
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


def main():
    """
    主函数 - 完整整合三个agent，并使用混合入学要求检索（Agent2 + Web Fallback）
    """
    import os
    import argparse

    # 延迟导入以避免在仅使用 MatchingModel 时强依赖 agent2
    try:
        from agent2 import AdmissionRequirementsLoader
    except Exception:
        AdmissionRequirementsLoader = None  # type: ignore

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Agent 3 - 成绩单与入学要求匹配评估")
    parser.add_argument(
        "--transcript",
        type=str,
        default="Student_Transcripts/transcript01.json",
        help="成绩单JSON文件路径（agent1的输出）",
    )
    parser.add_argument(
        "--university",
        type=str,
        default="Technical University of Munich",
        help="目标大学名称",
    )
    parser.add_argument(
        "--program",
        type=str,
        default="Computer Science",
        help="目标项目名称",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="university_programs.csv",
        help="CSV数据库路径（agent2使用）",
    )
    parser.add_argument(
        "--use-daad",
        action="store_true",
        help="是否使用DAAD爬虫（agent2）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-flash",
        help="LLM模型名称（推荐 gemini-1.5-flash 或 gemini-1.5-pro）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="matching_result.json",
        help="输出结果文件路径",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="如果agent2加载失败或数据不完整，启用Web搜索回退（Agent3）",
    )

    args = parser.parse_args()

    # Gemini API 密钥（实际使用中建议从环境变量读取）
    gemini_api_key = os.environ.get("GOOGLE_API_KEY", "")

    # 初始化 Gemini 提供者
    provider = GeminiProvider(gemini_api_key)

    # 初始化匹配模型
    matcher = MatchingModel(provider, model_name=args.model)

    print("=" * 60)
    print("Agent 3 - 成绩单与入学要求匹配评估")
    print("=" * 60)
    print(f"使用模型: {args.model}")
    print(f"成绩单文件: {args.transcript}")
    print(f"目标项目: {args.university} - {args.program}")
    print("=" * 60)

    # 步骤1: 使用agent1加载成绩单JSON
    print("\n[步骤1] 加载成绩单（Agent 1输出）...")
    if not os.path.exists(args.transcript):
        print(f"✗ 错误: 成绩单文件不存在: {args.transcript}")
        print("提示: 请先运行 agent1.py 生成成绩单JSON文件")
        return

    try:
        with open(args.transcript, "r", encoding="utf-8") as f:
            transcript_json = json.load(f)
        student_name = transcript_json.get("student_information", {}).get("name", "Unknown")
        print(f"✓ 成功加载成绩单: {student_name}")
    except Exception as e:
        print(f"✗ 加载成绩单失败: {str(e)}")
        return

    # 步骤2: 使用 Agent2 + Web Fallback 加载 / 补全入学要求
    print("\n[步骤2] 加载入学要求（Agent2 CSV/DAAD + Agent3 Web Fallback）...")
    requirement_json: Optional[Dict[str, Any]] = None
    agent2_raw: Optional[Dict[str, Any]] = None

    # 2.1 先尝试使用 Agent2
    if AdmissionRequirementsLoader is not None:
        try:
            loader = AdmissionRequirementsLoader(
                csv_path=args.csv,
                use_daad=args.use_daad,
                headless=True,
            )
            try:
                agent2_raw = loader.load_requirements(args.university, args.program)
            finally:
                loader.close()
        except Exception as e:
            print(f"⚠ 警告: Agent2 加载失败: {str(e)}")
            agent2_raw = None
    else:
        print("⚠ 警告: 无法导入 agent2，仅使用 Web Fallback")

    def _is_agent2_complete(data: Optional[Dict[str, Any]]) -> bool:
        """简单启发式：Agent2 是否提供了可用的入学要求文本"""
        if not data:
            return False
        content = (
            data.get("requirements", {})
            .get("Description/content", "")
            .strip()
        )
        if not content or content == "无法找到":
            return False
        # 要求文本足够长，且包含常见关键词时认为“相对完整”
        lower = content.lower()
        has_deadline = "deadline" in lower or "application" in lower
        has_lang = "ielts" in lower or "toefl" in lower or "language" in lower
        has_gpa = "gpa" in lower or "grade" in lower
        return has_deadline and has_lang and has_gpa

    # 2.2 使用 LLM 进行 Web 搜索 + 结构化要求抽取
    def _web_search_requirements(
        base_university: str,
        base_program: str,
        agent2_data: Optional[Dict[str, Any]],
        mode: str,
    ) -> Optional[Dict[str, Any]]:
        """
        使用 LLM 执行 Web 搜索回退 / 丰富入学要求。

        mode:
            - "verify_enrich": Agent2 已有较完整数据，要求LLM核验并补全缺失字段
            - "full_search": Agent2 数据为空或明显不完整，完全依赖 Web 搜索
        """
        try:
            prompt = f"""You are an admission requirements retrieval agent.

Your task:
- Find the OFFICIAL webpage of the programme
  "{base_program}" at "{base_university}".
- From the official source, extract and normalize the admission requirements.

You MUST:
- Prefer official university or DAAD pages.
- Extract at least:
  - GPA threshold (or equivalent grade requirement)
  - Required background / prerequisite degree
  - Key prerequisite courses or knowledge areas
  - Language requirement (e.g. IELTS/TOEFL, CEFR level)
  - Application deadlines

Agent2 data (may be incomplete, you MUST verify with web search):
{json.dumps(agent2_data or {}, ensure_ascii=False, indent=2)}

Mode: {mode}

Return STRICTLY a JSON object with the following schema:
{{
  "university": "{base_university}",
  "program": "{base_program}",
  "degree": "string or null",
  "teaching_language": "string or null",
  "gpa_threshold": {{
    "value": float or null,
    "scale": float or null,
    "description": "string explanation"
  }},
  "prerequisite_background": "string or null",
  "prerequisite_courses": ["course or topic 1", "course or topic 2", ...],
  "language_requirement": {{
    "language": "e.g. English",
    "level": "e.g. IELTS 6.5, TOEFL 90, CEFR C1",
    "description": "string explanation"
  }},
  "application_deadlines": ["deadline 1", "deadline 2", ...],
  "other_requirements": "string or null",
  "raw_requirements_text": "full text you used to derive the fields",
  "source": "agent2_only" | "web_only" | "agent2_plus_web"
}}"""
            raw = provider.generate(prompt, model=args.model)
            # 防御性处理：去掉代码块
            if "```json" in raw:
                s = raw.find("```json") + 7
                e = raw.find("```", s)
                raw = raw[s:e].strip()
            elif "```" in raw:
                s = raw.find("```") + 3
                e = raw.find("```", s)
                raw = raw[s:e].strip()
            data = json.loads(raw)
            return data
        except Exception as e:
            print(f"⚠ Web Fallback 解析失败: {str(e)}")
            return None

    # 2.3 组合 Agent2 + Web Fallback
    if agent2_raw is not None and _is_agent2_complete(agent2_raw):
        print("✓ Agent2 提供了相对完整的入学要求，将以其为主，并使用 Web 验证/补全缺失字段。")
        if args.fallback:
            enriched = _web_search_requirements(
                base_university=args.university,
                base_program=args.program,
                agent2_data=agent2_raw,
                mode="verify_enrich",
            )
            requirement_json = enriched or agent2_raw
        else:
            requirement_json = agent2_raw
    else:
        print("⚠ Agent2 入学要求为空或不完整。")
        if args.fallback:
            print("→ 启动 Agent3 Web Fallback 进行完整检索。")
            requirement_json = _web_search_requirements(
                base_university=args.university,
                base_program=args.program,
                agent2_data=agent2_raw,
                mode="full_search",
            )
        else:
            requirement_json = agent2_raw

    if requirement_json is None:
        print("✗ 错误: 无法通过 Agent2 或 Web Fallback 获取入学要求。")
        print("   提示: 请检查 CSV / 网络 / API 配置，或启用 --fallback 选项。")
        return

    # 步骤3: 使用Agent3进行匹配评估
    print("\n[步骤3] 进行匹配评估（Agent 3 - LLM）...")
    print(f"正在评估: {student_name} 的申请...")
    print(f"目标项目: {requirement_json.get('university') or requirement_json.get('University')} - "
          f"{requirement_json.get('program') or requirement_json.get('Program')}\n")

    try:
        result = matcher.evaluate(transcript_json, requirement_json)

        # 打印结果
        print("=" * 60)
        print("评估结果:")
        print("=" * 60)
        print(f"录取决定: {result['admission_decision']}")
        print(f"匹配分数: {result['matching_score']}/100")
        print(f"\n评估理由:\n{result['reasoning']}")

        # 保存结果
        matcher.save_result(result, args.output)
        print(f"\n✓ 结果已保存到: {args.output}")

        return result

    except Exception as e:
        print(f"✗ 评估失败: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()

