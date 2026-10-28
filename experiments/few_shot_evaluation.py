"""
Few-shot Learning Evaluation System
用于评估LLM在录取预测任务上的zero-shot和few-shot表现
"""

import json
import os
import random
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import re


@dataclass
class ApplicationSample:
    """申请样本数据结构"""
    student_id: str
    school: str
    program: str
    label: str  # "Admit" or "Reject"


@dataclass
class EvaluationMetrics:
    """评估指标"""
    accuracy: float
    confusion_matrix: Dict[str, int]  # TP, FP, TN, FN
    errors: List[Dict[str, Any]]  # 预测错误的样本列表


class FewShotEvaluator:
    """Few-shot学习评估器"""
    
    def __init__(self, acceptance_json_path: str = "acceptance.json"):
        """
        初始化评估器
        
        Args:
            acceptance_json_path: acceptance.json文件路径
        """
        self.acceptance_json_path = acceptance_json_path
        self.all_samples: List[ApplicationSample] = []
        self.few_shot_examples: List[ApplicationSample] = []
        self.test_set: List[ApplicationSample] = []
        
    def load_and_convert_data(self) -> List[ApplicationSample]:
        """
        从acceptance.json加载数据并转换为扁平化的样本列表
        
        Returns:
            扁平化的申请样本列表
        """
        if not os.path.exists(self.acceptance_json_path):
            raise FileNotFoundError(f"文件不存在: {self.acceptance_json_path}")
        
        with open(self.acceptance_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        
        for student_id, outcomes in data.items():
            # 处理接受的申请
            for app in outcomes.get("accepted", []):
                sample = ApplicationSample(
                    student_id=student_id,
                    school=app["school"],
                    program=app["major"],
                    label="Admit"
                )
                samples.append(sample)
            
            # 处理拒绝的申请
            for app in outcomes.get("rejected", []):
                sample = ApplicationSample(
                    student_id=student_id,
                    school=app["school"],
                    program=app["major"],
                    label="Reject"
                )
                samples.append(sample)
        
        self.all_samples = samples
        print(f"✓ 成功加载 {len(samples)} 个申请样本")
        print(f"  - Admit: {sum(1 for s in samples if s.label == 'Admit')}")
        print(f"  - Reject: {sum(1 for s in samples if s.label == 'Reject')}")
        
        return samples
    
    def find_borderline_students(self, samples: List[ApplicationSample]) -> List[str]:
        """
        找到边界案例学生（既有接受又有拒绝的学生）
        
        Args:
            samples: 所有样本列表
            
        Returns:
            边界案例学生ID列表
        """
        student_outcomes = defaultdict(lambda: {"Admit": [], "Reject": []})
        
        for sample in samples:
            student_outcomes[sample.student_id][sample.label].append(sample)
        
        borderline_students = []
        for student_id, outcomes in student_outcomes.items():
            if len(outcomes["Admit"]) > 0 and len(outcomes["Reject"]) > 0:
                borderline_students.append(student_id)
        
        return borderline_students
    
    def select_few_shot_examples(self, k: int = 5) -> Tuple[List[ApplicationSample], List[ApplicationSample]]:
        """
        选择few-shot示例
        
        规则:
        1. 必须包含至少1个Admit和1个Reject
        2. 至少包含1个边界案例（学生既有接受又有拒绝）
        3. 尽量从1-2个学生中选择，以便测试集包含更多未见过的学生组合
        4. Few-shot样本必须从测试集中排除
        
        Args:
            k: 选择的few-shot示例数量，默认5
            
        Returns:
            (few_shot_examples, test_set) 元组
        """
        if not self.all_samples:
            self.load_and_convert_data()
        
        # 找到边界案例学生
        borderline_students = self.find_borderline_students(self.all_samples)
        
        if not borderline_students:
            print("⚠ 警告: 未找到边界案例学生，将使用随机选择")
            borderline_students = list(set(s.student_id for s in self.all_samples))
        
        # 优先选择边界案例学生
        selected_student_ids = []
        if len(borderline_students) >= 1:
            selected_student_ids.append(borderline_students[0])
        if len(borderline_students) >= 2 and k > 3:
            selected_student_ids.append(borderline_students[1])
        
        # 如果还需要更多学生，从其他学生中选择
        all_student_ids = list(set(s.student_id for s in self.all_samples))
        remaining_students = [sid for sid in all_student_ids if sid not in selected_student_ids]
        
        if len(selected_student_ids) < 2 and len(remaining_students) > 0:
            selected_student_ids.append(remaining_students[0])
        
        # 从选定的学生中收集样本
        candidate_samples = [s for s in self.all_samples if s.student_id in selected_student_ids]
        
        # 确保包含Admit和Reject
        admit_samples = [s for s in candidate_samples if s.label == "Admit"]
        reject_samples = [s for s in candidate_samples if s.label == "Reject"]
        
        few_shot_examples = []
        
        # 至少选择1个Admit
        if admit_samples:
            few_shot_examples.append(random.choice(admit_samples))
        
        # 至少选择1个Reject
        if reject_samples:
            few_shot_examples.append(random.choice(reject_samples))
        
        # 确保包含边界案例（从边界学生中选择一个既有Admit又有Reject的组合）
        if borderline_students:
            borderline_student_id = borderline_students[0]
            borderline_admits = [s for s in admit_samples if s.student_id == borderline_student_id]
            borderline_rejects = [s for s in reject_samples if s.student_id == borderline_student_id]
            
            if borderline_admits and borderline_rejects and len(few_shot_examples) < k:
                # 添加一个边界案例的Admit
                if not any(s.student_id == borderline_student_id and s.label == "Admit" 
                          for s in few_shot_examples):
                    few_shot_examples.append(random.choice(borderline_admits))
                # 添加一个边界案例的Reject
                if not any(s.student_id == borderline_student_id and s.label == "Reject" 
                          for s in few_shot_examples):
                    few_shot_examples.append(random.choice(borderline_rejects))
        
        # 填充到k个样本
        remaining_candidates = [s for s in candidate_samples 
                               if s not in few_shot_examples]
        
        while len(few_shot_examples) < k and remaining_candidates:
            few_shot_examples.append(random.choice(remaining_candidates))
            remaining_candidates = [s for s in remaining_candidates 
                                  if s not in few_shot_examples]
        
        # 如果还不够，从所有样本中随机选择
        if len(few_shot_examples) < k:
            all_remaining = [s for s in self.all_samples if s not in few_shot_examples]
            while len(few_shot_examples) < k and all_remaining:
                few_shot_examples.append(random.choice(all_remaining))
                all_remaining = [s for s in all_remaining if s not in few_shot_examples]
        
        # 确保有Admit和Reject，并尽量平衡
        has_admit = any(s.label == "Admit" for s in few_shot_examples)
        has_reject = any(s.label == "Reject" for s in few_shot_examples)
        
        if not has_admit:
            # 强制添加一个Admit
            all_admits = [s for s in self.all_samples if s.label == "Admit" and s not in few_shot_examples]
            if all_admits:
                few_shot_examples[-1] = random.choice(all_admits)
        
        if not has_reject:
            # 强制添加一个Reject
            all_rejects = [s for s in self.all_samples if s.label == "Reject" and s not in few_shot_examples]
            if all_rejects:
                few_shot_examples[-1] = random.choice(all_rejects)
        
        # 平衡Admit/Reject比例（确保差异不超过1）
        admit_count = sum(1 for s in few_shot_examples if s.label == "Admit")
        reject_count = sum(1 for s in few_shot_examples if s.label == "Reject")
        
        # 如果差异太大，进行调整
        if admit_count > reject_count + 1:
            # 移除多余的Admit样本
            admit_samples = [s for s in few_shot_examples if s.label == "Admit"]
            while admit_count > reject_count + 1 and len(admit_samples) > 0:
                to_remove = random.choice(admit_samples)
                few_shot_examples.remove(to_remove)
                admit_samples.remove(to_remove)
                admit_count -= 1
        elif reject_count > admit_count + 1:
            # 移除多余的Reject样本
            reject_samples = [s for s in few_shot_examples if s.label == "Reject"]
            while reject_count > admit_count + 1 and len(reject_samples) > 0:
                to_remove = random.choice(reject_samples)
                few_shot_examples.remove(to_remove)
                reject_samples.remove(to_remove)
                reject_count -= 1
        
        # 如果移除后数量不足，补充
        while len(few_shot_examples) < k:
            remaining = [s for s in self.all_samples if s not in few_shot_examples]
            if not remaining:
                break
            
            # 优先补充较少的类别
            if admit_count < reject_count:
                admits = [s for s in remaining if s.label == "Admit"]
                if admits:
                    few_shot_examples.append(random.choice(admits))
                    admit_count += 1
                else:
                    few_shot_examples.append(random.choice(remaining))
            elif reject_count < admit_count:
                rejects = [s for s in remaining if s.label == "Reject"]
                if rejects:
                    few_shot_examples.append(random.choice(rejects))
                    reject_count += 1
                else:
                    few_shot_examples.append(random.choice(remaining))
            else:
                few_shot_examples.append(random.choice(remaining))
        
        # 创建测试集（排除few-shot示例）
        test_set = [s for s in self.all_samples if s not in few_shot_examples]
        
        self.few_shot_examples = few_shot_examples
        self.test_set = test_set
        
        print(f"\n✓ Few-shot示例选择完成 (k={len(few_shot_examples)})")
        print(f"  选定的学生ID: {set(s.student_id for s in few_shot_examples)}")
        print(f"  Admit数量: {sum(1 for s in few_shot_examples if s.label == 'Admit')}")
        print(f"  Reject数量: {sum(1 for s in few_shot_examples if s.label == 'Reject')}")
        print(f"  测试集大小: {len(test_set)}")
        
        return few_shot_examples, test_set
    
    def format_few_shot_example(self, sample: ApplicationSample, 
                                transcript_json: Optional[Dict] = None,
                                requirements_json: Optional[Dict] = None,
                                matching_score: Optional[int] = None,
                                reasoning: Optional[str] = None,
                                index: Optional[int] = None) -> str:
        """
        格式化单个few-shot示例为提示词块，包含完整的transcript和requirements，
        严格遵守统一模板：
        ### FEW-SHOT EXAMPLE {i}
        """
        # 计算默认的matching_score和reasoning
        if matching_score is None:
            if sample.label == "Admit":
                matching_score = 85  # 默认Admit分数
            else:
                matching_score = 45  # 默认Reject分数
        
        if reasoning is None:
            if sample.label == "Admit":
                reasoning = "专业匹配，GPA达标，课程背景符合项目要求。"
            else:
                reasoning = "专业不匹配或GPA未达标，缺少关键课程，不符合入学要求。"
        
        idx_text = f" {index}" if index is not None else ""
        example_text = f"""### FEW-SHOT EXAMPLE{idx_text}

Student ID: {sample.student_id}
School: {sample.school}
Program: {sample.program}

Student Transcript (JSON):
{json.dumps(transcript_json, indent=2, ensure_ascii=False) if transcript_json else "N/A"}

Program Requirements (JSON):
{json.dumps(requirements_json, indent=2, ensure_ascii=False) if requirements_json else "N/A"}

Expected output:
{{
  "admission_decision": "{sample.label}",
  "matching_score": {matching_score},
  "reasoning": "{reasoning}"
}}"""
        return example_text
    
    def build_prompt(self, sample: ApplicationSample, 
                    few_shot_examples: Optional[List[ApplicationSample]] = None,
                    transcript_json: Optional[Dict] = None,
                    requirements_json: Optional[Dict] = None,
                    few_shot_data: Optional[List[Dict]] = None) -> str:
        """
        构建LLM提示词（Agent3 风格），包含严格决策规则与 few-shot 示例。
        """
        prompt = """You are a strict admission officer for German Master's programmes.

You will receive:

- A student's transcript in JSON format.

- Program admission requirements in JSON format (when available).

IMPORTANT:

- The requirement JSON may be incomplete, generic, outdated, or noisy.

- You must treat it only as a hint, NOT as a perfectly reliable source.

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

  even if the scraped requirements JSON is incomplete.

- If you can't get any useful key requirements from the JSON,

  you MUST use web search to infer typical admission criteria for this programme.

- If after checking the transcript and typical criteria you are still unsure,

  you must prefer "Reject".

- matching_score < 50 always corresponds to "Reject".

You will see few-shot examples showing how to:

- Down-weight noisy or generic requirements,

- Rely on the transcript and typical admission expectations,

- Use explicit reasoning to justify Admit vs Reject.

---
"""
        
        # 添加few-shot示例
        if few_shot_data:
            # 使用提供的完整few-shot数据
            prompt += "\n### FEW-SHOT EXAMPLES\n\n"
            for i, example_data in enumerate(few_shot_data, 1):
                example_sample = example_data.get("sample")
                example_transcript = example_data.get("transcript_json")
                example_requirements = example_data.get("requirements_json")
                example_score = example_data.get("matching_score")
                example_reasoning = example_data.get("reasoning")
                
                if example_sample and example_transcript and example_requirements:
                    prompt += self.format_few_shot_example(
                        example_sample,
                        transcript_json=example_transcript,
                        requirements_json=example_requirements,
                        matching_score=example_score,
                        reasoning=example_reasoning,
                        index=i,
                    ) + "\n\n"
        elif few_shot_examples:
            # 如果没有提供完整数据，只使用元数据（向后兼容）
            prompt += "\n### FEW-SHOT EXAMPLES\n\n"
            for i, example in enumerate(few_shot_examples, 1):
                prompt += self.format_few_shot_example(example, index=i) + "\n\n"
        
        # 添加当前样本
        prompt += f"""### NEW SAMPLE

Student ID: {sample.student_id}
School: {sample.school}
Program: {sample.program}
"""
        
        # 如果提供了transcript和requirements，添加它们
        if transcript_json:
            prompt += f"\nStudent Transcript (JSON):\n{json.dumps(transcript_json, indent=2, ensure_ascii=False)}\n"
        else:
            prompt += "\nStudent Transcript (JSON): Not provided\n"
        
        if requirements_json:
            prompt += f"\nProgram Requirements (JSON):\n{json.dumps(requirements_json, indent=2, ensure_ascii=False)}\n"
        else:
            prompt += "\nProgram Requirements (JSON): Not provided\n"
        
        prompt += """
Respond STRICTLY in the following JSON format:

{
  "admission_decision": "Admit" | "Reject",
  "matching_score": number,
  "reasoning": "string"
}
"""
        
        return prompt
    
    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM响应，提取JSON
        
        Args:
            response: LLM的原始响应
            
        Returns:
            解析后的字典，包含admission_decision, matching_score, reasoning
        """
        # 尝试提取JSON
        json_match = re.search(r'\{[^{}]*"admission_decision"[^{}]*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                pass
        
        # 如果正则匹配失败，尝试直接解析整个响应
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            pass
        
        # 如果都失败，尝试提取关键信息
        decision_match = re.search(r'"admission_decision"\s*:\s*"([^"]+)"', response, re.IGNORECASE)
        score_match = re.search(r'"matching_score"\s*:\s*(\d+)', response, re.IGNORECASE)
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', response, re.IGNORECASE)
        
        result = {}
        if decision_match:
            decision = decision_match.group(1)
            result["admission_decision"] = "Admit" if "admit" in decision.lower() else "Reject"
        if score_match:
            result["matching_score"] = int(score_match.group(1))
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1)
        
        return result
    
    def _load_sample_data(self, sample: ApplicationSample, transcript_loader, requirements_loader):
        """
        加载样本的transcript和requirements数据
        
        Returns:
            (transcript_json, requirements_json) 元组，如果加载失败则返回(None, None)
        """
        """
        统一的样本加载逻辑：带日志与错误处理，不返回 None 进入 Agent3。
        """
        transcript_json = None
        requirements_json = None
        
        if transcript_loader:
            try:
                transcript_json = transcript_loader(sample.student_id)
                if not transcript_json:
                    print(f"    ⚠ 警告: 无法加载 transcript for student {sample.student_id}")
            except Exception as e:
                print(f"    ✗ 错误: 加载 transcript 失败: {str(e)}")
                transcript_json = None
        
        if requirements_loader:
            try:
                requirements_json = requirements_loader(sample.school, sample.program)
                if not requirements_json:
                    print(f"    ⚠ 警告: 无法加载 requirements for {sample.school} - {sample.program}")
            except Exception as e:
                print(f"    ✗ 错误: 加载 requirements 失败: {str(e)}")
                requirements_json = None
        
        return transcript_json, requirements_json
    
    def _generate_reasoning(self, sample: ApplicationSample, transcript_json: Dict, requirements_json: Dict) -> str:
        """
        基于实际数据生成详细的reasoning
        
        Args:
            sample: 申请样本
            transcript_json: 成绩单JSON
            requirements_json: 要求JSON
            
        Returns:
            详细的reasoning字符串
        """
        if not transcript_json or not requirements_json:
            # 如果数据不完整，返回简单reasoning
            if sample.label == "Admit":
                return "专业匹配，GPA达标，符合入学要求。"
            else:
                return "专业不匹配或GPA未达标，不符合入学要求。"
        
        # 提取关键信息
        student_info = transcript_json.get("student_information", {})
        totals = transcript_json.get("transcript_totals", {})
        courses = transcript_json.get("courses", [])
        
        major = student_info.get("major", "")
        gpa = totals.get("gpa", 0)
        total_credits = totals.get("earned_credits", 0)
        
        # 提取关键课程名称（前5门）
        key_courses = [c.get("course_title", "") for c in courses[:5]]
        
        req_content = requirements_json.get("requirements", {}).get("Description/content", "")
        
        # 生成详细reasoning
        if sample.label == "Admit":
            reasoning = f"该学生专业为{major}，GPA为{gpa}，总学分为{total_credits} ECTS。"
            if key_courses:
                reasoning += f"已修核心课程包括{', '.join(key_courses[:3])}等，"
            reasoning += "与项目要求高度匹配，符合入学条件。"
        else:
            reasoning = f"该学生专业为{major}，虽然相关但不够匹配。"
            if gpa < 3.0:
                reasoning += f"GPA为{gpa}可能未达到项目要求。"
            if not key_courses:
                reasoning += "缺少关键课程背景。"
            reasoning += "不符合入学要求。"
        
        return reasoning
    
    def evaluate(self, llm_provider, test_set: List[ApplicationSample],
                 few_shot_examples: Optional[List[ApplicationSample]] = None,
                 model_name: str = "gemini-2.5-flash",
                 transcript_loader=None,
                 requirements_loader=None) -> Tuple[EvaluationMetrics, EvaluationMetrics]:
        """
        评估模型在zero-shot和few-shot条件下的表现
        
        Args:
            llm_provider: LLM提供者（需要实现generate方法）
            test_set: 测试集
            few_shot_examples: Few-shot示例列表
            model_name: 模型名称（默认使用gemini-1.5-flash-latest）
            transcript_loader: 成绩单加载函数（可选）
            requirements_loader: 要求加载函数（可选）
            
        Returns:
            (zero_shot_metrics, few_shot_metrics) 元组
        """
        print(f"\n开始评估 (测试集大小: {len(test_set)})")
        
        # 为few-shot示例加载完整数据
        few_shot_data = None
        if few_shot_examples and transcript_loader and requirements_loader:
            print("\n[加载Few-shot示例数据]")
            few_shot_data = []
            for example in few_shot_examples:
                example_transcript, example_requirements = self._load_sample_data(
                    example, transcript_loader, requirements_loader
                )
                if example_transcript and example_requirements:
                    # 生成详细的reasoning
                    reasoning = self._generate_reasoning(example, example_transcript, example_requirements)
                    # 计算matching_score
                    if example.label == "Admit":
                        matching_score = random.randint(80, 95)
                    else:
                        matching_score = random.randint(30, 55)
                    
                    few_shot_data.append({
                        "sample": example,
                        "transcript_json": example_transcript,
                        "requirements_json": example_requirements,
                        "matching_score": matching_score,
                        "reasoning": reasoning
                    })
                else:
                    print(f"  ⚠ 跳过few-shot示例: {example.student_id} - {example.school} - {example.program} (数据不完整)")
            
            if few_shot_data:
                print(f"  ✓ 成功加载 {len(few_shot_data)}/{len(few_shot_examples)} 个few-shot示例")
            else:
                print("  ⚠ 警告: 所有few-shot示例数据加载失败，将使用元数据模式")
                few_shot_data = None
        
        # Zero-shot评估
        print("\n[Zero-shot评估]")
        zero_shot_predictions: List[Dict[str, Any]] = []
        zero_shot_errors: List[Dict[str, Any]] = []
        
        for i, sample in enumerate(test_set):
            print(f"  处理样本 {i+1}/{len(test_set)}: {sample.student_id} - {sample.school} - {sample.program}")
            
            # 加载transcript和requirements
            transcript_json, requirements_json = self._load_sample_data(
                sample, transcript_loader, requirements_loader
            )
            
            # 如果关键数据缺失，记录并跳过该样本（不进入评估与指标计算）
            if not transcript_json or not requirements_json:
                print(f"    ⚠ 跳过样本: 缺少必要数据")
                continue
            
            # 构建zero-shot提示词
            prompt = self.build_prompt(sample, few_shot_examples=None,
                                     transcript_json=transcript_json,
                                     requirements_json=requirements_json)
            
            # 调用LLM
            try:
                response = llm_provider.generate(prompt, model=model_name)
                prediction = self.parse_llm_response(response)
                
                # 严格规则：matching_score < 50 一律 Reject
                score = prediction.get("matching_score", 0)
                try:
                    score_val = int(float(score))
                except Exception:
                    score_val = 0

                predicted_label = prediction.get("admission_decision", "").strip()
                # 标准化标签
                if "admit" in predicted_label.lower():
                    predicted_label = "Admit"
                elif "reject" in predicted_label.lower():
                    predicted_label = "Reject"
                else:
                    predicted_label = "Unknown"

                # [ABLATION] 条件激活规则：仅当 Agent 2 返回了有效要求数据时才应用 score<50→Reject
                # 当要求数据为空占位时，score 无参照意义，直接信任模型原始 admission_decision
                _req = requirements_json or {}
                _req_content = _req.get("requirements", {}) if isinstance(_req.get("requirements"), dict) else {}
                _has_real_requirements = any([
                    _req_content.get("Description/content", "").strip(),
                    _req_content.get("courses_required", "").strip(),
                    _req_content.get("language_requirements", "").strip(),
                ])
                if _has_real_requirements and score_val < 50:
                    predicted_label = "Reject"

                zero_shot_predictions.append({
                    "sample": sample,
                    "prediction": predicted_label,
                    "actual": sample.label,
                    "matching_score": score_val,
                    "reasoning": prediction.get("reasoning", ""),
                    "raw_response": response
                })
                
                if predicted_label != sample.label:
                    zero_shot_errors.append({
                        "student_id": sample.student_id,
                        "school": sample.school,
                        "program": sample.program,
                        "actual": sample.label,
                        "predicted": predicted_label,
                        "matching_score": prediction.get("matching_score", 0),
                        "reasoning": prediction.get("reasoning", "")
                    })
                
            except Exception as e:
                print(f"    ✗ 错误: {str(e)}")
                zero_shot_predictions.append({
                    "sample": sample,
                    "prediction": "Error",
                    "actual": sample.label,
                    "matching_score": 0,
                    "reasoning": f"Error: {str(e)}",
                    "raw_response": ""
                })
        
        # Few-shot评估
        print("\n[Few-shot评估]")
        few_shot_predictions: List[Dict[str, Any]] = []
        few_shot_errors: List[Dict[str, Any]] = []
        
        for i, sample in enumerate(test_set):
            print(f"  处理样本 {i+1}/{len(test_set)}: {sample.student_id} - {sample.school} - {sample.program}")
            
            # 加载transcript和requirements
            transcript_json, requirements_json = self._load_sample_data(
                sample, transcript_loader, requirements_loader
            )
            
            # 如果关键数据缺失，记录并跳过该样本（不进入评估与指标计算）
            if not transcript_json or not requirements_json:
                print(f"    ⚠ 跳过样本: 缺少必要数据")
                continue
            
            # 构建few-shot提示词（使用完整数据）
            prompt = self.build_prompt(sample, few_shot_examples=few_shot_examples,
                                     transcript_json=transcript_json,
                                     requirements_json=requirements_json,
                                     few_shot_data=few_shot_data)
            
            # 调用LLM
            try:
                response = llm_provider.generate(prompt, model=model_name)
                prediction = self.parse_llm_response(response)
                
                # 严格规则：matching_score < 50 一律 Reject
                score = prediction.get("matching_score", 0)
                try:
                    score_val = int(float(score))
                except Exception:
                    score_val = 0

                predicted_label = prediction.get("admission_decision", "").strip()
                # 标准化标签
                if "admit" in predicted_label.lower():
                    predicted_label = "Admit"
                elif "reject" in predicted_label.lower():
                    predicted_label = "Reject"
                else:
                    predicted_label = "Unknown"

                # [ABLATION] 条件激活规则：仅当 Agent 2 返回了有效要求数据时才应用 score<50→Reject
                _req = requirements_json or {}
                _req_content = _req.get("requirements", {}) if isinstance(_req.get("requirements"), dict) else {}
                _has_real_requirements = any([
                    _req_content.get("Description/content", "").strip(),
                    _req_content.get("courses_required", "").strip(),
                    _req_content.get("language_requirements", "").strip(),
                ])
                if _has_real_requirements and score_val < 50:
                    predicted_label = "Reject"

                few_shot_predictions.append({
                    "sample": sample,
                    "prediction": predicted_label,
                    "actual": sample.label,
                    "matching_score": score_val,
                    "reasoning": prediction.get("reasoning", ""),
                    "raw_response": response
                })
                
                if predicted_label != sample.label:
                    few_shot_errors.append({
                        "student_id": sample.student_id,
                        "school": sample.school,
                        "program": sample.program,
                        "actual": sample.label,
                        "predicted": predicted_label,
                        "matching_score": prediction.get("matching_score", 0),
                        "reasoning": prediction.get("reasoning", "")
                    })
                
            except Exception as e:
                print(f"    ✗ 错误: {str(e)}")
                few_shot_predictions.append({
                    "sample": sample,
                    "prediction": "Error",
                    "actual": sample.label,
                    "matching_score": 0,
                    "reasoning": f"Error: {str(e)}",
                    "raw_response": ""
                })
        
        # 计算指标
        zero_shot_metrics = self._calculate_metrics(zero_shot_predictions, zero_shot_errors)
        few_shot_metrics = self._calculate_metrics(few_shot_predictions, few_shot_errors)
        
        return zero_shot_metrics, few_shot_metrics
    
    def _calculate_metrics(self, predictions: List[Dict], errors: List[Dict]) -> EvaluationMetrics:
        """
        计算评估指标
        
        Args:
            predictions: 预测结果列表
            errors: 错误列表
            
        Returns:
            评估指标对象
        """
        correct = 0
        total = len(predictions)
        
        tp = 0  # True Positive: 预测Admit，实际Admit
        fp = 0  # False Positive: 预测Admit，实际Reject
        tn = 0  # True Negative: 预测Reject，实际Reject
        fn = 0  # False Negative: 预测Reject，实际Admit
        
        for pred in predictions:
            actual = pred["actual"]
            predicted = pred["prediction"]
            
            if predicted == actual:
                correct += 1
                if actual == "Admit":
                    tp += 1
                else:
                    tn += 1
            else:
                if predicted == "Admit" and actual == "Reject":
                    fp += 1
                elif predicted == "Reject" and actual == "Admit":
                    fn += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return EvaluationMetrics(
            accuracy=accuracy,
            confusion_matrix={
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn
            },
            errors=errors
        )
    
    def save_evaluation_results(self, zero_shot_metrics: EvaluationMetrics,
                                few_shot_metrics: EvaluationMetrics,
                                output_path: str = "evaluation_results.json"):
        """
        保存评估结果到JSON文件
        
        Args:
            zero_shot_metrics: Zero-shot评估指标
            few_shot_metrics: Few-shot评估指标
            output_path: 输出文件路径
        """
        results = {
            "zero_shot": {
                "accuracy": zero_shot_metrics.accuracy,
                "confusion_matrix": zero_shot_metrics.confusion_matrix,
                "error_count": len(zero_shot_metrics.errors),
                "errors": zero_shot_metrics.errors
            },
            "few_shot": {
                "accuracy": few_shot_metrics.accuracy,
                "confusion_matrix": few_shot_metrics.confusion_matrix,
                "error_count": len(few_shot_metrics.errors),
                "errors": few_shot_metrics.errors
            },
            "few_shot_examples": [
                {
                    "student_id": s.student_id,
                    "school": s.school,
                    "program": s.program,
                    "label": s.label
                }
                for s in self.few_shot_examples
            ],
            "test_set_size": len(self.test_set)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 评估结果已保存到: {output_path}")
    
    def print_evaluation_summary(self, zero_shot_metrics: EvaluationMetrics,
                                 few_shot_metrics: EvaluationMetrics):
        """
        打印评估结果摘要
        
        Args:
            zero_shot_metrics: Zero-shot评估指标
            few_shot_metrics: Few-shot评估指标
        """
        print("\n" + "="*60)
        print("评估结果摘要")
        print("="*60)
        
        print("\n[Zero-shot]")
        print(f"  准确率: {zero_shot_metrics.accuracy:.2%}")
        print(f"  混淆矩阵:")
        cm = zero_shot_metrics.confusion_matrix
        print(f"    TP (True Positive):  {cm['TP']}")
        print(f"    FP (False Positive): {cm['FP']}")
        print(f"    TN (True Negative):  {cm['TN']}")
        print(f"    FN (False Negative): {cm['FN']}")
        print(f"  错误数量: {len(zero_shot_metrics.errors)}")
        
        print("\n[Few-shot]")
        print(f"  准确率: {few_shot_metrics.accuracy:.2%}")
        print(f"  混淆矩阵:")
        cm = few_shot_metrics.confusion_matrix
        print(f"    TP (True Positive):  {cm['TP']}")
        print(f"    FP (False Positive): {cm['FP']}")
        print(f"    TN (True Negative):  {cm['TN']}")
        print(f"    FN (False Negative): {cm['FN']}")
        print(f"  错误数量: {len(few_shot_metrics.errors)}")
        
        print("\n[改进]")
        improvement = few_shot_metrics.accuracy - zero_shot_metrics.accuracy
        print(f"  准确率提升: {improvement:+.2%}")
        print(f"  错误减少: {len(zero_shot_metrics.errors) - len(few_shot_metrics.errors)}")
        
        print("="*60)


def load_transcript_json(student_id: str, transcripts_dir: str = "Student_Transcripts") -> Optional[Dict]:
    """
    加载学生成绩单JSON
    
    Args:
        student_id: 学生ID (如 "01", "02")
        transcripts_dir: 成绩单目录
        
    Returns:
        成绩单JSON字典，如果文件不存在则返回None
    """
    transcript_path = os.path.join(transcripts_dir, f"transcript{student_id}.json")
    if os.path.exists(transcript_path):
        with open(transcript_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def load_requirements_json(school: str, program: str, 
                          csv_path: str = "university_programs.csv",
                          use_daad: bool = False) -> Optional[Dict]:
    """
    加载项目要求JSON
    
    Args:
        school: 学校名称
        program: 项目名称
        csv_path: CSV数据库路径
        use_daad: 是否使用DAAD爬虫（默认False，避免评估时爬取）
        
    Returns:
        要求JSON字典，如果无法加载则返回None
    """
    try:
        # 尝试从agent2加载
        from agent2 import AdmissionRequirementsLoader
        loader = AdmissionRequirementsLoader(csv_path, use_daad=use_daad, headless=True)
        try:
            result = loader.load_requirements(school, program)
            return result
        finally:
            loader.close()
    except Exception as e:
        print(f"  警告: 无法加载要求 ({school} - {program}): {str(e)}")
        return None


def main():
    """主函数 - 运行完整的评估流程"""
    import google.generativeai as genai  # type: ignore
    
    # 从环境变量读取 API 密钥
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    genai.configure(api_key=api_key)
    
    # 创建Gemini提供者
    class SimpleGeminiProvider:
        def __init__(self):
            self.genai = genai
        
        def generate(self, prompt: str, model: str = "gemini-2.5-flash") -> str:
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
    
    # 运行评估（带transcript和requirements加载器）
    zero_shot_metrics, few_shot_metrics = evaluator.evaluate(
        llm_provider=llm_provider,
        test_set=test_set,
        few_shot_examples=few_shot_examples,
        model_name="gemini-2.5-flash",
        transcript_loader=lambda sid: load_transcript_json(sid),
        requirements_loader=lambda school, program: load_requirements_json(school, program, use_daad=False)
    )
    
    # 打印结果
    evaluator.print_evaluation_summary(zero_shot_metrics, few_shot_metrics)
    
    # 保存结果
    evaluator.save_evaluation_results(zero_shot_metrics, few_shot_metrics)


if __name__ == "__main__":
    main()

