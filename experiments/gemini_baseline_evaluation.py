"""
Gemini Baseline Evaluation System
使用Gemini直接评估（不使用Agent系统）作为baseline对比
直接读取PDF成绩单，不使用预处理后的JSON
支持zero-shot和few-shot两种模式
与系统评估使用相同的Gemini模型，确保公平对比
"""

import json
import os
import random
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import re
from pathlib import Path

import google.generativeai as genai  # type: ignore

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

from experiments.few_shot_evaluation import (
    ApplicationSample,
    EvaluationMetrics,
    FewShotEvaluator,
)


class GeminiBaselineEvaluator:
    """Gemini Baseline评估器 - 不使用Agent系统，直接使用Gemini读取PDF"""
    
    def __init__(self, acceptance_json_path: str = "acceptance.json", 
                 gemini_api_key: str = None,
                 model_name: str = "gemini-2.5-flash",
                 transcripts_dir: str = "Student_Transcripts",
                 csv_path: str = "university_programs.csv"):
        """
        初始化评估器
        
        Args:
            acceptance_json_path: acceptance.json文件路径
            gemini_api_key: Gemini API密钥
            model_name: 使用的Gemini模型名称
            transcripts_dir: 成绩单PDF文件夹路径
            csv_path: 项目要求CSV文件路径
        """
        self.acceptance_json_path = acceptance_json_path
        self.model_name = model_name
        self.transcripts_dir = transcripts_dir
        self.csv_path = csv_path
        
        # 配置Gemini API
        if gemini_api_key:
            self.api_key = gemini_api_key
        else:
            self.api_key = os.environ.get("GOOGLE_API_KEY", "")
        
        genai.configure(api_key=self.api_key)
        self.genai = genai
        
        # 支持多个模型作为fallback
        self.models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]
        
        # 复用FewShotEvaluator的数据加载和few-shot选择逻辑
        self.base_evaluator = FewShotEvaluator(acceptance_json_path)
    
    def _pdf_to_images(self, pdf_path: str) -> List[str]:
        """将PDF转换为图像"""
        if not PDF2IMAGE_AVAILABLE:
            raise ImportError("需要安装pdf2image: pip install pdf2image")
        
        images = convert_from_path(pdf_path)
        image_paths = []
        
        for i, image in enumerate(images):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            image.save(temp_file.name, 'PNG')
            image_paths.append(temp_file.name)
        
        return image_paths
    
    def _load_transcript_pdf(self, student_id: str) -> Optional[str]:
        """获取成绩单PDF路径"""
        transcript_path = os.path.join(self.transcripts_dir, f"transcript{student_id}.pdf")
        if os.path.exists(transcript_path):
            return transcript_path
        return None
    
    def _load_requirements_from_csv(self, school: str, program: str) -> Optional[Dict]:
        """从CSV直接加载项目要求（简化版，不使用Agent2的复杂逻辑）"""
        try:
            import csv
            if not os.path.exists(self.csv_path):
                return None
            
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_school = row.get('University', '').strip()
                    row_program = row.get('Program', '').strip()
                    
                    # 模糊匹配
                    if (school.lower() in row_school.lower() or row_school.lower() in school.lower()):
                        if (program.lower() in row_program.lower() or row_program.lower() in program.lower()):
                            # 返回CSV中的原始数据
                            return {
                                "University": row_school,
                                "Program": row_program,
                                "requirements": {
                                    k: v for k, v in row.items() 
                                    if k not in ['University', 'Program'] and v
                                }
                            }
        except Exception as e:
            print(f"    ⚠ CSV加载错误: {str(e)}")
        
        return None
    
    def _generate_with_vision(self, prompt: str, image_paths: List[str], model: str = None) -> str:
        """
        使用Gemini Vision API生成响应，支持PDF图像
        """
        import time
        
        model_to_use = model or self.model_name
        models_to_try = [model_to_use] + [m for m in self.models if m != model_to_use]
        
        last_error = None
        for model_name in models_to_try:
            try:
                model_instance = self.genai.GenerativeModel(model_name)
                
                # 准备内容：文本 + 图像
                content = [prompt]
                for image_path in image_paths:
                    import PIL.Image
                    img = PIL.Image.open(image_path)
                    content.append(img)
                
                response = model_instance.generate_content(content)
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
        
        # 所有模型都失败
        print(f"    ✗ 所有模型都失败，最后错误: {str(last_error)[:200]}")
        return '{"admission_decision": "Unknown", "matching_score": 50, "reasoning": "API error"}'
    
    def _generate_with_retry(self, prompt: str, model: str = None) -> str:
        """使用Gemini生成响应（纯文本）"""
        import time
        
        model_to_use = model or self.model_name
        models_to_try = [model_to_use] + [m for m in self.models if m != model_to_use]
        
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
    
    def build_zero_shot_prompt(self, sample: ApplicationSample,
                               requirements_json: Dict) -> str:
        """
        构建zero-shot提示词（直接处理PDF图像）
        """
        requirements_text = json.dumps(requirements_json, indent=2, ensure_ascii=False) if requirements_json else "Not available"
        
        prompt = f"""You are a strict admission officer for German Master's programmes.

You will receive:

- A student's transcript in PDF format (as images).

- Program admission requirements in JSON format (when available).

IMPORTANT:

- The requirement JSON may be incomplete, generic, outdated, or noisy.

- You must treat it only as a hint, NOT as a perfectly reliable source.

- Your primary signals are:

  1) The student's transcript (GPA, major, core courses and their relevance) - read directly from the PDF images,

  2) Your knowledge about typical admission criteria for similar programmes

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

  you MUST base your decision on transcript quality and typical standards.

- If after checking the transcript and typical criteria you are still unsure,

  you must prefer "Reject".

- matching_score < 50 always corresponds to "Reject".

---

### NEW SAMPLE

Student ID: {sample.student_id}
School: {sample.school}
Program: {sample.program}

Program Requirements (JSON):
{requirements_text}

Please carefully read the transcript PDF images provided and evaluate the student's application.

Respond STRICTLY in the following JSON format:

{{
  "admission_decision": "Admit" | "Reject",
  "matching_score": number,
  "reasoning": "string"
}}
"""
        
        return prompt
    
    def build_few_shot_prompt(self, sample: ApplicationSample,
                             requirements_json: Dict,
                             few_shot_data: List[Dict]) -> str:
        """
        构建few-shot提示词（直接处理PDF图像）
        """
        requirements_text = json.dumps(requirements_json, indent=2, ensure_ascii=False) if requirements_json else "Not available"
        
        prompt = """You are a strict admission officer for German Master's programmes.

You will receive:

- A student's transcript in PDF format (as images).

- Program admission requirements in JSON format (when available).

IMPORTANT:

- The requirement JSON may be incomplete, generic, outdated, or noisy.

- You must treat it only as a hint, NOT as a perfectly reliable source.

- Your primary signals are:

  1) The student's transcript (GPA, major, core courses and their relevance) - read directly from the PDF images,

  2) Your knowledge about typical admission criteria for similar programmes

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

  you MUST base your decision on transcript quality and typical standards.

- If after checking the transcript and typical criteria you are still unsure,

  you must prefer "Reject".

- matching_score < 50 always corresponds to "Reject".

You will see few-shot examples showing how to:

- Down-weight noisy or generic requirements,

- Rely on the transcript and typical admission expectations,

- Use explicit reasoning to justify Admit vs Reject.

---

### FEW-SHOT EXAMPLES

"""
        
        # 添加few-shot示例（使用文本描述，因为不能同时发送多个PDF）
        for i, example_data in enumerate(few_shot_data, 1):
            example_sample = example_data.get("sample")
            example_requirements = example_data.get("requirements_json")
            example_score = example_data.get("matching_score")
            example_reasoning = example_data.get("reasoning")
            example_transcript_summary = example_data.get("transcript_summary", "Transcript data available")
            
            if example_sample and example_requirements:
                prompt += f"""### FEW-SHOT EXAMPLE {i}

Student ID: {example_sample.student_id}
School: {example_sample.school}
Program: {example_sample.program}

Transcript Summary: {example_transcript_summary}

Program Requirements (JSON):
{json.dumps(example_requirements, indent=2, ensure_ascii=False)}

Expected output:
{{
  "admission_decision": "{example_sample.label}",
  "matching_score": {example_score},
  "reasoning": "{example_reasoning}"
}}

"""
        
        # 添加当前样本
        prompt += f"""### NEW SAMPLE

Student ID: {sample.student_id}
School: {sample.school}
Program: {sample.program}

Program Requirements (JSON):
{requirements_text}

Please carefully read the transcript PDF images provided and evaluate the student's application.

Respond STRICTLY in the following JSON format:

{{
  "admission_decision": "Admit" | "Reject",
  "matching_score": number,
  "reasoning": "string"
}}
"""
        
        return prompt
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """解析Gemini响应，提取JSON"""
        # 尝试提取JSON（支持markdown代码块）
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                pass
        
        # 尝试提取JSON（不使用代码块）
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
    
    def evaluate(self, test_set: List[ApplicationSample],
                 few_shot_examples: Optional[List[ApplicationSample]] = None) -> Tuple[EvaluationMetrics, EvaluationMetrics]:
        """
        评估Gemini在zero-shot和few-shot条件下的表现（直接读取PDF）
        
        Args:
            test_set: 测试集
            few_shot_examples: Few-shot示例列表
            
        Returns:
            (zero_shot_metrics, few_shot_metrics) 元组
        """
        print(f"\n开始Gemini Baseline评估 (测试集大小: {len(test_set)})")
        print("⚠ 注意: 直接读取PDF，不使用Agent1/Agent2预处理")
        
        # 为few-shot示例准备数据（生成文本摘要，因为不能同时发送多个PDF）
        few_shot_data = None
        if few_shot_examples:
            print("\n[准备Few-shot示例数据]")
            few_shot_data = []
            for example in few_shot_examples:
                # 加载requirements
                requirements = self._load_requirements_from_csv(example.school, example.program)
                
                # 尝试读取PDF生成摘要
                transcript_pdf = self._load_transcript_pdf(example.student_id)
                transcript_summary = "Transcript data available"
                
                if transcript_pdf:
                    try:
                        # 使用Gemini快速读取PDF生成摘要
                        summary_prompt = f"""Please read this transcript PDF and provide a brief summary in one sentence:
- Student major/background
- GPA
- Key courses relevant to the program: {example.program}

Respond with just the summary text, no JSON."""
                        image_paths = self._pdf_to_images(transcript_pdf)
                        try:
                            summary = self._generate_with_vision(summary_prompt, image_paths[:1])  # 只读第一页
                            transcript_summary = summary[:200]  # 限制长度
                        finally:
                            # 清理临时文件
                            for img_path in image_paths:
                                try:
                                    os.unlink(img_path)
                                except:
                                    pass
                    except Exception as e:
                        print(f"    ⚠ 无法生成few-shot摘要: {str(e)}")
                
                # 生成reasoning和score
                if example.label == "Admit":
                    matching_score = random.randint(80, 95)
                    reasoning = "专业匹配，GPA达标，课程背景符合项目要求。"
                else:
                    matching_score = random.randint(30, 55)
                    reasoning = "专业不匹配或GPA未达标，缺少关键课程，不符合入学要求。"
                
                few_shot_data.append({
                    "sample": example,
                    "requirements_json": requirements or {},
                    "transcript_summary": transcript_summary,
                    "matching_score": matching_score,
                    "reasoning": reasoning
                })
            
            if few_shot_data:
                print(f"  ✓ 成功准备 {len(few_shot_data)}/{len(few_shot_examples)} 个few-shot示例")
            else:
                print("  ⚠ 警告: 所有few-shot示例数据准备失败")
                few_shot_data = None
        
        # Zero-shot评估
        print("\n[Zero-shot评估]")
        zero_shot_predictions = []
        zero_shot_errors = []
        
        for i, sample in enumerate(test_set):
            print(f"  处理样本 {i+1}/{len(test_set)}: {sample.student_id} - {sample.school} - {sample.program}")
            
            # 加载PDF和requirements
            transcript_pdf = self._load_transcript_pdf(sample.student_id)
            if not transcript_pdf:
                print(f"    ⚠ 跳过样本: PDF不存在")
                continue
            
            requirements = self._load_requirements_from_csv(sample.school, sample.program)
            if not requirements:
                requirements = {
                    "University": sample.school,
                    "Program": sample.program,
                    "requirements": {}
                }
            
            # 构建zero-shot提示词
            prompt = self.build_zero_shot_prompt(sample, requirements)
            
            # 将PDF转换为图像
            image_paths = []
            try:
                image_paths = self._pdf_to_images(transcript_pdf)
                
                # 调用Gemini Vision API
                response_text = self._generate_with_vision(prompt, image_paths)
                prediction = self.parse_response(response_text)
                
                predicted_label = prediction.get("admission_decision", "").strip()
                # 标准化标签
                if "admit" in predicted_label.lower():
                    predicted_label = "Admit"
                elif "reject" in predicted_label.lower():
                    predicted_label = "Reject"
                else:
                    predicted_label = "Unknown"
                
                # 强制规则：score < 50 → Reject
                score_val = prediction.get("matching_score", 0)
                if isinstance(score_val, str):
                    score_val = int(float(score_val))
                else:
                    score_val = int(score_val)
                score_val = max(0, min(100, score_val))
                
                if score_val < 50:
                    predicted_label = "Reject"
                
                zero_shot_predictions.append({
                    "sample": sample,
                    "prediction": predicted_label,
                    "actual": sample.label,
                    "matching_score": score_val,
                    "reasoning": prediction.get("reasoning", ""),
                    "raw_response": response_text
                })
                
                if predicted_label != sample.label:
                    zero_shot_errors.append({
                        "student_id": sample.student_id,
                        "school": sample.school,
                        "program": sample.program,
                        "actual": sample.label,
                        "predicted": predicted_label,
                        "matching_score": score_val,
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
            finally:
                # 清理临时图像文件
                for img_path in image_paths:
                    try:
                        os.unlink(img_path)
                    except:
                        pass
        
        # Few-shot评估
        print("\n[Few-shot评估]")
        few_shot_predictions = []
        few_shot_errors = []
        
        for i, sample in enumerate(test_set):
            print(f"  处理样本 {i+1}/{len(test_set)}: {sample.student_id} - {sample.school} - {sample.program}")
            
            # 加载PDF和requirements
            transcript_pdf = self._load_transcript_pdf(sample.student_id)
            if not transcript_pdf:
                print(f"    ⚠ 跳过样本: PDF不存在")
                continue
            
            requirements = self._load_requirements_from_csv(sample.school, sample.program)
            if not requirements:
                requirements = {
                    "University": sample.school,
                    "Program": sample.program,
                    "requirements": {}
                }
            
            # 构建few-shot提示词
            prompt = self.build_few_shot_prompt(sample, requirements, few_shot_data)
            
            # 将PDF转换为图像
            image_paths = []
            try:
                image_paths = self._pdf_to_images(transcript_pdf)
                
                # 调用Gemini Vision API
                response_text = self._generate_with_vision(prompt, image_paths)
                prediction = self.parse_response(response_text)
                
                predicted_label = prediction.get("admission_decision", "").strip()
                # 标准化标签
                if "admit" in predicted_label.lower():
                    predicted_label = "Admit"
                elif "reject" in predicted_label.lower():
                    predicted_label = "Reject"
                else:
                    predicted_label = "Unknown"
                
                # 强制规则：score < 50 → Reject
                score_val = prediction.get("matching_score", 0)
                if isinstance(score_val, str):
                    score_val = int(float(score_val))
                else:
                    score_val = int(score_val)
                score_val = max(0, min(100, score_val))
                
                if score_val < 50:
                    predicted_label = "Reject"
                
                few_shot_predictions.append({
                    "sample": sample,
                    "prediction": predicted_label,
                    "actual": sample.label,
                    "matching_score": score_val,
                    "reasoning": prediction.get("reasoning", ""),
                    "raw_response": response_text
                })
                
                if predicted_label != sample.label:
                    few_shot_errors.append({
                        "student_id": sample.student_id,
                        "school": sample.school,
                        "program": sample.program,
                        "actual": sample.label,
                        "predicted": predicted_label,
                        "matching_score": score_val,
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
            finally:
                # 清理临时图像文件
                for img_path in image_paths:
                    try:
                        os.unlink(img_path)
                    except:
                        pass
        
        # 计算指标
        zero_shot_metrics = self.base_evaluator._calculate_metrics(zero_shot_predictions, zero_shot_errors)
        few_shot_metrics = self.base_evaluator._calculate_metrics(few_shot_predictions, few_shot_errors)
        
        return zero_shot_metrics, few_shot_metrics
    
    def save_results(self, zero_shot_metrics: EvaluationMetrics,
                    few_shot_metrics: EvaluationMetrics,
                    few_shot_examples: List[ApplicationSample],
                    test_set: List[ApplicationSample],
                    output_path: str = "gemini_baseline_results.json"):
        """保存评估结果"""
        results = {
            "baseline_type": f"Gemini Direct (No Agent System) - {self.model_name} - Direct PDF Reading",
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
                for s in few_shot_examples
            ],
            "test_set_size": len(test_set)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Gemini Baseline评估结果已保存到: {output_path}")


def main():
    """主函数 - 运行Gemini Baseline评估"""
    import argparse
    
    parser = argparse.ArgumentParser(description='运行Gemini Baseline评估（不使用Agent系统，直接读取PDF）')
    parser.add_argument('--test-size', type=int, default=None,
                       help='测试样本数量（默认：全部）')
    parser.add_argument('--model', type=str, default="gemini-2.5-flash",
                       help='使用的Gemini模型（默认：gemini-2.5-flash）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Gemini Baseline评估（不使用Agent系统，直接读取PDF）")
    print("=" * 60)
    
    # 初始化评估器
    evaluator = GeminiBaselineEvaluator(
        acceptance_json_path="acceptance.json",
        model_name=args.model
    )
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    evaluator.base_evaluator.load_and_convert_data()
    
    # 选择few-shot示例
    print("\n[2/4] 选择few-shot示例...")
    few_shot_examples, test_set = evaluator.base_evaluator.select_few_shot_examples(k=5)
    
    # 如果指定了测试大小，只使用前N个样本
    if args.test_size is not None:
        test_set = test_set[:args.test_size]
        print(f"\n⚠ 仅评估前 {args.test_size} 个测试样本（总共 {len(evaluator.base_evaluator.test_set)} 个）")
    
    # 运行评估
    print("\n[3/4] 运行评估...")
    print(f"测试集大小: {len(test_set)}")
    print(f"Few-shot示例数量: {len(few_shot_examples)}")
    print(f"使用模型: {args.model}")
    
    zero_shot_metrics, few_shot_metrics = evaluator.evaluate(
        test_set=test_set,
        few_shot_examples=few_shot_examples
    )
    
    # 打印结果
    print("\n[4/4] 评估结果:")
    evaluator.base_evaluator.print_evaluation_summary(zero_shot_metrics, few_shot_metrics)
    
    # 保存结果
    output_file = f"gemini_baseline_results_{len(test_set)}samples.json"
    evaluator.save_results(zero_shot_metrics, few_shot_metrics, few_shot_examples, test_set, output_file)
    
    return zero_shot_metrics, few_shot_metrics


if __name__ == "__main__":
    main()
