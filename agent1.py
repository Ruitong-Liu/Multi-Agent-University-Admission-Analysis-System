"""
Agent 1 - Transcript Extraction
从PDF/JPG成绩单中提取信息并转换为统一的JSON格式
"""

import json
import os
import tempfile
from typing import Dict, List, Any
from pathlib import Path
from PIL import Image
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


class TranscriptExtractor:
    """成绩单提取器 - 使用 Gemini 进行信息提取"""

    def __init__(self, api_key: str, model_name: str = "gemini-3-pro-preview"):
        """
        初始化提取器
        
        Args:
            api_key: Gemini API密钥
            model_name: 使用的 Gemini 模型名称
        """
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=api_key)
            self.genai = genai
        except ImportError:
            raise ImportError("请安装 google-generativeai: pip install google-generativeai")
        self.model_name = model_name
        self.chinese_credit_to_ects_ratio = 1.5  # 1中国学分 = 1.5 ECTS
    
    def _pdf_to_images(self, pdf_path: str) -> List[str]:
        """
        将PDF转换为图像
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            图像文件路径列表
        """
        if not PDF2IMAGE_AVAILABLE:
            raise ImportError("pdf2image库未安装。请运行: pip install pdf2image")
        
        # 将PDF转换为图像
        images = convert_from_path(pdf_path)
        image_paths = []
        
        # 保存为临时图像文件
        for i, image in enumerate(images):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            image.save(temp_file.name, 'PNG')
            image_paths.append(temp_file.name)
        
        return image_paths
    
    def _extract_text_from_image(self, image_path: str) -> str:
        """
        使用 Gemini 从图像中提取文本
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            提取的文本内容
        """
        image = Image.open(image_path)
        prompt = (
            "请提取这个成绩单中的所有信息，包括学生姓名、院系、专业、总学分、GPA和所有课程信息"
            "（课程名称和学分）。请仔细识别所有文本内容。"
        )
        model_instance = self.genai.GenerativeModel(self.model_name)
        response = model_instance.generate_content([prompt, image])
        return response.text or ""

    @staticmethod
    def _clean_json_text(raw_text: str) -> str:
        """
        清理模型输出中的代码块包裹，提取JSON文本
        """
        text = raw_text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1].strip()
                if text.lower().startswith("json"):
                    text = text[4:].strip()
        return text
    
    def _parse_extracted_text(self, extracted_text: str) -> Dict[str, Any]:
        """
        解析提取的文本并转换为结构化JSON
        
        Args:
            extracted_text: Gemini 提取的文本
            
        Returns:
            结构化的JSON字典
        """
        prompt = """你是一个成绩单信息提取专家。请从给定的文本中提取以下信息，并以JSON格式返回：

1. 学生信息：姓名(name)、院系(department)、专业(major)
2. 成绩单总计：已获得学分(earned_credits) - 请提取原始的中国学分数值、GPA(gpa)
3. 课程列表：每门课程的课程名称(course_title)和学分(credit) - 请提取原始的中国学分数值

注意：
- 请提取成绩单中的原始学分数值（中国学分），不要进行任何转换
- 返回的JSON格式必须严格遵循以下结构：
{
  "student_information": {
    "name": "...",
    "department": "...",
    "major": "..."
  },
  "transcript_totals": {
    "earned_credits": ...,
    "gpa": ...
  },
  "courses": [
    {"course_title": "...", "credit": ...},
    ...
  ]
}

只返回JSON，不要添加任何其他文字说明。

原始文本如下：
"""
        model_instance = self.genai.GenerativeModel(self.model_name)
        response = model_instance.generate_content([prompt, extracted_text])
        json_text = self._clean_json_text(response.text or "")
        result = json.loads(json_text)
        
        # 转换学分为ECTS
        if "transcript_totals" in result and "earned_credits" in result["transcript_totals"]:
            chinese_credits = result["transcript_totals"]["earned_credits"]
            result["transcript_totals"]["earned_credits"] = round(chinese_credits * self.chinese_credit_to_ects_ratio)
        
        if "courses" in result:
            for course in result["courses"]:
                if "credit" in course:
                    chinese_credit = course["credit"]
                    course["credit"] = round(chinese_credit * self.chinese_credit_to_ects_ratio)
        
        return result
    
    def extract_transcript(self, transcript_path: str) -> Dict[str, Any]:
        """
        从成绩单文件中提取信息
        
        Args:
            transcript_path: 成绩单文件路径（PDF或JPG）
            
        Returns:
            结构化的JSON字典
        """
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"文件不存在: {transcript_path}")
        
        file_ext = Path(transcript_path).suffix.lower()
        temp_files = []
        
        try:
            # 如果是PDF，先转换为图像
            if file_ext == '.pdf':
                image_paths = self._pdf_to_images(transcript_path)
                temp_files.extend(image_paths)
                
                # 处理所有页面，合并文本
                all_extracted_text = []
                for image_path in image_paths:
                    text = self._extract_text_from_image(image_path)
                    all_extracted_text.append(text)
                extracted_text = "\n".join(all_extracted_text)
            else:
                # 直接处理图像文件
                extracted_text = self._extract_text_from_image(transcript_path)
            
            # 解析并结构化
            structured_data = self._parse_extracted_text(extracted_text)
            
            return structured_data
            
        finally:
            # 清理临时文件
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    def save_result(self, data: Dict[str, Any], output_path: str):
        """
        保存提取结果到JSON文件
        
        Args:
            data: 提取的数据
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    """主函数 - 用于测试"""
    # API密钥
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    # 初始化提取器
    extractor = TranscriptExtractor(api_key)
    
    # 测试文件路径
    transcripts_dir = "Student_Transcripts"
    transcript_files = ["transcript01.pdf", "transcript02.pdf", "transcript03.pdf"]
    
    # 处理每个成绩单
    for transcript_file in transcript_files:
        transcript_path = os.path.join(transcripts_dir, transcript_file)
        
        if os.path.exists(transcript_path):
            print(f"\n处理文件: {transcript_file}")
            try:
                result = extractor.extract_transcript(transcript_path)
                
                # 保存结果
                output_file = transcript_file.replace('.pdf', '.json').replace('.jpg', '.json')
                output_path = os.path.join(transcripts_dir, output_file)
                extractor.save_result(result, output_path)
                
                print(f"✓ 成功提取并保存到: {output_path}")
                print(f"  学生姓名: {result.get('student_information', {}).get('name', 'N/A')}")
                print(f"  总学分: {result.get('transcript_totals', {}).get('earned_credits', 'N/A')} ECTS")
                print(f"  GPA: {result.get('transcript_totals', {}).get('gpa', 'N/A')}")
                print(f"  课程数量: {len(result.get('courses', []))}")
                
            except Exception as e:
                print(f"✗ 处理失败: {str(e)}")
        else:
            print(f"✗ 文件不存在: {transcript_path}")


if __name__ == "__main__":
    main()

