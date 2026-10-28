"""
Agent 3 使用示例
演示如何使用 MatchingModel 进行成绩单与入学要求的匹配评估
"""

from agent3 import MatchingModel, GeminiProvider
import json
import os


def example_basic_usage():
    """基本使用示例 - 使用 Gemini"""
    print("=" * 60)
    print("示例1: 基本使用（使用 Gemini API）")
    print("=" * 60)
    
    # Gemini API 密钥
    gemini_api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    # 初始化 Gemini 提供者
    provider = GeminiProvider(gemini_api_key)
    
    # 初始化匹配模型（使用 gemini-3-pro-preview）
    matcher = MatchingModel(provider, model_name="gemini-3-pro-preview")
    
    # 加载成绩单
    transcript_path = "Student_Transcripts/transcript01.json"
    if not os.path.exists(transcript_path):
        print(f"错误: 成绩单文件不存在: {transcript_path}")
        return
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript_json = json.load(f)
    
    # 创建示例入学要求
    requirement_json = {
        "University": "Technical University of Munich",
        "Program": "Computer Science",
        "Degree": "Master of Science",
        "Teaching language": "English",
        "Application deadline": "Winter semester: 31 May; Summer semester: 30 November",
        "requirements": {
            "Description/content": "Bachelor's degree in Computer Science or related field. Strong background in algorithms, data structures, and software engineering. Minimum GPA of 3.5/4.0."
        }
    }
    
    # 进行评估
    print(f"\n正在评估: {transcript_json.get('student_information', {}).get('name', 'Unknown')} 的申请...")
    print(f"目标项目: {requirement_json.get('University')} - {requirement_json.get('Program')}\n")
    
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
        output_path = "matching_result_example1.json"
        matcher.save_result(result, output_path)
        print(f"\n结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"评估失败: {str(e)}")


def example_batch_processing():
    """批量处理示例"""
    print("\n" + "=" * 60)
    print("示例3: 批量处理多个申请")
    print("=" * 60)
    
    # Gemini API 密钥
    gemini_api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    # 初始化 Gemini 提供者
    provider = GeminiProvider(gemini_api_key)
    
    # 初始化匹配模型
    matcher = MatchingModel(provider, model_name="gemini-3-pro-preview")
    
    # 定义多个评估任务
    tasks = [
        {
            "transcript": "Student_Transcripts/transcript01.json",
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
            "output": "matching_result_batch1.json"
        },
        {
            "transcript": "Student_Transcripts/transcript02.json",
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
            "output": "matching_result_batch2.json"
        },
        {
            "transcript": "Student_Transcripts/transcript03.json",
            "requirement": {
                "University": "Heidelberg University",
                "Program": "Psychology",
                "Degree": "Master of Science",
                "Teaching language": "English",
                "Application deadline": "Winter semester: 15 July",
                "requirements": {
                    "Description/content": "Bachelor's degree in Psychology or related field. Strong background in research methods and statistics. Minimum GPA of 3.3/4.0."
                }
            },
            "output": "matching_result_batch3.json"
        }
    ]
    
    results = []
    
    for i, task in enumerate(tasks, 1):
        transcript_path = task["transcript"]
        if not os.path.exists(transcript_path):
            print(f"跳过任务 {i}: 成绩单文件不存在: {transcript_path}")
            continue
        
        print(f"\n处理任务 {i}/{len(tasks)}...")
        
        # 加载成绩单
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_json = json.load(f)
        
        requirement_json = task["requirement"]
        
        # 进行评估
        student_name = transcript_json.get('student_information', {}).get('name', 'Unknown')
        university = requirement_json.get('University')
        program = requirement_json.get('Program')
        
        print(f"学生: {student_name}")
        print(f"项目: {university} - {program}")
        
        try:
            result = matcher.evaluate(transcript_json, requirement_json)
            results.append(result)
            
            # 保存结果
            matcher.save_result(result, task["output"])
            print(f"✓ 评估完成: {result['admission_decision']} (分数: {result['matching_score']}/100)")
            print(f"  结果已保存到: {task['output']}")
            
        except Exception as e:
            print(f"✗ 评估失败: {str(e)}")
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("批量处理汇总:")
    print("=" * 60)
    print(f"总任务数: {len(tasks)}")
    print(f"成功: {len(results)}")
    print(f"失败: {len(tasks) - len(results)}")
    
    if results:
        admit_count = sum(1 for r in results if r['admission_decision'] == 'Admit')
        reject_count = sum(1 for r in results if r['admission_decision'] == 'Reject')
        avg_score = sum(r['matching_score'] for r in results) / len(results)
        
        print(f"\n录取决定统计:")
        print(f"  录取: {admit_count}")
        print(f"  拒绝: {reject_count}")
        print(f"  平均匹配分数: {avg_score:.2f}/100")
    
    return results


if __name__ == "__main__":
    # 运行示例
    example_basic_usage()
    # example_batch_processing()  # 取消注释以运行批量处理示例

