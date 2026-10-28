"""
三个Agent完整整合示例
演示如何使用 Agent1、Agent2 和 Agent3 进行完整的申请评估流程
"""

import os
import json
from agent1 import TranscriptExtractor
from agent2 import AdmissionRequirementsLoader
from agent3 import MatchingModel, GeminiProvider


def integrated_workflow():
    """
    完整的三个Agent整合工作流
    
    步骤1: Agent1 - 从PDF/JPG提取成绩单并转换为JSON
    步骤2: Agent2 - 从CSV/DAAD加载入学要求
    步骤3: Agent3 - 使用LLM进行匹配评估
    """
    print("=" * 80)
    print("三个Agent完整整合示例")
    print("=" * 80)
    
    # ========== 配置 ==========
    # Agent1/Agent3配置（同一模型）
    gemini_api_key = os.environ.get("GOOGLE_API_KEY", "")

    # Agent2配置
    csv_path = "university_programs.csv"
    use_daad = False  # 设置为True以使用DAAD爬虫（需要更长时间）
    
    # Agent3配置
    model_name = "gemini-3-pro-preview"
    
    # 目标项目
    target_university = "Technical University of Munich"
    target_program = "Computer Science"
    
    # 成绩单文件（如果已存在JSON，可以跳过Agent1）
    transcript_pdf = "Student_Transcripts/transcript01.pdf"
    transcript_json_path = "Student_Transcripts/transcript01.json"
    
    # ========== 步骤1: Agent1 - 提取成绩单 ==========
    print("\n" + "=" * 80)
    print("步骤1: Agent1 - 从PDF提取成绩单信息")
    print("=" * 80)
    
    transcript_json = None
    
    # 检查是否已有JSON文件
    if os.path.exists(transcript_json_path):
        print(f"✓ 发现已存在的成绩单JSON: {transcript_json_path}")
        try:
            with open(transcript_json_path, 'r', encoding='utf-8') as f:
                transcript_json = json.load(f)
            student_name = transcript_json.get('student_information', {}).get('name', 'Unknown')
            print(f"✓ 成功加载成绩单: {student_name}")
        except Exception as e:
            print(f"✗ 加载JSON失败: {str(e)}")
            transcript_json = None
    
    # 如果没有JSON，尝试从PDF提取
    if transcript_json is None and os.path.exists(transcript_pdf):
        print(f"正在从PDF提取: {transcript_pdf}")
        try:
            extractor = TranscriptExtractor(gemini_api_key, model_name=model_name)
            transcript_json = extractor.extract_transcript(transcript_pdf)
            
            # 保存提取的结果
            extractor.save_result(transcript_json, transcript_json_path)
            student_name = transcript_json.get('student_information', {}).get('name', 'Unknown')
            print(f"✓ 成功提取并保存成绩单: {student_name}")
            print(f"  保存路径: {transcript_json_path}")
        except Exception as e:
            print(f"✗ Agent1提取失败: {str(e)}")
            return None
    
    if transcript_json is None:
        print("✗ 错误: 无法获取成绩单数据")
        print("提示: 请确保PDF文件存在或JSON文件有效")
        return None
    
    # ========== 步骤2: Agent2 - 加载入学要求 ==========
    print("\n" + "=" * 80)
    print("步骤2: Agent2 - 从CSV/DAAD加载入学要求")
    print("=" * 80)
    
    requirement_json = None
    
    try:
        loader = AdmissionRequirementsLoader(
            csv_path=csv_path,
            use_daad=use_daad,
            headless=True
        )
        
        try:
            print(f"正在加载: {target_university} - {target_program}")
            requirement_json = loader.load_requirements(target_university, target_program)
            
            if requirement_json:
                req_content = requirement_json.get('requirements', {}).get('Description/content', '')
                if req_content and req_content != "无法找到":
                    print(f"✓ 成功加载入学要求")
                    print(f"  大学: {requirement_json.get('University', 'N/A')}")
                    print(f"  项目: {requirement_json.get('Program', 'N/A')}")
                    print(f"  学位: {requirement_json.get('Degree', 'N/A')}")
                    print(f"  教学语言: {requirement_json.get('Teaching language', 'N/A')}")
                    print(f"  申请截止: {requirement_json.get('Application deadline', 'N/A')}")
                else:
                    print("⚠ 警告: 未找到详细的入学要求内容")
            else:
                print("⚠ 警告: Agent2返回空数据")
        finally:
            loader.close()
            
    except Exception as e:
        print(f"✗ Agent2加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    if requirement_json is None:
        print("✗ 错误: 无法加载入学要求")
        return None
    
    # ========== 步骤3: Agent3 - 匹配评估 ==========
    print("\n" + "=" * 80)
    print("步骤3: Agent3 - LLM匹配评估")
    print("=" * 80)
    
    try:
        # 初始化Agent3
        provider = GeminiProvider(gemini_api_key)
        matcher = MatchingModel(provider, model_name=model_name)
        
        student_name = transcript_json.get('student_information', {}).get('name', 'Unknown')
        print(f"正在评估: {student_name} 的申请...")
        print(f"目标项目: {requirement_json.get('University')} - {requirement_json.get('Program')}\n")
        
        # 进行评估
        result = matcher.evaluate(transcript_json, requirement_json)
        
        # 打印结果
        print("=" * 80)
        print("评估结果")
        print("=" * 80)
        print(f"录取决定: {result['admission_decision']}")
        print(f"匹配分数: {result['matching_score']}/100")
        print(f"\n评估理由:\n{result['reasoning']}")
        
        # 保存结果
        output_path = "integrated_matching_result.json"
        matcher.save_result(result, output_path)
        print(f"\n✓ 结果已保存到: {output_path}")
        
        # 保存完整的工作流结果（包含所有三个agent的数据）
        full_result = {
            "workflow": "integrated_three_agents",
            "student": {
                "name": student_name,
                "transcript_source": transcript_json_path,
                "transcript_data": transcript_json
            },
            "program": {
                "university": target_university,
                "program": target_program,
                "requirements_source": "agent2_csv_daad",
                "requirements_data": requirement_json
            },
            "evaluation": {
                "model": model_name,
                "result": result
            }
        }
        
        full_output_path = "integrated_full_result.json"
        with open(full_output_path, 'w', encoding='utf-8') as f:
            json.dump(full_result, f, ensure_ascii=False, indent=2)
        print(f"✓ 完整工作流结果已保存到: {full_output_path}")
        
        return result
        
    except Exception as e:
        print(f"✗ Agent3评估失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def batch_integrated_workflow():
    """
    批量处理多个学生的申请评估
    """
    print("=" * 80)
    print("批量整合工作流示例")
    print("=" * 80)
    
    # 配置
    gemini_api_key = os.environ.get("GOOGLE_API_KEY", "")
    csv_path = "university_programs.csv"
    
    # 批量任务配置
    tasks = [
        {
            "student_id": "01",
            "university": "Technical University of Munich",
            "program": "Computer Science"
        },
        {
            "student_id": "02",
            "university": "Heidelberg University",
            "program": "Physics"
        },
        {
            "student_id": "03",
            "university": "Technical University of Munich",
            "program": "Computer Science"
        }
    ]
    
    # 初始化Agent3
    provider = GeminiProvider(gemini_api_key)
    matcher = MatchingModel(provider, model_name="gemini-3-pro-preview")
    
    results = []
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*80}")
        print(f"处理任务 {i}/{len(tasks)}: 学生{task['student_id']} - {task['university']} - {task['program']}")
        print(f"{'='*80}")
        
        # 加载成绩单（Agent1输出）
        transcript_path = f"Student_Transcripts/transcript{task['student_id']}.json"
        if not os.path.exists(transcript_path):
            print(f"✗ 跳过: 成绩单文件不存在: {transcript_path}")
            continue
        
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_json = json.load(f)
        except Exception as e:
            print(f"✗ 加载成绩单失败: {str(e)}")
            continue
        
        # 加载入学要求（Agent2输出）
        try:
            loader = AdmissionRequirementsLoader(
                csv_path=csv_path,
                use_daad=False,
                headless=True
            )
            try:
                requirement_json = loader.load_requirements(
                    task['university'],
                    task['program']
                )
            finally:
                loader.close()
        except Exception as e:
            print(f"✗ 加载入学要求失败: {str(e)}")
            continue
        
        if not requirement_json:
            print("✗ 跳过: 无法加载入学要求")
            continue
        
        # 进行评估（Agent3）
        try:
            student_name = transcript_json.get('student_information', {}).get('name', 'Unknown')
            print(f"正在评估: {student_name}")
            
            result = matcher.evaluate(transcript_json, requirement_json)
            
            print(f"✓ 评估完成: {result['admission_decision']} (分数: {result['matching_score']}/100)")
            
            results.append({
                "task": task,
                "student_name": student_name,
                "result": result
            })
            
        except Exception as e:
            print(f"✗ 评估失败: {str(e)}")
    
    # 打印汇总
    print("\n" + "=" * 80)
    print("批量处理汇总")
    print("=" * 80)
    print(f"总任务数: {len(tasks)}")
    print(f"成功完成: {len(results)}")
    
    if results:
        admit_count = sum(1 for r in results if r['result']['admission_decision'] == 'Admit')
        reject_count = sum(1 for r in results if r['result']['admission_decision'] == 'Reject')
        avg_score = sum(r['result']['matching_score'] for r in results) / len(results)
        
        print(f"\n录取决定统计:")
        print(f"  录取: {admit_count}")
        print(f"  拒绝: {reject_count}")
        print(f"  平均匹配分数: {avg_score:.2f}/100")
        
        # 保存批量结果
        batch_output = "batch_integrated_results.json"
        with open(batch_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 批量结果已保存到: {batch_output}")


if __name__ == "__main__":
    # 运行单个工作流
    integrated_workflow()
    
    # 取消注释以运行批量处理
    # batch_integrated_workflow()





