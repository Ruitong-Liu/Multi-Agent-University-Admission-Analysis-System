"""
Agent 2 使用示例
演示如何使用AdmissionRequirementsLoader加载入学要求
"""

from agent2 import AdmissionRequirementsLoader
import json

def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("示例1: 基本使用（使用DAAD爬虫）")
    print("=" * 60)
    
    # 初始化加载器
    loader = AdmissionRequirementsLoader(
        csv_path="university_programs.csv",
        use_daad=True,
        headless=False  # 设置为True可以隐藏浏览器窗口
    )
    
    try:
        # 加载入学要求
        result = loader.load_requirements(
            university="Technical University of Munich",
            program="Computer Science"
        )
        
        # 打印结果
        print("\n提取的结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    finally:
        loader.close()


def example_csv_only():
    """仅使用CSV数据示例"""
    print("\n" + "=" * 60)
    print("示例2: 仅使用CSV数据（不使用DAAD爬虫）")
    print("=" * 60)
    
    # 初始化加载器（禁用DAAD）
    loader = AdmissionRequirementsLoader(
        csv_path="university_programs.csv",
        use_daad=False
    )
    
    try:
        # 加载入学要求
        result = loader.load_requirements(
            university="Heidelberg University",
            program="Physics"
        )
        
        # 打印结果
        print("\n提取的结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    finally:
        loader.close()


def example_batch_processing():
    """批量处理示例"""
    print("\n" + "=" * 60)
    print("示例3: 批量处理多个项目")
    print("=" * 60)
    
    # 要查询的项目列表
    programs = [
        ("Technical University of Munich", "Computer Science"),
        ("Heidelberg University", "Physics"),
        ("Free University of Berlin", "Mathematics"),
    ]
    
    # 初始化加载器
    loader = AdmissionRequirementsLoader(
        csv_path="university_programs.csv",
        use_daad=True,
        headless=False
    )
    
    results = []
    
    try:
        for university, program in programs:
            print(f"\n处理: {university} - {program}")
            result = loader.load_requirements(university, program)
            results.append(result)
            
            # 保存每个结果
            filename = f"requirements_{university.replace(' ', '_')}_{program.replace(' ', '_')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {filename}")
    
    finally:
        loader.close()
    
    return results


if __name__ == "__main__":
    # 运行示例
    example_basic_usage()
    example_csv_only()
    # example_batch_processing()  # 取消注释以运行批量处理示例





