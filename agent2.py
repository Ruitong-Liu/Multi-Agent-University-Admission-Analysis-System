"""
Agent 2 - Admission Requirements Loader
从CSV数据库和DAAD网站加载并标准化入学要求信息
"""

import json
import csv
import time
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
try:
    from webdriver_manager.chrome import ChromeDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError:
    WEBDRIVER_MANAGER_AVAILABLE = False


class CSVLoader:
    """CSV数据库加载器"""
    
    def __init__(self, csv_path: str):
        """
        初始化CSV加载器
        
        Args:
            csv_path: CSV文件路径
        """
        self.csv_path = csv_path
        self.data = []
    
    def load(self) -> List[Dict[str, Any]]:
        """
        加载CSV文件并返回数据列表
        
        Returns:
            包含大学和项目信息的字典列表
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {self.csv_path}")
        
        self.data = []
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 清理空值
                cleaned_row = {k: v.strip() if v else "" for k, v in row.items()}
                self.data.append(cleaned_row)
        
        return self.data
    
    def find_program(self, university: str, program: str) -> Optional[Dict[str, Any]]:
        """
        在CSV数据中查找匹配的项目
        
        Args:
            university: 大学名称
            program: 项目名称
            
        Returns:
            匹配的项目字典，如果未找到则返回None
        """
        university_lower = university.lower().strip()
        program_lower = program.lower().strip()
        
        for row in self.data:
            row_university = row.get('University', '').lower().strip()
            row_program = row.get('Program', '').lower().strip()
            
            if university_lower in row_university or row_university in university_lower:
                if program_lower in row_program or row_program in program_lower:
                    return row
        
        return None
    
    def update_or_add_program(self, program_data: Dict[str, Any]) -> bool:
        """
        更新或添加项目数据到CSV
        
        Args:
            program_data: 包含项目信息的字典，必须包含"University"和"Program"字段
            
        Returns:
            如果成功返回True，否则返回False
        """
        if not program_data.get("University") or not program_data.get("Program"):
            print("错误: 程序数据必须包含University和Program字段")
            return False
        
        university = program_data.get("University", "").strip()
        program = program_data.get("Program", "").strip()
        
        # 查找是否已存在
        existing_index = None
        for i, row in enumerate(self.data):
            row_university = row.get('University', '').lower().strip()
            row_program = row.get('Program', '').lower().strip()
            
            if university.lower() in row_university or row_university in university.lower():
                if program.lower() in row_program or row_program in program.lower():
                    existing_index = i
                    break
        
        # 准备要保存的数据
        # 确保requirements字段是字符串格式（CSV中存储为字符串）
        requirements_str = ""
        if "requirements" in program_data:
            if isinstance(program_data["requirements"], dict):
                requirements_str = program_data["requirements"].get("Description/content", "")
            elif isinstance(program_data["requirements"], str):
                requirements_str = program_data["requirements"]
        
        csv_row = {
            "University": university,
            "Program": program,
            "Degree": program_data.get("Degree", "").strip(),
            "Teaching language": program_data.get("Teaching language", "English").strip(),
            "Application deadline": program_data.get("Application deadline", "").strip(),
            "requirements": requirements_str
        }
        
        # 更新或添加
        if existing_index is not None:
            # 更新现有记录
            self.data[existing_index] = csv_row
            print(f"已更新CSV中的记录: {university} - {program}")
        else:
            # 添加新记录
            self.data.append(csv_row)
            print(f"已添加新记录到CSV: {university} - {program}")
        
        # 保存到文件
        return self.save()
    
    def save(self) -> bool:
        """
        将数据保存到CSV文件
        
        Returns:
            如果成功返回True，否则返回False
        """
        try:
            # 确定CSV文件的列名
            if not self.data:
                print("警告: 没有数据可保存")
                return False
            
            # 获取所有可能的列名
            all_keys = set()
            for row in self.data:
                all_keys.update(row.keys())
            
            # 标准列名顺序
            fieldnames = ["University", "Program", "Degree", "Teaching language", "Application deadline", "requirements"]
            # 添加其他可能存在的列
            for key in sorted(all_keys):
                if key not in fieldnames:
                    fieldnames.append(key)
            
            # 写入CSV文件
            with open(self.csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in self.data:
                    # 确保所有字段都存在
                    complete_row = {field: row.get(field, "") for field in fieldnames}
                    writer.writerow(complete_row)
            
            print(f"数据已成功保存到: {self.csv_path}")
            return True
        except Exception as e:
            print(f"保存CSV文件时出错: {str(e)}")
            return False


class DAADScraper:
    """DAAD网站爬虫 - 模拟人类交互"""
    
    def __init__(self, headless: bool = False, delay: float = 1.0):
        """
        初始化DAAD爬虫
        
        Args:
            headless: 是否使用无头模式
            delay: 操作之间的延迟时间（秒）
        """
        self.headless = headless
        self.delay = delay
        self.driver = None
        self.base_url = "https://www2.daad.de/deutschland/studienangebote/international-programmes/en/"
    
    def _init_driver(self):
        """初始化Selenium WebDriver"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # 使用webdriver-manager自动管理驱动
        if WEBDRIVER_MANAGER_AVAILABLE:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
        else:
            self.driver = webdriver.Chrome(options=chrome_options)
        
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    def _human_type(self, element, text: str):
        """
        模拟人类打字（逐字符输入，带随机延迟）
        
        Args:
            element: Selenium WebElement
            text: 要输入的文本
        """
        element.clear()
        for char in text:
            element.send_keys(char)
            time.sleep(0.1 + (self.delay * 0.1))  # 随机延迟
    
    def _wait_and_find(self, by: By, value: str, timeout: int = 10):
        """
        等待并查找元素
        
        Args:
            by: 定位方式
            value: 定位值
            timeout: 超时时间（秒）
            
        Returns:
            WebElement
        """
        return WebDriverWait(self.driver, timeout).until(
            EC.presence_of_element_located((by, value))
        )
    
    def search_program(self, university: str, program: str) -> Optional[Dict[str, Any]]:
        """
        在DAAD网站搜索项目并提取信息
        
        Args:
            university: 大学名称
            program: 项目名称
            
        Returns:
            提取的项目信息字典，如果未找到则返回None
        """
        try:
            if self.driver is None:
                self._init_driver()
            
            # 导航到DAAD搜索页面
            print(f"正在访问DAAD网站...")
            self.driver.get(self.base_url)
            time.sleep(self.delay)
            
            # 处理cookie弹窗（如果存在）
            try:
                # 尝试查找并点击cookie接受按钮
                cookie_selectors = [
                    "button[id*='accept']",
                    "button[class*='accept']",
                    "button[class*='cookie']",
                    ".cookie-accept",
                    ".accept-all",
                    "button:contains('Accept')",
                    "button:contains('Akzeptieren')"
                ]
                
                for selector in cookie_selectors:
                    try:
                        cookie_buttons = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if cookie_buttons:
                            # 使用JavaScript点击，避免元素被遮挡
                            self.driver.execute_script("arguments[0].click();", cookie_buttons[0])
                            print("已处理cookie弹窗")
                            time.sleep(1)
                            break
                    except:
                        continue
                
                # 也尝试通过文本查找
                try:
                    accept_buttons = self.driver.find_elements(By.XPATH, 
                        "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept')]")
                    if accept_buttons:
                        self.driver.execute_script("arguments[0].click();", accept_buttons[0])
                        print("已处理cookie弹窗（通过文本查找）")
                        time.sleep(1)
                except:
                    pass
            except Exception as e:
                print(f"处理cookie弹窗时出错（可能不存在）: {str(e)}")
            
            # 查找搜索表单元素
            try:
                # DAAD网站只有一个搜索框，需要将大学和项目名称组合输入
                search_query = f"{university} {program}"
                print(f"正在搜索: {search_query}")
                
                # 查找搜索框（可能是input[type='text']或input[type='search']）
                search_field = None
                
                # 尝试多种选择器
                search_selectors = [
                    "input[type='text']",
                    "input[type='search']",
                    "input[placeholder*='search']",
                    "input[placeholder*='Search']",
                    "input[name*='search']",
                    "input[id*='search']",
                    "input[aria-label*='search']"
                ]
                
                for selector in search_selectors:
                    try:
                        fields = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if fields:
                            search_field = fields[0]
                            break
                    except:
                        continue
                
                # 如果还是找不到，尝试所有input元素
                if search_field is None:
                    all_inputs = self.driver.find_elements(By.TAG_NAME, "input")
                    for inp in all_inputs:
                        inp_type = inp.get_attribute("type") or ""
                        if inp_type in ["text", "search", ""]:
                            search_field = inp
                            break
                
                if search_field is None:
                    print("错误: 无法找到搜索框")
                    return None
                
                # 输入搜索关键词
                print(f"正在输入搜索关键词: {search_query}")
                try:
                    # 尝试滚动到元素
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", search_field)
                    time.sleep(0.5)
                    # 使用JavaScript点击，避免元素被遮挡
                    self.driver.execute_script("arguments[0].click();", search_field)
                    time.sleep(0.5)
                    # 清空字段
                    search_field.clear()
                    time.sleep(0.3)
                    # 输入搜索关键词
                    self._human_type(search_field, search_query)
                    time.sleep(self.delay)
                except Exception as e:
                    print(f"输入搜索关键词时出错，尝试直接输入: {str(e)}")
                    # 如果点击失败，直接输入
                    try:
                        search_field.clear()
                        search_field.send_keys(search_query)
                        time.sleep(self.delay)
                    except Exception as e2:
                        print(f"直接输入也失败: {str(e2)}")
                        return None
                
                # 查找并点击搜索按钮
                search_button = None
                buttons = self.driver.find_elements(By.CSS_SELECTOR, "button[type='submit'], input[type='submit'], button")
                for button in buttons:
                    button_text = button.text.lower()
                    if any(keyword in button_text for keyword in ["search", "suchen", "find", "finden"]):
                        search_button = button
                        break
                
                if search_button is None and buttons:
                    search_button = buttons[0]
                
                if search_button:
                    print("正在点击搜索按钮...")
                    search_button.click()
                    time.sleep(self.delay * 2)  # 等待结果加载
                else:
                    # 尝试按Enter键
                    if search_field:
                        search_field.send_keys(Keys.RETURN)
                        time.sleep(self.delay * 2)
                
                # 等待搜索结果加载
                time.sleep(self.delay * 2)
                
                # 等待搜索结果加载
                time.sleep(self.delay * 2)
                
                # 检查当前URL，如果已经在搜索结果页面，需要查找详情链接
                current_url = self.driver.current_url
                if "/result/" in current_url:
                    print("当前在搜索结果页面，正在查找第一个程序详情链接...")
                    # 在搜索结果页面中查找包含"/detail/"的链接
                    result_links = []
                    all_links = self.driver.find_elements(By.TAG_NAME, "a")
                    
                    for link in all_links:
                        href = link.get_attribute("href")
                        if href:
                            href_lower = href.lower()
                            # 只选择包含"/detail/"的DAAD内部链接
                            if "/detail/" in href_lower and "daad.de" in href_lower and \
                               not any(external in href_lower for external in ["myguide", "utm_source", "facebook", "linkedin", "instagram", "youtube", "threads", "bluesky"]):
                                result_links.append(link)
                    
                    if not result_links:
                        print("未找到程序详情链接")
                        return None
                    
                    # 点击第一个详情链接
                    first_result = result_links[0]
                    result_url = first_result.get_attribute("href")
                    print(f"找到程序详情链接，正在访问: {result_url}")
                    self.driver.get(result_url)
                    time.sleep(self.delay * 2)
                else:
                    # 如果不在搜索结果页面，说明可能直接跳转到了详情页
                    print(f"当前URL: {current_url}")
                    if "/detail/" not in current_url:
                        print("警告: 未找到程序详情页面")
                        return None
                
                time.sleep(self.delay * 2)
                
                # 提取详细信息
                return self._extract_program_details()
                
            except (TimeoutException, NoSuchElementException) as e:
                print(f"搜索过程中出错: {str(e)}")
                return None
                
        except Exception as e:
            print(f"DAAD爬虫错误: {str(e)}")
            return None
    
    def _extract_program_details(self) -> Dict[str, Any]:
        """
        从详情页面提取项目信息
        
        Returns:
            包含项目信息的字典
        """
        details = {
            "University": "",
            "Program": "",
            "Degree": "",
            "Teaching language": "English",  # 默认值，因为这是国际项目数据库
            "Application deadline": "",
            "requirements": {
                "Description/content": ""
            }
        }
        
        try:
            # 等待页面加载
            time.sleep(self.delay)
            
            # 提取页面标题（通常包含项目名称）
            try:
                page_title = self.driver.title
                if page_title:
                    # 从标题中提取项目名称（通常在"-"之前）
                    if " - " in page_title:
                        details["Program"] = page_title.split(" - ")[0].strip()
                    else:
                        details["Program"] = page_title.strip()
            except:
                pass
            
            # 尝试从h1或h2标签提取项目名称
            if not details["Program"]:
                try:
                    h1_elements = self.driver.find_elements(By.TAG_NAME, "h1")
                    if h1_elements:
                        details["Program"] = h1_elements[0].text.strip()
                except:
                    pass
            
            # 提取大学名称（通常在Contact部分或页面标题中）
            try:
                # 查找包含"University"或"Technical University"的元素
                university_elements = self.driver.find_elements(By.XPATH, 
                    "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'technical university') or contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'university of')]")
                if university_elements:
                    for elem in university_elements:
                        text = elem.text.strip()
                        if "university" in text.lower() and len(text) < 100:
                            details["University"] = text
                            break
            except:
                pass
            
            # 从项目名称中提取学位信息
            if details["Program"]:
                program_lower = details["Program"].lower()
                if "master" in program_lower or "msc" in program_lower or "m.sc" in program_lower:
                    details["Degree"] = "Master of Science"
                elif "bachelor" in program_lower or "bsc" in program_lower or "b.sc" in program_lower:
                    details["Degree"] = "Bachelor of Science"
                elif "phd" in program_lower or "doctorate" in program_lower:
                    details["Degree"] = "PhD"
            
            # 点击"Requirements / Registration"标签页以获取入学要求
            try:
                print("正在查找并点击'Requirements / Registration'标签页...")
                # 尝试多种方式查找标签页
                req_tab_selector = None
                
                # 方法1: 通过文本查找（使用XPath，但先获取选择器字符串）
                try:
                    req_tabs = self.driver.find_elements(By.XPATH, 
                        "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'requirement') or contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'registration')]")
                    if req_tabs:
                        # 使用JavaScript点击，避免stale element reference
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", req_tabs[0])
                        time.sleep(0.5)
                        self.driver.execute_script("arguments[0].click();", req_tabs[0])
                        print("已点击'Requirements / Registration'标签页（方法1）")
                        time.sleep(self.delay * 2)
                        req_tab_selector = "found"
                except Exception as e1:
                    # 方法2: 通过href查找
                    try:
                        req_tabs = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='registration'], a[href*='requirement']")
                        if req_tabs:
                            self.driver.execute_script("arguments[0].scrollIntoView(true);", req_tabs[0])
                            time.sleep(0.5)
                            self.driver.execute_script("arguments[0].click();", req_tabs[0])
                            print("已点击'Requirements / Registration'标签页（方法2）")
                            time.sleep(self.delay * 2)
                            req_tab_selector = "found"
                    except Exception as e2:
                        # 方法3: 查找所有标签页
                        try:
                            all_tabs = self.driver.find_elements(By.CSS_SELECTOR, "a[role='tab'], .tab, [role='tab']")
                            for tab in all_tabs:
                                try:
                                    tab_text = tab.text.lower()
                                    if "requirement" in tab_text or "registration" in tab_text:
                                        self.driver.execute_script("arguments[0].scrollIntoView(true);", tab)
                                        time.sleep(0.5)
                                        self.driver.execute_script("arguments[0].click();", tab)
                                        print("已点击'Requirements / Registration'标签页（方法3）")
                                        time.sleep(self.delay * 2)
                                        req_tab_selector = "found"
                                        break
                                except:
                                    continue
                        except:
                            pass
                
                if req_tab_selector is None:
                    print("警告: 无法找到'Requirements / Registration'标签页")
            except Exception as e:
                print(f"点击标签页时出错: {str(e)}")
            
            # 提取入学要求信息
            try:
                # 等待标签页内容加载
                time.sleep(2)
                
                # 方法1: 直接查找包含"Bachelor"的ul元素（requirements通常在ul中）
                req_content = None
                try:
                    # 查找所有ul元素
                    all_uls = self.driver.find_elements(By.TAG_NAME, "ul")
                    for ul in all_uls:
                        ul_text = ul.text
                        # 检查是否包含入学要求相关的关键词
                        if "Bachelor" in ul_text and "Applicant" in ul_text:
                            # 进一步验证：检查是否包含多个列表项
                            list_items = ul.find_elements(By.TAG_NAME, "li")
                            if len(list_items) >= 3:  # 至少有3个列表项
                                req_content = ul
                                print(f"找到requirements ul，包含{len(list_items)}个列表项")
                                break
                except Exception as e:
                    print(f"方法1查找失败: {str(e)}")
                    req_content = None
                
                # 方法2: 如果没找到ul，查找包含"Bachelor"的tabpanel
                if req_content is None:
                    try:
                        all_panels = self.driver.find_elements(By.CSS_SELECTOR, "[role='tabpanel']")
                        for panel in all_panels:
                            panel_text = panel.text
                            # 检查是否包含入学要求相关的多个关键词
                            if "Bachelor" in panel_text and "Applicant" in panel_text and \
                               ("GRE" in panel_text or "language" in panel_text.lower() or "test" in panel_text.lower()):
                                req_content = panel
                                print("找到requirements tabpanel")
                                break
                    except Exception as e:
                        print(f"方法2查找失败: {str(e)}")
                        pass
                
                # 无论是否找到req_content，都尝试直接提取包含"Bachelor"的列表项
                requirements_text = []
                
                # 直接查找所有包含"Bachelor"或"Applicant"的li元素
                try:
                    req_li_elements = self.driver.find_elements(By.XPATH, 
                        "//li[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'bachelor') or contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'applicant')]")
                    for li in req_li_elements:
                        text = li.text.strip()
                        if text and len(text) > 10:
                            requirements_text.append(text)
                    print(f"找到{len(requirements_text)}个requirements列表项")
                except Exception as e:
                    print(f"直接查找li元素失败: {str(e)}")
                
                # 如果找到了req_content，也从中提取
                if req_content:
                    # 提取列表项
                    try:
                        list_items = req_content.find_elements(By.TAG_NAME, "li")
                        for item in list_items:
                            text = item.text.strip()
                            if text and len(text) > 10 and \
                               not any(skip in text.lower() for skip in ["home", "back", "previous", "next", "pagination", "services", "more info"]):
                                if text not in requirements_text:
                                    requirements_text.append(text)
                    except:
                        pass
                    
                    # 提取段落
                    try:
                        paragraphs = req_content.find_elements(By.TAG_NAME, "p")
                        for p in paragraphs:
                            text = p.text.strip()
                            if text and len(text) > 20 and \
                               "http" not in text.lower() and \
                               not any(skip in text.lower() for skip in ["home", "back", "previous", "next", "pagination", "services", "more info", "cookie"]):
                                if text not in requirements_text:
                                    requirements_text.append(text)
                    except:
                        pass
                
                if requirements_text:
                    # 去重并合并
                    unique_text = []
                    seen = set()
                    for text in requirements_text:
                        text_lower = text.lower()
                        if text_lower not in seen:
                            seen.add(text_lower)
                            unique_text.append(text)
                    
                    # 检查提取的内容是否像真正的入学要求
                    # 如果包含太多导航文本，视为无效
                    combined_text = "\n".join(unique_text).lower()
                    navigation_keywords = ["services", "more info", "pagination", "home", "back", "previous", "next", "international programmes", "cookie"]
                    navigation_count = sum(1 for keyword in navigation_keywords if keyword in combined_text)
                    
                    # 如果导航关键词太多，或者没有包含入学要求相关的关键词，视为无效
                    if navigation_count > 2 or not any(keyword in combined_text for keyword in ["bachelor", "applicant", "requirement", "degree", "language"]):
                        print("警告: 提取的内容看起来不像入学要求，视为无法找到")
                        details["requirements"]["Description/content"] = "无法找到"
                    else:
                        details["requirements"]["Description/content"] = "\n".join(unique_text)
                        print(f"成功提取{len(unique_text)}条入学要求")
                else:
                    print("警告: 无法找到入学要求内容")
            except Exception as e:
                print(f"提取入学要求时出错: {str(e)}")
            
            # 如果描述为空，尝试从整个页面提取（但需要验证）
            if not details["requirements"]["Description/content"]:
                try:
                    page_text = self.driver.find_element(By.TAG_NAME, "body").text
                    if page_text:
                        page_text_lower = page_text.lower()
                        # 检查是否包含入学要求相关的关键词
                        if any(keyword in page_text_lower for keyword in ["bachelor", "applicant", "requirement", "degree", "language", "gre", "test"]):
                            # 检查是否包含太多导航文本
                            navigation_keywords = ["services", "more info", "pagination", "home", "back", "previous", "next"]
                            navigation_count = sum(1 for keyword in navigation_keywords if keyword in page_text_lower[:500])
                            if navigation_count <= 3:  # 允许少量导航文本
                                details["requirements"]["Description/content"] = page_text[:500].strip()
                except:
                    pass
            
            # 如果最终仍然无法找到入学要求，设置为"无法找到"
            if not details["requirements"]["Description/content"]:
                details["requirements"]["Description/content"] = "无法找到"
                print("无法找到入学要求，已设置为'无法找到'")
            
        except Exception as e:
            print(f"提取详情时出错: {str(e)}")
            # 如果出错，也设置为"无法找到"
            if not details["requirements"]["Description/content"]:
                details["requirements"]["Description/content"] = "无法找到"
        
        return details
    
    def close(self):
        """关闭浏览器"""
        if self.driver:
            self.driver.quit()
            self.driver = None


class JSONNormalizer:
    """JSON数据标准化器"""
    
    @staticmethod
    def normalize(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化数据格式
        
        Args:
            data: 原始数据字典
            
        Returns:
            标准化后的JSON字典
        """
        # 处理requirements字段（可能是字符串或字典）
        requirements_content = ""
        if "requirements" in data:
            if isinstance(data["requirements"], str):
                requirements_content = data["requirements"].strip()
            elif isinstance(data["requirements"], dict):
                requirements_content = data["requirements"].get("Description/content", "").strip()
        
        # 如果requirements内容为空，设置为"无法找到"
        if not requirements_content:
            requirements_content = "无法找到"
        
        normalized = {
            "University": data.get("University", "").strip(),
            "Program": data.get("Program", "").strip(),
            "Degree": data.get("Degree", "").strip(),
            "Teaching language": data.get("Teaching language", "English").strip(),
            "Application deadline": data.get("Application deadline", "").strip(),
            "requirements": {
                "Description/content": requirements_content
            }
        }
        
        # 清理空值
        if not normalized["Teaching language"]:
            normalized["Teaching language"] = "English"
        
        # 如果requirements为空，设置为"无法找到"
        if not normalized["requirements"]["Description/content"]:
            normalized["requirements"]["Description/content"] = "无法找到"
        
        return normalized
    
    @staticmethod
    def merge(csv_data: Dict[str, Any], daad_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        合并CSV和DAAD数据，DAAD数据优先
        
        Args:
            csv_data: CSV数据
            daad_data: DAAD数据（可选）
            
        Returns:
            合并后的标准化数据
        """
        # 从CSV数据开始
        merged = csv_data.copy()
        
        # 如果DAAD数据存在，用它覆盖CSV数据
        if daad_data:
            for key in ["University", "Program", "Degree", "Teaching language", "Application deadline"]:
                if daad_data.get(key):
                    merged[key] = daad_data[key]
            
            # 合并requirements
            if "requirements" in daad_data:
                if isinstance(daad_data["requirements"], dict):
                    merged["requirements"] = daad_data["requirements"]
                else:
                    merged["requirements"] = {"Description/content": str(daad_data["requirements"])}
        
        return JSONNormalizer.normalize(merged)


class AdmissionRequirementsLoader:
    """入学要求加载器 - 主类"""
    
    def __init__(self, csv_path: str, use_daad: bool = True, headless: bool = False):
        """
        初始化加载器
        
        Args:
            csv_path: CSV数据库路径
            use_daad: 是否使用DAAD爬虫
            headless: DAAD爬虫是否使用无头模式
        """
        self.csv_loader = CSVLoader(csv_path)
        self.use_daad = use_daad
        self.daad_scraper = DAADScraper(headless=headless) if use_daad else None
        self.normalizer = JSONNormalizer()
    
    def load_requirements(self, university: str, program: str, auto_save: bool = True) -> Dict[str, Any]:
        """
        加载指定大学和项目的入学要求
        
        Args:
            university: 大学名称
            program: 项目名称
            auto_save: 是否在成功从DAAD获取数据后自动保存到CSV（默认True）
            
        Returns:
            标准化的JSON字典
        """
        # 加载CSV数据
        self.csv_loader.load()
        csv_data = self.csv_loader.find_program(university, program)
        
        if not csv_data:
            csv_data = {
                "University": university,
                "Program": program,
                "Degree": "",
                "Teaching language": "English",
                "Application deadline": "",
                "requirements": {"Description/content": ""}
            }
        
        # 尝试从DAAD获取数据
        daad_data = None
        if self.use_daad and self.daad_scraper:
            try:
                print(f"正在从DAAD获取 {university} - {program} 的信息...")
                daad_data = self.daad_scraper.search_program(university, program)
            except Exception as e:
                print(f"DAAD爬取失败: {str(e)}")
                daad_data = None
        
        # 合并并标准化数据
        result = self.normalizer.merge(csv_data, daad_data)
        
        # 如果成功从DAAD获取了数据，且auto_save为True，则保存到CSV
        if auto_save and daad_data is not None:
            try:
                # 检查DAAD数据是否有效（不是"无法找到"）
                requirements_content = ""
                if isinstance(daad_data.get("requirements"), dict):
                    requirements_content = daad_data["requirements"].get("Description/content", "")
                elif isinstance(daad_data.get("requirements"), str):
                    requirements_content = daad_data["requirements"]
                
                # 只有当requirements不是"无法找到"时才保存
                if requirements_content and requirements_content != "无法找到":
                    print(f"正在将DAAD爬取的数据保存到CSV数据库...")
                    success = self.csv_loader.update_or_add_program(result)
                    if success:
                        print("✓ DAAD数据已成功保存到CSV数据库")
                    else:
                        print("✗ 保存DAAD数据到CSV数据库失败")
                else:
                    print("DAAD数据中的入学要求为'无法找到'，跳过保存到CSV")
            except Exception as e:
                print(f"保存DAAD数据到CSV时出错: {str(e)}")
        
        return result
    
    def close(self):
        """关闭资源"""
        if self.daad_scraper:
            self.daad_scraper.close()


def main():
    """主函数 - 用于测试"""
    # CSV数据库路径
    csv_path = "university_programs.csv"
    
    # 创建示例CSV文件（如果不存在）
    if not os.path.exists(csv_path):
        print("创建示例CSV文件...")
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["University", "Program", "Degree", "Teaching language", "Application deadline", "requirements"])
            writer.writerow([
                "Technical University of Munich",
                "Computer Science",
                "Master of Science",
                "English",
                "Winter semester: 31 May; Summer semester: 30 November",
                "Bachelor's degree in Computer Science or related field"
            ])
            writer.writerow([
                "Heidelberg University",
                "Physics",
                "Master of Science",
                "English",
                "Winter semester: 15 July",
                "Bachelor's degree in Physics or related field"
            ])
        print(f"示例CSV文件已创建: {csv_path}")
    
    # 初始化加载器
    loader = AdmissionRequirementsLoader(csv_path, use_daad=True, headless=False)
    
    # 测试用例
    test_cases = [
        ("Technical University of Munich", "Computer Science"),
        ("Heidelberg University", "Physics"),
    ]
    
    results = []
    
    try:
        for university, program in test_cases:
            print(f"\n{'='*60}")
            print(f"处理: {university} - {program}")
            print(f"{'='*60}")
            
            result = loader.load_requirements(university, program)
            results.append(result)
            
            # 打印结果
            print("\n提取的结果:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
            # 保存结果
            output_file = f"requirements_{university.replace(' ', '_')}_{program.replace(' ', '_')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {output_file}")
    
    finally:
        loader.close()


if __name__ == "__main__":
    main()

