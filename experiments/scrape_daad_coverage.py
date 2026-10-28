"""
批量爬取 38 个 school-program 组合的 DAAD 入学要求
基于当前 DAAD 页面结构重写提取逻辑：
  - 要求内容在 <p> 段落中，位于 "Academic admission requirements"
    和 "Language requirements" 等 heading 之下
  - 不依赖旧版 <li> 提取逻辑

用法：python scrape_daad_coverage.py
中断后可重新运行，自动断点续传（scrape_checkpoint.json）
"""

import json
import csv
import os
import time
import re
from typing import Dict, List, Tuple, Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
try:
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.service import Service
    WDM_OK = True
except ImportError:
    WDM_OK = False

# ── 配置 ───────────────────────────────────────────────────────────────────
ACCEPTANCE_JSON  = "acceptance.json"
OUTPUT_CSV       = "university_programs_daad_full.csv"
CHECKPOINT_FILE  = "scrape_checkpoint.json"
REPORT_JSON      = "results/daad_coverage_report.json"
DELAY            = 2.5   # 操作间隔（秒）
REQUEST_DELAY    = 4     # 爬取间隔（秒）
DAAD_BASE        = "https://www2.daad.de/deutschland/studienangebote/international-programmes/en/"

CSV_FIELDNAMES = ["University", "Program", "Degree", "Teaching language",
                  "Application deadline", "requirements"]


# ── 工具函数 ────────────────────────────────────────────────────────────────

def load_programs() -> List[Tuple[str, str]]:
    with open(ACCEPTANCE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    programs = set()
    for student, outcomes in data.items():
        for app in outcomes.get("accepted", []) + outcomes.get("rejected", []):
            programs.add((app["school"], app["major"]))
    return sorted(programs)


def load_checkpoint() -> Dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(cp: Dict):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(cp, f, ensure_ascii=False, indent=2)


def append_to_csv(row: Dict):
    """追加一行到 CSV（已存在则追加，不存在则创建含表头的文件）"""
    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists or os.path.getsize(OUTPUT_CSV) == 0:
            writer.writeheader()
        req = row.get("requirements", "")
        if isinstance(req, dict):
            req = req.get("Description/content", "")
        writer.writerow({
            "University":           row.get("University", ""),
            "Program":              row.get("Program", ""),
            "Degree":               row.get("Degree", ""),
            "Teaching language":    row.get("Teaching language", ""),
            "Application deadline": row.get("Application deadline", ""),
            "requirements":         req,
        })


def has_real_content(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and t not in ("无法找到", "Not available", "")


# ── 浏览器操作 ───────────────────────────────────────────────────────────────

def init_driver() -> webdriver.Chrome:
    opts = Options()
    # 不使用 headless，方便观察；改为 True 可隐藏窗口
    # opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    if WDM_OK:
        svc = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=svc, options=opts)
    else:
        driver = webdriver.Chrome(options=opts)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


def dismiss_cookies(driver: webdriver.Chrome):
    """
    处理 DAAD 网站 cookie 弹窗。
    DAAD 使用 CookieConsent 或自有弹窗，通常有 "Accept all" / "Alle akzeptieren" 按钮。
    每次导航后调用，已同意则秒过，有弹窗则自动点击。
    """
    # CSS selector 列表（从最具体到最宽泛）
    css_selectors = [
        # DAAD 特有
        "button.c-button--primary[data-testid*='cookie']",
        "button[data-consent*='accept']",
        ".cc-btn.cc-allow",
        ".cc-allow",
        # 通用
        "button[id*='accept']",
        "button[id*='consent']",
        "button[class*='accept']",
        "button[class*='consent']",
        ".cookie-accept",
        ".accept-all",
        "#cookieAccept",
        "#acceptCookies",
    ]
    for sel in css_selectors:
        try:
            btns = driver.find_elements(By.CSS_SELECTOR, sel)
            visible = [b for b in btns if b.is_displayed()]
            if visible:
                driver.execute_script("arguments[0].click();", visible[0])
                print("  [cookie] 已点击同意按钮")
                time.sleep(1)
                return
        except Exception:
            pass

    # XPath 兜底：匹配按钮文本含 accept / allow / akzeptieren / agree / alle
    try:
        btns = driver.find_elements(By.XPATH,
            "//button[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'accept')"
            " or contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'allow')"
            " or contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'akzeptieren')"
            " or contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'agree')"
            " or contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'alle')"
            "]")
        visible = [b for b in btns if b.is_displayed()]
        if visible:
            driver.execute_script("arguments[0].click();", visible[0])
            print("  [cookie] 已点击同意按钮（XPath兜底）")
            time.sleep(1)
    except Exception:
        pass


def search_daad(driver: webdriver.Chrome, university: str, program: str) -> Optional[str]:
    """
    在 DAAD 搜索页面查找程序，返回 detail URL（或 None）
    先收集所有 href，再直接用 driver.get() 导航，避免 stale element 问题
    """
    driver.get(DAAD_BASE)
    time.sleep(DELAY)
    dismiss_cookies(driver)

    query = f"{university} {program}"
    print(f"  搜索关键词: {query}")

    # 找搜索框
    search_field = None
    for sel in ["input[type='text']", "input[type='search']",
                 "input[placeholder*='search' i]", "input[name*='search' i]"]:
        try:
            fields = driver.find_elements(By.CSS_SELECTOR, sel)
            if fields:
                search_field = fields[0]
                break
        except Exception:
            pass

    if search_field is None:
        all_inputs = driver.find_elements(By.TAG_NAME, "input")
        visible = [i for i in all_inputs if i.is_displayed()]
        if visible:
            search_field = visible[0]

    if search_field is None:
        print("  ✗ 找不到搜索框")
        return None

    # 输入并搜索
    search_field.clear()
    for ch in query:
        search_field.send_keys(ch)
        time.sleep(0.05)
    time.sleep(0.5)
    search_field.send_keys(Keys.RETURN)
    time.sleep(DELAY * 2)

    # 当前页面如果已经是 detail 页
    if "/detail/" in driver.current_url:
        return driver.current_url

    # ── 用 JavaScript 一次性收集所有 href，避免循环中 stale element ─────────
    try:
        # 额外等待 JS 渲染
        time.sleep(3)
        all_hrefs: list = driver.execute_script(
            "return Array.from(document.querySelectorAll('a[href]'))"
            ".map(a => a.href).filter(h => h.includes('/detail/'));"
        )
        # 过滤掉社交媒体等干扰链接
        skip_keywords = ["facebook", "linkedin", "instagram", "youtube", "myguide", "utm_"]
        for href in all_hrefs:
            if not any(kw in href for kw in skip_keywords):
                print(f"  → 找到详情页: {href}")
                return href
    except Exception as e:
        print(f"  ✗ JS 收集 href 失败: {e}")

    print("  ✗ 未找到 detail 链接")
    return None


def extract_requirements(driver: webdriver.Chrome, detail_url: str) -> Dict:
    """
    从 DAAD detail 页面提取入学要求信息
    当前页面结构：要求文本在 <p> 段落中，位于相关 heading 之下
    """
    driver.get(detail_url)
    time.sleep(DELAY * 2)
    dismiss_cookies(driver)  # 详情页也可能弹出 cookie 弹窗

    result = {
        "University": "", "Program": "", "Degree": "",
        "Teaching language": "English", "Application deadline": "",
        "requirements": "",
    }

    # ── 基本字段 ────────────────────────────────────────────────────────────
    try:
        h1s = driver.find_elements(By.TAG_NAME, "h1")
        if h1s:
            result["Program"] = h1s[0].text.strip()
    except Exception:
        pass

    try:
        h2s = driver.find_elements(By.TAG_NAME, "h2")
        for h2 in h2s:
            t = h2.text.strip()
            if "university" in t.lower() and len(t) < 120:
                result["University"] = t
                break
    except Exception:
        pass

    # ── 点击 Requirements / Registration 标签页 ─────────────────────────────
    # 用 JS 找 href 再直接 driver.get()，完全避免 stale element
    try:
        tab_hrefs: list = driver.execute_script(
            "return Array.from(document.querySelectorAll('a[href]'))"
            ".map(a => a.href)"
            ".filter(h => h.match(/registration|requirement/i));"
        )
        if tab_hrefs:
            tab_url = tab_hrefs[0]
            driver.get(tab_url)
            time.sleep(DELAY)
        else:
            # 降级：找包含 requirement/registration 文本的 <a>，JS 点击
            driver.execute_script("""
                var links = document.querySelectorAll('a');
                for (var i = 0; i < links.length; i++) {
                    var t = links[i].textContent.toLowerCase();
                    if (t.includes('requirement') || t.includes('registration')) {
                        links[i].click();
                        break;
                    }
                }
            """)
            time.sleep(DELAY)
    except Exception:
        pass

    # ── 提取要求内容（当前页面用 <p> 段落） ─────────────────────────────────
    req_texts = []

    # 策略1：找 "Academic admission requirements" / "Language requirements" heading，
    #         提取后续 <p> 内容
    target_headings = [
        "academic admission requirements",
        "language requirements",
        "language requirement",
        "admission requirements",
        "entry requirements",
    ]
    for tag in ["h3", "h4", "h2", "strong", "b"]:
        try:
            elems = driver.find_elements(By.TAG_NAME, tag)
            for elem in elems:
                txt = elem.text.strip().lower()
                if any(kw in txt for kw in target_headings):
                    # 获取同级或子级段落
                    parent = elem.find_element(By.XPATH, "..")
                    paras = parent.find_elements(By.TAG_NAME, "p")
                    for p in paras:
                        t = p.text.strip()
                        if t and len(t) > 30:
                            req_texts.append(t)
        except Exception:
            pass

    # 策略2：tabpanel 中的所有 <p>（标签页点击后加载）
    if not req_texts:
        try:
            panels = driver.find_elements(By.CSS_SELECTOR, "[role='tabpanel']")
            for panel in panels:
                paras = panel.find_elements(By.TAG_NAME, "p")
                for p in paras:
                    t = p.text.strip()
                    if t and len(t) > 30:
                        req_texts.append(t)
        except Exception:
            pass

    # 策略3：全页面搜索含"Bachelor"/"applicant"/"GPA"/"IELTS"/"language"的 <p>
    if not req_texts:
        try:
            all_paras = driver.find_elements(By.TAG_NAME, "p")
            for p in all_paras:
                t = p.text.strip()
                t_lower = t.lower()
                if len(t) > 40 and any(kw in t_lower for kw in [
                    "bachelor", "applicant", "gpa", "ielts", "toefl",
                    "language", "degree", "admission", "requirement",
                ]):
                    if not any(skip in t_lower for skip in [
                        "cookie", "privacy", "navigation", "menu",
                    ]):
                        req_texts.append(t)
        except Exception:
            pass

    # 去重并合并
    seen, unique = set(), []
    for t in req_texts:
        key = t[:80].lower()
        if key not in seen:
            seen.add(key)
            unique.append(t)

    combined = "\n\n".join(unique[:20])  # 最多取 20 段

    # ── 其他字段（截止日期、学位等） ────────────────────────────────────────
    try:
        body_text = driver.find_element(By.TAG_NAME, "body").text
        # 截止日期
        dl_match = re.search(
            r'Application deadline[:\s]*([^\n]{10,80})', body_text, re.IGNORECASE)
        if dl_match:
            result["Application deadline"] = dl_match.group(1).strip()
        # 学位
        deg_match = re.search(
            r'Degree[:\s]*(Master[^\n]{0,50}|Bachelor[^\n]{0,50}|PhD[^\n]{0,30})',
            body_text, re.IGNORECASE)
        if deg_match:
            result["Degree"] = deg_match.group(1).strip()
        # 教学语言
        lang_match = re.search(
            r'Teaching language[:\s]*([^\n]{2,30})', body_text, re.IGNORECASE)
        if lang_match:
            result["Teaching language"] = lang_match.group(1).strip()
    except Exception:
        pass

    result["requirements"] = combined if has_real_content(combined) else "无法找到"
    return result


# ── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    programs = load_programs()
    total = len(programs)
    checkpoint = load_checkpoint()
    os.makedirs("results", exist_ok=True)

    # 初始化 CSV 表头
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDNAMES).writeheader()

    print(f"共 {total} 个 school-program 组合")
    print(f"输出 CSV : {OUTPUT_CSV}")
    print(f"覆盖报告 : {REPORT_JSON}")
    print("=" * 70)

    driver = init_driver()
    results_log = []
    n_success = n_empty = n_failed = 0

    # 统计断点中已完成的数量
    for key, status in checkpoint.items():
        if status == "success":
            n_success += 1
        elif status == "empty":
            n_empty += 1
        else:
            n_failed += 1

    try:
        for i, (school, program) in enumerate(programs):
            key = f"{school}|||{program}"
            tag = f"[{i+1}/{total}]"

            if key in checkpoint:
                print(f"{tag} 跳过（{checkpoint[key]}）: {school} — {program}")
                results_log.append({
                    "school": school, "program": program,
                    "status": checkpoint[key], "content_preview": "(断点恢复)",
                })
                continue

            print(f"\n{tag} {school}")
            print(f"     {program}")

            try:
                detail_url = search_daad(driver, school, program)
                if not detail_url:
                    raise ValueError("未找到 detail URL")

                row = extract_requirements(driver, detail_url)
                # 补充我们自己知道的 school/program（DAAD 提取的名称可能不同）
                if not row["University"]:
                    row["University"] = school
                if not row["Program"]:
                    row["Program"] = program

                content = row["requirements"]
                if has_real_content(content):
                    preview = content[:120].replace("\n", " ")
                    print(f"  ✓ 成功: {preview}…")
                    checkpoint[key] = "success"
                    n_success += 1
                    append_to_csv(row)
                    results_log.append({
                        "school": school, "program": program,
                        "status": "success",
                        "content_preview": preview,
                        "detail_url": detail_url,
                        "degree": row.get("Degree", ""),
                        "teaching_language": row.get("Teaching language", ""),
                        "deadline": row.get("Application deadline", ""),
                    })
                else:
                    print("  ⚠ 找到页面但要求内容为空")
                    checkpoint[key] = "empty"
                    n_empty += 1
                    results_log.append({
                        "school": school, "program": program,
                        "status": "empty", "detail_url": detail_url,
                        "content_preview": "",
                    })

            except Exception as e:
                print(f"  ✗ 失败: {e}")
                checkpoint[key] = "failed"
                n_failed += 1
                results_log.append({
                    "school": school, "program": program,
                    "status": "failed", "content_preview": str(e)[:100],
                })

            save_checkpoint(checkpoint)

            if i < total - 1:
                print(f"  等待 {REQUEST_DELAY}s …")
                time.sleep(REQUEST_DELAY)

    finally:
        driver.quit()

    # ── 统计报告 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"爬取完成！  ✓{n_success}  ⚠{n_empty}  ✗{n_failed}  / {total}")
    print(f"覆盖率: {n_success}/{total} = {n_success/total*100:.1f}%")

    report = {
        "total": total, "success": n_success,
        "empty": n_empty, "failed": n_failed,
        "coverage_rate_pct": round(n_success / total * 100, 1),
        "results": results_log,
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告: {REPORT_JSON}")

    print("\n✓ 成功项目：")
    for r in results_log:
        if r["status"] == "success":
            print(f"    {r['school']} — {r['program']}")
    print("\n⚠/✗ 未成功：")
    for r in results_log:
        if r["status"] != "success":
            print(f"    [{r['status']}] {r['school']} — {r['program']}")


if __name__ == "__main__":
    # 断点续传：保留旧 checkpoint；如需全部重跑，手动删除 scrape_checkpoint.json
    main()
