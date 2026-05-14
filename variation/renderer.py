"""
variation/renderer.py
변형문제 HTML 렌더링 - 두 템플릿 스타일 보존 결합

핵심 설계:
- A 템플릿과 B 템플릿의 <style>을 모두 보존
- 각 템플릿 출력을 <section class="variation-section">으로 감싸서 격리
- 답지(.ans-block)는 page-break-inside: avoid로 페이지 분리 방지
- 인쇄 안내(.print-hint)는 화면에만, 인쇄 시 완전 숨김
"""
import os
import re
import base64
from typing import List
from jinja2 import Environment, FileSystemLoader

# 템플릿/스태틱 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = BASE_DIR
STATIC_DIR = os.path.join(BASE_DIR, "static")
LOGO_PATH = os.path.join(STATIC_DIR, "logo2.png")


# ============ 로고 → data URI ============
_logo_data_uri_cache = None

def get_logo_data_uri() -> str:
    global _logo_data_uri_cache
    if _logo_data_uri_cache is not None:
        return _logo_data_uri_cache
    if not os.path.exists(LOGO_PATH):
        _logo_data_uri_cache = ""
        return ""
    try:
        with open(LOGO_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        _logo_data_uri_cache = f"data:image/png;base64,{b64}"
        return _logo_data_uri_cache
    except Exception:
        _logo_data_uri_cache = ""
        return ""


# ============ 마커 렌더링 ============
def render_marks(text: str) -> str:
    circles = ['①', '②', '③', '④', '⑤']
    for i in range(1, 6):
        text = text.replace(f"<MARK{i}>", f'<span class="pos-mark">{circles[i-1]}</span>')
    return text


def convert_chunks_a(chunks):
    out = []
    for label, text in chunks:
        text = text.replace("<BLANK_A>", '<span class="blank-mark">(A)</span>')
        text = text.replace("<BLANK_B>", '<span class="blank-mark">(B)</span>')
        text = text.replace("<CORE_BLANK>", '<span class="core-blank-inline">______________</span>')
        out.append([label, text])
    return out


def convert_lead_a(lead: str) -> str:
    return lead.replace("<CORE_BLANK>", '<span class="core-blank-inline">______________</span>')


# ============ 데이터 정규화 ============
def prepare_a_passage(data: dict, label: str) -> dict:
    n_false = sum(1 for _, _, ok in data["statements"] if not ok)
    return {
        "label": label,
        "data": {
            "lead": convert_lead_a(data["lead"]),
            "chunks": convert_chunks_a(data["chunks"]),
            "topic_options": data["topic_options"],
            "topic_correct": data["topic_correct"],
            "order_options": data["order_options"],
            "order_correct": data["order_correct"],
            "statements": [list(s) for s in data["statements"]],
            "statements_kr": data.get("statements_kr", []),
            "mismatch_count": data.get("mismatch_count", n_false),
            "blank_A": data["blank_A"],
            "blank_B": data["blank_B"],
            "bogi": data["bogi"],
            "topic_explain": data.get("topic_explain", ""),
            "order_explain": data.get("order_explain", ""),
            "blank_explain_A": data.get("blank_explain_A", ""),
            "blank_explain_B": data.get("blank_explain_B", ""),
            "core_blank_target": data.get("core_blank_target"),
            "core_blank_options": data.get("core_blank_options"),
            "core_blank_correct": data.get("core_blank_correct"),
            "core_blank_explain": data.get("core_blank_explain", ""),
        },
    }


def prepare_b_passage(data: dict, label: str) -> dict:
    return {
        "label": label,
        "data": {
            **data,
            "passage_rendered": render_marks(data["passage_with_marks"]),
        },
    }


def get_jinja_env():
    return Environment(loader=FileSystemLoader(TEMPLATE_DIR))


# ============ 인쇄 안내 (헤더 위 띄움) ============
PRINT_HINT_HTML = """
<div id="print-hint-banner" class="print-hint" style="position:fixed;top:0;left:0;right:0;background:#fef3c7;color:#92400e;padding:10px 20px;text-align:center;font-family:'Malgun Gothic',sans-serif;font-size:13px;border-bottom:2px solid #fbbf24;z-index:99999;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
  💡 인쇄/PDF 저장: <b>Ctrl+P</b> (Mac: Cmd+P) → '대상'에서 PDF로 저장
  <button onclick="document.getElementById('print-hint-banner').style.display='none';document.body.style.paddingTop='0';" style="margin-left:15px;padding:3px 10px;background:#92400e;color:white;border:none;border-radius:4px;cursor:pointer;font-weight:600;">✕ 닫기</button>
</div>
"""

PRINT_HINT_STYLE = """
<style id="print-hint-style">
  body { padding-top: 45px; }
  @media print {
    #print-hint-banner { display: none !important; }
    body { padding-top: 0 !important; }
  }
  
  /* ★ footer-logo는 WeasyPrint 전용 - 브라우저에서는 화면에서만 숨김 (인쇄 시 보임 X) */
  .footer-logo {
    display: none !important;
  }
  
  /* ★ 빈칸 underline 명확하게 (Q3 핵심빈칸 등) */
  .core-blank-inline {
    border-bottom: 2px solid #6A1B9A !important;
    min-width: 120px !important;
    display: inline-block !important;
    font-size: 0 !important;  /* '_____' 문자 안 보이게 - 밑줄만 보임 */
    height: 1.2em;
    vertical-align: middle;
  }
  
  /* 일반 빈칸 (밑줄 표시) */
  .blank-line, .blank-underline {
    display: inline-block;
    min-width: 100px;
    border-bottom: 2px solid #333;
  }
  
  /* 답지 페이지 처리 */
  .ans-block, .answer-block, .ans-row, .ans-item, .answer-section {
    page-break-inside: avoid;
  }
  /* 새 변형문제 섹션은 새 페이지에서 시작 */
  .variation-section + .variation-section { page-break-before: always; }
  
  /* q-page는 페이지 단위 (원래 템플릿 정의) - print 시에만 페이지 break */
  .q-page {
    page-break-after: auto;  /* 화면에서는 break 강제 X */
  }
  @media print {
    .q-page { page-break-after: always; }
  }
</style>
"""


# ============ HTML 파싱 헬퍼 ============
def _extract_head_styles(html: str) -> str:
    """HTML <head>의 <style>, <link> 태그만 모두 추출"""
    head_match = re.search(r"<head[^>]*>(.*?)</head>", html, re.DOTALL | re.IGNORECASE)
    if not head_match:
        return ""
    head = head_match.group(1)
    
    # <style>...</style> 모두 추출
    styles = re.findall(r"<style[^>]*>.*?</style>", head, re.DOTALL | re.IGNORECASE)
    # <link rel="stylesheet" ...> 추출
    links = re.findall(r'<link[^>]*rel=["\']stylesheet["\'][^>]*/?>', head, re.IGNORECASE)
    # <meta charset>, <meta viewport> 등
    metas = re.findall(r'<meta[^>]*/?>', head, re.IGNORECASE)
    
    return "\n".join(metas + links + styles)


def _extract_body_content(html: str) -> str:
    """<body>...</body> 내부만 추출"""
    body_match = re.search(r"<body[^>]*>(.*?)</body>", html, re.DOTALL | re.IGNORECASE)
    if body_match:
        return body_match.group(1)
    # body 태그 없으면 전체 반환
    return html


# ============ 메인 렌더링 ============
def render_variation_html(
    a_items: List[dict],
    b_items: List[dict],
    mode: str = "by-type",
    school_name: str = "레벨미업학원",
) -> str:
    """
    변형문제 HTML 페이지 생성
    
    두 템플릿(A=보라 테마, B=청록 테마)의 <head>를 모두 합쳐서
    스타일 충돌 없이 한 페이지에 모두 표시.
    """
    env = get_jinja_env()
    logo_url = get_logo_data_uri()
    
    try:
        tmpl_a = env.get_template("variation.html")
    except Exception as e:
        raise RuntimeError(f"variation.html 템플릿 로드 실패: {e}")
    try:
        tmpl_b = env.get_template("variation_b.html")
    except Exception as e:
        raise RuntimeError(f"variation_b.html 템플릿 로드 실패: {e}")
    
    # 각 섹션(템플릿 출력)을 따로 저장 - 디버깅 + 페이지 분리 위해
    sections = []
    
    if mode == "by-type":
        if a_items:
            html_a = tmpl_a.render(
                passages=a_items, school_name=school_name, logo_url=logo_url
            )
            sections.append(("A", html_a))
        if b_items:
            html_b = tmpl_b.render(
                passages=b_items, school_name=school_name, logo_url=logo_url
            )
            sections.append(("B", html_b))
    elif mode == "by-passage":
        a_by_label = {item["label"]: item for item in a_items}
        b_by_label = {item["label"]: item for item in b_items}
        all_labels = []
        seen = set()
        for it in a_items + b_items:
            if it["label"] not in seen:
                all_labels.append(it["label"])
                seen.add(it["label"])
        for label in all_labels:
            if label in a_by_label:
                sections.append(("A", tmpl_a.render(
                    passages=[a_by_label[label]],
                    school_name=school_name, logo_url=logo_url
                )))
            if label in b_by_label:
                sections.append(("B", tmpl_b.render(
                    passages=[b_by_label[label]],
                    school_name=school_name, logo_url=logo_url
                )))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    if not sections:
        return "<html><body><p>변형문제 데이터가 없습니다.</p></body></html>"
    
    return _build_final_html(sections)


def _build_final_html(sections: List[tuple]) -> str:
    """
    각 섹션의 <head> 스타일을 모두 합치고, <body>는 .variation-section으로 감싸서 결합.
    
    sections: [("A", html_a), ("B", html_b), ...]
    """
    # 1) 모든 섹션의 head 스타일 + meta + link 추출
    all_head_parts = []
    seen_styles = set()
    for type_name, html in sections:
        head_extracted = _extract_head_styles(html)
        # 중복 제거 (똑같은 <style> 블록이 여러 번 들어가지 않게)
        for tag in re.findall(r"(<(?:style|link|meta)[^>]*(?:>.*?</style>|/?>))", head_extracted, re.DOTALL | re.IGNORECASE):
            if tag not in seen_styles:
                seen_styles.add(tag)
                all_head_parts.append(tag)
    
    # 2) 각 섹션 body 내용을 .variation-section[data-type="A"]로 감싸기
    body_parts = []
    for type_name, html in sections:
        body_content = _extract_body_content(html)
        body_parts.append(
            f'<section class="variation-section" data-type="{type_name}">\n'
            f'{body_content}\n'
            f'</section>'
        )
    
    # 3) 최종 HTML
    final = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>변형문제 - 레벨미업학원</title>
{chr(10).join(all_head_parts)}
{PRINT_HINT_STYLE}
</head>
<body>
{PRINT_HINT_HTML}
{chr(10).join(body_parts)}
</body>
</html>"""
    return final
