"""
variation/renderer.py
변형문제 PDF 렌더링 (Jinja2 + WeasyPrint)
"""
import os
from typing import List
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

# 템플릿/스태틱 경로
# variation/ 폴더의 부모 = 워크북 루트 (기존 template.html, secret_note_*.html과 같은 위치)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = BASE_DIR  # 루트에서 variation.html, variation_b.html 찾기
STATIC_DIR = os.path.join(BASE_DIR, "static")
LOGO_PATH = os.path.join(STATIC_DIR, "logo2.png")


# ============ 위치 마커 렌더링 (유형 B) ============
def render_marks(text: str) -> str:
    """<MARK1>...<MARK5>를 동그라미 숫자로 변환"""
    circles = ['①', '②', '③', '④', '⑤']
    for i in range(1, 6):
        text = text.replace(f"<MARK{i}>", f'<span class="pos-mark">{circles[i-1]}</span>')
    return text


# ============ Q5 빈칸 / Q4 빈칸 렌더링 (유형 A) ============
def convert_chunks_a(chunks):
    """chunks의 <BLANK_A>, <BLANK_B>, <CORE_BLANK>를 시각 마크로 변환"""
    out = []
    for label, text in chunks:
        text = text.replace("<BLANK_A>", '<span class="blank-mark">(A)</span>')
        text = text.replace("<BLANK_B>", '<span class="blank-mark">(B)</span>')
        text = text.replace("<CORE_BLANK>", '<span class="core-blank-inline">______________</span>')
        out.append([label, text])
    return out


def convert_lead_a(lead: str) -> str:
    """lead의 <CORE_BLANK> 변환"""
    return lead.replace("<CORE_BLANK>", '<span class="core-blank-inline">______________</span>')


# ============ 데이터 정규화 (템플릿이 기대하는 형태로) ============
def prepare_a_passage(data: dict, label: str) -> dict:
    """유형 A 데이터를 템플릿 입력 형식으로"""
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
    """유형 B 데이터를 템플릿 입력 형식으로"""
    return {
        "label": label,
        "data": {
            **data,
            "passage_rendered": render_marks(data["passage_with_marks"]),
        },
    }


# ============ Jinja2 환경 ============
def get_jinja_env():
    return Environment(loader=FileSystemLoader(TEMPLATE_DIR))


# ============ 메인 렌더링 함수 ============
def render_variation_pdf(
    a_items: List[dict],
    b_items: List[dict],
    mode: str = "by-type",
    school_name: str = "레벨미업학원",
) -> bytes:
    """
    변형문제 PDF 생성
    
    a_items: [{label: str, data: dict}, ...] (유형 A 데이터)
    b_items: [{label: str, data: dict}, ...] (유형 B 데이터)
    mode: 'by-type' (AAAA...BBBB...) or 'by-passage' (AB AB AB)
    
    by-type: A 유형 모든 지문 → B 유형 모든 지문 (한 PDF에)
    by-passage: 지문1 A+B → 지문2 A+B (한 PDF에)
    """
    env = get_jinja_env()
    
    logo_url = f"file://{LOGO_PATH}" if os.path.exists(LOGO_PATH) else None
    
    # 두 템플릿 로드
    tmpl_a = env.get_template("variation.html")
    tmpl_b = env.get_template("variation_b.html")
    
    if mode == "by-type":
        # 유형 A 전체 → 유형 B 전체
        html_parts = []
        if a_items:
            html_a = tmpl_a.render(
                passages=a_items,
                school_name=school_name,
                logo_url=logo_url,
            )
            html_parts.append(html_a)
        if b_items:
            html_b = tmpl_b.render(
                passages=b_items,
                school_name=school_name,
                logo_url=logo_url,
            )
            html_parts.append(html_b)
        
        # HTML들을 한 PDF에 결합 - WeasyPrint는 각각 따로 만든 후 합칠 수 있지만,
        # 가장 간단한 방법: 한 HTML로 결합
        combined_html = _combine_htmls(html_parts)
        return HTML(string=combined_html).write_pdf()
    
    elif mode == "by-passage":
        # 지문별로 묶기 — 각 지문에 대해 A+B를 한 묶음
        # a_items와 b_items의 label로 매칭
        a_by_label = {item["label"]: item for item in a_items}
        b_by_label = {item["label"]: item for item in b_items}
        all_labels = []
        seen = set()
        for it in a_items + b_items:
            if it["label"] not in seen:
                all_labels.append(it["label"])
                seen.add(it["label"])
        
        html_parts = []
        for label in all_labels:
            if label in a_by_label:
                html_parts.append(tmpl_a.render(
                    passages=[a_by_label[label]],
                    school_name=school_name,
                    logo_url=logo_url,
                ))
            if label in b_by_label:
                html_parts.append(tmpl_b.render(
                    passages=[b_by_label[label]],
                    school_name=school_name,
                    logo_url=logo_url,
                ))
        
        combined_html = _combine_htmls(html_parts)
        return HTML(string=combined_html).write_pdf()
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _combine_htmls(html_parts: List[str]) -> str:
    """여러 HTML 문자열을 하나로 결합 (body 내용만 추출 후 합침)"""
    if not html_parts:
        return "<html><body></body></html>"
    if len(html_parts) == 1:
        return html_parts[0]
    
    # 첫 번째 HTML의 <head>를 보존하고, body 내용만 추가
    import re
    
    # 첫 번째 HTML에서 <head>...</head> 추출
    head_match = re.search(r"<head[^>]*>(.*?)</head>", html_parts[0], re.DOTALL)
    head_content = head_match.group(1) if head_match else ""
    
    # 각 HTML에서 body 내용 추출
    body_contents = []
    for h in html_parts:
        m = re.search(r"<body[^>]*>(.*?)</body>", h, re.DOTALL)
        if m:
            body_contents.append(m.group(1))
        else:
            body_contents.append(h)
    
    combined = (
        '<!DOCTYPE html><html lang="ko"><head>'
        + head_content
        + "</head><body>"
        + "\n".join(body_contents)
        + "</body></html>"
    )
    return combined
