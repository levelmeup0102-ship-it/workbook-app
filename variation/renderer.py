"""
variation/renderer.py
변형문제 HTML 렌더링 (WeasyPrint 없이, Jinja2만)

비밀노트와 동일 패턴 — HTML 페이지를 만들어서
사용자가 브라우저에서 Ctrl+P로 인쇄/PDF 저장
"""
import os
import base64
from typing import List
from jinja2 import Environment, FileSystemLoader

# 템플릿/스태틱 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = BASE_DIR  # 루트에서 variation.html, variation_b.html 찾기
STATIC_DIR = os.path.join(BASE_DIR, "static")
LOGO_PATH = os.path.join(STATIC_DIR, "logo2.png")


# ============ 로고 → data URI (HTML에 인라인) ============
_logo_data_uri_cache = None

def get_logo_data_uri() -> str:
    """로고 파일을 base64 data URI로 변환 (HTML에 인라인 임베드)"""
    global _logo_data_uri_cache
    if _logo_data_uri_cache:
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
        return ""


# ============ 위치 마커 렌더링 (유형 B) ============
def render_marks(text: str) -> str:
    """<MARK1>...<MARK5>를 동그라미 숫자로"""
    circles = ['①', '②', '③', '④', '⑤']
    for i in range(1, 6):
        text = text.replace(f"<MARK{i}>", f'<span class="pos-mark">{circles[i-1]}</span>')
    return text


# ============ Q5 빈칸 / Q4 빈칸 렌더링 (유형 A) ============
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
    """유형 A → 템플릿 입력 형식"""
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
    """유형 B → 템플릿 입력 형식"""
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


# ============ 인쇄 안내 헤더 ============
PRINT_HEADER = """
<!-- 인쇄 안내 (화면에만 보임, 인쇄 시 숨김) -->
<div class="print-hint" style="position:fixed;top:0;left:0;right:0;background:#fef3c7;color:#92400e;padding:10px 20px;text-align:center;font-family:'Malgun Gothic',sans-serif;font-size:13px;border-bottom:2px solid #fbbf24;z-index:9999;">
  💡 인쇄/PDF 저장: 키보드 <b>Ctrl+P</b> (Mac: <b>Cmd+P</b>) → '대상'에서 PDF로 저장 선택
  <button onclick="this.parentElement.style.display='none'" style="margin-left:15px;padding:3px 10px;background:#92400e;color:white;border:none;border-radius:4px;cursor:pointer;">닫기</button>
</div>
<style>
  body { padding-top: 50px !important; }
  @media print {
    .print-hint { display: none !important; }
    body { padding-top: 0 !important; }
  }
</style>
"""


# ============ 메인 렌더링 ============
def render_variation_html(
    a_items: List[dict],
    b_items: List[dict],
    mode: str = "by-type",
    school_name: str = "레벨미업학원",
) -> str:
    """
    변형문제 HTML 페이지 생성
    
    a_items, b_items: prepare_a_passage / prepare_b_passage로 가공된 데이터 리스트
    mode: 'by-type' (A 전체 → B 전체) or 'by-passage' (지문별 A+B 묶음)
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
    
    html_parts = []
    
    if mode == "by-type":
        if a_items:
            html_parts.append(tmpl_a.render(
                passages=a_items, school_name=school_name, logo_url=logo_url
            ))
        if b_items:
            html_parts.append(tmpl_b.render(
                passages=b_items, school_name=school_name, logo_url=logo_url
            ))
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
                html_parts.append(tmpl_a.render(
                    passages=[a_by_label[label]],
                    school_name=school_name, logo_url=logo_url
                ))
            if label in b_by_label:
                html_parts.append(tmpl_b.render(
                    passages=[b_by_label[label]],
                    school_name=school_name, logo_url=logo_url
                ))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    combined = _combine_htmls(html_parts)
    # 인쇄 안내 추가
    combined = combined.replace("<body>", "<body>\n" + PRINT_HEADER, 1)
    return combined


def _combine_htmls(html_parts: List[str]) -> str:
    """여러 HTML을 하나로 결합 (head는 첫 번째 것 사용, body는 모두 합침)"""
    if not html_parts:
        return "<html><body><p>변형문제 데이터가 없습니다.</p></body></html>"
    if len(html_parts) == 1:
        return html_parts[0]
    
    import re
    head_match = re.search(r"<head[^>]*>(.*?)</head>", html_parts[0], re.DOTALL)
    head_content = head_match.group(1) if head_match else ""
    
    body_contents = []
    for h in html_parts:
        m = re.search(r"<body[^>]*>(.*?)</body>", h, re.DOTALL)
        if m:
            body_contents.append(m.group(1))
        else:
            body_contents.append(h)
    
    return (
        '<!DOCTYPE html><html lang="ko"><head>'
        + head_content
        + "</head><body>"
        + "\n".join(body_contents)
        + "</body></html>"
    )
