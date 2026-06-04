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
import random
import hashlib
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
def shuffle_bogi(bogi: list, seed_str: str = "") -> list:
    """보기 단어들을 셔플 (seed로 동일 데이터엔 동일 결과 보장)"""
    if not bogi:
        return bogi
    shuffled = list(bogi)
    # seed: 정답 텍스트 기반으로 deterministic하지만 정답 순서와는 다르게
    seed_int = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) if seed_str else 42
    rng = random.Random(seed_int)
    # 정답 순서와 다른지 확인하면서 셔플 (최대 5번 시도)
    for _ in range(5):
        rng.shuffle(shuffled)
        if shuffled != list(bogi):
            return shuffled
    return shuffled


def prepare_a_passage(data: dict, label: str) -> dict:
    n_false = sum(1 for _, _, ok in data["statements"] if not ok)
    # Q5 bogi 셔플
    bogi_shuffled = shuffle_bogi(
        data["bogi"],
        seed_str=(data.get("blank_A", "") + data.get("blank_B", ""))
    )
    return {
        "label": label,
        "data": {
            "intro": convert_lead_a(data["intro"]),
            "paragraphs": convert_chunks_a(data["paragraphs"]),
            "topic_options": data["topic_options"],
            "topic_correct": data["topic_correct"],
            "order_correct": data["order_correct"],
            "statements": [list(s) for s in data["statements"]],
            "statements_kr": data.get("statements_kr", []),
            "statements_evidence": data.get("statements_evidence", []),
            "mismatch_count": data.get("mismatch_count", n_false),
            "blank_A": data["blank_A"],
            "blank_B": data["blank_B"],
            "bogi": bogi_shuffled,
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
    new_data = {**data, "passage_rendered": render_marks(data["passage_with_marks"])}
    
    # Q4 blank_summary_bogi 셔플
    if "blank_summary_bogi" in new_data:
        new_data["blank_summary_bogi"] = shuffle_bogi(
            new_data["blank_summary_bogi"],
            seed_str=(data.get("blank_A", "") + data.get("blank_B", ""))
        )
    # Q5 topic_writing_bogi 셔플
    if "topic_writing_bogi" in new_data:
        new_data["topic_writing_bogi"] = shuffle_bogi(
            new_data["topic_writing_bogi"],
            seed_str=data.get("topic_writing_answer", "")
        )
    
    return {
        "label": label,
        "data": new_data,
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
  
  /* ★ 빈칸 underline 명확하게 (Q3 핵심빈칸 등) - 글자 baseline 높이에 맞춤 */
  .core-blank-inline {
    border-bottom: 2px solid #6A1B9A !important;
    min-width: 120px !important;
    display: inline-block !important;
    font-size: 0 !important;  /* '_____' 문자 안 보이게 - 밑줄만 보임 */
    height: 0.95em;            /* 글자 높이만큼 - 글자가 있는 것처럼 */
    vertical-align: baseline;  /* 글자 baseline에 정렬 → 밑줄이 글자 아래쪽 */
  }
  
  /* 일반 빈칸 (밑줄 표시) */
  .blank-line, .blank-underline {
    display: inline-block;
    min-width: 100px;
    border-bottom: 2px solid #333;
  }
  
  /* ★★★ A 답지 압축 — 한 페이지에 더 많은 문제 답지 들어가게 ★★★ */
  .ans-block {
    padding: 6px 10px !important;
    margin-bottom: 8px !important;
    page-break-inside: avoid;
  }
  .ans-block-title {
    font-size: 9.5pt !important;
    margin-bottom: 4px !important;
    padding-bottom: 2px !important;
  }
  .ans-row {
    margin: 2px 0 !important;
    padding: 1px 0 !important;
    line-height: 1.4 !important;
  }
  .ans-row .ans-detail {
    font-size: 7.5pt !important;
    line-height: 1.4 !important;
  }
  .ans-row b.q-name {
    font-size: 7.5pt !important;
  }
  
  /* Q1~Q5 한국어 해설 박스 압축 (매우 강하게) */
  .expl-box {
    padding: 2px 6px !important;
    margin: 1px 0 !important;
    font-size: 7pt !important;
    line-height: 1.35 !important;
    background: #f8f8f8;
    border-left: 2px solid #ddd;
  }
  .expl-label {
    font-size: 6.5pt !important;
    padding: 0 4px !important;
    margin-right: 3px !important;
  }
  
  /* Q4 진술 테이블 압축 (매우 강하게) */
  .match-tbl {
    font-size: 6.5pt !important;
    line-height: 1.25 !important;
    border-collapse: collapse;
    width: 100%;
    margin: 1px 0 !important;
  }
  .match-tbl th, .match-tbl td {
    padding: 1px 3px !important;
    vertical-align: top;
  }
  .stmt-kr {
    font-size: 6pt !important;
    color: #999 !important;
    font-style: italic;
  }
  .stmt-why, .stmt-why-true {
    font-size: 6pt !important;
    line-height: 1.25 !important;
  }
  
  /* Q5 grammar-note 압축 */
  .grammar-note {
    font-size: 6.5pt !important;
    line-height: 1.3 !important;
    margin: 1px 0 3px !important;
    padding: 2px 4px !important;
  }
  
  /* blank-ans-grid 압축 */
  .blank-ans-grid {
    font-size: 8pt !important;
    margin: 3px 0 !important;
    gap: 4px !important;
  }
  
  /* 답지 페이지 처리 */
  /* ans-row, expl-box 같은 작은 항목은 페이지 중간에 안 끊김 */
  .ans-row, .expl-box, .ans-block-title, .grammar-note {
    page-break-inside: avoid;
    break-inside: avoid;
  }
  /* ans-block은 너무 크면 페이지 나눠도 OK (강제 avoid 제거) */
  
  /* ★ 답지 전체(헤더+본문)는 가능하면 한 페이지에 - 너무 크면 어쩔 수 없이 break */
  .ans-start.ans-merged {
    /* page-break-inside: auto로 변경 - 답지 내부 break 허용 */
  }
  
  /* ★★★ 답지 페이지 합치기 - 원본 템플릿의 page-break 모두 override ★★★ */
  .answers-combined > div.ans-start.ans-merged {
    page-break-before: auto !important;
    break-before: auto !important;
  }
  /* 첫 번째 답지만 새 페이지에서 시작 */
  .answers-combined > div.ans-start.ans-merged:first-child {
    page-break-before: always !important;
    break-before: page !important;
  }
  /* 두 번째 이후 답지는 이전 답지 바로 다음에 */
  .answers-combined > div.ans-start.ans-merged + div.ans-start.ans-merged {
    page-break-before: auto !important;
    break-before: auto !important;
    margin-top: 18px !important;
    padding-top: 8px !important;
    border-top: 1px dashed #ccc;
  }
  
  /* 답지 내부 q-page는 무력화 */
  .answers-combined .q-page {
    page-break-after: auto !important;
    break-after: auto !important;
  }
  
  /* q-page는 페이지 단위 - 마지막 q-page는 break 안 함 (빈 페이지 방지) */
  .q-page {
    page-break-after: auto;
  }
  @media print {
    .q-page { page-break-after: always; }
    /* 마지막 q-page (다음에 answers-combined가 옴) → break 제거 */
    .q-page:has(+ .answers-combined) {
      page-break-after: auto !important;
      break-after: auto !important;
    }
    /* 마지막 q-page (다음 형제 없음) → break 제거 */
    .q-page:last-child {
      page-break-after: auto !important;
      break-after: auto !important;
    }
    /* 마지막 섹션의 마지막 q-page도 안전하게 */
    section.variation-section:last-of-type .q-page:last-child {
      page-break-after: auto !important;
      break-after: auto !important;
    }
    /* 유형 A 끝 → 유형 B 시작: 새 섹션은 항상 새 페이지에서 시작 */
    section.variation-section + section.variation-section {
      page-break-before: always !important;
      break-before: page !important;
    }
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
    각 섹션의 <head> 스타일을 모두 합치고, <body>는 다음 순서로 재배치:
    1) 모든 문제 페이지 (.q-page) 먼저
    2) 모든 답지 (.ans-start) 마지막에 한 곳에 모음 - 페이지 낭비 방지
    
    sections: [("A", html_a), ("B", html_b), ...]
    """
    # 1) 모든 섹션의 head 스타일 + meta + link 추출
    all_head_parts = []
    seen_styles = set()
    for type_name, html in sections:
        head_extracted = _extract_head_styles(html)
        for tag in re.findall(r"(<(?:style|link|meta)[^>]*(?:>.*?</style>|/?>))", head_extracted, re.DOTALL | re.IGNORECASE):
            if tag not in seen_styles:
                seen_styles.add(tag)
                all_head_parts.append(tag)
    
    # 2) 각 섹션의 body 내용에서 페이지 단위로 문제/답지 분리
    #    ★ 답지 시작 div의 실제 클래스는 "ans-start" (ans-page 아님!) 이므로
    #      page_pat에 ans-start를 반드시 포함해야 답지가 페이지로 인식되어
    #      answers-combined로 취합된다. q-page는 문제, ans-start/ans-page는 답지.
    #      (지문이 여러 개여서 [문제][답지][문제][답지] 순서여도 정확히 분리됨)
    question_parts = []  # 문제만 (.q-page)
    answer_parts = []    # 답지만 (.ans-start)

    # 최상위 페이지 div 시작 태그 — 클래스 순서/추가 클래스에 무관하게 매칭
    page_pat = re.compile(r'<div\s+class="(?P<kind>q-page|ans-page|ans-start)[^"]*"', re.IGNORECASE)

    for type_name, html in sections:
        body_content = _extract_body_content(html)
        matches = list(page_pat.finditer(body_content))

        if not matches:
            # 페이지 div를 못 찾으면 전체를 문제로 처리 (안전장치)
            question_parts.append(
                f'<section class="variation-section" data-type="{type_name}">\n'
                f'{body_content}\n'
                f'</section>'
            )
            continue

        q_blocks = []  # 이 섹션의 문제 페이지들
        a_blocks = []  # 이 섹션의 답지 페이지들
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body_content)
            block = body_content[start:end]
            if m.group("kind").lower() == "q-page":
                q_blocks.append(block)
            else:
                a_blocks.append(block)

        if q_blocks:
            question_parts.append(
                f'<section class="variation-section" data-type="{type_name}">\n'
                + "\n".join(q_blocks) +
                "\n</section>"
            )
        if a_blocks:
            answer_parts.append("\n".join(a_blocks))

    # 3) 최종 HTML — 문제들 먼저, 답지들 마지막에 한 묶음으로
    body_html = "\n".join(question_parts)
    if answer_parts:
        combined = "\n".join(answer_parts)
        # 답지 시작 div(.ans-start)에 .ans-merged 부여 → CSS의 .answers-combined 규칙이
        # page-break를 제어 (첫 답지만 새 페이지, 나머지는 자연스럽게 이어짐)
        combined = re.sub(
            r'(<div\s+class="[^"]*\bans-start\b)([^"]*")',
            r'\1 ans-merged\2',
            combined,
            flags=re.IGNORECASE,
        )
        body_html += '\n<div class="answers-combined">\n' + combined + '\n</div>'
    
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
{body_html}
</body>
</html>"""
    return final
