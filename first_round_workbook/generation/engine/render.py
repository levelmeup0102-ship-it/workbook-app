"""
워크북 최종 조립+렌더
ㄴ> step 결과 → 템플릿 변수 → HTML 문자열

웹 API 이므로 디스크 저장/WeasyPrint 없이 HTML 문자열만 생성 (프론트가 PDF 인쇄).
(구 pipeline.merge_to_template_data + render_pdf)
"""
import math
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

# 프로젝트 루트 (template.html 위치): .../first_round_workbook/generation/engine/render.py → parents[3]
TEMPLATE_DIR = Path(__file__).resolve().parents[3]

try:
    from qa_check import fix_single_html
    _QA_AVAILABLE = True
except Exception:
    _QA_AVAILABLE = False


def _split_sentences_chunks(sentences: list, max_per_page: int = 8) -> list:
    """문장 리스트를 균등 분배하여 페이지별 청크로 나눈다."""
    total = len(sentences)
    if total <= max_per_page:
        return [sentences]
    num_pages = math.ceil(total / max_per_page)
    base = total // num_pages
    extra = total % num_pages
    sizes = [base + 1] * extra + [base] * (num_pages - extra)
    chunks, idx = [], 0
    for size in sizes:
        chunks.append(sentences[idx:idx + size])
        idx += size
    return chunks


def merge_to_template_data(passage: str, meta: dict, all_steps: dict) -> dict:
    """모든 단계 결과(step1~8)를 템플릿 변수로 병합."""
    s1 = all_steps["step1"]
    s2 = all_steps["step2"]
    s3 = all_steps["step3"]
    s4 = all_steps["step4"]
    s5 = all_steps["step5"]
    s6 = all_steps["step6"]
    s7 = all_steps["step7"]
    s8 = all_steps["step8"]

    return {
        # 메타 정보
        "subject": meta.get("subject", ""),
        "publisher": meta.get("publisher", ""),
        "lesson_num": meta.get("lesson_num", ""),
        "lesson_n": meta.get("lesson_n", ""),
        "challenge_title": meta.get("challenge_title", ""),
        # 지문/번역
        "passage": passage,
        "translation": s1.get("translation", ""),
        "sentence_translations": s1.get("sentence_translations", []),
        # Lv.1 어휘
        "vocab": s1.get("vocab", []),
        "test_a": s1.get("test_a", []),
        "test_b": s1.get("test_b", []),
        "test_c": s1.get("test_c", []),
        # Lv.3 문장분석 + 핵심문장
        "sentences": s1.get("sentences", []),
        "sentence_chunks": _split_sentences_chunks(s1.get("sentences", [])),
        "key_sentences": s1.get("key_sentences", []),
        # Lv.5 순서/삽입
        "order_intro": s2.get("order_intro", ""),
        "order_paragraphs": s2.get("order_paragraphs", []),
        "insert_sentence": s2.get("insert_sentence", ""),
        "insert_passage": s2.get("insert_passage", ""),
        "full_order_blocks": s2.get("full_order_blocks", []),
        # Lv.6 빈칸
        "blank_passage": s3.get("blank_passage", ""),
        "blank_options": s3.get("blank_options", []),
        # Lv.7 주제
        "topic_passage": s4.get("topic_passage", ""),
        "topic_options": s4.get("topic_options", []),
        # Lv.8 어법
        "grammar_bracket_passage": s5.get("grammar_bracket_passage", ""),
        "grammar_bracket_count": s5.get("grammar_bracket_count", 13),
        "grammar_error_passage": s5.get("grammar_error_passage", ""),
        "grammar_error_count": s5.get("grammar_error_count", 8),
        # Lv.9
        "vocab_advanced_passage": s6.get("vocab_advanced_passage", ""),
        "vocab_parta_answers": s6.get("vocab_parta_answers", []),
        "vocab_partb": s6.get("vocab_partb", []),
        "content_match_kr": s6.get("content_match_kr", []),
        "content_match_en": s6.get("content_match_en", []),
        # Stage 10 영작
        "writing_items": s7.get("writing_items", []),
        "writing_chunks": _split_sentences_chunks(s7.get("writing_items", []), max_per_page=8),
        # 정답
        "answers_html": s8.get("answers_html", ""),
    }


def render_workbook_html(template_data: dict, levels=None) -> str:
    """template.html(Jinja2) 렌더 + QA 자동보정 → HTML 문자열 반환.
    (QA bracket_missing 재생성 루프는 TODO — 우선 QA 보정만)
    """
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    tmpl = env.get_template("template.html")
    template_data["levels"] = levels  # None이면 전체 출력
    html = tmpl.render(**template_data)

    if _QA_AVAILABLE:
        html, qa_issues = fix_single_html(html)
        if qa_issues:
            logger.warning("render: QA 이슈 감지 %s", qa_issues)

    return html
