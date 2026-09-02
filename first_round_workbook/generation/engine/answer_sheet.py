"""답안지(정답 페이지) 조립 — 생성이 아닌 출력/조립 계층.

step1~7 결과(all_data)에서 정답만 추출해 레벨별 HTML 블록으로 조립.
LLM 없음. 캐시 안 함(orchestrator가 직접 호출) → 항상 최신 step 결과로 재생성.
(구 pipeline.step8_answers)
"""
import logging

logger = logging.getLogger(__name__)


def build_answer_sheet(all_data: dict, levels=None) -> dict:
    """step1~7 결과에서 정답 추출 → 답안지 HTML. 반환: {"answers_html": ...}

    levels: 선택된 레벨 목록(None=전체). 선택된 레벨의 답안 블록만 포함한다.
    """
    logger.info("답안지 생성")

    def want(level: int) -> bool:
        """해당 레벨이 선택됐는지(None이면 전체 포함) → 답안 블록 포함 여부."""
        return levels is None or level in levels

    blocks = []

    # Lv.4 주제 (step4)
    if want(4):
        stage4_data = all_data.get("step4") or {}
        correct_number = ', '.join(stage4_data.get("topic_correct") or [])
        wrong_list = stage4_data.get("topic_wrong_translation") or []
        wrong_translation = ''.join(f'<p>{i}</p>' for i in wrong_list)

        blocks.append(
            '<div class="ablock">'
            '<p class="ast">Stage 4 수업 직후 정리</p>'
            f'<p>[STEP 1 - 주제문 직접 쓰기]<br>정답: {correct_number}</p>'
            '<p>[오답 선지 해석]</p>'
            f'{wrong_translation}'
            '</div>'
        )

    # Lv.5 순서/삽입 (step2)
    if want(5):
        s2 = all_data.get("step2", {})
        blocks.append('<div class="ablock">'
                      '<p class="ast">Stage 5 순서 배열</p>'
                      f'<p>[A. 순서 배열]<br>정답: {s2.get("order_answer","")}</p>'
                      f'<p>[B. 문장 삽입]<br>정답: {s2.get("insert_answer","")}</p>'
                      f'<p>[심화 - 문장 순서 배열]<br>정답: {s2.get("full_order_answer","")}</p>'
                      '</div>')

    # Lv.6 빈칸 추론 (step3)
    if want(6):
        s3 = all_data.get("step3") or {}
        correct = ', '.join(s3.get("blank_correct") or [])
        stage6_wrong = ''.join(f'<p>{i}</p>' for i in (s3.get("blank_wrong_translation") or []))
        blocks.append('<div class="ablock">'
                      '<p class="ast">Stage 6 빈칸 추론</p>'
                      f'<p>[STEP 2]<br>정답: {correct}</p>'
                      '<p>[오답 선지 해석]</p>'
                      f'{stage6_wrong}'
                      '</div>')

    # Lv.7 어법 (step5) — STEP1 괄호형 3단, STEP2 서술형 2단 나열
    if want(7):
        stage7_data = all_data.get("step5") or {}
        stage7_step1_correct_list = stage7_data.get("grammar_bracket_answers") or []
        stage7_step2_error_list = stage7_data.get("grammar_error_answers") or []

        step1_items = []
        for i in range(0, len(stage7_step1_correct_list), 3):
            chunk = stage7_step1_correct_list[i:i+3]
            line = ' '.join(f'({c.get("num")}) {c.get("answer")}' for c in chunk)
            step1_items.append(f'<p>{line}</p>')
        stage7_step1_answer = ''.join(step1_items)

        step2_items = []
        for i in range(0, len(stage7_step2_error_list), 2):
            chunk = stage7_step2_error_list[i:i+2]
            line = ' '.join(f'({c.get("num")}) {c.get("error")} → {c.get("original")}' for c in chunk)
            step2_items.append(f'<p>{line}</p>')
        stage7_step2_answer = ''.join(step2_items)

        blocks.append('<div class="ablock"><p class="ast">Stage 7-1 어법</p>'
                      '<p>[STEP 1]</p>'
                      f'<ul>{stage7_step1_answer}</ul>'
                      '<p>[STEP 2]</p>'
                      f'<ul>{stage7_step2_answer}</ul>'
                      '</div>')

    # Lv.8 어휘 (step6) — Part A / Part B 정답
    if want(8):
        stage8_data = all_data.get("step6") or {}
        stage8_partA_list = stage8_data.get("vocab_parta_answers") or []
        stage8_partB_list = stage8_data.get("vocab_partb_answers") or []

        partA_items = []
        for i in range(0, len(stage8_partA_list), 3):
            chunk = stage8_partA_list[i:i+3]
            stage8_partA_answers = ' '.join(f'({c.get("num")}) {c.get("answer")}' for c in chunk)
            partA_items.append(f'<p>{stage8_partA_answers}</p>')
        insert_data8_A = ''.join(partA_items)

        partB_items = []
        for i in range(0, len(stage8_partB_list), 2):
            chunk = stage8_partB_list[i:i+2]
            stage8_partB_answers = ' '.join(f'({c.get("num")}) {", ".join(c.get("correct"))}' for c in chunk)
            partB_items.append(f'<p>{stage8_partB_answers}</p>')
        insert_data_8_B = ''.join(partB_items)

        blocks.append('<div class="ablock"><p class="ast">Stage 8 어휘</p>'
                      '<p>[Part A]</p>'
                      f'<ul>{insert_data8_A}</ul>'
                      '<p>[Part B]</p>'
                      f'<ul>{insert_data_8_B}</ul>'
                      '</div>')

    # Lv.9 내용 일치 (step6) — 정답 번호 / 오답 번호+한글해석
    if want(9):
        stage9_data = all_data.get("step6") or {}
        correct_numbers_kor = ', '.join(stage9_data.get("content_match_kr_answer") or [])
        stage9_wrong_kor = ''.join(f'<p>{i}</p>' for i in (stage9_data.get("content_match_kr_wrong_trans") or []))
        correct_numbers_eng = ', '.join(stage9_data.get("content_match_en_answer") or [])
        stage9_wrong_eng = ''.join(f'<p>{i}</p>' for i in (stage9_data.get("content_match_en_wrong_trans") or []))

        blocks.append(
            '<div class="ablock"><p class="ast">Stage 9 내용 일치</p>'
            '<p>[STEP 2 - Part A. 한국어]</p>'
            f'<p>정답: {correct_numbers_kor}</p>'
            '<p>[오답 선지 해설 및 정답]</p>'
            f'{stage9_wrong_kor}'
            '<p>[STEP 2 - Part B. English]</p>'
            f'<p>정답: {correct_numbers_eng}</p>'
            '<p>[오답 선지 해설 및 정답]</p>'
            f'{stage9_wrong_eng}'
            '</div>'
        )

    # Lv.10 영작 (step7)
    if want(10):
        s7 = all_data.get("step7", {})
        lv10 = ['<div class="ablock"><p class="ast">Stage 10 영작</p>']
        for idx, item in enumerate(s7.get("writing_items", []), start=1):
            lv10.append(f'<p>{idx}. {item.get("answer","")}</p>')
        lv10.append('</div>')
        blocks.append(''.join(lv10))

    return {"answers_html": '\n'.join(blocks)}
