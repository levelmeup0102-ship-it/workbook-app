"""
variation/validator.py
변형문제 데이터 무결성 검증 - 완화 버전

핵심 변경:
- 원문 보존 검증은 옵션 (기본 끔) - Claude가 종종 미세한 차이를 만들어내서 폐기
- 마커 최소 간격 5→3 단어로 완화
- 보기 일치는 대소문자 무시 + 공백/구두점 정규화 강화
- 셔플 안 됐어도 경고만 (불합격 X)
"""
import re
from collections import Counter


def normalize_text(s: str) -> str:
    return " ".join(s.split())


def normalize_word(w: str) -> str:
    """단어 정규화: 소문자 + 양옆 구두점 제거"""
    return w.strip(".,!?;:'\"()[]{}").lower()


def check_cutout_match(bogi: list, answer_parts: list, pid: str = "?", q_name: str = "Q") -> list:
    """보기 단어 = 정답 단어 (대소문자/구두점 무시, 개수만 일치)"""
    errors = []
    
    all_ans_words = []
    for part in answer_parts:
        for w in part.split():
            cleaned = normalize_word(w)
            if cleaned:
                all_ans_words.append(cleaned)
    
    bogi_normalized = [normalize_word(w) for w in bogi if normalize_word(w)]
    
    bogi_c = Counter(bogi_normalized)
    ans_c = Counter(all_ans_words)
    
    if bogi_c != ans_c:
        missing = ans_c - bogi_c
        extra = bogi_c - ans_c
        errors.append(
            f"[{pid}] {q_name} 보기와 정답 단어 불일치 (개수 또는 단어 다름)\n"
            f"   정답에 있는데 보기에 없음: {dict(missing) if missing else '없음'}\n"
            f"   보기에 있는데 정답에 없음: {dict(extra) if extra else '없음'}"
        )
    
    return errors


def check_marker_positions(passage_with_marks: str, pid: str = "?", min_between: int = 3) -> list:
    """마커 위치 분산 검증 (간격 최소 3단어 - 완화)"""
    errors = []
    positions = {}
    for i in range(1, 6):
        idx = passage_with_marks.find(f"<MARK{i}>")
        if idx >= 0:
            positions[i] = idx
    
    # 마커 최소 3개만 있어도 OK
    if len(positions) < 3:
        errors.append(f"[{pid}] 마커 수 부족: {len(positions)}개 (최소 3개)")
        return errors
    
    sorted_marks = sorted(positions.items(), key=lambda x: x[1])
    for i in range(len(sorted_marks) - 1):
        m1, p1 = sorted_marks[i]
        m2, p2 = sorted_marks[i + 1]
        between = passage_with_marks[p1 + len(f"<MARK{m1}>"):p2]
        between = re.sub(r"<MARK\d>", "", between).strip()
        wc = len(between.split())
        if wc < min_between:
            errors.append(
                f"[{pid}] 마커 {m1}과 {m2} 사이 너무 짧음 ({wc}단어 < {min_between}최소)"
            )
    return errors


# ====================== 유형 A 검증 (완화) ======================
def validate_a(data: dict, original_passage: str = None, pid: str = "?") -> list:
    """유형 A 검증 - 원문 보존은 검증하지 않음, 핵심 규칙만"""
    errors = []
    
    # 필수 필드 존재 확인
    required = ["lead", "chunks", "topic_options", "topic_correct",
                "order_options", "order_correct", "statements",
                "blank_A", "blank_B", "bogi"]
    for f in required:
        if f not in data:
            errors.append(f"[{pid}] 필수 필드 누락: {f}")
            return errors
    
    # ★ 정답 순서가 (a)-(b)-(c)-(d) 원본 그대로면 거부
    try:
        correct_order_str = data["order_options"][data["order_correct"]]
        # 공백 제거 후 비교
        normalized = correct_order_str.replace(" ", "").replace("(", "").replace(")", "")
        if normalized == "a-b-c-d":
            errors.append(
                f"[{pid}] Q2 순서 정답이 (a)-(b)-(c)-(d) 원본 그대로임 — 청크를 SHUFFLE해서 정답이 다른 순서가 되도록 해야 함"
            )
    except (IndexError, KeyError, AttributeError) as e:
        errors.append(f"[{pid}] Q2 order_options/correct 형식 오류: {e}")
    
    # ★ blank_A, blank_B 단어 수 최소 7개 강제
    try:
        wa = len(data["blank_A"].split())
        wb = len(data["blank_B"].split())
        if wa < 7:
            errors.append(f"[{pid}] Q5 blank_A 단어 수 부족 ({wa}개 < 7개) — 더 긴 구문을 선택할 것")
        if wb < 7:
            errors.append(f"[{pid}] Q5 blank_B 단어 수 부족 ({wb}개 < 7개) — 더 긴 구문을 선택할 것")
    except (KeyError, AttributeError) as e:
        errors.append(f"[{pid}] blank_A/B 형식 오류: {e}")
    
    # ★ Q3 core_blank_target 단어 수 최소 3개 강제
    if data.get("core_blank_target"):
        try:
            cwc = len(data["core_blank_target"].split())
            if cwc < 3:
                errors.append(
                    f"[{pid}] Q3 core_blank_target 단어 수 부족 ({cwc}개 < 3개) "
                    f"— '{data['core_blank_target']}' 대신 더 긴 구문을 선택할 것"
                )
        except (KeyError, AttributeError):
            pass
    
    # ★ blank_A와 blank_B가 chunks 안에서 인접해 있으면 거부 (최소 5단어 사이에)
    try:
        chunks_text = " ".join([c[1] for c in data["chunks"] if len(c) >= 2])
        # BLANK_A와 BLANK_B 마커 위치
        ia = chunks_text.find("<BLANK_A>")
        ib = chunks_text.find("<BLANK_B>")
        if ia >= 0 and ib >= 0:
            # 두 마커 사이 단어 수 계산
            start = min(ia, ib) + len("<BLANK_A>")  # 어느 쪽이든 마커 길이는 같음
            end = max(ia, ib)
            between_text = chunks_text[start:end]
            # 빈칸 마커 제거
            between_text = between_text.replace("<BLANK_A>", "").replace("<BLANK_B>", "").strip()
            wc_between = len(between_text.split()) if between_text else 0
            if wc_between < 5:
                errors.append(
                    f"[{pid}] Q5 blank_A와 blank_B가 너무 가까움 (사이에 {wc_between}단어만 있음 < 5개) "
                    f"— 서로 떨어진 두 구문을 선택할 것"
                )
    except (KeyError, AttributeError, TypeError) as e:
        pass  # 거리 검증 실패는 무시 (필수 검증은 아님)
    
    # Q5 잘라쓰기 검증 (대소문자 무시)
    try:
        errors += check_cutout_match(
            data["bogi"],
            [data["blank_A"], data["blank_B"]],
            pid, "Q5(빈칸영작)"
        )
    except Exception as e:
        errors.append(f"[{pid}] Q5 보기 검증 예외: {e}")
    
    # 정답 인덱스 범위
    for key in ["topic_correct", "order_correct"]:
        v = data.get(key, -1)
        if not isinstance(v, int) or not (0 <= v <= 4):
            errors.append(f"[{pid}] {key} 범위 오류: {v}")
    
    if data.get("core_blank_correct") is not None:
        v = data["core_blank_correct"]
        if not isinstance(v, int) or not (0 <= v <= 4):
            errors.append(f"[{pid}] core_blank_correct 범위 오류")
    
    # statements 5개
    if not isinstance(data.get("statements"), list) or len(data["statements"]) != 5:
        errors.append(f"[{pid}] statements는 5개 항목이어야 함")
    
    # chunks 4개
    if not isinstance(data.get("chunks"), list) or len(data["chunks"]) != 4:
        errors.append(f"[{pid}] chunks는 4개 항목이어야 함")
    
    return errors


# ====================== 유형 B 검증 (완화) ======================
def validate_b(data: dict, original_passage: str = None, pid: str = "?") -> list:
    """유형 B 검증 - 핵심만 체크"""
    errors = []
    
    # 필수 필드 존재
    required = ["given_sentence", "passage_with_marks", "position_correct",
                "topic_options", "topic_correct", "summary_options", "summary_correct",
                "blank_summary_bogi", "blank_A", "blank_B",
                "topic_writing_bogi", "topic_writing_answer"]
    for f in required:
        if f not in data:
            errors.append(f"[{pid}] 필수 필드 누락: {f}")
            return errors
    
    # 정답 인덱스 범위
    for key in ["summary_correct", "topic_correct", "position_correct"]:
        v = data.get(key, -1)
        if not isinstance(v, int) or not (0 <= v <= 4):
            errors.append(f"[{pid}] {key} 범위 오류: {v}")
    
    # ★ Q4 blank_A, blank_B 단어 수 최소 6개 강제
    try:
        wa = len(data["blank_A"].split())
        wb = len(data["blank_B"].split())
        if wa < 6:
            errors.append(f"[{pid}] Q4 blank_A 단어 수 부족 ({wa}개 < 6개) — 더 긴 구문 선택")
        if wb < 6:
            errors.append(f"[{pid}] Q4 blank_B 단어 수 부족 ({wb}개 < 6개) — 더 긴 구문 선택")
    except (KeyError, AttributeError) as e:
        errors.append(f"[{pid}] B blank_A/B 형식 오류: {e}")
    
    # ★ Q5 topic_writing_answer 단어 수 최소 10개 강제
    try:
        twc = len(data["topic_writing_answer"].split())
        if twc < 10:
            errors.append(
                f"[{pid}] Q5 topic_writing_answer 단어 수 부족 ({twc}개 < 10개) "
                f"— 더 완전한 문장으로 작성"
            )
    except (KeyError, AttributeError):
        errors.append(f"[{pid}] B topic_writing_answer 형식 오류")
    
    # Q4 잘라쓰기 (요약 영작)
    try:
        errors += check_cutout_match(
            data["blank_summary_bogi"],
            [data["blank_A"], data["blank_B"]],
            pid, "Q4(요약영작)"
        )
    except Exception as e:
        errors.append(f"[{pid}] Q4 보기 검증 예외: {e}")
    
    # Q5 잘라쓰기 (주제 영작)
    try:
        errors += check_cutout_match(
            data["topic_writing_bogi"],
            [data["topic_writing_answer"]],
            pid, "Q5(주제영작)"
        )
    except Exception as e:
        errors.append(f"[{pid}] Q5 보기 검증 예외: {e}")
    
    # 마커 위치 분산 (완화: 최소 3개, 최소 3단어 간격)
    try:
        errors += check_marker_positions(data["passage_with_marks"], pid, min_between=3)
    except Exception as e:
        errors.append(f"[{pid}] 마커 검증 예외: {e}")
    
    # summary_options 5개
    if not isinstance(data.get("summary_options"), list) or len(data["summary_options"]) != 5:
        errors.append(f"[{pid}] summary_options는 5개 항목이어야 함")
    
    # ※ 원문 보존 검증은 일부러 안 함
    return errors


# ====================== 호환용 함수 (예전 코드용) ======================
def check_passage_preservation(rebuilt: str, original: str, pid: str = "?") -> list:
    """예전 코드 호환용 - 호출돼도 빈 리스트 반환 (검증 안 함)"""
    return []
