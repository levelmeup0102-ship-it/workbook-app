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


def tokenize_for_comparison(text: str) -> list:
    """비교용 토큰화: 하이픈을 공백으로 처리해서 split
    'south-facing slopes' → ['south', 'facing', 'slopes']
    이 함수로 정답과 보기 양쪽 다 정규화하면 'south-facing' vs ['south','facing'] 문제 해결
    """
    # 하이픈을 공백으로
    text = text.replace("-", " ").replace("—", " ").replace("–", " ")
    words = []
    for w in text.split():
        cleaned = normalize_word(w)
        if cleaned:
            words.append(cleaned)
    return words


def check_cutout_match(bogi: list, answer_parts: list, pid: str = "?", q_name: str = "Q") -> list:
    """보기 단어 = 정답 단어 (대소문자/구두점/하이픈 무시, 개수만 일치)"""
    errors = []
    
    # 정답을 토큰화 (하이픈 분리 포함)
    all_ans_words = []
    for part in answer_parts:
        all_ans_words.extend(tokenize_for_comparison(part))
    
    # 보기도 토큰화 (혹시 보기에 하이픈 단어 있을 경우 대비)
    bogi_normalized = []
    for w in bogi:
        bogi_normalized.extend(tokenize_for_comparison(w))
    
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


def check_marker_positions(passage_with_marks: str, pid: str = "?", min_between: int = 3, position_correct: int = None) -> list:
    """마커 위치 분산 검증 (완화):
    - 정답 마커 좌우는 반드시 분산되어야 함 (앞뒤 3단어+)
    - 나머지 마커들은 1단어 이상이면 OK (distractor라서 너무 까다롭게 안 함)
    """
    errors = []
    positions = {}
    for i in range(1, 6):
        idx = passage_with_marks.find(f"<MARK{i}>")
        if idx >= 0:
            positions[i] = idx
    
    # 마커 최소 3개
    if len(positions) < 3:
        errors.append(f"[{pid}] 마커 수 부족: {len(positions)}개 (최소 3개)")
        return errors
    
    sorted_marks = sorted(positions.items(), key=lambda x: x[1])
    # position_correct는 0-based 인덱스 → 정답 마커 번호 = sorted_marks[idx][0]
    correct_mark_num = None
    if position_correct is not None and 0 <= position_correct < len(sorted_marks):
        correct_mark_num = sorted_marks[position_correct][0]
    
    for i in range(len(sorted_marks) - 1):
        m1, p1 = sorted_marks[i]
        m2, p2 = sorted_marks[i + 1]
        between = passage_with_marks[p1 + len(f"<MARK{m1}>"):p2]
        between = re.sub(r"<MARK\d>", "", between).strip()
        wc = len(between.split())
        
        # 정답 마커가 m1 또는 m2일 때만 엄격 (3단어 필요)
        # 그 외 distractor끼리 가까운 건 OK (1단어 이상이면 패스)
        is_correct_adjacent = (correct_mark_num is not None and 
                              (m1 == correct_mark_num or m2 == correct_mark_num))
        threshold = min_between if is_correct_adjacent else 1
        
        if wc < threshold:
            label = "정답 마커 인접" if is_correct_adjacent else "마커"
            errors.append(
                f"[{pid}] {label} {m1}과 {m2} 사이 너무 짧음 ({wc}단어 < {threshold}최소)"
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
    
    # ★ blank_A, blank_B 단어 수 최소 5개 (사용자 요구 7개에서 완화 - 5회 실패 방지)
    try:
        wa = len(data["blank_A"].split())
        wb = len(data["blank_B"].split())
        if wa < 5:
            errors.append(f"[{pid}] Q5 blank_A 단어 수 부족 ({wa}개 < 5개) — 더 긴 구문을 선택할 것")
        if wb < 5:
            errors.append(f"[{pid}] Q5 blank_B 단어 수 부족 ({wb}개 < 5개) — 더 긴 구문을 선택할 것")
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
    
    # ★ blank_A와 blank_B가 chunks 안에서 인접해 있으면 거부 (최소 3단어 사이에 - 완화)
    try:
        chunks_text = " ".join([c[1] for c in data["chunks"] if len(c) >= 2])
        ia = chunks_text.find("<BLANK_A>")
        ib = chunks_text.find("<BLANK_B>")
        if ia >= 0 and ib >= 0:
            start = min(ia, ib) + len("<BLANK_A>")
            end = max(ia, ib)
            between_text = chunks_text[start:end]
            between_text = between_text.replace("<BLANK_A>", "").replace("<BLANK_B>", "").strip()
            wc_between = len(between_text.split()) if between_text else 0
            if wc_between < 3:
                errors.append(
                    f"[{pid}] Q5 blank_A와 blank_B가 너무 가까움 (사이에 {wc_between}단어만 있음 < 3개) "
                    f"— 서로 떨어진 두 구문을 선택할 것"
                )
    except (KeyError, AttributeError, TypeError) as e:
        pass
    
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
    else:
        # ★ 각 chunk의 텍스트가 비어있지 않은지 확인 (Claude가 종종 (d)를 비움)
        for idx, ch in enumerate(data["chunks"]):
            if not isinstance(ch, list) or len(ch) < 2:
                errors.append(f"[{pid}] chunks[{idx}] 형식 오류 (label, text 필요)")
                continue
            label, text = ch[0], ch[1]
            if not text or not text.strip():
                errors.append(
                    f"[{pid}] chunks[{idx}] {label} 텍스트가 비어있음 — "
                    f"4개 chunk 모두 원문 텍스트를 채워야 함"
                )
            elif len(text.split()) < 5:
                errors.append(
                    f"[{pid}] chunks[{idx}] {label} 텍스트 너무 짧음 ({len(text.split())}단어) — "
                    f"최소 5단어 이상으로 4개 chunk 균등 분할"
                )
    
    return errors


# ====================== 유형 B 검증 (완화) ======================
def validate_b(data: dict, original_passage: str = None, pid: str = "?", strict: bool = True) -> list:
    """유형 B 검증 - 핵심만 체크
    strict=False면 단어 수 / 중복 검증을 풀어줌 (마지막 retry용)
    """
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
    
    # 정답 인덱스 범위 (필수)
    for key in ["summary_correct", "topic_correct", "position_correct"]:
        v = data.get(key, -1)
        if not isinstance(v, int) or not (0 <= v <= 4):
            errors.append(f"[{pid}] {key} 범위 오류: {v}")
    
    # ★ Q4 blank_A, blank_B 단어 수 (strict 6개, soft 2개)
    min_blank_words = 6 if strict else 2
    try:
        wa = len(data["blank_A"].split())
        wb = len(data["blank_B"].split())
        if wa < min_blank_words:
            errors.append(f"[{pid}] Q4 blank_A 단어 수 부족 ({wa}개 < {min_blank_words}개) — 더 긴 구문 선택")
        if wb < min_blank_words:
            errors.append(f"[{pid}] Q4 blank_B 단어 수 부족 ({wb}개 < {min_blank_words}개) — 더 긴 구문 선택")
    except (KeyError, AttributeError) as e:
        errors.append(f"[{pid}] B blank_A/B 형식 오류: {e}")
    
    # ★ Q5 topic_writing_answer 단어 수 (strict 10개, soft 3개)
    min_topic_words = 10 if strict else 3
    try:
        twc = len(data["topic_writing_answer"].split())
        if twc < min_topic_words:
            errors.append(
                f"[{pid}] Q5 topic_writing_answer 단어 수 부족 ({twc}개 < {min_topic_words}개) "
                f"— 더 완전한 문장으로 작성"
            )
    except (KeyError, AttributeError):
        errors.append(f"[{pid}] B topic_writing_answer 형식 오류")
    
    # Q4 잘라쓰기 (요약 영작) - strict일 때만 필수, soft는 통과
    if strict:
        try:
            errors += check_cutout_match(
                data["blank_summary_bogi"],
                [data["blank_A"], data["blank_B"]],
                pid, "Q4(요약영작)"
            )
        except Exception as e:
            errors.append(f"[{pid}] Q4 보기 검증 예외: {e}")
    
    # Q5 잘라쓰기 (주제 영작) - strict일 때만 필수, soft는 통과
    if strict:
        try:
            errors += check_cutout_match(
                data["topic_writing_bogi"],
                [data["topic_writing_answer"]],
                pid, "Q5(주제영작)"
            )
        except Exception as e:
            errors.append(f"[{pid}] Q5 보기 검증 예외: {e}")
    
    # 마커 위치 분산 - strict일 때만 필수, soft는 마커 개수만 체크
    try:
        if strict:
            errors += check_marker_positions(
                data["passage_with_marks"], pid, min_between=3,
                position_correct=data.get("position_correct")
            )
        else:
            # soft: 마커가 최소 2개만 있어도 OK
            positions = {}
            for i in range(1, 6):
                if data["passage_with_marks"].find(f"<MARK{i}>") >= 0:
                    positions[i] = True
            if len(positions) < 2:
                errors.append(f"[{pid}] 마커 수 부족: {len(positions)}개 (최소 2개)")
    except Exception as e:
        errors.append(f"[{pid}] 마커 검증 예외: {e}")
    
    # summary_options 5개 (필수)
    if not isinstance(data.get("summary_options"), list) or len(data["summary_options"]) < 2:
        errors.append(f"[{pid}] summary_options는 최소 2개 항목 필요")
    elif strict and len(data["summary_options"]) != 5:
        errors.append(f"[{pid}] summary_options는 5개 항목이어야 함")
    else:
        # ★ Q3 각 (A), (B) 슬롯 단어 수 (strict 1단어, soft 5단어 이하)
        max_slot_words = 1 if strict else 5
        a_words = []
        b_words = []
        for idx, opt in enumerate(data["summary_options"]):
            if not isinstance(opt, list) or len(opt) != 2:
                errors.append(f"[{pid}] Q3 summary_options[{idx}] 형식 오류 (2개 슬롯 필요)")
                continue
            a_val, b_val = opt[0], opt[1]
            if not isinstance(a_val, str) or not isinstance(b_val, str):
                errors.append(f"[{pid}] Q3 summary_options[{idx}] 문자열 아님")
                continue
            a_wc = len(a_val.strip().split())
            b_wc = len(b_val.strip().split())
            if a_wc > max_slot_words:
                errors.append(
                    f"[{pid}] Q3 summary_options[{idx}][A]는 단어 {max_slot_words}개 이하여야 함 "
                    f"({a_wc}단어: '{a_val}')"
                )
            if b_wc > max_slot_words:
                errors.append(
                    f"[{pid}] Q3 summary_options[{idx}][B]는 단어 {max_slot_words}개 이하여야 함 "
                    f"({b_wc}단어: '{b_val}')"
                )
            a_words.append(a_val.strip().lower())
            b_words.append(b_val.strip().lower())
        
        # 5개 (A) 모두 다른 단어, 5개 (B) 모두 다른 단어 (strict일 때만 체크)
        if strict:
            if len(set(a_words)) < len(a_words):
                errors.append(f"[{pid}] Q3 summary_options의 (A) 값들이 중복됨: {a_words}")
            if len(set(b_words)) < len(b_words):
                errors.append(f"[{pid}] Q3 summary_options의 (B) 값들이 중복됨: {b_words}")
    
    return errors


# ====================== 호환용 함수 (예전 코드용) ======================
def check_passage_preservation(rebuilt: str, original: str, pid: str = "?") -> list:
    """예전 코드 호환용 - 호출돼도 빈 리스트 반환 (검증 안 함)"""
    return []
