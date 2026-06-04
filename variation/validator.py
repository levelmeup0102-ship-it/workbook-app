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


def check_marker_positions(passage_with_marks: str, pid: str = "?", min_between: int = 3, position_correct: int = None, position_count: int = None, strict: bool = True) -> list:
    """마커 위치 검증:
    - strict: 개수=position_count, 1..N 연속, 문장 경계에만, 한곳 몰림 금지
    - 정답 마커 좌우는 항상 3단어+ 분산
    """
    errors = []
    positions = {}
    for i in range(1, 6):
        idx = passage_with_marks.find(f"<MARK{i}>")
        if idx >= 0:
            positions[i] = idx
    n = len(positions)

    # 개수/연속성
    if position_count in (4, 5):
        if n != position_count:
            errors.append(f"[{pid}] 마커 수 불일치: {n}개 (position_count={position_count})")
        missing = [i for i in range(1, position_count + 1) if i not in positions]
        if missing:
            errors.append(f"[{pid}] 마커 누락: MARK{missing}")
    if n < 4:
        errors.append(f"[{pid}] 마커 수 부족: {n}개 (최소 4개)")
    if n < 3:
        return errors

    sorted_marks = sorted(positions.items(), key=lambda x: x[1])

    if strict:
        # 문장 경계 검증: 각 마커 직전(다른 마커 제거 후)이 문장부호로 끝나야
        for mnum, pos in positions.items():
            before = re.sub(r"<MARK\d>", "", passage_with_marks[:pos]).rstrip()
            if before and before[-1] not in '.!?"\')]':
                errors.append(f"[{pid}] MARK{mnum} 문장 중간 배치 (직전: ...{before[-25:]})")
        # 분산 검증: 첫~마지막 마커가 본문 전체의 40%+ 에 걸쳐야 (몰림 방지)
        clean_len = len(re.sub(r"<MARK\d>", "", passage_with_marks))
        span = sorted_marks[-1][1] - sorted_marks[0][1]
        if clean_len > 0 and span < clean_len * 0.40:
            errors.append(f"[{pid}] 마커들이 한곳에 몰림 (분산 부족: {span}/{clean_len})")

    # 정답 마커 좌우 간격 (항상)
    correct_mark_num = None
    if position_correct is not None and 0 <= position_correct < len(sorted_marks):
        correct_mark_num = sorted_marks[position_correct][0]
    for i in range(len(sorted_marks) - 1):
        m1, p1 = sorted_marks[i]
        m2, p2 = sorted_marks[i + 1]
        between = passage_with_marks[p1 + len(f"<MARK{m1}>"):p2]
        between = re.sub(r"<MARK\d>", "", between).strip()
        wc = len(between.split())
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
    """유형 A 검증 - 평가원 순서형 (intro + (A)(B)(C) 3문단 + 고정 5선지)"""
    errors = []

    required = ["intro", "paragraphs", "topic_options", "topic_correct",
                "order_correct", "statements", "blank_A", "blank_B", "bogi"]
    for f in required:
        if f not in data:
            errors.append(f"[{pid}] 필수 필드 누락: {f}")
            return errors

    # Q2 order_correct: 고정 5선지 인덱스(0-4). (A)-(B)-(C)는 선지에 없어 자동 배제됨.
    v = data.get("order_correct", -1)
    if not isinstance(v, int) or not (0 <= v <= 4):
        errors.append(f"[{pid}] Q2 order_correct 범위 오류(0-4여야 함): {v}")

    # Q5 blank_A/B 단어 수 최소 5
    try:
        wa = len(data["blank_A"].split()); wb = len(data["blank_B"].split())
        if wa < 5:
            errors.append(f"[{pid}] Q5 blank_A 단어 수 부족 ({wa}개 < 5개)")
        if wb < 5:
            errors.append(f"[{pid}] Q5 blank_B 단어 수 부족 ({wb}개 < 5개)")
    except (KeyError, AttributeError) as e:
        errors.append(f"[{pid}] blank_A/B 형식 오류: {e}")

    # Q3 core_blank: 단어 수 최소 3 + <CORE_BLANK> 마커가 intro에 존재
    if data.get("core_blank_target"):
        try:
            cwc = len(data["core_blank_target"].split())
            if cwc < 3:
                errors.append(f"[{pid}] Q3 core_blank_target 단어 수 부족 ({cwc}개 < 3개)")
        except (KeyError, AttributeError):
            pass
        if "<CORE_BLANK>" not in data.get("intro", ""):
            errors.append(f"[{pid}] Q3 <CORE_BLANK> 마커가 intro에 없음 — intro 안에 표시할 것")
        # ★ Q3 정답 선지가 core_blank_target과 동일해야 함 (빈칸 범위 침범/중복 방지)
        opts = data.get("core_blank_options"); ci = data.get("core_blank_correct"); tgt = data.get("core_blank_target")
        if isinstance(opts, list) and isinstance(ci, int) and 0 <= ci < len(opts) and tgt:
            _w = lambda t: re.sub(r"[^a-z0-9 ]", " ", str(t).lower()).split()
            if _w(opts[ci]) != _w(tgt):
                errors.append(
                    f"[{pid}] Q3 정답 선지가 core_blank_target과 불일치 — "
                    f"선지에 빈칸 밖 단어가 섞였거나 범위가 다름 (정답='{opts[ci]}' vs 빈칸='{tgt}')"
                )

    # paragraphs 3개 + 각 텍스트 5단어 이상
    paras = data.get("paragraphs")
    if not isinstance(paras, list) or len(paras) != 3:
        errors.append(f"[{pid}] paragraphs는 정확히 3개((A)(B)(C))여야 함")
        paras = paras if isinstance(paras, list) else []
    para_texts = []
    for idx, ch in enumerate(paras):
        if not isinstance(ch, list) or len(ch) < 2:
            errors.append(f"[{pid}] paragraphs[{idx}] 형식 오류 (label, text 필요)")
            continue
        label, text = ch[0], ch[1]
        para_texts.append(text or "")
        if not text or not text.strip():
            errors.append(f"[{pid}] paragraphs[{idx}] {label} 텍스트가 비어있음")
        elif len(text.split()) < 5:
            errors.append(f"[{pid}] paragraphs[{idx}] {label} 텍스트 너무 짧음 ({len(text.split())}단어 < 5)")

    # ★★★ 중복 차단 (이번 핵심 버그): intro 문장이 (A)(B)(C)에 통째로 재등장하면 거부
    def _norm(t):
        t = re.sub(r"<[^>]+>", " ", t)
        t = re.sub(r"[^a-zA-Z0-9 ]", " ", t.lower())
        return re.sub(r"\s+", " ", t).strip()
    intro_n = _norm(data.get("intro", ""))
    if intro_n:
        iw = intro_n.split()
        probe = " ".join(iw[:8]) if len(iw) >= 8 else intro_n
        for idx, txt in enumerate(para_texts):
            if probe and probe in _norm(txt):
                errors.append(
                    f"[{pid}] intro 문장이 paragraphs[{idx}]에 중복 등장 — "
                    f"주어진 글은 (A)/(B)/(C)에 다시 넣지 말 것 (누락·중복 0)"
                )

    # Q5 blank_A/B 인접 검증 (paragraphs 기준, 최소 3단어 사이)
    try:
        ptext = " ".join(para_texts)
        ia = ptext.find("<BLANK_A>"); ib = ptext.find("<BLANK_B>")
        if ia >= 0 and ib >= 0:
            start = min(ia, ib) + len("<BLANK_A>"); end = max(ia, ib)
            between = ptext[start:end].replace("<BLANK_A>", "").replace("<BLANK_B>", "").strip()
            wc = len(between.split()) if between else 0
            if wc < 3:
                errors.append(f"[{pid}] Q5 blank_A와 blank_B가 너무 가까움 (사이 {wc}단어 < 3개)")
    except Exception:
        pass

    # Q5 잘라쓰기(보기) 검증
    try:
        errors += check_cutout_match(data["bogi"], [data["blank_A"], data["blank_B"]], pid, "Q5(빈칸영작)")
    except Exception as e:
        errors.append(f"[{pid}] Q5 보기 검증 예외: {e}")

    # 정답 인덱스 범위
    v = data.get("topic_correct", -1)
    if not isinstance(v, int) or not (0 <= v <= 4):
        errors.append(f"[{pid}] topic_correct 범위 오류: {v}")
    if data.get("core_blank_correct") is not None:
        v = data["core_blank_correct"]
        if not isinstance(v, int) or not (0 <= v <= 4):
            errors.append(f"[{pid}] core_blank_correct 범위 오류")

    # statements 5개
    if not isinstance(data.get("statements"), list) or len(data["statements"]) != 5:
        errors.append(f"[{pid}] statements는 5개 항목이어야 함")

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

    # position_count(4 or 5) + position_correct가 그 범위 안인지
    pc = data.get("position_count")
    if pc is not None:
        if pc not in (4, 5):
            errors.append(f"[{pid}] position_count는 4 또는 5여야 함: {pc}")
        else:
            pcorr = data.get("position_correct", -1)
            if isinstance(pcorr, int) and not (0 <= pcorr < pc):
                errors.append(f"[{pid}] position_correct({pcorr})가 position_count({pc}) 범위 밖")
    
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
                position_correct=data.get("position_correct"),
                position_count=data.get("position_count"),
                strict=True,
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
