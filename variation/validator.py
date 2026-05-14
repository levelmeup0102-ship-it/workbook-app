"""
variation/validator.py
변형문제 데이터 무결성 검증 모듈

검증 항목:
1. 지문 변형/축소 금지 (원문 100% 보존)
2. 잘라쓰기 보기 = 정답 정확 일치 (대소문자 무시, 개수 일치)
3. 위치 마커 분산 (같은 자리 X)
4. 정답 인덱스 범위 (0~4)
5. BLANK_A/B 각각 정확히 1번씩
6. 셔플 강제 (정답 순서와 같으면 안 됨)
"""
import re
from collections import Counter


def normalize_text(s: str) -> str:
    """공백·줄바꿈 정규화"""
    return " ".join(s.split())


def check_passage_preservation(rebuilt: str, original: str, pid: str = "?") -> list:
    """원문 보존 - 단어 단위 100% 일치"""
    orig_norm = normalize_text(original)
    rebuilt_norm = normalize_text(rebuilt)
    if orig_norm == rebuilt_norm:
        return []
    
    orig_words = orig_norm.split()
    rebuilt_words = rebuilt_norm.split()
    
    for i, (rw, ow) in enumerate(zip(rebuilt_words, orig_words)):
        if rw != ow:
            ctx_start = max(0, i - 3)
            ctx_end = min(len(rebuilt_words), i + 5)
            return [
                f"[{pid}] 원문 변형/축소 감지 (단어 #{i})\n"
                f"   재구성: ...{' '.join(rebuilt_words[ctx_start:ctx_end])}...\n"
                f"   원문:   ...{' '.join(orig_words[ctx_start:ctx_end])}..."
            ]
    
    if len(rebuilt_words) != len(orig_words):
        diff = len(rebuilt_words) - len(orig_words)
        sign = "+" if diff > 0 else "-"
        return [f"[{pid}] 원문 길이 변경 ({sign}{abs(diff)}단어)"]
    return []


def check_cutout_match(bogi: list, answer_parts: list, pid: str = "?", q_name: str = "Q") -> list:
    """잘라쓰기 보기 vs 정답 일치 검증 (대소문자 무시)"""
    errors = []
    
    all_ans_words = []
    for part in answer_parts:
        for w in part.split():
            cleaned = w.rstrip(".,!?;:").lower()
            if cleaned:
                all_ans_words.append(cleaned)
    
    bogi_lower = [w.lower() for w in bogi]
    
    bogi_c = Counter(bogi_lower)
    ans_c = Counter(all_ans_words)
    
    if bogi_c != ans_c:
        missing = ans_c - bogi_c
        extra = bogi_c - ans_c
        errors.append(
            f"[{pid}] {q_name} 보기 ≠ 정답 단어\n"
            f"   누락: {dict(missing) if missing else '없음'}\n"
            f"   잉여: {dict(extra) if extra else '없음'}"
        )
    
    if bogi_lower == all_ans_words:
        errors.append(f"[{pid}] {q_name} 보기가 셔플되지 않음")
    
    return errors


def check_marker_positions(passage_with_marks: str, pid: str = "?", min_between: int = 5) -> list:
    """마커 위치 분산 검증 (같은 자리에 두 마커 X)"""
    errors = []
    positions = {}
    for i in range(1, 6):
        idx = passage_with_marks.find(f"<MARK{i}>")
        if idx >= 0:
            positions[i] = idx
    
    if len(positions) < 4:
        errors.append(f"[{pid}] 마커 수 부족: {len(positions)}개 (최소 4개)")
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
                f"[{pid}] 마커 {m1}과 {m2} 사이 텍스트 너무 짧음 ({wc}단어 < {min_between}최소)"
            )
    return errors


# ====================== 유형 A 통합 검증 ======================
def validate_a(data: dict, original_passage: str = None, pid: str = "?") -> list:
    """유형 A 전체 검증"""
    errors = []
    
    # Q5 잘라쓰기 검증
    errors += check_cutout_match(
        data["bogi"],
        [data["blank_A"], data["blank_B"]],
        pid, "Q5(빈칸영작)"
    )
    
    # 정답 인덱스 범위
    for key in ["topic_correct", "order_correct"]:
        if not (0 <= data.get(key, -1) <= 4):
            errors.append(f"[{pid}] {key} 범위 오류: {data.get(key)}")
    
    if "core_blank_correct" in data and data["core_blank_correct"] is not None:
        if not (0 <= data["core_blank_correct"] <= 4):
            errors.append(f"[{pid}] core_blank_correct 범위 오류")
    
    # BLANK 개수
    chunks_text = " ".join(c[1] for c in data["chunks"])
    if chunks_text.count("<BLANK_A>") != 1:
        errors.append(f"[{pid}] <BLANK_A> 개수 ≠ 1")
    if chunks_text.count("<BLANK_B>") != 1:
        errors.append(f"[{pid}] <BLANK_B> 개수 ≠ 1")
    
    # 진술 5개 중 불일치 개수 = mismatch_count
    n_false = sum(1 for _, _, ok in data["statements"] if not ok)
    if "mismatch_count" in data and data["mismatch_count"] != n_false:
        errors.append(
            f"[{pid}] mismatch_count({data['mismatch_count']}) ≠ 실제 불일치 ({n_false})"
        )
    
    # 원문 보존 (제공된 경우)
    if original_passage:
        order_correct = data["order_correct"]
        correct_order = data["order_options"][order_correct].split("-")
        chunks_dict = {c[0]: c[1] for c in data["chunks"]}
        try:
            reordered = " ".join(
                chunks_dict[label]
                    .replace("<BLANK_A>", data["blank_A"])
                    .replace("<BLANK_B>", data["blank_B"])
                    .replace("<CORE_BLANK>", data.get("core_blank_target", ""))
                for label in correct_order
            )
            full_rebuilt = data["lead"].replace(
                "<CORE_BLANK>", data.get("core_blank_target", "")
            ) + " " + reordered
            errors += check_passage_preservation(full_rebuilt, original_passage, pid)
        except KeyError as e:
            errors.append(f"[{pid}] 순서 라벨 오류: {e}")
    
    return errors


# ====================== 유형 B 통합 검증 ======================
def validate_b(data: dict, original_passage: str = None, pid: str = "?") -> list:
    """유형 B 전체 검증"""
    errors = []
    
    # 정답 인덱스 범위
    for key in ["summary_correct", "topic_correct", "position_correct"]:
        if not (0 <= data.get(key, -1) <= 4):
            errors.append(f"[{pid}] {key} 범위 오류")
    
    # Q4 잘라쓰기 (요약 영작)
    errors += check_cutout_match(
        data["blank_summary_bogi"],
        [data["blank_A"], data["blank_B"]],
        pid, "Q4(요약영작)"
    )
    
    # Q5 잘라쓰기 (주제 영작)
    errors += check_cutout_match(
        data["topic_writing_bogi"],
        [data["topic_writing_answer"]],
        pid, "Q5(주제영작)"
    )
    
    # 마커 위치 분산
    errors += check_marker_positions(data["passage_with_marks"], pid)
    
    # 원문 보존 (마커 제거 + 주어진 문장 삽입 후 비교)
    if original_passage:
        target_mark = data["position_correct"] + 1
        full = data["passage_with_marks"].replace(
            f"<MARK{target_mark}>", data["given_sentence"]
        )
        full = re.sub(r"<MARK\d>", "", full)
        errors += check_passage_preservation(full, original_passage, pid)
    
    return errors
