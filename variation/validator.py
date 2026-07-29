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


# ── 문법 성립 검사용 (LLM이 만든 패러프레이즈/영작이 비문이 되는 경우 차단) ──
# 흔한 동사 원형: 주절 주어 자리에 오면 "주어 없는 비문" 의심 (gerund -ing/과거 -ed는 set에 없어 자동 통과)
_BASE_VERBS = {
    "partition", "break", "create", "make", "build", "find", "choose", "direct", "enable",
    "divide", "split", "develop", "produce", "provide", "reduce", "increase", "improve",
    "change", "control", "convert", "transform", "consider", "focus", "turn", "keep", "help",
    "allow", "cause", "lead", "drive", "shape", "form", "use", "apply", "achieve", "gain",
    "maintain", "manage", "handle", "address", "tackle", "solve", "approach", "treat", "regard",
    "view", "perceive", "recognize", "identify", "determine", "foster", "promote", "prevent",
    "ensure", "require",
}
# 정형동사 신호 (주절에 이미 동사가 있다는 표시 → 앞의 동사원형은 주어 자리였다는 뜻)
_FINITE_HINT = re.compile(
    r'\b(enables?|is|are|was|were|makes?|allows?|leads?|helps?|creates?|becomes?|results?|'
    r'provides?|requires?|drives?|causes?|fosters?|promotes?|ensures?|reduces?|increases?|'
    r'improves?|transforms?|builds?|produces?|maintains?|manages?)\b', re.I)
# 동사 존재 판정용(보수적): 이게 하나도 없으면 '동사 없는 명사구'로 본다
_VERB_WORDS = _BASE_VERBS | {
    "is", "are", "was", "were", "be", "been", "plays", "play", "have", "has", "had", "do", "does",
    "exert", "exerts", "hold", "holds", "exist", "exists", "occur", "occurs", "matter", "matters",
    "work", "works", "fluctuate", "fluctuates", "remain", "remains", "stem", "stems", "depend", "depends",
}


def looks_bare_verb_subject(sentence: str) -> bool:
    """영작 정답 문장의 주절이 '동사원형 + ... + 정형동사' 꼴이면 주어 없는 비문으로 본다.
    예: 'partition those into pieces enables responses' (X)  /  'Partitioning ... enables ...' (O)"""
    s = str(sentence or "").strip()
    m = re.match(
        r'^\s*(rather than|instead of|by|when|while|if|although|though|because|since|after|before)\b.*?,\s*(.+)$',
        s, re.I)
    main = m.group(2) if m else s
    words = main.split()
    if not words:
        return False
    w0 = re.sub(r'[^a-zA-Z]', '', words[0]).lower()
    if w0 in _BASE_VERBS:
        rest = " ".join(words[1:])
        if _FINITE_HINT.search(rest):
            return True
    return False


def _phrase_has_verb(phrase: str) -> bool:
    toks = re.sub(r'[^a-zA-Z ]', ' ', str(phrase or "").lower()).split()
    if any(t in _VERB_WORDS for t in toks):
        return True
    # 목록에 없어도: 3인칭 단수 동사는 -s로 끝남 (departs, deviates, requires, stimulates...).
    # 명사 복수도 -s지만, that/which 절 뒤 정답이라 동사일 확률이 높다 → 오탐(정상 거부)을 줄이려 동사로 간주.
    # 단, 흔한 복수명사 어미(-ies는 명사 가능성↑이나 동사도 있음)는 그대로 두고 일반 -s를 본다.
    for t in toks:
        if len(t) >= 4 and t.endswith('s') and not t.endswith('ss') and not t.endswith('us') and not t.endswith('is'):
            return True
    # -ing/-ed 형(동사 활용)도 동사 신호로 본다
    if any(t.endswith('ing') or t.endswith('ed') for t in toks if len(t) >= 4):
        return True
    return False


def q3_blank_is_nounphrase_after_clause(intro: str, correct_opt: str) -> bool:
    """Q3 빈칸 바로 앞이 that/which/who/because/whether(절 유도)인데 정답에 동사가 전혀 없으면
    'believe that [명사구]' 같은 비문 → True. (절 유도어 뒤가 아니면 검사 안 함: 보수적)"""
    filled = str(intro or "").replace("<CORE_BLANK>", " \u27e6OPT\u27e7 ")
    m = re.search(r'\b(that|which|who|because|whether)\s+\u27e6OPT\u27e7', filled, re.I)
    if not m:
        return False
    return not _phrase_has_verb(correct_opt)


# 정형동사 신호 (be/조동사/have/do + 흔한 3인칭 동사)
_FINITE_VERB = re.compile(
    r'\b(is|am|are|was|were|be|been|has|have|had|do|does|did|'
    r'can|could|will|would|shall|should|may|might|must|'
    r'enables?|requires?|involves?|makes?|allows?|leads?|becomes?|remains?|'
    r'provides?|needs?|takes?|gives?|helps?|results?|reduces?|increases?|'
    r'improves?|creates?|drives?|causes?|fosters?|promotes?|ensures?)\b', re.I)


def fill_boundary_dup(template: str, pairs) -> str:
    """빈칸에 정답을 넣었을 때 ① 정답 구절이 빈칸 앞/뒤 텍스트와 겹쳐 반복되거나
    ② 경계에서 같은 단어가 연달아 나오면(예: 'aspects aspects') 그 중복 텍스트를 반환. 없으면 None.
    (빈칸 범위가 앞/뒤 단어·구절을 먹은 경우 감지)"""
    template = template or ""
    # ② 구절 겹침: 정답 끝 k단어 == 빈칸 뒤 앞 k단어, 또는 정답 앞 k단어 == 빈칸 앞 끝 k단어
    for mk, ans in pairs:
        parts = template.split(mk, 1)
        if len(parts) < 2:
            continue
        pre, suf = parts[0], parts[1]
        for omk, _ in pairs:
            suf = suf.replace(omk, " "); pre = pre.replace(omk, " ")
        aw = re.sub(r'[^A-Za-z0-9 ]', ' ', (ans or "").lower()).split()
        sw = re.sub(r'[^A-Za-z0-9 ]', ' ', suf.lower()).split()
        pw = re.sub(r'[^A-Za-z0-9 ]', ' ', pre.lower()).split()
        for k in range(min(len(aw), len(sw)), 1, -1):
            if aw[-k:] == sw[:k]:
                return " ".join(aw[-k:])
        for k in range(min(len(aw), len(pw)), 1, -1):
            if aw[:k] == pw[-k:]:
                return " ".join(aw[:k])
    # ① 경계 인접 중복 (k=1) — 빈칸이 '맞닿는 자리'에서만 검사한다.
    #   ★ 원문 자체에 원래 있던 인접 중복(예: "questions (questions ...")은 빈칸 탓이 아니므로
    #     무시한다. 전체를 훑던 옛 방식은 그런 원문 중복까지 오탐으로 잡아 멀쩡한 지문을 거부했다.
    for mk, ans in pairs:
        parts = template.split(mk, 1)
        if len(parts) < 2:
            continue
        pre, suf = parts[0], parts[1]
        for omk, _ in pairs:
            suf = suf.replace(omk, " "); pre = pre.replace(omk, " ")
        aw = re.sub(r'[^A-Za-z0-9 ]', ' ', (ans or "").lower()).split()
        sw = re.sub(r'[^A-Za-z0-9 ]', ' ', suf.lower()).split()
        pw = re.sub(r'[^A-Za-z0-9 ]', ' ', pre.lower()).split()
        if aw and sw and aw[-1] == sw[0] and len(aw[-1]) > 2:
            return aw[-1]
        if aw and pw and aw[0] == pw[-1] and len(aw[0]) > 2:
            return aw[0]
    return None


_END_BAD = {"through", "to", "of", "in", "on", "at", "for", "with", "by", "from", "into", "onto",
            "upon", "about", "and", "or", "but", "nor", "that", "which", "who", "whose",
            "the", "a", "an", "their", "its", "his", "her",
            "can", "will", "must", "should", "would", "could", "may", "might",
            "is", "are", "was", "were", "be"}


def ends_with_function_word(s: str) -> bool:
    """문장/구절이 전치사·접속사·관사·조동사·be로 끝나면 비문 (예: '... toward solutions through')"""
    w = re.sub(r'[^A-Za-z ]', ' ', str(s or "").lower()).split()
    return bool(w) and w[-1] in _END_BAD


def despite_with_clause(s: str) -> bool:
    """Despite / In spite of + (주어+동사 절) → 비문 (Despite는 명사구만 이끔; 절은 Although)"""
    m = re.match(r'^\s*(despite|in spite of)\s+(.+)$', str(s or ""), re.I)
    if not m:
        return False
    head = " ".join(m.group(2).split()[:7])
    return bool(re.search(
        r'\b(is|are|was|were|can|could|will|would|must|should|may|might|has|have|had|'
        r'enables?|requires?|makes?|involves?|provides?|leads?|becomes?|'
        r'contains?|seeks?|needs?|wants?|offers?|shows?|includes?|lacks?|faces?|'
        r'fails?|tends?|appears?|seems?|remains?|suggests?|claims?|argues?|'
        r'believes?|finds?|gives?|takes?|uses?|helps?|allows?|causes?|creates?|'
        r'produces?|reduces?|increases?|carries?|holds?|keeps?|brings?)\b', head, re.I))


def modal_no_verb(s: str) -> bool:
    """조동사(can/will/must...) 뒤에 동사원형이 아니라 형용사가 오면 비문 (예: 'can controllable')"""
    m = re.search(r'\b(can|will|must|should|would|could|may|might)\s+(\w+)', str(s or ""), re.I)
    if not m:
        return False
    return bool(re.search(r'(able|ible|ous|ive|ful|less)$', m.group(2).lower()))


def grammar_flaws(s: str):
    """위 세 가지 흔한 비문 패턴을 한 번에 검사해 사유 리스트를 반환."""
    out = []
    if ends_with_function_word(s):
        out.append("전치사·접속사로 끝나 문장이 끊김")
    if despite_with_clause(s):
        out.append("Despite 뒤에 절(주어+동사)이 옴 — Although를 쓰거나 명사구로")
    if modal_no_verb(s):
        out.append("조동사 뒤에 동사원형이 아닌 형용사가 옴")
    return out


def gerund_start_no_finite(sentence: str) -> bool:
    """문장이 동명사(-ing)로 시작하는데 정형동사가 하나도 없으면 비문 의심.
    (분사·동명사만 나열된 'directing energy preventing reactions' 류).
    -ing로 시작하지 않으면 검사하지 않음(단순현재 문장 오탐 회피)."""
    s = str(sentence or "").strip()
    m = re.match(
        r'^\s*(rather than|instead of|by|when|while|if|although|though|because|since)\b.*?,\s*(.+)$',
        s, re.I)
    main = m.group(2) if m else s
    parts = main.split()
    if not parts:
        return False
    w0 = re.sub(r'[^a-zA-Z]', '', parts[0]).lower()
    if not w0.endswith("ing"):
        return False
    return not _FINITE_VERB.search(main)


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

    # 개수/연속성 (기본 4~5개, 지문이 짧아 4개가 안 되면 3개까지 허용)
    if position_count in (3, 4, 5):
        if n != position_count:
            errors.append(f"[{pid}] 마커 수 불일치: {n}개 (position_count={position_count})")
        missing = [i for i in range(1, position_count + 1) if i not in positions]
        if missing:
            errors.append(f"[{pid}] 마커 누락: MARK{missing}")
    if n < 3:
        errors.append(f"[{pid}] 마커 수 부족: {n}개 (최소 3개 — 4~5개 우선, 지문이 너무 짧을 때만 3개)")
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


def blank_has_punct(s: str) -> bool:
    """영작 빈칸 정답에 쉼표·세미콜론·콜론이 들어 있는가.
    보기(bogi)는 구두점을 떼고 만들어지므로, 정답 안에 쉼표가 있으면 학생이
    그것을 복원할 근거가 없어 채점이 갈린다. 끝에 붙은 것도 마찬가지."""
    t = str(s or "")
    return bool(re.search(r'[,;:]', t)) or bool(re.search(r'[.,;:!?]\s*$', t))


def _ngrams(s, n=4):
    """비교용 n-gram 집합. 소문자·영문자만 남긴다."""
    w = re.sub(r"[^a-z ]", " ", str(s or "").lower()).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def option_echoes_answer(answer: str, option: str, n: int = 4):
    """선지/진술이 다른 문항의 정답을 그대로 베꼈는지 (n-gram 겹침).

    같은 지문으로 여러 문항을 만들다 보면 Q3 정답 구절이 Q4 진술에 그대로 들어가,
    한 문항이 다른 문항의 답을 알려주는 일이 생긴다. 겹친 n-gram을 반환(없으면 빈 집합)."""
    return _ngrams(answer, n) & _ngrams(option, n)


# ── 제목 유형(B Q2) 검사 ──────────────────────────────────────
# 주제 상투구로 시작하면 '제목'이 아니라 '주제'를 쓴 것이다.
_TOPIC_OPENERS = re.compile(
    r"^\s*(the|a|an)?\s*(importance|necessity|need|role|benefit|benefits|advantage|advantages|"
    r"significance|value|way|ways|method|methods|reason|reasons|effect|effects|impact|"
    r"difficulty|difficulties|challenge|challenges|problem|problems|process|nature|"
    r"characteristic|characteristics|feature|features|function|purpose|principle|"
    r"relationship|connection|influence|consequence|consequences|limitation|limitations|"
    r"strategy|strategies|prevalence|superiority|cognitive)\s+(of|for|in|to|why|demands?)\b",
    re.I)

# 제목에서 대문자로 안 써도 되는 기능어 (관사·전치사·접속사)
_TITLE_LOWER_OK = {"a", "an", "the", "and", "or", "but", "nor", "for", "so", "yet",
                   "at", "by", "in", "of", "on", "to", "up", "as", "if", "vs",
                   "from", "into", "over", "with", "without", "than", "that"}


def title_form_flaw(s: str) -> str:
    """B Q2 선지가 제목 형식인지. 문제가 있으면 사유 문자열, 없으면 빈 문자열.

    ★ 단어 목록만으로는 계속 샌다('difficulty'처럼 빠진 명사가 나옴).
      그래서 대문자화(title case)를 함께 본다 — 제목은 주요 단어를 대문자로 쓰고,
      주제 명사구는 소문자로 쓴다. 이게 형태를 가르는 가장 견고한 신호다."""
    t = str(s or "").strip()
    if not t:
        return "비어 있음"
    ws = t.split()
    n = len(ws)
    if n < 3:
        return f"{n}단어 — 제목으로 너무 짧음 (4~10단어)"
    if n > 12:
        return f"{n}단어 — 제목으로 너무 김 (4~10단어)"
    if _TOPIC_OPENERS.match(t):
        return "주제 형식('the importance of ~')으로 시작 — 제목이 아님"
    # 대문자화: 콜론 뒤·첫 단어를 뺀 '내용어' 중 대문자로 시작한 비율
    content = [w for w in ws[1:]
               if re.sub(r"[^A-Za-z]", "", w).lower() not in _TITLE_LOWER_OK
               and len(re.sub(r"[^A-Za-z]", "", w)) > 1]
    if content:
        cap = sum(1 for w in content if w[:1].isupper())
        if cap / len(content) < 0.5:
            return ("주요 단어가 대문자로 시작하지 않음 — 주제 명사구 형태다 "
                    f"(내용어 {len(content)}개 중 대문자 {cap}개)")
    return ""


def title_shape(s: str) -> str:
    """제목의 형태 분류: colon / question / plain"""
    t = str(s or "").strip()
    if ":" in t:
        return "colon"
    if t.endswith("?"):
        return "question"
    return "plain"


def blank_order_wrong(text: str, mk_a: str, mk_b: str) -> bool:
    """본문/요약문에서 (B) 빈칸이 (A)보다 먼저 나오면 True.
    학생은 (A)→(B) 순으로 답을 쓰는데 지문이 반대면 읽기 흐름이 어긋난다."""
    t = str(text or "")
    ia, ib = t.find(mk_a), t.find(mk_b)
    return ia >= 0 and ib >= 0 and ib < ia


# ====================== 유형 A 검증 (완화) ======================
def validate_a(data: dict, original_passage: str = None, pid: str = "?", lenient: bool = False) -> list:
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

    # Q5 blank_A/B 단어 수 최소 4, 최대 12 (백스톱 — 픽커가 9 초과를 거름. 그래도 새면 여기서 막아 재생성)
    min_bw = 4  # 빈칸은 자연스러운 핵심 구절(4~6단어) 허용 (1회독처럼 하한 완화)
    max_bw = 12
    try:
        wa = len(data["blank_A"].split()); wb = len(data["blank_B"].split())
        if wa < min_bw:
            errors.append(f"[{pid}] [CRITICAL] Q5 blank_A 단어 수 부족 ({wa}개 < {min_bw}개) — 4단어 이상 필수")
        if wb < min_bw:
            errors.append(f"[{pid}] [CRITICAL] Q5 blank_B 단어 수 부족 ({wb}개 < {min_bw}개) — 4단어 이상 필수")
        if wa > max_bw:
            errors.append(f"[{pid}] [CRITICAL] Q5 blank_A 단어 수 과다 ({wa}개 > {max_bw}개) — 영작 빈칸으로 너무 김, 5~7단어로 줄일 것")
        if wb > max_bw:
            errors.append(f"[{pid}] [CRITICAL] Q5 blank_B 단어 수 과다 ({wb}개 > {max_bw}개) — 영작 빈칸으로 너무 김, 5~7단어로 줄일 것")
    except (KeyError, AttributeError) as e:
        errors.append(f"[{pid}] blank_A/B 형식 오류: {e}")

    # ★ Q5 (A)(B) 빈칸 마커가 본문에 실제로 찍혔는지 — 마킹 실패 시 한쪽만/둘 다 사라지는 버그 방지
    if data.get("blank_A") and data.get("blank_B"):
        joined = " ".join(
            p[1] for p in data.get("paragraphs", [])
            if isinstance(p, (list, tuple)) and len(p) > 1
        )
        if "<BLANK_A>" not in joined:
            errors.append(f"[{pid}] [CRITICAL] Q5 (A) 빈칸이 본문에 표시되지 않음 — blank_A 구절을 원문에서 찾지 못함 (구절을 원문 그대로 고를 것)")
        if "<BLANK_B>" not in joined:
            errors.append(f"[{pid}] [CRITICAL] Q5 (B) 빈칸이 본문에 표시되지 않음 — blank_B 구절을 원문에서 찾지 못함 (구절을 원문 그대로 고를 것)")

    # Q3 core_blank: 단어 수 최소 3 + <CORE_BLANK> 마커가 intro에 존재
    if data.get("core_blank_target"):
        try:
            cwc = len(data["core_blank_target"].split())
            min_cw = 2 if lenient else 3
            if cwc < min_cw:
                errors.append(f"[{pid}] Q3 core_blank_target 단어 수 부족 ({cwc}개 < {min_cw}개)")
        except (KeyError, AttributeError):
            pass
        if "<CORE_BLANK>" not in data.get("intro", ""):
            errors.append(f"[{pid}] [CRITICAL] Q3 핵심빈칸이 지문에 표시되지 않음 — core_blank_target 구절을 intro(첫 문장)에서 찾지 못함 (구절을 첫 문장 원문 그대로 고를 것)")
        # ★ Q3 정답은 빈칸 원문(core_blank_target)의 패러프레이즈여야 함 — 글자 그대로 베끼면 거부 (strict만)
        opts = data.get("core_blank_options"); ci = data.get("core_blank_correct"); tgt = data.get("core_blank_target")
        if not lenient and isinstance(opts, list) and isinstance(ci, int) and 0 <= ci < len(opts) and tgt:
            _w = lambda t: re.sub(r"[^a-z0-9 ]", " ", str(t).lower()).split()
            if _w(opts[ci]) == _w(tgt):
                errors.append(
                    f"[{pid}] Q3 정답이 빈칸 원문을 그대로 베낌 — 유의어/비유로 패러프레이즈할 것 "
                    f"(정답='{opts[ci]}' = 원문 '{tgt}')"
                )
        # ★ Q3 문법 성립: 빈칸 앞이 that/which 등 절 유도어인데 정답이 동사 없는 명사구면 비문 (lenient에서도 검사 — CRITICAL)
        if isinstance(opts, list) and isinstance(ci, int) and 0 <= ci < len(opts):
            if q3_blank_is_nounphrase_after_clause(data.get("intro", ""), opts[ci]):
                errors.append(
                    f"[{pid}] [CRITICAL] Q3 정답이 빈칸에 문법적으로 안 맞음 — 빈칸 앞이 절 유도어(that 등)인데 "
                    f"정답('{opts[ci]}')이 동사 없는 명사구임. 원문이 절(주어+동사)이면 정답도 절로 패러프레이즈할 것."
                )

    # ★ A Q5 영작 정답에 구두점이 들어갔는지 — 보기엔 구두점이 없어 학생이 복원 불가 (CRITICAL)
    for key in ("blank_A", "blank_B"):
        v = data.get(key)
        if v and blank_has_punct(v):
            errors.append(
                f"[{pid}] [CRITICAL] Q5 {key}에 구두점 포함 — 보기엔 구두점이 없어 학생이 복원 불가 "
                f"('{v}'). 구두점 사이의 깨끗한 구절로 다시 고를 것.")

    # ★ A Q5 (A)(B)가 본문 등장 순서와 어긋나는지
    _joined_ab = " ".join(p[1] for p in data.get("paragraphs", [])
                          if isinstance(p, (list, tuple)) and len(p) > 1)
    if blank_order_wrong(_joined_ab, "<BLANK_A>", "<BLANK_B>"):
        errors.append(
            f"[{pid}] Q5 본문에서 (B) 빈칸이 (A)보다 먼저 나옴 — 학생은 (A)→(B) 순으로 쓰므로 "
            f"라벨을 등장 순서에 맞게 바꿀 것")

    # ★ A Q5 영작 정답(blank_A/B) 비문/중복 검사 — lenient(마지막 시도)에서도 실행 (CRITICAL은 끝까지 막아야 함)
    for key in ("blank_A", "blank_B"):
        v = data.get(key)
        # blank은 원문에서 떼온 '구절'이라 동사로 시작/전치사로 끝나는 게 정상.
        # 구절에서도 명백한 비문인 '조동사+형용사'(can controllable)만 검사한다.
        if v and modal_no_verb(v):
            errors.append(f"[{pid}] [CRITICAL] Q5 {key} 비문 — 조동사 뒤에 동사원형이 아닌 형용사가 옴 ('{v}')")
    # ★ A Q5 빈칸에 정답을 넣었을 때 경계 단어/구절이 중복되는지 (빈칸이 앞/뒤 단어를 먹음)
    if data.get("blank_A") and data.get("blank_B"):
        dup = fill_boundary_dup(
            " ".join(p[1] for p in data.get("paragraphs", []) if isinstance(p, list) and len(p) > 1),
            [("<BLANK_A>", data["blank_A"]), ("<BLANK_B>", data["blank_B"])])
        if dup:
            errors.append(
                f"[{pid}] [CRITICAL] Q5 빈칸에 정답을 넣으면 '{dup}'이(가) 중복됨 "
                f"— 빈칸 범위가 앞/뒤 단어를 먹었음 (구절을 빈칸 자리에 맞게 다시 고를 것)."
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
    # ★★★ 순서형 복원 대조를 '먼저' 계산 (이게 순서 정답의 권위 있는 검사다).
    #     intro+(A)(B)(C)를 정답순서로 이으면 원문과 100% 일치해야 함. 일치하면 순서가 증명된 것.
    _recon_ok = None  # None=계산불가, True=원문과 일치, False=불일치
    if original_passage and isinstance(data.get("order_correct"), int) \
       and 0 <= data["order_correct"] <= 4 and len(para_texts) == 3:
        FIXED = [["A", "C", "B"], ["B", "A", "C"], ["B", "C", "A"], ["C", "A", "B"], ["C", "B", "A"]]
        l2t = {}
        for ch in paras:
            if isinstance(ch, list) and len(ch) >= 2:
                l2t[str(ch[0]).strip("() ").upper()] = ch[1] or ""
        restored = (data.get("intro", "") or "").replace("<CORE_BLANK>", data.get("core_blank_target", "") or "")
        for lbl in FIXED[data["order_correct"]]:
            t = l2t.get(lbl, "")
            t = t.replace("<BLANK_A>", data.get("blank_A", "") or "").replace("<BLANK_B>", data.get("blank_B", "") or "")
            restored += " " + t
        _recon_ok = (_norm(restored) == _norm(original_passage))

    # intro 중복 (보조 휴리스틱) — ★ 복원 대조가 통과(_recon_ok True)면 순서가 원문과 100% 복원되어
    #   중복이 있을 수 없으므로 이 휴리스틱은 오탐이다. 그때는 검사하지 않는다.
    intro_n = _norm(data.get("intro", ""))
    if intro_n and _recon_ok is not True:
        iw = intro_n.split()
        probe = " ".join(iw[:8]) if len(iw) >= 8 else intro_n
        for idx, txt in enumerate(para_texts):
            if probe and probe in _norm(txt):
                errors.append(
                    f"[CRITICAL][{pid}] intro 문장이 paragraphs[{idx}]에 중복 등장 — "
                    f"주어진 글은 (A)/(B)/(C)에 다시 넣지 말 것 (누락·중복 0)"
                )

    # ★★★ Q3 빈칸 겹침 방지: intro의 <CORE_BLANK>를 정답으로 채우면 원문에 그 문장이 그대로 존재해야 함.
    _intro_raw = data.get("intro", "") or ""
    _tgt = data.get("core_blank_target", "") or ""
    if original_passage and _tgt and "<CORE_BLANK>" in _intro_raw:
        _filled = _intro_raw.replace("<CORE_BLANK>", _tgt)
        if _norm(_filled) not in _norm(original_passage):
            errors.append(
                f"[CRITICAL][{pid}] Q3 빈칸 위치/범위 오류 — 빈칸을 정답으로 채우면 원문과 어긋남 "
                f"(빈칸이 앞/뒤 본문 단어를 먹었을 가능성). 빈칸은 실제로 빠진 부분만 덮을 것."
            )

    # 복원 불일치 — 계산 가능했고(_recon_ok is False) 실제로 안 맞을 때만 (진짜 순서 오류)
    if _recon_ok is False:
        errors.append(
            f"[CRITICAL][{pid}] 순서형 복원 불일치 — intro+(A)(B)(C)를 정답순서로 이어붙여도 원문과 다름 "
            f"(문장 재배치/병합/누락 또는 Q3 빈칸 위치 오류 의심). (A)(B)(C)는 원문 연속 구간만 담을 것."
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

    # ★ Q3 핵심빈칸 정답이 Q4 진술에 그대로 들어갔는지 — 한 문항이 다른 문항의 답을 흘린다.
    #   Q3 빈칸을 못 풀어도 Q4 진술만 읽으면 정답 구절을 알 수 있게 되는 구조적 결함.
    _opts3 = data.get("core_blank_options"); _ci3 = data.get("core_blank_correct")
    if isinstance(_opts3, list) and isinstance(_ci3, int) and 0 <= _ci3 < len(_opts3) \
       and isinstance(data.get("statements"), list):
        _ans3 = str(_opts3[_ci3])
        for _s in data["statements"]:
            if not (isinstance(_s, (list, tuple)) and len(_s) >= 2):
                continue
            _ov = option_echoes_answer(_ans3, str(_s[1]))
            if _ov:
                errors.append(
                    f"[{pid}] [CRITICAL] Q4 진술 '{_s[0]}'이 Q3 정답 구절을 그대로 베낌 "
                    f"({sorted(_ov)}) — Q3를 안 풀어도 답이 보인다. 진술을 빈칸 밖 문장 근거로 다시 쓸 것")

    # statements 5개
    if not isinstance(data.get("statements"), list) or len(data["statements"]) != 5:
        errors.append(f"[{pid}] statements는 5개 항목이어야 함")
    else:
        # ★ Q4 개수 정합성 — 답지는 mismatch_count를 선지 번호로 그대로 쓰는데(mismatch_count-1),
        #   이 값이 statements의 실제 불일치 개수와 다르면 "3개인데 정답은 ②" 꼴로 어긋난다.
        #   0이면 음수 인덱스가 되어 정답이 ⑤로 뒤집히고, 6 이상이면 렌더가 터진다.
        actual = sum(1 for s in data["statements"]
                     if isinstance(s, (list, tuple)) and len(s) >= 3 and not s[2])
        mc = data.get("mismatch_count")
        if not isinstance(mc, int) or not (1 <= mc <= 5):
            errors.append(
                f"[{pid}] [CRITICAL] Q4 mismatch_count 범위 오류({mc}) — 선지는 1~5개뿐. "
                f"0이면 답지에 ⑤로 잘못 찍힘")
        elif mc != actual:
            errors.append(
                f"[{pid}] [CRITICAL] Q4 개수 불일치 — mismatch_count={mc}인데 "
                f"statements의 실제 불일치는 {actual}개. 정답 번호와 나열 항목이 어긋남")
        if actual == 0:
            errors.append(
                f"[{pid}] [CRITICAL] Q4 불일치 진술이 하나도 없음 — 최소 1개는 거짓이어야 함")

    return errors


# ====================== 유형 B 검증 (완화) ======================
def validate_b(data: dict, original_passage: str = None, pid: str = "?", strict: bool = True,
               a_data: dict = None) -> list:
    """유형 B 검증 - 핵심만 체크
    strict=False면 단어 수 / 중복 검증을 풀어줌 (마지막 retry용)
    a_data: 같은 지문의 유형 A 결과(있으면). A Q1 주제와 B Q2 주제가 같은 명제를 쓰는지 본다.
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

    # ★ Q2는 '제목' 문항이다 (A Q1이 '주제'를 맡으므로 형태가 달라야 한다).
    #   같은 지문으로 A·B를 만들면 주제 명제가 겹치는데, 요구 형태를 나눠 구조적으로 해소한다.
    _topts = data.get("topic_options")
    if isinstance(_topts, list) and _topts:
        for _i, _t in enumerate(_topts):
            _flaw = title_form_flaw(_t)
            if _flaw:
                errors.append(
                    f"[{pid}] Q2 제목 선지 {_i + 1}번 형식 오류 — {_flaw}: '{_t}'")
        if strict and len(_topts) >= 5:
            _shapes = {title_shape(_t) for _t in _topts}
            if len(_shapes) < 2:
                errors.append(
                    f"[{pid}] Q2 제목 5개가 전부 같은 형태({_shapes.pop()}) — "
                    f"콜론형·의문형·동명사형 중 최소 두 가지를 섞을 것")

    # position_count(3~5) + position_correct가 그 범위 안인지 (4~5 우선, 짧으면 3 허용)
    pc = data.get("position_count")
    if pc is not None:
        if pc not in (3, 4, 5):
            errors.append(f"[{pid}] position_count는 3~5여야 함 (4~5 우선): {pc}")
        else:
            pcorr = data.get("position_correct", -1)
            if isinstance(pcorr, int) and not (0 <= pcorr < pc):
                errors.append(f"[{pid}] position_correct({pcorr})가 position_count({pc}) 범위 밖")

    # ★ Q1 삽입 정답 위치 검증 — 정답 마커 자리에 given_sentence를 도로 넣으면 원문이 복원돼야 함
    #   (LLM이 위치를 잘못 세면 답이 틀리게 나오던 문제 방지. 따옴표/공백 차이는 무시하고 위치만 검사)
    if original_passage and data.get("passage_with_marks") and data.get("given_sentence"):
        pcorr = data.get("position_correct")
        if isinstance(pcorr, int) and 0 <= pcorr <= 4:
            pwm = str(data["passage_with_marks"])
            gs = str(data["given_sentence"]).strip()
            correct_mark = f"<MARK{pcorr + 1}>"
            def _alnum(t):
                return re.sub(r"[^a-z0-9]", "", str(t).lower())
            if correct_mark not in pwm:
                errors.append(f"[{pid}] [CRITICAL] Q1 정답 마커 {correct_mark}가 본문에 없음")
            else:
                recon = pwm.replace(correct_mark, " " + gs + " ")
                recon = re.sub(r"<MARK\d>", "", recon)
                if _alnum(recon) != _alnum(original_passage):
                    errors.append(
                        f"[{pid}] [CRITICAL] Q1 삽입 정답 위치 오류 — 정답({pcorr + 1}번) 자리에 "
                        f"주어진 문장을 넣어도 원문이 복원되지 않음 (주어진 문장이 실제로 빠진 위치를 정답으로 표시할 것)"
                    )

    # ★ Q4 요약문 / Q3 요약문에 (A)(B) 빈칸 표시가 반드시 있어야 함 (완성문 금지)
    #   strict/soft 무관 필수 — 빈칸이 없으면 문제 자체가 성립하지 않음
    bst = str(data.get("blank_summary_template", "") or "")
    if "(A)" not in bst or "(B)" not in bst:
        errors.append(f"[{pid}] Q4 blank_summary_template에 (A)/(B) 빈칸이 없음 — 완성문 말고 (A)(B) placeholder를 남길 것")
    sst = str(data.get("summary_template", "") or "")
    if "(A)" not in sst or "(B)" not in sst:
        errors.append(f"[{pid}] Q3 summary_template에 (A)/(B) 빈칸이 없음 — (A)(B) placeholder를 남길 것")

    # ★ Q4 blank_A, blank_B 단어 수 (최소 4, 최대 12 백스톱)
    min_blank_words = 4  # 빈칸은 자연스러운 핵심 구절(4~6단어) 허용
    max_blank_words = 12
    try:
        wa = len(data["blank_A"].split())
        wb = len(data["blank_B"].split())
        if wa < min_blank_words:
            errors.append(f"[{pid}] [CRITICAL] Q4 blank_A 단어 수 부족 ({wa}개 < {min_blank_words}개) — 4단어 이상 필수")
        if wb < min_blank_words:
            errors.append(f"[{pid}] [CRITICAL] Q4 blank_B 단어 수 부족 ({wb}개 < {min_blank_words}개) — 4단어 이상 필수")
        if wa > max_blank_words:
            errors.append(f"[{pid}] [CRITICAL] Q4 blank_A 단어 수 과다 ({wa}개 > {max_blank_words}개) — 영작 빈칸으로 너무 김, 5~7단어로 줄일 것")
        if wb > max_blank_words:
            errors.append(f"[{pid}] [CRITICAL] Q4 blank_B 단어 수 과다 ({wb}개 > {max_blank_words}개) — 영작 빈칸으로 너무 김, 5~7단어로 줄일 것")
    except (KeyError, AttributeError) as e:
        errors.append(f"[{pid}] B blank_A/B 형식 오류: {e}")

    # ★ Q5 topic_writing_answer 단어 수 (strict 14개, soft 3개)
    min_topic_words = 10  # 너무 짧은 것만 차단 (자연스러움 우선)
    try:
        twc = len(data["topic_writing_answer"].split())
        if twc < min_topic_words:
            errors.append(
                f"[{pid}] Q5 topic_writing_answer 단어 수 부족 ({twc}개 < {min_topic_words}개) "
                f"— 더 완전한 문장으로 작성"
            )
    except (KeyError, AttributeError):
        errors.append(f"[{pid}] B topic_writing_answer 형식 오류")

    # ★ B Q5 주제문 / Q4 요약영작 비문·중복 검사 — soft(마지막 시도)에서도 실행 (CRITICAL은 끝까지 막아야 함)
    tw = data.get("topic_writing_answer")
    if tw and looks_bare_verb_subject(tw):
        errors.append(
            f"[{pid}] [CRITICAL] Q5 주제문이 동사원형으로 시작하는 비문 꼴 "
            f"('{tw}') — 동명사/명사구 주어로 쓸 것 (예: 'Partitioning ...' not 'partition ...')."
        )
    for flaw in grammar_flaws(tw):
        errors.append(f"[{pid}] [CRITICAL] Q5 주제문 비문 — {flaw} ('{tw}')")
    for key in ("blank_A", "blank_B"):
        v = data.get(key)
        # blank은 원문에서 떼온 '구절'이라 동사로 시작/전치사로 끝나는 게 정상.
        # 구절에서도 명백한 비문인 '조동사+형용사'(can controllable)만 검사한다.
        if v and modal_no_verb(v):
            errors.append(f"[{pid}] [CRITICAL] Q4 {key} 비문 — 조동사 뒤에 동사원형이 아닌 형용사가 옴 ('{v}')")
    # ★ B Q4 영작 정답에 구두점이 들어갔는지 (A Q5와 동일 사유, CRITICAL)
    for key in ("blank_A", "blank_B"):
        v = data.get(key)
        if v and blank_has_punct(v):
            errors.append(
                f"[{pid}] [CRITICAL] Q4 {key}에 구두점 포함 — 보기엔 구두점이 없어 학생이 복원 불가 "
                f"('{v}'). 구두점 사이의 깨끗한 구절로 다시 고를 것.")

    # ★ B Q4 (A)(B) 등장 순서 + 두 빈칸 간격
    _bst = str(data.get("blank_summary_template", "") or "")
    if blank_order_wrong(_bst, "(A)", "(B)"):
        errors.append(
            f"[{pid}] Q4 요약문에서 (B) 빈칸이 (A)보다 먼저 나옴 — 라벨을 등장 순서에 맞게 바꿀 것")
    if "(A)" in _bst and "(B)" in _bst:
        _ia, _ib = _bst.find("(A)"), _bst.find("(B)")
        _lo, _hi = (_ia, _ib) if _ia < _ib else (_ib, _ia)
        _between = _bst[_lo + 3:_hi].strip()
        _wc = len(_between.split()) if _between else 0
        if _wc < 3:
            errors.append(
                f"[{pid}] [CRITICAL] Q4 (A)와 (B) 빈칸 사이 {_wc}단어 — 붙어 있으면 사실상 빈칸 하나다. "
                f"최소 3단어 떨어뜨릴 것")

    # ★ B Q4 빈칸에 정답을 넣었을 때 경계 단어/구절 중복 (빈칸이 앞/뒤 단어를 먹음)
    if data.get("blank_A") and data.get("blank_B") and data.get("blank_summary_template"):
        dup = fill_boundary_dup(
            data["blank_summary_template"],
            [("(A)", data["blank_A"]), ("(B)", data["blank_B"])])
        if dup:
            errors.append(
                f"[{pid}] [CRITICAL] Q4 요약문 빈칸에 정답을 넣으면 '{dup}'이(가) 중복됨 "
                f"— 빈칸 범위가 앞/뒤 단어를 먹었음 (정답 구절을 빈칸 자리에 맞게 다시 고를 것)."
            )

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

    # ★ Q3 정답 (A)(B)가 본문 단어 그대로면 거부 (패러프레이즈/추상화 강제, strict만)
    if strict and original_passage:
        sc = data.get("summary_correct")
        opts = data.get("summary_options")
        if isinstance(opts, list) and isinstance(sc, int) and 0 <= sc < len(opts) \
           and isinstance(opts[sc], list) and len(opts[sc]) >= 2:
            psg_tokens = set(tokenize_for_comparison(original_passage))
            for slot, w in zip(["(A)", "(B)"], opts[sc][:2]):
                wt = tokenize_for_comparison(str(w))
                if wt and all(t in psg_tokens for t in wt):
                    errors.append(
                        f"[{pid}] Q3 정답 {slot} '{w}'가 본문에 그대로 등장 — "
                        f"본문 단어를 베끼지 말고 한 단계 추상화/패러프레이즈할 것"
                    )

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

        # ★ Q3 오답 설계 — 정답 단어가 다른 선지에도 나타나야 '한쪽만 보고 찍기'를 막는다.
        #   (A) 5개가 전부 다르고 (B) 5개도 전부 다르면, 정답의 (A)만 알아도 답이 결정된다.
        #   정답 단어를 오답 행에도 한 번씩 흘려야 (A)(B)를 교차 확인하게 된다.
        sc = data.get("summary_correct")
        if isinstance(sc, int) and 0 <= sc < len(a_words) and len(a_words) == 5:
            if a_words.count(a_words[sc]) == 1 and b_words.count(b_words[sc]) == 1:
                errors.append(
                    f"[{pid}] Q3 오답 설계 약함 — 정답 (A)'{a_words[sc]}'·(B)'{b_words[sc]}'가 "
                    f"각각 한 번씩만 나와 한쪽만 보고 답이 결정된다. 정답 단어를 오답 행에도 "
                    f"한 번씩 배치해 교차 확인이 필요하게 만들 것")

        # 5개 (A) 모두 다른 단어, 5개 (B) 모두 다른 단어 (strict일 때만 체크)
        #   ※ 위 교차 배치와 상충하므로, 교차 배치를 도입하면 이 검사는 완화한다.
        if strict:
            if len(set(a_words)) < 3:
                errors.append(f"[{pid}] Q3 summary_options의 (A) 값이 너무 겹침: {a_words}")
            if len(set(b_words)) < 3:
                errors.append(f"[{pid}] Q3 summary_options의 (B) 값이 너무 겹침: {b_words}")

    # ★ B Q2 주제 선지가 같은 지문 A Q1 주제 선지와 같은 명제인지 (A 결과가 있을 때만)
    #   같은 원문으로 A·B를 각각 만들다 보면 주제 명제가 겹쳐, 한쪽 답이 다른 쪽 힌트가 된다.
    if isinstance(a_data, dict):
        _ao = a_data.get("topic_options"); _aci = a_data.get("topic_correct")
        _bo = data.get("topic_options"); _bci = data.get("topic_correct")
        if isinstance(_ao, list) and isinstance(_bo, list) \
           and isinstance(_aci, int) and isinstance(_bci, int) \
           and 0 <= _aci < len(_ao) and 0 <= _bci < len(_bo):
            # B 정답 ↔ A 오답, A 정답 ↔ B 오답 양방향
            for _lab, _ans, _opts in (("B Q2 정답 ↔ A Q1 선지", _bo[_bci], _ao),
                                      ("A Q1 정답 ↔ B Q2 선지", _ao[_aci], _bo)):
                for _i, _o in enumerate(_opts):
                    _ov = option_echoes_answer(_ans, str(_o))
                    if _ov:
                        errors.append(
                            f"[{pid}] {_lab} {_i + 1}번이 같은 명제 ({sorted(_ov)}) — "
                            f"한 유형의 답이 다른 유형의 힌트가 된다. 명제를 갈 것")

    return errors


# ====================== 호환용 함수 (예전 코드용) ======================
def check_passage_preservation(rebuilt: str, original: str, pid: str = "?") -> list:
    """예전 코드 호환용 - 호출돼도 빈 리스트 반환 (검증 안 함)"""
    return []
