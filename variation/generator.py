"""
variation/generator.py
변형문제 데이터 생성 - Claude API + httpx Supabase REST + 자동 검증 + 재시도

기존 워크북 시스템과 일관성 유지:
- supa.py처럼 httpx 사용
- anthropic, supabase 라이브러리 추가 의존성 없음
"""
import os
import hashlib
import random
import re
import traceback
from typing import Optional

import httpx

from variation.prompts import SYSTEM_PROMPT_A, SYSTEM_PROMPT_B, extract_json_from_response, TOPIC_SENTENCE_SYS, build_topic_sentence_prompt, SUMMARY_SENTENCE_SYS, build_summary_sentence_prompt, TRANSLATE_SYS, build_translate_prompt, VOCAB_SYS, build_vocab_prompt, Q5_BLANK_SYS, build_q5_blank_prompt, INSERT_SYS, build_insert_prompt, SOLVE_SYS, build_solve_prompt, GRAMMAR_ERROR_SYS, build_grammar_error_prompt
from variation.validator import validate_a, validate_b, check_marker_positions, fill_boundary_dup, modal_no_verb, grammar_count, grammar_replace_once
import variation.validator as _validator


# ════════════════════════════════════════════════════════════════
# blank_has_punct 정책 통일 (_s95)
#   renderer.bogi_words 는 A·B 둘 다 중간 구두점을 앞 단어에 붙여 제시한다
#   ('original,' 'signals,'). _s66부터 그렇다. 그러니 빈칸 안쪽 쉼표는
#   학생이 복원할 수 있고, 배열 결과가 원문과 글자까지 일치한다.
#   그런데 validator.blank_has_punct 만 _s65의 '쉼표 통째 배제'에 남아 있었다.
#   실측: generator 가 통과시킨 'desert, which is Bir Tawil' 을 validator 가
#   CRITICAL 로 쳐서 03번 A 가 3회 재시도 끝에 통째로 누락됐다.
#   B Q4 도 같은 문제를 안고 있다 — _span_from_marks_summary 는 안쪽 쉼표를
#   허용하는데 validator 가 막는다(잠재 버그).
#   → 여기서 한 번에 맞춘다. 끝 구두점과 문장 경계는 계속 거부한다.
#   ※ validator.py 를 직접 고칠 수 있게 되면 이 블록을 지우고 그쪽에 옮길 것.
# ════════════════════════════════════════════════════════════════
def _blank_has_punct_v2(s) -> bool:
    t = str(s or "")
    if re.search(r'[.,;:!?]\s*$', t):     # 끝 구두점 — 배열 결과가 갈린다
        return True
    return bool(re.search(r'[.!?]', t))    # 문장 경계 — 빈칸은 한 문장 안에


_validator.blank_has_punct = _blank_has_punct_v2

from variation.vocab_q3 import (normalize_llm_vocab, validate_vocab,
                                blank_token_spans)



# ============================================================
# 문장 분리 (0회독 pipeline.py와 동일 — 경칭/약어/따옴표 보호, 무손실)
# ============================================================
def split_sentences(text: str) -> list:
    """영어 지문 문장 분리 (원문 무손실). 0회독 워크북과 동일 로직."""
    protected = text

    def protect_quote_internals(match):
        inner = match.group(1)
        open_q = match.group(0)[0]
        close_q = match.group(0)[-1]
        protected_inner = re.sub(
            r'([.!?])\s+([A-Z])',
            lambda m2: f"{m2.group(1)}§QSEP§{m2.group(2)}",
            inner
        )
        return open_q + protected_inner + close_q

    protected = re.sub(r'["\u201c](.*?)["\u201d]', protect_quote_internals, protected, flags=re.DOTALL)

    abbrevs = [
        'Dr.', 'Mr.', 'Ms.', 'Mrs.', 'Prof.', 'Jr.', 'Sr.', 'St.',
        'vs.', 'etc.', 'No.', 'Vol.', 'Fig.', 'Gen.', 'Gov.', 'Rev.',
        'Sgt.', 'Cpl.', 'Lt.', 'Co.', 'Inc.', 'Ltd.', 'Corp.', 'Dept.',
        'Est.', 'al.', 'e.g.', 'i.e.', 'U.S.', 'U.K.', 'U.N.',
    ]
    replacements = {}
    for ab in abbrevs:
        token = ab.replace('.', '§DOT§')
        pattern = r'(?<!\w)' + re.escape(ab)
        if re.search(pattern, protected):
            replacements[token] = ab
            protected = re.sub(pattern, token, protected)

    def protect_initial(m):
        return m.group(0).replace('.', '§DOT§')
    protected = re.sub(r'(?<!\w)([A-Z])\.\s*(?=[A-Z][\.\s]|[A-Z][a-z])', protect_initial, protected)

    sentences = [s.strip() for s in re.split(
        r'(?<=[.!?])\s+(?=[\u201c\u201d\u0022]?[A-Z])|(?<=[.!?][\u201c\u201d\u0022])\s+(?=[\u201c\u201d\u0022]?[A-Z])',
        protected
    ) if s.strip()]

    restored = []
    for s in sentences:
        for token, original in replacements.items():
            s = s.replace(token, original)
        s = s.replace('§DOT§', '.')
        s = s.replace('§QSEP§', ' ')
        restored.append(s)
    return restored


# 순서형 5선지: order_correct 인덱스 → 복원(원문) 라벨 순서
FIXED_ORDER = [["A", "C", "B"], ["B", "A", "C"], ["B", "C", "A"], ["C", "A", "B"], ["C", "B", "A"]]


def build_order_blocks_a(en_text: str, pid: str = "?", seed_extra: str = "") -> Optional[dict]:
    """
    원문을 코드로 분할 → intro + (A)(B)(C) + order_correct 구성 (원문 무손실).
    LLM이 단락을 만들지 않으므로 복원검증이 절대 깨지지 않는다.
    반환: {"intro", "paragraphs":[["A",t],["B",t],["C",t]], "order_correct"} 또는 None(분할 불가).
    """
    sents = split_sentences(en_text)
    if len(sents) < 3:
        print(f"[VAR][A][{pid}] 문장 {len(sents)}개 — 순서배열 불가 (최소 3문장 필요)")
        return None

    # ★ 제목 분리: 첫 문장이 짧은 라벨(≤4단어)이면 제목으로 보고 떼어 둠.
    #   연속된 라벨(제목+소제목+"Where:"...)도 모두 흡수. 단 본문이 최소 4문장은 남아야 함.
    title = ""
    while len(sents) >= 5 and len(sents[0].split()) <= 4:
        title = (title + " " + sents[0]).strip()
        sents = sents[1:]

    # ★ 문장이 3개뿐이면 intro 를 두지 않고 3문장을 그대로 (A)(B)(C)로 쓴다.
    #   intro 를 떼면 남는 게 2문장이라 3단락을 못 만들고, 그러면 순서 문제가
    #   통째로 빠져 A 유형이 4문항으로 줄어든다. 짧은 지문에서 자주 생긴다.
    if len(sents) == 3:
        intro_text = title.strip()          # 제목이 있으면 그것만, 없으면 빈 문자열
        rest = sents
        print(f"[VAR][A][{pid}] 문장 3개 — intro 없이 (A)(B)(C) 구성")
    else:
        # intro = (제목 +) 첫 진짜 문장. 제목은 intro 앞에 붙여 원문 무손실 유지.
        if len(sents) < 4:
            return None
        intro_text = (title + " " + sents[0]).strip() if title else sents[0].strip()
        rest = sents[1:]
    if len(rest) < 3:
        return None

    # rest를 연속 3덩어리로 (앞쪽에 +1)
    k = len(rest)
    sizes = [k // 3, k // 3, k // 3]
    for i in range(k % 3):
        sizes[i] += 1
    blocks, idx = [], 0
    for s in sizes:
        blocks.append(" ".join(rest[idx:idx + s]).strip())
        idx += s
    # blocks[0]=원문1번째덩어리, [1]=2번째, [2]=3번째

    # 라벨 셔플: 원문순서 그대로(A-B-C)는 선지에 없으니 제외
    seed = int(hashlib.md5((pid + seed_extra + en_text[:40]).encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    perm = [0, 1, 2]
    for _ in range(10):
        rng.shuffle(perm)
        # (A)=blocks[perm[0]], (B)=blocks[perm[1]], (C)=blocks[perm[2]]
        label_of = {}
        for li, bi in enumerate(perm):
            label_of[bi] = ["A", "B", "C"][li]
        restore = [label_of[0], label_of[1], label_of[2]]  # 원문순서대로의 라벨
        if restore in FIXED_ORDER:
            break
    else:
        restore = ["B", "A", "C"]
        perm = [1, 0, 2]

    paragraphs = [
        ["A", blocks[perm[0]]],
        ["B", blocks[perm[1]]],
        ["C", blocks[perm[2]]],
    ]
    order_correct = FIXED_ORDER.index(restore)
    return {"intro": intro_text, "paragraphs": paragraphs, "order_correct": order_correct}

# ═══════════════════════════════════════════════════════════════════════
# 안내문·공고문 판별 (_s152)
#
#   수능 27·28번(안내문·공고문)에서 유형 A 는 통째로 죽던 자리다.
#   Q3 어휘는 '방향을 가진 낱말' 다섯 개가 필요한데 안내문엔 그런 말이 없다
#   (participation·registration·notice…). Q2 순서배열도 성립하지 않는다 —
#   항목 나열이라 순서를 바꿔도 논리가 깨지지 않는다.
#   → 안내문은 주제·일치·빈칸영작 3문항으로 낸다.
#
#   판별 기준은 짐작이 아니라 실측으로 맞췄다. passages 전수 1332개 대조:
#     · 안내문 14개 전부 적중(모의고사 27·28번 11 + 수특 2 + 교과서 1)
#     · 오탐 0개
#
#   ★ 가장 위험한 오탐은 대화문이다. 'Beth:' 'Neil:' 'Host:' 도 머리말이라
#     머리말 개수만 세면 대화문이 9개까지 나온다(영어1 YBM(박) 02번).
#     그래서 머리말은 ①안내문 어휘를 담았거나 ②두 단어 이상 전부 대문자일
#     때만 센다. 사람 이름은 둘 다 아니다.
#   ★ 그것만으론 모자란다. MYTH:/FACT:(비상(홍) Read More) 같은 대문자
#     머리말도 있고, VITRA FIRE STATION.(지학사 Read More 2) 같은 소제목도
#     있다. 그래서 '실무 정보' 다섯 갈래 중 두 갈래 이상이 함께 있어야 한다.
#     안내문이 아니면 요금과 신청이 같이 나오지 않는다. 실측에서 이 셋 다
#     실무정보 0갈래였다.

_NOTICE_HEADER_WORDS = frozenset("""
when where date dates day days time times hour hours place location venue site sites
schedule program programs course courses class classes session sessions event events
activity activities highlight highlights offering offerings promotion promotions
fee fees cost costs price prices payment admission ticket tickets refund discount
participant participants participation eligibility age ages level levels who
registration register signup sign-up reservation reservations booking apply application
deadline submission submit entry entries contact address email phone website
note notes notice details detail guideline guidelines rule rules requirement requirements
prize prizes award awards criteria judging benefit benefits theme topic format
instructor menu duration capacity parking transportation dress food drinks
information more what how why special offer offers membership members
""".split())

# 실무 정보 다섯 갈래 — 안내문이면 이 중 둘 이상이 함께 나온다.
_NOTICE_LOGI_PATTERNS = (
    ("요금", r"\$\d|\bfees?\b|\bfree\b|\bdiscounts?\b|\bprices?\b|\badmission\b"),
    ("일시", r"\d{1,2}:\d{2}|\d ?[ap]\.?m\.?\b|"
             r"\b(?:January|February|March|April|May|June|July|August|September|"
             r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\.? ?\d{1,2}"),
    ("연락", r"www\.|https?://|[A-Za-z0-9]@[A-Za-z0-9]|visit (?:our|the|us)\b|"
             r"for more (?:information|details)"),
    ("신청", r"sign ?-?up|registration|\bregister\b|reservation|\bbooking\b|book now|"
             r"first-come|submission|\bdeadline\b|\bentries\b|entry per|\bparticipants?\b|\btickets?\b"),
    ("자격", r"\bages? \d|\d+ years old|per person|\bmaximum\b|limited to|open to (?:all|the)"),
)


def _notice_headers(en_text: str) -> list:
    """문장 첫머리의 머리말을 (시작위치, 문자열)로 모은다. 같은 자리는 한 번만."""
    found = {}
    # ① 콜론형:  'Registration:'  'When & Where:'  'What You'll Learn:'
    for m in re.finditer(r"(?:^|(?<=[.!?\n]))\s*([A-Z][A-Za-z'&/ ]{0,40}?)\s*:", en_text):
        found.setdefault(m.start(1), m.group(1).strip())
    # ② 전부 대문자형:  'PARTICIPATION.'  'ON THE BUS.'  'WHEN AND WHERE.'
    for m in re.finditer(r"(?:^|(?<=[.!?\n]))\s*([A-Z][A-Z'&/ ]{2,60}?)\s*[.:](?=\s|$)", en_text):
        found.setdefault(m.start(1), m.group(1).strip())
    return sorted(found.items())


def notice_signals(en_text: str) -> dict:
    """안내문 판별 신호를 그대로 돌려준다 (로그·테스트용)."""
    hdrs = []
    for _pos, h in _notice_headers(en_text):
        words = [w.strip("'&/.").lower() for w in h.split()]
        words = [w for w in words if w]
        if not words:
            continue
        if any(w in _NOTICE_HEADER_WORDS for w in words):
            hdrs.append(h)
        elif h.isupper() and len(words) >= 2:
            # 사람 이름 한 단어(MYTH/FACT)는 제외 — 두 단어 이상만
            hdrs.append(h)
    logi = [name for name, pat in _NOTICE_LOGI_PATTERNS
            if re.search(pat, en_text, re.IGNORECASE)]
    return {"headers": hdrs, "hdr": len(hdrs), "logi": logi, "n_logi": len(logi)}


# ════════════════════════════════════════════════════════════════
# ★ 도표 지문도 순서·어휘가 성립하지 않는다 (_s163)
#   수능 도표형(25번)은 "The graph above shows …" 로 시작해 수치를 비교한다.
#   · 순서배열 — 항목 나열이라 섞어도 논리가 안 깨진다
#   · 어휘 — 방향을 가진 내용어가 거의 없다. 최상급·수치 표현뿐이라
#     모델이 계속 'most' 같은 기능어를 정답으로 고른다.
#   실측(25년 고1 9월 25번): Q3 어휘가 세 번 다 'most' 를 골라 3회 실패,
#   그 재시도가 크레딧을 태우다 400(잔액 부족)으로 26번까지 통째로 못 만들었다.
#   → 안내문과 **같은 3문항 경로**로 보낸다 (주제·불일치·빈칸영작).
#   ★ 판별은 '도표 자기 언급' 한 줄이면 충분하다. 도표 지문은 예외 없이
#     그 문장으로 시작하고, 일반 지문에는 나올 수 없는 표현이라 오탐이 없다.
# ════════════════════════════════════════════════════════════════
_CHART_RE = re.compile(
    r"\b(?:the\s+)?(?:above\s+)?(?:graph|chart|table|figure|diagram)s?\s+"
    r"(?:above\s+)?(?:shows?|illustrates?|presents?|displays?|compares?|"
    r"indicates?|represents?)\b"
    r"|\b(?:graph|chart|table|figure)s?\s+above\b"
    r"|\bshown\s+in\s+the\s+(?:graph|chart|table|figure)\b"
    r"|\baccording\s+to\s+the\s+(?:graph|chart|table|figure)\b",
    re.IGNORECASE)


def is_chart(en_text: str, pid: str = "?") -> bool:
    """도표·표 지문이면 True. 도표를 가리키는 자기 언급 한 곳이면 확정."""
    m = _CHART_RE.search(en_text or "")
    print(f"[VAR][A][{pid}] 도표 판별 {'✔ 도표' if m else '— 도표 아님'}"
          + (f" ({m.group(0)!r})" if m else ""))
    return bool(m)


def is_notice(en_text: str, pid: str = "?") -> bool:
    """안내문·공고문이면 True. 머리말 2개 이상 + 실무정보 2갈래 이상."""
    sig = notice_signals(en_text or "")
    ok = sig["hdr"] >= 2 and sig["n_logi"] >= 2
    print(f"[VAR][A][{pid}] 안내문 판별 {'✔ 안내문' if ok else '— 일반 지문'} "
          f"(머리말 {sig['hdr']}개 {sig['headers'][:4]} / 실무정보 {sig['logi']})")
    return ok


def build_notice_blocks_a(en_text: str, pid: str = "?") -> Optional[dict]:
    """
    안내문용 단락 분할. 순서배열을 내지 않으므로 **섞지 않는다** — 원문 순서 그대로.
    라벨은 A/B/C 를 유지한다(Q5·검증 기계가 라벨로 단락을 찾는다).
    화면에서 라벨을 감추는 일은 템플릿이 layout=='notice' 로 처리한다.
    """
    sents = split_sentences(en_text)
    if len(sents) < 4:
        print(f"[VAR][A][{pid}] 안내문 문장 {len(sents)}개 — 4문장 미만이라 분할 불가")
        return None
    intro = sents[0].strip()
    rest = sents[1:]
    if len(rest) < 3:
        return None
    k = len(rest)
    sizes = [k // 3, k // 3, k // 3]
    for i in range(k % 3):
        sizes[i] += 1
    blocks, idx = [], 0
    for s in sizes:
        blocks.append(" ".join(rest[idx:idx + s]).strip())
        idx += s
    return {"intro": intro,
            "paragraphs": [["A", blocks[0]], ["B", blocks[1]], ["C", blocks[2]]],
            "order_correct": None,
            "layout": "notice"}


def build_insert_blocks_b(en_text: str, pid: str = "?", preferred: int = None) -> Optional[dict]:
    """원문에서 문장 하나를 떼어 given_sentence로, 나머지에 마커를 균등 배치해
    삽입문제(Q1)를 무손실로 재구성한다. (1회독 pipeline.step2_order 방식 이식)

    마커 배치 원칙 — 1회독과 동일:
      · 남은 문장이 5개 이상이면 마커 5개를 균등 5분할로 배치한다.
        _interval = n/5, 위치 = int(_interval * (i + 0.5))
      · 정답 자리(삽입문이 원래 있던 곳)가 그 5개에 없으면, 가장 가까운 마커를
        정답 자리로 옮긴다. 정답은 반드시 선지 안에 있어야 하므로.
      · 남은 문장이 4개면 마커 4개, 3개면 3개 — 자리 수가 모자라 5개를 못 만든다.
        (삽입문을 빼고 4문장만 남는 지문이 실제로 있다)

    LLM이 preferred 로 문장을 지목하면 그것부터 시도하고, 복원·마커 검증을 통과하는
    첫 구성을 반환한다. 재구성 불가하면 None(→ 기존 LLM 결과 유지).
    """
    def _alnum_ib(t):
        return re.sub(r"[^a-z0-9]", "", str(t).lower())

    sents = split_sentences(en_text)
    m = len(sents)
    if m < 4:
        return None

    mid = m // 2
    # LLM 픽을 최우선, 없으면 가운데 문장부터 (첫 문장은 제외 — 앞을 가리킬 대상이 없다)
    order = sorted(range(1, m), key=lambda g: abs(g - mid))
    if isinstance(preferred, int) and 1 <= preferred < m:
        order = [preferred] + [g for g in order if g != preferred]

    for g in order:
        given = sents[g]
        remaining = sents[:g] + sents[g + 1:]
        n = len(remaining)
        if n < 3:
            continue
        real_gap = g                      # 삽입문이 원래 있던 자리(= 갭 인덱스)

        # ── 1회독 방식: 균등 분할 ──────────────────────────────
        #   갭은 0..n 까지 n+1 곳이지만, 맨 앞(0)은 문장 앞이라 쓰지 않는다.
        #   1회독은 int(interval*(i+0.5)) 로 각 구간의 중앙을 집는다.
        target = 5 if n >= 5 else n       # 4문장이면 4개, 3문장이면 3개
        if n >= 5:
            interval = n / 5
            positions = [int(interval * (i + 0.5)) for i in range(5)]
        else:
            positions = list(range(1, n + 1))[:target]

        positions = sorted(set(x for x in positions if 1 <= x <= n))
        # 정답 자리를 반드시 포함 — 없으면 가장 가까운 마커를 옮긴다
        if 1 <= real_gap <= n and real_gap not in positions and positions:
            closest = min(range(len(positions)),
                          key=lambda x: abs(positions[x] - real_gap))
            positions[closest] = real_gap
            positions = sorted(set(positions))
        # 개수가 모자라면 빈 자리로 채운다
        if len(positions) < target:
            for cand in range(1, n + 1):
                if cand not in positions:
                    positions.append(cand)
                if len(positions) >= target:
                    break
            positions = sorted(set(positions))[:target]

        if real_gap not in positions or len(positions) < 3:
            continue

        pos_correct = positions.index(real_gap)
        pos_count = len(positions)

        # ── 지문 재구성 ────────────────────────────────────────
        parts, mi = [], 0
        pset = set(positions)
        for si in range(n + 1):
            if si in pset and mi < pos_count:
                parts.append(f"<MARK{mi + 1}>")
                mi += 1
            if si < n:
                parts.append(remaining[si])
        pwm = " ".join(parts).strip()

        # ── 검증: 정답 자리에 도로 넣으면 원문이 복원되는가 ─────
        recon = pwm.replace(f"<MARK{pos_correct + 1}>", " " + given + " ")
        recon = re.sub(r"<MARK\d>", "", recon)
        if _alnum_ib(recon) != _alnum_ib(en_text):
            continue
        errs = check_marker_positions(pwm, pid, min_between=3,
                                      position_correct=pos_correct,
                                      position_count=pos_count, strict=True)
        if errs:
            continue
        if pos_count < 5:
            print(f"[VAR][B][{pid}] Q1 삽입 — 남은 문장 {n}개라 선지 {pos_count}개")
        return {"given_sentence": given, "passage_with_marks": pwm,
                "position_correct": pos_correct, "position_count": pos_count}
    return None


_Q5_MODALS = {"can", "will", "must", "should", "would", "could", "may", "might", "shall"}


def _quote_ok(s: str, allow_comma: bool = False) -> bool:
    """빈칸 후보 검사: 문장경계(.!?) 없고, 따옴표 '짝'이 갈리지 않음(균형).
    따옴표가 있어도 쌍이 맞으면 허용 — "good student" 통째는 OK, 여는 짝만 먹으면 제외.

    ★ allow_comma=True 면 구절 '안쪽'의 쉼표·세미콜론·콜론을 허용한다 (_s94, A Q5 전용).
      옛 정책(_s65)은 쉼표를 통째로 배제했다 — 보기에서 구두점을 떼니 학생이 쉼표
      자리를 복원할 수 없다는 이유였다. 그런데 renderer.bogi_words 는 _s66부터
      중간 구두점을 앞 단어에 붙여 제시한다('signals,' 'first,'). B Q5 주제영작이
      이미 그 방식으로 돌고 있고, A Q5만 옛 정책에 남아 있었다.
      실측: LLM 픽 거부 5건 중 3건이 쉼표 하나 때문이었다
      ('uses not one, but multiple routes' — not A but B 구문이라 좋은 빈칸인데 버려졌다).
    ★ B Q4(_b_candidates)는 기본값(False) 그대로다. 그쪽 validator 가 아직 쉼표를
      거부하므로 같이 풀면 어긋난다. B는 별도로 결정한다."""
    if re.search(r'[.!?]', _mask_abbrev_dots(s)):   # 약어의 점은 문장 경계가 아니다 (_s144)
        return False
    if not allow_comma and re.search(r'[,;:]', s):
        return False
    # 대시(─ — –)도 배제. 하이픈(-)은 south-facing 같은 복합어라 허용한다.
    if re.search(r'[\u2500\u2014\u2013]', s):
        return False
    if s.count('"') % 2 != 0:
        return False
    if s.count('\u201c') != s.count('\u201d'):
        return False
    return True


def _strip_edge_punct(s: str) -> str:
    """구절 양끝의 구두점·공백 제거. 픽커가 span을 자를 때 끝에 붙어 오는
    쉼표/마침표를 떨어낸다. 안쪽 구두점은 _quote_ok가 이미 거른다."""
    return str(s or "").strip().strip(",.;:!?").strip()


# (더) 빈칸 경계 어휘 — 시작/끝 공통 사용
_BAD_EDGE = {"the","a","an","of","for","to","in","on","at","by","with","from","into","onto",
             # ★ 전치사 보강 (_s116) — 절반이 빠져 있었다.
             #   실측: 'Rival claims over' 가 통과했다('over' 가 목록에 없었다).
             #   전치사로 끝나면 목적어가 빈칸 밖에 남아 갈린다.
             # ★ 부사로도 쓰여 문장을 끝맺을 수 있는 말은 뺀다 (_s116) —
             #   around / off / near / past / behind 등. 기출
             #   'tracks pretty closely with how she gets around' 가 정답이다.
             "over","under","above","below","about","through","across",
             "between","among","during","against","toward","towards","within",
             "without","beside","besides","upon","until","till",
             "along","amid","despite","except","inside","outside",
             "per","since","throughout","underneath","unlike","versus","via",
             # ★ 종속접속사·관계사·의문사 보강 (_s118) — 절반이 빠져 있었다.
             #   실측: 'conventions determine whether' 가 통과했다.
             #   이런 말로 끝나면 뒤에 와야 할 절이 빈칸 밖에 남아 갈린다.
             "whether","although","though","unless","where","how","why","what",
             "whom","whose","before","after","nor","yet","whereas","wherever",
             "whenever","whatever","whoever","lest","provided","supposing",
             "and","or","but","that","which","who","whose","whom","as","than","is","are",
             "was","were","be","been","being","this","these","those","their","her","his","its",
             "our","your","my","not","no","so","if","when","while","because",
             "they","we","i","he","she","it","you",
             "have","has","had","having","do","does","did",
             "may","might","can","could","will","would","shall","should","must",
             # ★ _s104 — 뒤에 명사가 와야 하는 한정사·수식어. 여기서 끊으면 잘린 것이다.
             #   실측: 'region forces both' (both nations 에서 nations 가 잘림)
             "both","each","every","either","neither","another","other","such",
             "more","most","less","least","many","much","few","several","some","any",
             "very","own","same","only","just","also","even","rather","instead",
             "like","there","then",
             # ★ _s154 — 실측(25년 고1 9월 21번): 'later leave your body by various'
             #   에서 'means' 가 밖에 남아 시험지가 'they ⬜ means.' 가 됐다.
             #   'various·certain·numerous…' 는 뒤에 명사가 반드시 온다.
             "various","certain","numerous","countless","multiple","plenty","lots"}
# 완화 모드에서 시작으로 절대 허용하지 않는 것 (기존 bad_start와 동일)
# ★ 시작 경계 (_s98) — 기출 23개 정답 빈칸 실측으로 다시 잡았다.
#   첫 단어가 기능어인 것이 7/23(30%)이다:
#     'the less similarity is required...'  'the real product being sold is you'
#     'the commonalities between us...'     'a justification for converting...'
#     'a comprehensive description of...'   'is often counterproductive in cases...'
#     'not in the perception of the figure but...'
#   → 관사·전치사·be동사·부정어로 시작하는 건 정상이다. 막으면 안 된다.
#   막아야 하는 것은 '앞 절과 이어붙는 접속사·관계사'뿐이다. 그런 걸로 시작하면
#   빈칸 앞이 잘린 것처럼 읽힌다.
_BAD_START_MIN = {"and", "or", "but", "nor", "yet", "so",
                  "that", "which", "who", "whom", "whose",
                  "because", "although", "though", "while", "since", "unless",
                  "as", "if", "when", "whereas"}


def _clean_boundary_ok(phrase: str, full_text: str, strict: bool = True) -> bool:
    """빈칸 경계가 깔끔한지 검사한다.

    ★ 시작과 끝의 기준이 다르다 (_s98, 기출 23개 실측).
        시작 — 거의 안 가린다. 기능어 시작이 7/23(30%)다.
               막는 것은 접속사·관계사뿐 — 그걸로 시작하면 앞이 잘린 것처럼 읽힌다.
        끝   — 엄격하다. 기능어로 끝나는 기출 빈칸은 0/23이다.
    옛 코드는 양쪽을 같은 목록(_BAD_EDGE)으로 봐서 'is known as vicarious functioning'
    같은 정상 빈칸을 거부했다 — 기출에 'is often counterproductive in cases of conflict'
    가 있다.

    strict 는 이제 '시작'에만 영향을 준다:
      strict=True  : 시작도 접속사·관계사 + 주격대명사·조동사까지 본다(코드 픽용)
      strict=False : 시작은 접속사·관계사만 본다(LLM 픽용 — 기출 기준)
    끝은 어느 모드에서나 _BAD_EDGE 로 엄격히 본다."""
    # ★ 끝 검사에서 대명사는 뺀다 (_s98). 대명사로 끝나는 구절은 목적어·보어가
    #   제자리에 있는 완결된 형태다 — 기출 'the real product being sold is you'.
    #   막아야 하는 건 소유격 한정사(their/its/your...)로 끝나는 것이다. 뒤에
    #   명사가 와야 하므로 잘린 게 맞다.
    # ★★ 주격 전용 대명사로 끝나면 뒤에 동사가 남는다 (_s144).
    #   _s98 이 대명사를 통째로 풀어준 근거는 기출 'the real product being sold is you'
    #   였는데, 그건 보어 자리라 문장이 거기서 끝난다. 'knowing I' 는 다르다 —
    #   주어만 넣고 동사('was in')를 빈칸 밖에 남겼다(실측 25년 고1 9월 19번).
    #   목적격 형태가 따로 있는 대명사(I/he/she/we/they)는 끝에 오면 절이 잘린 것이고,
    #   주격·목적격이 같은 you/it 은 문장을 끝맺을 수 있으므로 계속 허용한다.
    bad_end = (_BAD_EDGE - {"they", "we", "he", "she", "it", "you", "i"}) | {
        "i", "he", "she", "we", "they"}
    # ★ 하이픈 복합어로 끝나면 잘린 것이다 (_s105).
    #   'twenty-second' 'well-known' 'long-term' 은 뒤에 꾸밀 명사가 따라온다.
    #   실측: 'straight line of the twenty-second' — twenty-second parallel 을 쪼갰다.
    #   ※ 하이픈이 있어도 홀로 쓰는 말은 예외로 둔다(self-esteem, one-of-a-kind 등 명사).
    _HYPHEN_OK_END = {"one-of-a-kind", "self-esteem", "well-being", "know-how",
                      "trade-off", "by-product", "side-effect", "vice-versa"}
    # ★ strict=True 는 **코드가 기계적으로 잘라내는 경로**용이다 (_s116).
    #   거기엔 '기출 30%가 기능어로 시작한다'는 근거가 없다 — 그 통계는 LLM 이
    #   논지를 보고 고른 자리 얘기다. 코드가 자를 때는 전치사·관사로 시작하면
    #   앞이 잘린 조각이 된다. 실측: 'over a coastal'(전치사 시작)이 나갔다.
    #   strict=False 는 LLM 픽용이라 접속사·관계사만 막는다(기출 근거 그대로).
    bad_start = (_BAD_EDGE | {"they", "we", "he", "she", "it", "you", "i",
                              "have", "has", "had", "do", "does", "did",
                              "may", "might", "can", "could", "will",
                              "would", "shall", "should", "must"}
                 ) if strict else _BAD_START_MIN
    ws = phrase.split()
    if not ws:
        return False
    if phrase.count("(") != phrase.count(")") or phrase.count("[") != phrase.count("]"):
        return False
    bare = lambda w: re.sub(r"[^A-Za-z'-]", "", w).lower()
    if bare(ws[0]) in bad_start or bare(ws[-1]) in bad_end:
        return False
    _last = bare(ws[-1])
    if "-" in _last and _last not in _HYPHEN_OK_END:
        return False
    m = re.search(re.escape(phrase), full_text)
    if m:
        before = full_text[:m.start()].split()
        after = full_text[m.end():].split()
        if ws[-1][:1].isupper() and after and after[0][:1].isupper():
            return False
        if ws[0][:1].isupper() and before and before[-1][:1].isupper():
            return False
    return True


# 약어·이니셜의 마침표 — 문장 경계가 아니다 (_s144)
#   자리(인덱스)를 유지해야 하므로 지우지 않고 같은 길이의 '§' 로 가린다.
_ABBREV_DOT = re.compile(
    r"\b(?:[ap]\.m|u\.s|u\.k|u\.n|ph\.d|e\.g|i\.e|et\.al|"
    r"dr|mr|mrs|ms|prof|jr|sr|st|vs|etc|no|vol|fig|inc|ltd|corp|dept|co)\.",
    re.I)
_INITIAL_DOT = re.compile(r"\b[A-Za-z]\.")
# 점을 이미 품고 있어 오인 여지가 없는 약어 — 잘린 끝에서 마지막 점을 되살릴 때 쓴다
_ABBREV_CUT = re.compile(r"\b(?:[ap]\.m|u\.s|u\.k|u\.n|ph\.d|e\.g|i\.e)$", re.I)


def _mask_abbrev_dots(t: str) -> str:
    """약어·1글자 이니셜의 마침표를 '§' 로 가린 사본. 길이·인덱스는 그대로.

    ★ 약어의 **마지막** 점이 '공백+대문자' 앞이면 문장도 거기서 끝날 수 있다
      ('… to 7 p.m. This change …'). 그 점은 가리지 않는다 — 가리면 두 문장에
      걸친 빈칸이 만들어진다. 대신 그 자리에서 자르면 'p.m' 이 남으므로
      _cut_before_punct 가 점을 되붙인다.
    """
    out = list(t)
    for pat in (_ABBREV_DOT, _INITIAL_DOT):
        for m in pat.finditer(t):
            for k in range(m.start(), m.end()):
                if out[k] != ".":
                    continue
                if k == m.end() - 1 and re.match(r'\s+["\u201c(]?[A-Z]', t[k + 1:k + 4]):
                    continue          # 문장 끝일 수 있다 — 열어둔다
                out[k] = "§"
    return "".join(out)


def _cut_before_punct(sub: str, min_w: int = 4, sentence_only: bool = False) -> str:
    """구절 안·끝의 구두점 직전까지 자른다. 남은 단어가 min_w 미만이면 빈 문자열.

    구두점은 빈칸 밖에 남는다 — 지문에 그대로 인쇄되고 학생은 그 앞부분만 배열한다.
      'dwindle and trail off, over the course'  →  'dwindle and trail off'
      'convince more readers for the whole story.' → 'convince more readers for the whole story'
    쉼표 위치를 학생이 알 수 없으므로 구두점을 정답에 포함시키면 채점이 갈린다."""
    t = str(sub or "").strip()
    if not t:
        return ""
    # ★★ 약어 안의 마침표는 문장 경계가 아니다 (_s144).
    #   실측: 'extend the library's operating hours to 7 p.m.' 를 'p.' 에서 잘라
    #   빈칸이 '... to 7 p' 가 됐다. 본문엔 '.m.' 만 덜렁 남고 보기엔 'p' 가
    #   단독 토큰으로 나갔다(25년 고1 9월 18번). 기준 (라)의 'U.S.·100,000'
    #   구두점 분리와 같은 유형이다.
    #   → 자를 자리를 찾을 때만 약어의 점을 가려두고, 자르는 것은 원문 t 에서 한다.
    probe = _mask_abbrev_dots(t)
    m = re.search(r'[.!?]' if sentence_only else r'[.!?,;:]', probe)
    if m:
        t = t[:m.start()].strip()
        # 문장을 끝내는 마침표가 약어의 마지막 점이기도 하면('… 7 p.m. This change')
        # 자른 자리에 'p.m' 이 남는다. 약어는 온전해야 하므로 점을 되붙인다.
        if _ABBREV_CUT.search(t):
            t += "."
            return t if len(t.split()) >= min_w else ""
    # ★ 약어로 끝나면 그 마침표는 단어의 일부다 — 떼면 'p.m' 이 된다 (_s144).
    _tail = t.rstrip()
    _last = _tail.split()[-1] if _tail.split() else ""
    if _last and (_ABBREV_DOT.fullmatch(_last) or _INITIAL_DOT.fullmatch(_last)):
        t = _tail
    else:
        t = t.rstrip('.,;:!?').strip()      # 끝 구두점은 잘림의 신호라 떼어낸다
    return t if len(t.split()) >= min_w else ""


# ═══════════════════════════════════════════════════════════════════════
# 빈칸이 절 경계를 넘는가 (_s153)
#
#   실측(공통영어2 비상(홍) 1과 6번) — 한 지문에서 두 빈칸이 다 이랬다:
#     (A) "map, each gap you fill and dot you"
#         원문: Since you've designed your own [map], [each gap you fill and dot you] make …
#         종속절의 목적어부터 시작해 주절 주어까지 먹고 동사 make 만 밖에 남겼다.
#         학생이 보는 것: "your own ⬜ make" — 뼈대가 안 남아 복원 불가.
#     (B) "up to you, but your dot map"
#         원문: … is [up to you], but [your dot map] should show …
#         등위접속사 but 이 빈칸 안에 숨어 문장이 둘이라는 것조차 안 보인다.
#
#   기출 23문항 115선지 실측: **두 절에 걸친 빈칸이 하나도 없다.** 전부 성분 하나다.
#
#   ★ 쉼표 자체를 막지 않는다. 그건 _s94·_s95 에서 되돌린 실수다 —
#     'claimed the original, straight border of 1899' 나
#     'uses not one, but multiple routes' 는 좋은 빈칸이다.
#     막는 것은 **쉼표 뒤에서 새 절이 시작되는 경우**뿐이다.

_CLAUSE_COORD = {"but", "so", "yet", "nor"}
_CLAUSE_SUBORD = {"because", "although", "though", "while", "since", "when",
                  "whenever", "if", "unless", "whereas", "which", "who", "whom",
                  "whose", "that", "where"}
# 쉼표 바로 뒤의 한정사·대명사는 거의 언제나 새 절(또는 동격절)의 시작이다.
_CLAUSE_HEAD = {"i", "you", "he", "she", "it", "we", "they",
                "each", "every", "this", "these", "those",
                "my", "your", "his", "her", "its", "our", "their",
                # ★ 관사도 넣는다 (_s157). 실측(24년 고1 9월 22번):
                #   'evolutionary perspective, an emotion is a kind' 가 뽑혀
                #   전치사구 끝 + 주절 시작을 한 빈칸으로 물었다. 쉼표 뒤가
                #   'an' 이라 _CLAUSE_HEAD 에 없어 통과했다.
                "a", "an", "the"}
# not A but B / either A or B 는 상관접속이지 절 접속이 아니다.
_CORRELATIVE = {"not", "either", "neither", "both"}


def crosses_clause(span: str) -> bool:
    """빈칸 구절이 절 경계를 넘으면 True (_s153)."""
    toks = str(span or "").split()
    for i, t in enumerate(toks[:-1]):
        if not t.endswith(","):
            continue
        nxt = re.sub(r"[^A-Za-z']", "", toks[i + 1]).lower()
        if not nxt:
            continue
        before = {re.sub(r"[^A-Za-z']", "", w).lower() for w in toks[:i + 1]}
        if nxt in _CLAUSE_COORD and not (before & _CORRELATIVE):
            return True
        if nxt in _CLAUSE_SUBORD:
            return True
        if nxt in _CLAUSE_HEAD:
            return True
        if nxt in ("and", "or") and not (before & _CORRELATIVE):
            # ', and she eventually became' 처럼 뒤에 주어+동사가 오면 절 접속이다.
            tail = [re.sub(r"[^A-Za-z']", "", w).lower() for w in toks[i + 2:i + 4]]
            if tail and tail[0] in _CLAUSE_HEAD:
                return True
    return False


# ── 절 꼬리만 물고 술부를 밖에 두는 빈칸 (_s160) ────────────────────
#   crosses_clause 는 **쉼표가 있을 때만** 본다. 쉼표 없이 두 절이 이어지면
#   그대로 통과했다.
#     원문: … is up to you but [your dot map] should show …
#     쉼표를 빼면 'up to you but your dot map' 이 후보로 살아남는다.
#     학생이 보는 것: "is ⬜ should show" — 둘째 절의 주어만 빈칸 안이고
#     술부는 밖이라, 무엇을 넣어야 문장이 되는지 단서가 없다.
#   ★ 등위 명사구(soil and water pollution)와 구별하는 열쇠는 **빈칸 밖**이다.
#     명사구 등위는 뒤에 동사가 안 오고, 절이 잘린 경우는 반드시 동사가 온다.
#     그래서 span 만으로는 못 본다 — 지문을 같이 받아야 한다.
_CLAUSE_JOIN = {"but", "and", "or", "so", "yet", "because", "although", "though",
                "while", "since", "unless", "whereas", "that", "which", "who", "when"}
_SUBJ_HEAD = {"i", "you", "he", "she", "it", "we", "they", "this", "these", "those",
              "the", "a", "an", "my", "your", "his", "her", "its", "our", "their"}
_AUXV = {"is", "are", "was", "were", "be", "been", "am", "do", "does", "did",
         "have", "has", "had", "can", "could", "will", "would", "may", "might",
         "must", "should", "shall", "need", "seems", "seem", "becomes", "become",
         # ★ 불규칙 과거형 (_s160) — 어미 규칙(-ed/-ing/-s)으로는 안 잡힌다.
         #   실측: 'was tired and the whole team' 뒤의 'knew' 를 못 봤다.
         #   현재형 원형(break/show/make)은 넣지 않는다 — 명사로도 흔히 쓰여
         #   등위 명사구까지 절로 오인한다.
         "knew", "said", "made", "took", "saw", "came", "went", "got", "gave",
         "found", "thought", "told", "felt", "left", "kept", "brought", "began",
         "grew", "held", "meant", "met", "ran", "sat", "stood", "won", "wrote",
         "spoke", "broke", "chose", "drove", "fell", "forgot", "heard", "led",
         "lost", "paid", "sent", "sold", "taught", "understood", "built",
         "spent", "drew", "threw", "rose", "drank", "ate", "knew", "wore"}


def _verbish(w: str) -> bool:
    w = re.sub(r"[^A-Za-z']", "", w).lower()
    return bool(w) and (w in _AUXV or re.search(r"(ed|ing|es|s)$", w) is not None)


def clause_tail_cut(span: str, ptext: str) -> bool:
    """빈칸이 둘째 절의 주어까지만 물고 술부를 밖에 두면 True (_s160)."""
    toks = str(span or "").split()
    if len(toks) < 4:
        return False
    ks = [k for k, t in enumerate(toks)
          if 1 <= k <= len(toks) - 2
          and re.sub(r"[^A-Za-z']", "", t).lower() in _CLAUSE_JOIN]
    if not ks:
        return False
    k = ks[-1]
    tail = toks[k + 1:]
    head = re.sub(r"[^A-Za-z']", "", tail[0]).lower()
    if head not in _SUBJ_HEAD:          # 접속사 뒤가 주어류가 아니면 절이 아니다
        return False
    if any(_verbish(t) for t in tail[1:]):
        return False                    # 술부가 빈칸 안에 있다 — 온전한 절
    # 빈칸 **밖** 첫 낱말이 동사면 그 절의 술부가 잘린 것이다
    i = ptext.find(span)
    if i < 0:
        return False
    nxt = ptext[i + len(span):].split()
    return bool(nxt) and _verbish(nxt[0])


def _q5_candidates(ptext: str, min_w: int = 4, max_w: int = 8) -> list:
    """단락에서 '문장 중간 연속 구절'(verbatim) 후보 생성. 가운데 우선.
    문장경계/따옴표 포함 제외, 조동사 시작 제외, 단락 내 유일 등장만.

    (더) 2단계: 깐깐한 경계(strict)로 먼저 훑고, 후보가 하나도 없으면 완화 경계로
    한 번 더 훑는다. 짧은 단락에서 후보가 소진돼 A가 통째로 누락되던 위험 없이
    경계 품질만 올린다."""
    spans = [(m.start(), m.end()) for m in re.finditer(r'\S+', ptext)]
    toks = [ptext[s:e] for s, e in spans]
    n = len(toks)

    def _scan(strict):
        cands = []
        for L in range(min_w, max_w + 1):
            for i in range(1, n - L):  # 양끝 한 토큰씩 비워 경계 확보
                j = i + L - 1
                sub = ptext[spans[i][0]:spans[j][1]]
                # ★ 구두점은 빈칸 밖에 남긴다. 'dwindle and trail off, over the course'에서
                #   쉼표 든 구절을 통째로 버리면 'trail off over'(쉼표 건너뛴 자리)가 뽑힌다.
                #   대신 구두점 직전까지 잘라 'dwindle and trail off'를 후보로 만든다.
                #   지문에는 쉼표가 그대로 인쇄되고 학생은 그 앞부분만 배열한다.
                sub = _cut_before_punct(sub, min_w, sentence_only=True)
                if not sub:
                    continue
                if not _quote_ok(sub, allow_comma=True):
                    continue
                # ★ 절 경계를 넘는 구절은 뺀다 (_s153). 실측: 후보 8.3% 감소,
                #   후보 0 단락 0개 — 소진 위험 없이 품질만 오른다.
                if crosses_clause(sub):
                    continue
                if clause_tail_cut(sub, ptext):      # ★ _s160
                    continue
                if not _clean_boundary_ok(sub, ptext, strict=strict):
                    continue
                fw = re.sub(r'[^a-z]', '', toks[i].lower())
                if fw in _Q5_MODALS:
                    continue
                if ptext.count(sub) != 1:
                    continue
                mid = abs((i + j) / 2 - n / 2)
                cands.append((mid, sub))
        cands.sort(key=lambda x: x[0])
        return [c[1] for c in cands]

    out = _scan(True)
    if not out:
        out = _scan(False)
    return out


# ════════════════════════════════════════════════════════════════
# 영작 빈칸 상한 (_s98) — A 와 B 는 근거가 되는 기출이 달라 값도 다르다.
#   A Q5 : 수능 32~34번 빈칸 115선지 실측 3~12단어, 평균 7.4, 12초과 0개  → 12
#   B Q4 : 학교 기말 기출 실측 3~9단어, 평균 5.9                          → 9
#   옛 코드는 하나(9)로 묶어 A Q5 의 10~12단어 정상 빈칸이 코드 픽 경로에서
#   거부됐다. LLM 픽 경로는 12를 쓰는데 코드 픽만 9라 기준이 어긋나 있었다.
# ════════════════════════════════════════════════════════════════
MAX_BLANK_WORDS_A = 12   # A Q5 빈칸영작
MAX_BLANK_WORDS_B = 9    # B Q4 요약영작
MAX_BLANK_WORDS = MAX_BLANK_WORDS_B   # 하위호환 (옛 이름을 참조하는 코드가 있으면 B 기준)


def _span_from_marks(paragraphs, mark, pid="?", lab="?") -> Optional[str]:
    """LLM이 지목한 {para, starts_with, ends_with}로 원문에서 구절을 잘라낸다.

    ★ 문자열을 통째로 받으면 LLM이 논지를 자기 말로 요약해 원문에 없는 구절을 만든다
      (실측: 'The territory known as Bir Tawil'). 시작·끝 단어만 받고 그 사이는
      코드가 원문에서 그대로 떼어내면 창작이 원천 차단된다.
    Q3 어휘가 para/idx 를 받아 코드가 그 자리 단어를 쓰는 것과 같은 방식."""
    if not isinstance(mark, dict):
        return None
    try:
        pi = int(mark.get("para", -1))
    except Exception:
        return None
    if pi < 0 or pi >= len(paragraphs):
        print(f"[VAR][A][{pid}] Q5 지목({lab}) para={pi} 범위 밖")
        return None
    text = paragraphs[pi][1]
    st = re.sub(r"\s+", " ", str(mark.get("starts_with", "")).strip())
    en = re.sub(r"\s+", " ", str(mark.get("ends_with", "")).strip())
    if not st or not en:
        print(f"[VAR][A][{pid}] Q5 지목({lab}) starts_with/ends_with 비어 있음")
        return None

    # 단어 경계로 시작 위치 찾기 (여러 번 나오면 첫 번째)
    ms = re.search(r"(?<!\w)" + re.escape(st), text)
    if not ms:
        # 다른 단락에 있으면 para 를 잘못 적은 것 — 어느 단락인지 알려준다
        _elsewhere = [k for k, (_l, _t) in enumerate(paragraphs)
                      if k != pi and re.search(r"(?<!\w)" + re.escape(st), _t)]
        _hint = f" (실제로는 para={_elsewhere[0]}에 있음)" if _elsewhere else ""
        print(f"[VAR][A][{pid}] Q5 지목({lab}) 시작어 '{st}' 가 para={pi}"
              f"({paragraphs[pi][0]}) 안에 없음{_hint}")
        return None
    me = re.search(r"(?<!\w)" + re.escape(en) + r"(?!\w)", text[ms.start():])
    if not me:
        print(f"[VAR][A][{pid}] Q5 지목({lab}) 끝어 '{en}' 시작어 뒤에 없음")
        return None
    span = text[ms.start():ms.start() + me.end()].strip()

    # ★ LLM 자기신고(word_count)와 실제가 다르면 남긴다 — 세지 않았다는 뜻이다.
    _wc_said = mark.get("word_count")
    _wc_real = len(span.split())
    if isinstance(_wc_said, int) and _wc_said != _wc_real:
        print(f"[VAR][A][{pid}] Q5 지목({lab}) 자기신고 {_wc_said}단어 ≠ 실제 {_wc_real}단어 "
              f"— 세지 않고 답했다")

    # ★ 코드는 자르지 않는다 — 조건을 지켜서 지목하는 것이 LLM의 일이다.
    #   뒤에서 잘라 맞추면 'very few of your readers would make it to your dramatic'처럼
    #   목적어(conclusion)가 빠진 어정쩡한 구절이 나온다. 조건 위반은 거부하고
    #   폴백(코드 픽)으로 넘긴다. 프롬프트가 STEP 4에서 같은 조건을 명시한다.
    ws = span.split()
    if len(ws) > 12:
        print(f"[VAR][A][{pid}] Q5 지목({lab}) {len(ws)}단어 — 12단어 초과, ends_with를 앞으로 당겨야 함")
        return None
    if len(ws) < 4:
        print(f"[VAR][A][{pid}] Q5 지목({lab}) {len(ws)}단어 — 4단어 미만")
        return None
    if re.search(r"[.!?,;:]", span):
        print(f"[VAR][A][{pid}] Q5 지목({lab}) 구두점 포함 — 구두점 앞에서 끊어야 함: '{span[:50]}'")
        return None
    return span


# 마지막 Q5 거부 사유 — 재시도 프롬프트에 그대로 돌려준다.
#   "다시 골라라"만 하면 LLM은 같은 실수를 반복한다. 무엇이 왜 걸렸는지 알려줘야 한다.
_Q5_FAIL_REASONS = []

# ★ 강 안에서 이미 정답으로 쓴 어휘 (_s138)
#   실측: 비상(홍) 1과에서 'trivial' 이 4번·5번 두 지문의 정답으로 나왔다.
#   같은 과에서 같은 단어가 반복되면 학생이 눈치챈다.
#   {(book, unit): {소문자 단어, ...}} — 프로세스가 살아 있는 동안만 유지된다.
#   캐시 히트로 나온 것도 기록해야 하므로 반환 직전에 담는다.
_USED_ANSWER_WORDS = {}


def _unit_key(book, unit):
    return (str(book or "").strip(), str(unit or "").strip())


def _answer_stem(word):
    """어근 비교용 — 어미를 넓게 떼어낸다 (_s139).

    ★ 소문자 비교만으로는 'ignorance' 와 'ignoring' 을 다른 단어로 본다.
      실측: 비상(홍) 1과 2번 정답 'ignorance', 5번 정답 'ignoring' —
      학생 눈엔 같은 말이다. 어미를 넓게 떼어 둘 다 'ignor' 로 만든다.
    ★ 파생 접미사까지 떼므로 'trivial'→'trivi' 처럼 과하게 잘리기도 하지만,
      **다른 단어끼리 우연히 같아지는 일은 드물다**(길이 4 이상만 뗀다).
    """
    x = re.sub(r"[^a-z]", "", str(word or "").lower())
    for suf in ("ationally", "ization", "ational", "fulness", "iveness",
                "ability", "ibility", "ousness", "lessness", "ance", "ence",
                "ment", "tion", "sion", "ness", "ship", "hood", "ing", "ies",
                "ied", "ive", "ous", "ful", "less", "able", "ible", "ally",
                "ity", "ize", "ise", "ate", "ant", "ent", "ed", "es", "er",
                "ly", "al", "ic", "y", "s"):
        if x.endswith(suf) and len(x) - len(suf) >= 4:
            return x[:-len(suf)]
    return x


def note_answer_word(book, unit, word):
    """이 강에서 정답으로 쓴 단어를 기록한다. 표기와 어근을 함께 담는다."""
    w = re.sub(r"[^a-z]", "", str(word or "").lower())
    if w:
        _USED_ANSWER_WORDS.setdefault(_unit_key(book, unit), set()).add(w)


def used_answer_words(book, unit):
    """프롬프트에 보여줄 목록 — 실제 표기 그대로."""
    return sorted(_USED_ANSWER_WORDS.get(_unit_key(book, unit), set()))


def answer_word_clash(book, unit, word):
    """이미 쓴 정답과 어근이 겹치는가 (_s139). 겹치면 그 단어를 반환."""
    _st = _answer_stem(word)
    if not _st or len(_st) < 4:
        return ""
    for w in _USED_ANSWER_WORDS.get(_unit_key(book, unit), set()):
        if _answer_stem(w) == _st:
            return w
        # 'advanced' / 'advance' 처럼 어미 처리가 갈리는 경우 — 앞 5글자로 보완
        if len(_st) >= 5 and len(w) >= 5 and _answer_stem(w)[:5] == _st[:5]:
            return w
    return ""


# ════════════════════════════════════════════════════════════════
# 절대어 오답 차단 (_s108)
#   기출 28개 오답 중 all/always/never/only 를 쓴 것은 0개다. 절대어가 있으면
#   학생이 지문을 안 읽고 그 선지를 소거한다 — 오답이 오답 구실을 못 한다.
#   프롬프트에는 규칙이 있었지만 **코드 검사가 아예 없어** 새어 나갔다.
#   실측: B Q2 오답 ④ 'Why One Sensory Pathway Is Never Enough' — 대문자 Never.
#   ★ 대소문자를 무시한다. 제목 선지는 각 단어가 대문자로 시작해 'Never' 가 된다.
# ════════════════════════════════════════════════════════════════
_ABSOLUTE_WORDS = {
    "all", "always", "never", "none", "nothing", "every", "everything",
    "only", "cannot", "impossible", "entirely", "completely",
    "absolutely", "totally", "invariably", "universally",
}


def absolute_word_in_option(opt: str) -> str:
    """선지에 절대어가 있으면 그 단어, 없으면 빈 문자열. 대소문자 무시.

    ★ 하이픈 복합어 안의 것은 세지 않는다 (_s119).
      'One-Size-Fits-All' 은 관용구지 절대어가 아니다 — 'all' 이 들어 있다고
      막으면 정상 제목이 거부된다(실측 1건).
      'all-in-one' 'know-it-all' 'once-and-for-all' 도 마찬가지다."""
    _raw = str(opt or "")
    # 하이픈으로 이어진 덩어리는 통째로 지우고 본다
    _raw = re.sub(r"\b[A-Za-z]+(?:-[A-Za-z]+)+\b", " ", _raw)
    # ★ 관용구 안의 절대어도 세지 않는다 (_s120).
    #   'One Reasoning Fits All' 'The Anchor of All Emergency Communication' 처럼
    #   'all' 이 문법적으로 필요한 자리다. 절대어로 티내는 것과 다르다.
    #   실측: 두 지문이 이것 때문에 재시도를 소진하고 관대 모드로 떨어졌다.
    for _idiom in (r"fits\s+all", r"one\s+size\s+fits\s+all", r"all\s+in\s+one",
                   r"once\s+and\s+for\s+all", r"all\s+of\s+us", r"above\s+all",
                   r"after\s+all", r"all\s+the\s+while", r"in\s+all",
                   r"anchor\s+of\s+all", r"heart\s+of\s+all", r"root\s+of\s+all",
                   r"know\s+it\s+all", r"end\s+all", r"all\s+but"):
        _raw = re.sub(_idiom, " ", _raw, flags=re.I)
    t = " " + re.sub(r"[^A-Za-z ]", " ", _raw.lower()) + " "
    t = re.sub(r"\s+", " ", t)
    for w in _ABSOLUTE_WORDS:
        if f" {w} " in t:
            return w
    return ""


def place_vocab_answer(items, want_n):
    """Q3 어휘 정답 자리 이동 — ★ 쓰지 않는다 (_s137에서 검토 후 보류).

    ★ 왜 못 하나
      어휘는 주제·제목과 다르다. 주제 선지는 다섯 개가 **서로 독립**이라
      순서를 섞어도 되지만, 어휘는 각 항목이 **지문의 특정 자리에 묶여 있다.**
        - n 만 바꾸면 → 본문 밑줄이 지문 앞쪽에 ③, 뒤쪽에 ① 이 붙어 어색하다
        - 좌표까지 바꾸면 → 정답 단어가 원래 문맥이 아닌 문장에 놓여 뜻이 깨진다
      결국 **어느 자리를 정답으로 삼을지는 LLM 이 고를 때 정해야 한다.**

    ★ 대신 프롬프트를 고쳤다 — 출력 예시 JSON 이 계속 n:3 을 정답으로 보여준 게
      앵커였다(실측: 비상 1과 다섯 지문이 ③③③③④). 예시에서 그 앵커를 없앴다.
    """
    return items


def shuffle_correct_position(options, correct_idx, book, unit, pid, salt):
    """선지 순서를 섞고 정답을 강 단위 순환 자리에 놓는다 (_s136).

    ★ 정답 번호를 LLM 이 정하면 한 자리에 몰린다 —
      실측: 16강 주제 ①에 4/6, YBM 1과 ④에 3/4.
      어휘(want_n)와 B Q3(summary_design)는 이미 코드가 자리를 정하는데
      주제·제목만 LLM 에 맡겨 있었다.
    ★ 같은 강·같은 번호는 항상 같은 자리라 이미 배포한 답지가 안 흔들린다.

    반환: (새 선지 리스트, 새 정답 인덱스). 못 하면 (원본, 원본인덱스).
    """
    if not isinstance(options, list) or len(options) != 5:
        return options, correct_idx
    if not isinstance(correct_idx, int) or not (0 <= correct_idx < 5):
        return options, correct_idx
    _seq = re.findall(r"\d+", str(pid))
    _seq = int(_seq[0]) if _seq else 0
    _start = int(hashlib.md5(
        (str(book) + "|" + str(unit) + "|" + salt).encode()
    ).hexdigest()[:8], 16) % 5
    _pos = (_start + _seq) % 5
    _correct = options[correct_idx]
    _rest = [o for k, o in enumerate(options) if k != correct_idx]
    random.Random(f"{book}|{unit}|{pid}|{salt}|ord").shuffle(_rest)
    return _rest[:_pos] + [_correct] + _rest[_pos:], _pos


def check_absolute_words(options, correct_idx, label, pid="?"):
    """★ _s133 부터 생성 경로에서는 호출하지 않는다. 진단·검사기 참고용으로만 남긴다.
    "이 'never' 가 절대 주장인가 관용구인가"는 의미 판단이라 코드가 못 한다."""
    """A Q1 주제 / B Q2 제목 오답에 절대어가 있는지. 정답은 안 본다
    (정답에 필요한 말이면 쓸 수 있다)."""
    out = []
    if not isinstance(options, list):
        return out
    for k, o in enumerate(options):
        if k == correct_idx:
            continue
        w = absolute_word_in_option(o)
        if w:
            out.append(f"[{pid}] {label} 오답 {k+1}번에 절대어 '{w}' — "
                       f"지문을 안 읽고 소거된다. 절대어를 빼라: '{str(o)[:60]}'")
    return out


def _q5_text_of(raw, paragraphs, pid="?", lab="?"):
    """LLM이 준 blank_A/blank_B 에서 구절 문자열을 꺼낸다.

    ★ _s93부터 기본은 {"para": n, "text": "구절"} 이다. 구절을 통째로 받는다.
      · 창작 방지 — validate_llm_q5_spans 가 '원문에 없으면 거부'로 잡는다.
        _s70에서 이걸 구조(위치 지목)로 막으려다 길이 통제를 잃었다.
      · 길이 통제 — 구절을 직접 쓰면 LLM 눈에 길이가 보인다. 프롬프트가
        '한 문법 단위' 기준을 주므로 저절로 4~12단어가 된다.
    구 형식({starts_with, ends_with})과 순수 문자열도 그대로 받는다(하위호환)."""
    if isinstance(raw, dict):
        t = str(raw.get("text") or "").strip()
        if t:
            return re.sub(r"\s+", " ", t)
        if raw.get("starts_with") or raw.get("ends_with"):
            return _span_from_marks(paragraphs, raw, pid, lab)
        return ""
    return re.sub(r"\s+", " ", str(raw or "")).strip()


def validate_llm_q5_spans(paragraphs, span_a: str, span_b: str, pid: str = "?") -> Optional[dict]:
    """LLM이 고른 Q5 빈칸 두 구절을 검증한다. 통과하면 마킹된 paragraphs를 반환.

    LLM은 '논지가 착지하는 자리'를 판단하고(코드로는 불가), 코드는 복원 가능성만 본다.
    ★ 실패하면 어느 조건에서 걸렸는지 로그로 남긴다 — 로그에 '검증 실패'만 찍히면
      8단계 중 어디가 문제인지 알 수 없어 프롬프트를 고칠 수 없다.
    """
    def _fail(why):
        print(f"[VAR][A][{pid}] Q5 LLM 픽 거부: {why}")
        _Q5_FAIL_REASONS.append(str(why))
        return None

    try:
        texts = [p[1] for p in paragraphs]
    except Exception:
        return _fail("paragraphs 형식 오류")
    a = re.sub(r"\s+", " ", str(span_a or "")).strip()
    b = re.sub(r"\s+", " ", str(span_b or "")).strip()
    if not a or not b:
        return _fail("빈 값")
    if a == b:
        return _fail("(A)(B) 동일")

    # 문장 경계(. ! ?)가 붙어 오면 그 직전까지 잘라 쓴다.
    # ★ 쉼표는 자르지 않는다 (_s94) — 빈칸 안에 남고 보기에 붙어 제시된다.
    a = _cut_before_punct(a, 4, sentence_only=True) or a
    b = _cut_before_punct(b, 4, sentence_only=True) or b

    # ★ 절 경계를 넘는 구절 거부 (_s153). LLM 픽이 유효하면 코드 픽보다 먼저
    #   채택되므로 여기서 안 걸면 _q5_candidates 에 넣은 규칙이 통째로 우회된다.
    #   실측(공통영어2 비상(홍) 1과 6번): 두 빈칸이 다 절을 걸쳐 'your own ⬜ make',
    #   'is ⬜ should show' 가 나갔다. 거부 사유는 재시도 프롬프트로 돌아간다.
    for v, lab in ((a, "A"), (b, "B")):
        if clause_tail_cut(v, " ".join(texts)):      # ★ _s160
            return _fail(f"({lab}) '{v[:45]}' 가 둘째 절의 주어까지만 물고 "
                         f"술부를 빈칸 밖에 두었다 — 접속사 뒤 절을 통째로 "
                         f"넣거나 접속사 앞에서 끊을 것")
        if crosses_clause(v):
            return _fail(f"({lab}) '{v[:45]}' 가 절 경계를 넘는다 — "
                         f"쉼표 뒤에서 새 절이 시작된다. 성분 하나만 고를 것")

    # 단어 중간에서 잘린 구절 거부 ('arbitrarily' → 'arily')
    def _word_bounded(v, whole):
        k = whole.find(v)
        if k < 0:
            return False
        before_ok = (k == 0) or whole[k - 1].isspace()
        e = k + len(v)
        after_ok = (e >= len(whole)) or (not whole[e].isalnum())
        return before_ok and after_ok

    _whole = " ".join(texts)
    for v, lab in ((a, "A"), (b, "B")):
        if v not in _whole:
            return _fail(f"({lab}) 원문에 없음 — verbatim 아님: '{v[:50]}'")
        if not _word_bounded(v, _whole):
            return _fail(f"({lab}) 단어 중간에서 잘림: '{v[:50]}'")
        n = len(v.split())
        if n < 4:
            return _fail(f"({lab}) {n}단어 — 4단어 미만: '{v[:50]}'")
        if n > 12:
            # ★ 상한 12 (_s96) — 기출 115선지 실측 최대가 정확히 12다. 11로 두면
            #   12단어 정상 빈칸이 거부돼 코드 픽으로 떨어진다(실측 2건).
            return _fail(f"({lab}) {n}단어 — 12단어 초과: '{v[:50]}'")
        # ★ 문장 경계만 거부. 안쪽 쉼표는 허용한다 (_s94) — 보기에 'original,' 로 붙어 나간다.
        if re.search(r"[.!?]", v):
            return _fail(f"({lab}) 문장 경계(. ! ?) 포함 — 한 문장 안에서 고를 것: '{v[:50]}'")
        if re.search(r"[,;:]\s*$", v):
            return _fail(f"({lab}) 쉼표로 끝남 — 끝 구두점은 빈칸 밖에 남길 것: '{v[:50]}'")
        if not _quote_ok(v, allow_comma=True):
            return _fail(f"({lab}) 따옴표 짝이 안 맞음: '{v[:50]}'")
        # ★ LLM 픽에는 완화 경계(strict=False). strict 는 코드가 단어 수만 보고
        #   아무 데나 자를 때 쓰는 안전장치다. 기출도 관사·전치사로 시작한다.
        if not _clean_boundary_ok(v, _whole, strict=False):
            ws = v.split()
            return _fail(f"({lab}) 경계 어정쩡 (시작 '{ws[0]}' / 끝 '{ws[-1]}'): '{v[:50]}'")

    def _locate(v):
        tot = sum(t.count(v) for t in texts)
        if tot != 1:
            return None, tot
        for i, t in enumerate(texts):
            if t.count(v) == 1:
                return i, 1
        return None, tot

    ia, na = _locate(a)
    ib, nb = _locate(b)
    if ia is None:
        return _fail(f"(A) 전체에서 {na}회 등장 — 유일해야 복원이 확정된다: '{a[:50]}'")
    if ib is None:
        return _fail(f"(B) 전체에서 {nb}회 등장: '{b[:50]}'")
    # ★ (A)(B)는 서로 다른 단락에서 하나씩. intro 포함 네 곳(intro/A/B/C) 중 두 곳.
    #   같은 단락에 몰면 지문 한쪽만 비고 나머지는 온전해 읽기 균형이 깨진다.
    if ia == ib:
        _lab = paragraphs[ia][0] if ia < len(paragraphs) else ia
        return _fail(f"(A)(B)가 같은 단락({_lab}) — intro/A/B/C 중 서로 다른 두 곳에서 골라야 함")
    _joined_all = " ".join(texts)
    if _joined_all.find(a) > _joined_all.find(b):     # (A)가 지문에서 먼저 나오게
        a, b, ia, ib = b, a, ib, ia

    new_paras = [list(p) for p in paragraphs]
    new_paras[ia][1] = texts[ia].replace(a, "<BLANK_A>", 1)
    new_paras[ib][1] = texts[ib].replace(b, "<BLANK_B>", 1)

    joined = " ".join(p[1] for p in new_paras)
    if "<BLANK_A>" not in joined or "<BLANK_B>" not in joined:
        return _fail("마커 삽입 실패")
    dup = fill_boundary_dup(joined, [("<BLANK_A>", a), ("<BLANK_B>", b)])
    if dup:
        return _fail(f"빈칸 경계에서 '{dup}' 중복")
    for _i in {ia, ib}:
        _rec = new_paras[_i][1].replace("<BLANK_A>", a).replace("<BLANK_B>", b)
        if _rec != texts[_i]:
            return _fail(f"단락{_i} 되넣어도 원문 복원 안 됨")
    return {"paragraphs": new_paras, "blank_A": a, "blank_B": b}


def _q5_leaks(statements, va, vb) -> bool:
    """Q4 진술이 이 빈칸 답을 그대로 말해주면 True (_s152).

    ★ 왜 픽커 안으로 들여왔나.
      이 검사는 원래 바깥 검증 블록에만 있었고 `not is_last` 로 묶여 있었다.
      마지막 시도에서는 검사가 통째로 꺼지므로, 3회를 다 쓴 지문은 그대로 나갔다.
      실측(25년 고2 9월 39번 _s151): Q4 '나' = "Paper can be torn easily and
      accurately along a crease…" 가 Q5(B) 정답과 사실상 같은 문장인데 그대로 배포됐다.
      재시도로 고치려 하면 시도를 다 쓰면 못 막고, 막으면 지문이 통째로 빠진다.
      → 애초에 **다른 후보를 고르게** 한다. 픽커에는 후보가 여럿이라 공짜다.
    """
    try:
        from variation.vocab_q3 import statements_leak_blanks
        return bool(statements_leak_blanks(statements, va, vb))
    except Exception as _e:
        print(f"[VAR][A] ⚠ 검사 건너뜀 (_q5_leaks): {_e}")
        return False


def pick_a_q5_blanks(paragraphs, llm_a: str = "", llm_b: str = "", pid: str = "?",
                     statements=None, avoid=None) -> Optional[dict]:
    """A Q5 빈칸을 코드가 (A)(B)(C)에서 직접 골라 마킹 (B 빈칸뚫기와 같은 철학).
    LLM이 고른 구절(blank_A/B)이 유효하면 우선 사용, 아니면 코드가 깨끗한 구절 선택.
    fill_boundary_dup None + verbatim 복원 + 서로 다른 단락이 보장되는 조합만 반환. 실패 시 None.
    → 빈칸 짧음/원문 미발견/경계 단어중복(예: 'questions' 중복) 원천 차단."""
    try:
        texts = [p[1] for p in paragraphs]
    except Exception:
        return None
    if len(texts) < 2:
        return None
    cand = [_q5_candidates(t) for t in texts]

    def _valid_llm(val, idx):
        if not val:
            return None
        v = str(val).strip()
        v = _cut_before_punct(v, 4, sentence_only=True) or v   # 문장 경계만 자른다(_s94)
        if len(v.split()) < 4:
            return None
        if len(v.split()) > MAX_BLANK_WORDS_A:   # A Q5 상한 12 (_s98)
            return None
        if texts[idx].count(v) != 1:
            return None
        if not _quote_ok(v, allow_comma=True):
            return None
        if modal_no_verb(v):
            return None
        # ★ LLM 구절도 경계 검사를 거친다. 이게 없으면 LLM 픽이 우선 채택되면서
        #   _clean_boundary_ok가 통째로 우회돼, 코드 픽에만 걸린 경계 규칙이 무의미해진다.
        #   (거부되면 아래에서 코드 픽 후보가 대신 쓰이므로 항목 누락은 안 생긴다)
        if not _clean_boundary_ok(v, texts[idx], strict=True):
            return None
        if crosses_clause(v):          # ★ _s153
            return None
        if clause_tail_cut(v, texts[idx]):   # ★ _s160
            return None
        return v

    order = sorted(range(len(texts)), key=lambda k: -len(texts[k].split()))
    pool = []
    for k in order:
        lst = []
        for llm in (llm_a, llm_b):
            vv = _valid_llm(llm, k)
            if vv and vv not in lst:
                lst.append(vv)
        # ★ 코드 후보를 재정렬한다 (_s152) — 거르지 않고 **순서만** 바꾼다.
        #   목록에서 빼면 짧은 단락에서 후보가 말라 항목이 통째로 빠진다((더) 함정).
        #   순서만 바꾸면 그 위험이 없다. 뒤로 미루는 것:
        #     · 담화표지로 시작 ('Indeed, in this state')
        #     · 4단어짜리 최소 길이 (기출 평균 7.4단어)
        def _rank(c):
            w = c.split()
            head = re.sub(r"[^A-Za-z]", "", w[0]).lower() if w else ""
            try:
                from variation.vocab_q3 import is_discourse_marker
                disc = is_discourse_marker(head)
            except Exception:
                disc = head in ("indeed", "however", "moreover", "thus",
                                "therefore", "furthermore", "nevertheless")
            return (1 if disc else 0, 1 if len(w) <= 4 else 0)
        # ★ 중복 제거 (_s158) — cand[k] 안에 같은 구절이 여러 번 들어 있어
        #   `c not in lst` 가 못 걸렀다(sorted 중에는 lst 가 안 늘어나므로).
        #   그래서 아래 [:N] 슬라이스가 같은 후보로 채워져 실질 후보가 2~3개뿐이었다.
        _seen = set(lst)
        for _c in sorted((c for c in cand[k] if c not in lst), key=_rank):
            if _c not in _seen:
                _seen.add(_c); lst.append(_c)
        pool.append((k, lst))

    for ai in range(len(pool)):
        ka, la = pool[ai]
        for bi in range(len(pool)):
            if bi == ai:
                continue
            kb, lb = pool[bi]
            # ★ 회피 조건이 걸리면 후보를 더 넓게 본다 (_s158). 좁게 보면
            #   대체 자리를 못 찾아 결국 겹친 채로 나간다.
            _N = 18 if avoid else 6
            for va in la[:_N]:
                for vb in lb[:_N]:
                    new = [list(p) for p in paragraphs]
                    new[ka][1] = texts[ka].replace(va, "<BLANK_A>", 1)
                    new[kb][1] = texts[kb].replace(vb, "<BLANK_B>", 1)
                    joined = " ".join(p[1] for p in new)
                    if "<BLANK_A>" not in joined or "<BLANK_B>" not in joined:
                        continue
                    if fill_boundary_dup(joined, [("<BLANK_A>", va), ("<BLANK_B>", vb)]):
                        continue
                    if new[ka][1].replace("<BLANK_A>", va) != texts[ka]:
                        continue
                    if new[kb][1].replace("<BLANK_B>", vb) != texts[kb]:
                        continue
                    # ★ Q4 진술이 이 답을 그대로 말해주면 다른 후보로 넘어간다 (_s152)
                    if statements and _q5_leaks(statements, va, vb):
                        continue
                    # ★ Q3 정답이 든 문장은 빈칸으로 뚫지 않는다 (_s158)
                    #   실측(업로드 22개 파일 58문항): Q3 정답과 Q5 빈칸이 **같은 문장**
                    #   인 것이 7건. 두 문항이 한 판단을 공유해 출제 포인트가 겹친다.
                    if avoid and any(va in _s or vb in _s for _s in avoid):
                        continue
                    # ★ (A)(B) 라벨을 본문 등장 순서로 재배정 — 학생은 (A)→(B) 순으로 쓰는데
                    #   지문이 (B)→(A) 순이면 읽기 흐름이 어긋난다. 마커만 맞바꾸면 되므로
                    #   verbatim·복원 검증에는 영향이 없다.
                    joined2 = " ".join(p[1] for p in new)
                    if joined2.find("<BLANK_B>") < joined2.find("<BLANK_A>"):
                        for p in new:
                            p[1] = (p[1].replace("<BLANK_A>", "\x00")
                                        .replace("<BLANK_B>", "<BLANK_A>")
                                        .replace("\x00", "<BLANK_B>"))
                        va, vb = vb, va
                    return {"paragraphs": new, "blank_A": va, "blank_B": vb}
    return None


def _b_candidates(hn: str, min_w: int = 4, max_w: int = 7) -> list:
    """(더) 2단계: _q5_candidates와 동일. 깐깐한 경계로 먼저, 0개면 완화.

    ★ 완화(strict=False)는 **시작 경계만** 푼다. 끝은 어느 모드에서나 _BAD_EDGE 로
      엄격히 본다 — 전치사·관사로 끝나면 목적어가 빈칸 밖에 남아 갈리기 때문이다.
      실측: 'over a coastal' 이 완화 모드로 나왔다('over' 가 _BAD_EDGE 에 없어서였고,
      _s116 에서 전치사 30여 개를 채웠다).
    """
    spans = [(m.start(), m.end()) for m in re.finditer(r'\S+', hn)]
    toks = [hn[s:e] for s, e in spans]
    n = len(toks)

    def _scan(strict):
        out = []
        for L in range(min_w, max_w + 1):
            for i in range(0, n - L + 1):
                j = i + L - 1
                sub = hn[spans[i][0]:spans[j][1]]
                if not _quote_ok(sub):
                    continue
                if not _clean_boundary_ok(sub, hn, strict=strict):
                    continue
                fw = re.sub(r'[^a-z]', '', toks[i].lower())
                if fw in _Q5_MODALS:
                    continue
                if hn.count(sub) != 1:
                    continue
                out.append((i, j, sub))
        return out

    res = _scan(True)
    if not res:
        res = _scan(False)
    return res


def _span_from_marks_summary(summary: str, mark, pid="?", lab="?") -> Optional[str]:
    """LLM이 지목한 {starts_with, ends_with}로 요약문에서 구절을 잘라낸다.

    ★ A Q5와 같은 이유 — 문자열로 받으면 LLM이 방금 쓴 문장을 다시 타이핑하면서
      단어를 바꾼다. 시작·끝만 받고 그 사이는 코드가 떼어내면 창작이 차단된다.
    요약문은 우리가 만든 한 문장이라 단락 개념이 없고, 조건도 A Q5보다 좁다(4~8단어)."""
    if not isinstance(mark, dict):
        return None
    text = re.sub(r"\s+", " ", str(summary or "")).strip()
    st = re.sub(r"\s+", " ", str(mark.get("starts_with", "")).strip())
    en = re.sub(r"\s+", " ", str(mark.get("ends_with", "")).strip())
    if not text or not st or not en:
        return None

    ms = re.search(r"(?<!\w)" + re.escape(st), text)
    if not ms:
        print(f"[VAR][B][{pid}] Q4 지목({lab}) 시작어 '{st}' 요약문에 없음")
        return None
    me = re.search(r"(?<!\w)" + re.escape(en) + r"(?!\w)", text[ms.start():])
    if not me:
        print(f"[VAR][B][{pid}] Q4 지목({lab}) 끝어 '{en}' 시작어 뒤에 없음")
        return None
    span = text[ms.start():ms.start() + me.end()].strip()
    span = span.strip(".,;:!?").strip()          # 양끝 구두점만 제거(안쪽 쉼표는 허용)
    n = len(span.split())
    if n < 3:
        print(f"[VAR][B][{pid}] Q4 지목({lab}) {n}단어 — 3단어 미만")
        return None
    if n > MAX_BLANK_WORDS_B:
        print(f"[VAR][B][{pid}] Q4 지목({lab}) {n}단어 — "
              f"{MAX_BLANK_WORDS_B}단어 초과, ends_with를 앞으로 당길 것")
        return None
    # ★ 경계 검사 (_s105) — 여기엔 원래 없었다. 코드 픽(_b_candidates)에만 있어
    #   LLM 지목은 'twenty-second' 'both' 같은 끊김이 그대로 통과했다.
    if not _clean_boundary_ok(span, text, strict=False):
        print(f"[VAR][B][{pid}] Q4 지목({lab}) 경계 어정쩡 "
              f"(시작 '{span.split()[0]}' / 끝 '{span.split()[-1]}'): '{span[:50]}'")
        return None
    return span


# ★ 짝을 이루는 표현 (_s101). 한쪽만 빈칸 밖에 남으면 홀로 떠서 문장이 어색해진다.
#   실측: 'demand (A) rather (B)' — 'rather' 를 밖에 두고 'than delayed revelation' 을 뚫었다.
#   (앞말, 뒷말) — 둘이 갈라지면 안 된다.
#   ★ 앞말은 '그것만 나오면 뒷말이 반드시 따라오는' 것만 넣는다. 'to' 'not' 'as'
#     처럼 홀로도 흔히 쓰이는 말을 넣으면 오탐이 폭발한다(실측: 정상 문장까지 전부 거부).
_PAIRED_PHRASES = [
    ("rather", "than"),
    ("not only", "but"),
    ("no sooner", "than"),
    ("either", "or"),
    ("neither", "nor"),
    ("whether", "or"),
    ("so as", "to"),
    ("in order", "to"),
    ("as well", "as"),
]


def _orphan_pair(template: str) -> str:
    """빈칸 밖에 남은 텍스트에 짝을 잃고 홀로 뜬 앞말이 있는지 (_s101).

    template 은 (A)/(B) 가 뚫린 상태다. 앞말은 밖에 남았는데 뒷말이 안 보이면
    뒷말이 빈칸 안에 먹힌 것이다 — 'rather' 만 남고 'than' 이 빈칸으로 들어간 꼴.
    ★ 한 방향만 본다. 뒷말('than')만 밖에 남는 건 정상일 때가 많다
      ('more … than', 'less … than' 처럼 앞말이 목록에 없는 비교 구문).
    """
    _out = re.sub(r"\(A\)|\(B\)", " \u2588 ", str(template or ""))
    low = " " + re.sub(r"[^a-z ]", " ", _out.lower()) + " "
    low = re.sub(r"\s+", " ", low)
    for a, b in _PAIRED_PHRASES:
        if f" {a} " in low and f" {b} " not in low:
            return a
    return ""


def pick_b_q4_blanks(full_summary, llm_a: str = "", llm_b: str = "", min_w: int = 3, max_w: int = 7) -> Optional[dict]:
    """B Q4 빈칸 — LLM이 지목한 자리를 최대한 그대로 쓴다.

    ★ 자리 판단은 LLM 몫이다. 코드는 '학생이 복원할 수 있는가'만 본다:
        · 요약문에 verbatim 으로 있는가 (보기 단어로 되맞춰야 하므로 필수)
        · 4단어 이상인가
        · 두 빈칸이 겹치지 않는가
        · 되넣으면 요약문이 정확히 복원되는가
      경계 어휘·간격·비율 같은 '품질' 조건은 프롬프트가 진다. 코드가 그걸 검사하면
      LLM 지목이 거의 전부 거부돼 코드픽으로 대체된다(실측 1/3만 사용).
    LLM 지목이 없거나 위 최소 조건도 못 맞추면 코드가 요약문에서 직접 고른다."""
    hn = re.sub(r'\s+', ' ', str(full_summary or "")).strip()
    if len(hn.split()) < (2 * min_w + 1):
        return None

    def _locate(v):
        """요약문에서 구절 위치(토큰 인덱스)를 찾는다. 없으면 None."""
        v = re.sub(r'\s+', ' ', str(v or "")).strip()
        if len(v.split()) < min_w:
            return None
        if len(v.split()) > MAX_BLANK_WORDS_B:   # B Q4 상한 9 (_s98)
            return None
        _bare = lambda w: w.strip('.,;:!?"\'\u201c\u201d()')
        toks = hn.split(); pw = v.split()
        btoks = [_bare(t) for t in toks]; bpw = [_bare(t) for t in pw]
        hits = [i for i in range(len(toks) - len(pw) + 1)
                if toks[i:i + len(pw)] == pw or btoks[i:i + len(pw)] == bpw]
        if len(hits) != 1:                      # 두 번 나오면 복원이 모호
            return None
        i = hits[0]
        span = " ".join(toks[i:i + len(pw)])
        # ★ 끝 구두점은 빈칸 밖에 남긴다 — 'symptoms,' 의 쉼표까지 물면
        #   연결어(so/whereas) 앞 구두점이 정답에 들어가 배열이 어색해진다.
        #   지문에는 쉼표가 그대로 인쇄되고 학생은 그 앞부분만 배열한다.
        span = span.rstrip(".,;:!?")
        if len(span.split()) < min_w:
            return None
        # ★ 경계 검사 (_s107) — 여기엔 없었다. _s105 에서 _span_from_marks_summary
        #   (지목 경로)에만 넣고 이 경로를 놓쳤다. LLM 이 준 구절이 그대로 통과해
        #   'When contradictory claims prioritize a more valuable prize' 처럼
        #   종속접속사로 시작하는 빈칸이 나갔다(뒤에 주절이 와야 하는 자리다).
        # ★ 시작도 엄격히 본다 (_s116). 완화 모드는 '기출 30%가 기능어로 시작'이라는
        #   근거로 둔 것인데, 그건 **LLM 이 논지를 보고 고른 자리**에 해당한다.
        #   여기는 코드가 요약문에서 기계적으로 잘라내는 경로라 그 근거가 없다 —
        #   실측: 'over a coastal' 처럼 전치사로 시작하는 조각이 나왔다.
        if not _clean_boundary_ok(span, hn, strict=True):
            return None
        return (i, i + len(pw) - 1, span)

    def _try(ca, cb):
        """두 자리 조합이 성립하면 결과 dict, 아니면 None."""
        if not ca or not cb:
            return None
        lo, hi = (ca, cb) if ca[0] < cb[0] else (cb, ca)
        if lo[1] >= hi[0]:                      # 겹치면 불가
            return None
        # ★ 두 빈칸이 딱 붙으면 사실상 빈칸 하나다 — 사이에 최소 1단어.
        #   'Online writing must (A) (B) by revealing...' 같은 꼴을 막는다.
        #   이것만 코드가 막고, 나머지 품질 조건은 프롬프트가 진다.
        if hi[0] - lo[1] < 2:
            return None
        va, vb = lo[2], hi[2]                   # (A)가 앞, (B)가 뒤
        if va == vb:
            return None
        tmpl = hn.replace(va, "(A)", 1)
        if "(A)" not in tmpl:
            return None
        tmpl = tmpl.replace(vb, "(B)", 1)
        if "(B)" not in tmpl:
            return None
        if tmpl.replace("(A)", va).replace("(B)", vb) != hn:   # 복원 확인
            return None
        if fill_boundary_dup(tmpl, [("(A)", va), ("(B)", vb)]):
            return None
        # ★ 짝 표현이 갈라지면 안 된다 (_s101)
        _orp = _orphan_pair(tmpl)
        if _orp:
            return None
        return {"blank_A": va, "blank_B": vb}

    # 1순위 — LLM 지목 그대로
    la, lb = _locate(llm_a), _locate(llm_b)
    r = _try(la, lb)
    if r:
        return r

    # 2순위 — 한쪽만 살리고 나머지는 코드가 채움
    #   ★ _b_candidates 는 이미 (start, end, text) 튜플을 준다. _locate 로 다시 감싸면
    #     문자열이 아니라 튜플이 들어가 전부 None 이 되어 폴백이 통째로 죽는다.
    cands = [c for c in _b_candidates(hn, min_w, max_w)
             if isinstance(c, (list, tuple)) and len(c) >= 3]
    if la:
        for cb in cands:
            r = _try(la, cb)
            if r:
                return r
    if lb:
        for ca in cands:
            r = _try(ca, lb)
            if r:
                return r

    # 3순위 — 코드가 둘 다 고름
    for ca in cands:
        for cb in cands:
            if ca[0] >= cb[0]:
                continue
            r = _try(ca, cb)
            if r:
                return r
    return None




# ============================================================
# (버) 객관식 정답 위치 셔플 — 핵심빈칸·요약빈칸·주제 정답이 ①에 쏠리던 문제 교정
#   LLM/프롬프트가 정답을 0번에 두고 내려보내므로, 코드가 보기를 셔플하고 정답
#   인덱스를 재계산한다. 인덱스 순열로 옮기므로 중복 원소가 있어도(원소 ordinality
#   추적) 정답 의미는 불변. pid+태그 기반 seed라 같은 지문은 항상 같은 배열이고
#   네 유형은 태그가 달라 서로 다르게 섞인다(deterministic).
# ============================================================
def _shuffle_choices(options, correct, seed):
    if not isinstance(options, list) or not isinstance(correct, int):
        return options, correct
    n = len(options)
    if n <= 1 or not (0 <= correct < n):
        return options, correct
    order = list(range(n))
    random.Random(seed).shuffle(order)
    new_options = [options[i] for i in order]
    new_correct = order.index(correct)  # 정답 원소가 옮겨간 새 위치
    return new_options, new_correct


def _choice_seed(pid, tag, payload=""):
    return int(hashlib.md5((str(pid) + tag + str(payload)).encode()).hexdigest()[:8], 16)


# ============ 환경 변수 ============
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5")
ANTHROPIC_VERSION = "2023-06-01"
MAX_RETRIES = 3
# ★ B 는 검사가 훨씬 많다 (_s122) — 제목 형식·절대어·Q3 본문베끼기·복수정답·
#   요약문 겹침·역할 배치·풀이 검증. 3회로는 매번 다른 사유로 소진돼
#   정작 복수정답을 고칠 기회가 없었다(실측: B 하나가 통째로 누락).
#   문항이 빠지면 시험지가 안 되므로 B 만 넉넉히 준다.
MAX_RETRIES_B = 6

SB_URL = os.environ.get("SUPABASE_URL", "")
SB_KEY = (
    os.environ.get("SUPABASE_SERVICE_KEY")
    or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    or os.environ.get("SUPABASE_KEY", "")
)


# ============ Supabase REST 헬퍼 ============
def _supabase_enabled() -> bool:
    return bool(SB_URL and SB_KEY)


def _sb_headers() -> dict:
    return {
        "apikey": SB_KEY,
        "Authorization": f"Bearer {SB_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


# ============ 캐시 키 ============
# 변형문제 step_cache 의 step_name 은 이 둘뿐이다.
# 1회독 워크북은 같은 테이블에 step1~stepN 으로 들어가므로,
# 캐시를 지우거나 셀 때는 반드시 step_name 을 같이 걸어 1회독을 건드리지 않는다.
VARIATION_STEP_NAMES = ("variation_a", "variation_b")

# cache_key 뒤에 붙는 본문 md5 길이 (make_cache_key 의 txt_hash)
_TXT_HASH_LEN = 8


def make_cache_key_prefix(book: str, unit: str, pid: str) -> str:
    """캐시 키의 앞부분 '{책}_{단원}_{번호}_' 만 만든다.

    뒤에 붙는 본문 md5·유형·버전(_s141)은 지문 원문과 코드 버전에 따라 달라져서
    바깥(프런트·삭제 API)에서는 알 수 없다. 캐시 삭제와 '생성됨' 표시는
    이 앞부분으로 훑는다 — 본문이 바뀌어 남은 옛 캐시까지 같이 잡힌다.
    """
    book_safe = book[:15].replace(" ", "_").replace("/", "_")
    unit_safe = unit[:8].replace(" ", "_").replace("/", "_")
    pid_safe = pid[:6].replace(" ", "_").replace("/", "_")
    return f"{book_safe}_{unit_safe}_{pid_safe}_"


def cache_key_to_prefix(cache_key: str) -> Optional[str]:
    """make_cache_key 가 만든 키에서 앞부분(prefix)만 되돌린다.

    키 구조: {prefix}{md5 8자리}_var{a|b}_s{버전}
    → 오른쪽 끝의 '_var' 를 찾아 자르고, 그 앞 8자리(md5)를 떼면 prefix 다.
    구조가 다르면(=변형문제 키가 아니면) None.
    """
    idx = cache_key.rfind("_var")
    if idx <= _TXT_HASH_LEN:
        return None
    base = cache_key[:idx]              # {prefix}{md5}
    prefix = base[:-_TXT_HASH_LEN]      # {prefix}
    if not prefix.endswith("_"):
        return None
    return prefix


# 캐시 버전 — 유형별로 따로 움직인다 (_s161)
_A_VER = "s160"
_B_VER = "s161"


def make_cache_key(book: str, unit: str, pid: str, passage_text: str, variation_type: str) -> str:
    """캐시 키: {책}_{단원}_{번호}_{md5}_v{유형}"""
    txt_hash = hashlib.md5(passage_text.encode("utf-8")).hexdigest()[:8]
    prefix = make_cache_key_prefix(book, unit, pid)  # {책}_{단원}_{번호}_ (끝에 _ 포함)
    # _s151 = 검사가 **조용히 꺼지는 것**을 막는다. 오늘 가장 뼈아픈 발견이다.
    #        실측(25년 고2 9월 37·39번): 캐시 키가 _s148 이라 새 코드로 만든 게
    #        분명한데 _s148 의 두 검사(Q5 정답 노출·Q3 정답이 빈칸 문장)가
    #        하나도 안 걸렸다. 원인은 그 검사들이
    #            try: from variation.vocab_q3 import ...
    #            except Exception: pass
    #        로 감싸여 있었던 것. generator.py 는 새 것이고 vocab_q3.py 가 옛
    #        것이면 import 가 실패하는데, pass 가 그걸 삼켜 **검사가 통째로
    #        안 돈 채 아무 일도 없는 것처럼** 배포된다.
    #        → 전부 `except Exception as _e: print("⚠ 검사 건너뜀 (이름): …")` 로
    #          바꿨다. 로그만 보면 어느 검사가 왜 안 돌았는지 바로 보인다.
    #        ★ 교훈은 _s135·_s140 과 같다 — "검사대가 있는 줄 알았는데 없었다".
    #          그때는 목록만 있고 부르는 자리가 없었고, 이번엔 부르다 실패한 걸
    #          삼켰다. 실패는 시끄러워야 한다.
    #        ★ 운영: generator.py 와 vocab_q3.py 는 **항상 같이** 올려야 한다.
    # (구) _s150 = 내부 마커가 산출물로 새는 것을 막는다.
    #        프롬프트는 Q5 자리를 [[[여기는 Q5 빈칸 …]]] 로 가리고 지문에는
    #        <BLANK_A>/<BLANK_B> 를 심는다. 전부 코드 내부용이다.
    #        실측(25년 고2 9월 37·39번): LLM 이 그 가림 문자열을 근거 인용문에
    #        그대로 베껴 답지에 찍혔다 — 근거 “… too large for the nail, or [[[...]]]”.
    #        선생님·학생이 보는 답지에 내부 문자열이 나온다.
    #        지문(paragraphs)은 마커를 담는 게 정상이라 검사에서 뺀다.
    #        닫힌 목록이라 오탐이 없다 — 정상 산출물엔 나올 수 없는 문자열이다.
    # (구) _s149 = Q4 근거 인용문이 지문에 실제로 있어야 한다.
    #        Q4 의 O/X 판정이 틀리는 사고가 실제로 있었다(사용자 보고).
    #        판정 자체는 의미 판단이라 코드가 못 본다. 그런데 그 앞 단계 —
    #        근거로 든 문장이 지문에 있기는 한가 — 는 볼 수 있고, 지어낸 근거는
    #        판정도 대개 틀린다. 오늘 산출물 75개 근거 전수 통과(오탐 0),
    #        일부러 지어낸 근거는 정확히 걸린다.
    #        ★ 글자 그대로 비교하지 않는다 — 'and eventually' → 'he eventually'
    #          처럼 살짝 바꾼 인용이 실제로 있다(2건, 둘 다 정상). 내용어 4연속
    #          일치로 본다. 지어낸 것은 어느 4연속도 안 맞는다.
    #        + 시험지 머리말에 학교명 출력(variation.html, 캐시 무관).
    #        ★ 남은 층: O/X 판정이 맞는가는 여전히 코드 밖이다. _s121(B Q3
    #          복수정답)처럼 지문과 진술만 주고 답지를 감춘 채 다시 풀려
    #          대조하는 방법이 있다 — 호출이 지문당 1회 는다.
    # (구) _s148 = 문항끼리 답을 흘리는 것 둘. 오늘 검수에서 제일 자주 났다.
    #        (1) Q4 진술이 Q5 빈칸 정답을 그대로 담는다 — 16지문 중 6건.
    #        실측: 진술 "Ocean Alliance was founded to protect whales and the
    #        oceans" ↔ Q5(A) 'founded Ocean Alliance to protect whales and the
    #        earth's oceans'. Q4 만 읽어도 영작 답이 나온다. 내용어 4연속이
    #        겹치면 재시도 — 낱말 몇 개 겹치는 건 같은 지문이니 당연해서 안 본다.
    #        (2) Q3 정답 단어가 Q5 빈칸과 같은 문장에 있다 — 실측 능률(오) 03번
    #        'To ③hinder them, local doctors and nurses <BLANK_B>.' 술부가 통째로
    #        빈칸이라 hinder 가 틀렸는지 판단할 근거가 그 문장에 없다.
    #        ★ 금지 구역을 문장 단위로 넓히는 방법은 버렸다 — 후보가 33% 줄어
    #          _s112 가 되돌린 실수를 반복한다(실측 12지문). **정답 자리에만** 건다.
    #          오답 넷은 동의어라 판단할 게 없으니 빈칸 문장에 있어도 무해하다.
    #        둘 다 관대 모드(마지막 시도)에서는 통과 — 지문을 버리는 게 더 손해다.
    # (구) _s147 = Q4 진술이 Q3 어휘 정답 문장을 근거로 삼는 것을 막는다.
    #        Q4 진술·해설은 원문 기준인데 학생이 보는 지문은 Q3 로 뒤집혀 있다.
    #        실측(능률(민) 04과 01번): 지문엔 'Achieving these goals seems easy'
    #        인데 Q4 '라'가 "…seems easy" 이고 해설은 원문 'difficult' 를 근거로
    #        (X) 거짓이라 했다. 학생이 지문대로 읽으면 참이다 — 정답 시비가 난다.
    #        덤으로 Q4 쪽에서 Q3 정답을 역산할 수도 있다(두 문항이 답을 흘린다).
    #        정답 단어가 든 문장과 각 진술의 근거 문장을 내용어로 대조해 70% 이상
    #        겹치면 재시도시킨다. 관대 모드(마지막 시도)에서는 통과시킨다 —
    #        지문을 통째로 버리는 편이 더 손해다.
    # (구) _s146 = Q3어휘 구동사 자리 차단. 동사만 바꾸면 불변화사가 남는다.
    #        실측: 'These bacteria can break down plastic.' → 'can decompose down
    #        plastic'(능률(민) 04과 02번 ④). _s141 의 'giving up' → 'relinquishing up'
    #        과 같은 사고인데, 그때는 프롬프트에만 적고 검사 코드를 안 넣어 또 샜다.
    #        up/down/off/out/away/back 뒤따르는 자리를 세 경로 모두에서 뺀다.
    #        실측(실지문 12개 220후보): 2개(0.9%)만 줄고 5개 미만 지문 없음.
    # (구) _s145 = Q3어휘를 조동사로 바꿔 뒤의 to 가 남던 것을 막는다.
    #        실측: 'Participants need to be…' → 'Participants must to be…'
    #        (25년 고1 9월 28번 ⑤). 조동사 뒤에는 to 가 오지 않는다.
    #        ⑤는 오답 자리인데 학생 눈엔 명백한 비문이라 그걸 답으로 찍는다.
    #        ★ 원문 대조로는 안 잡힌다 — shown 을 original 로 되돌리면 원문과
    #          같아지기 때문이다. 화면 문장을 따로 봐야 하는 부류다(관사 _s143 과 같다).
    #        normalize(재시도 사유 전달)와 validate(옛 캐시 백스톱) 양쪽에 둔다.
    #        + 시험지 라벨에서 내부 분류값 'etc' 제거(variation/api.py, 캐시 무관).
    # (구) _s144 = 구두점 셋. (1) A Q5 빈칸이 약어 마침표에서 잘리던 것 —
    #        실측 'to 7 p.m.' 을 'p.' 에서 잘라 빈칸이 '… to 7 p' 가 됐다
    #        (본문엔 '.m.' 만 남고 보기엔 'p' 가 단독 토큰. 25년 고1 9월 18번).
    #        _cut_before_punct·_quote_ok 가 약어의 점을 문장 경계로 봤다.
    #        약어 점을 가린 사본으로 자를 자리를 찾되, 그 약어가 문장도 끝내는
    #        경우('p.m. This')는 점을 열어두고 자른 뒤 되붙인다.
    #        (2) Q3어휘 밑줄에서 원문 끝 구두점이 사라지던 것 —
    #        'use them to communicate.' → 'interact' 로 마침표가 증발해
    #        순서대로 이으면 문장이 안 끊겼다(26번). 원문 토큰의 끝 구두점을
    #        밑줄 **밖**에 붙인다. 덤으로 'responded.' 처럼 밑줄이 구두점까지
    #        덮던 것도 없어진다 — 기출은 단어에만 긋는다.
    #        (3) 빈칸이 주격 전용 대명사로 끝나면 거부 — 'knowing I' 는 동사를
    #        빈칸 밖에 남긴다(19번). _s98 이 근거로 든 'is you' 는 보어라
    #        문장이 거기서 끝난다. 목적격이 따로 있는 I/he/she/we/they 만 막고
    #        주격·목적격이 같은 you/it 은 계속 허용한다.
    #        ★ 후보 소진 측정(실지문 24단락): 후보 0인 단락 0개 — 안 늘어났다.
    # (구) _s143 = Q3어휘 앞의 부정관사 a/an 을 shown 에 맞춰 고친다.
    #        어휘는 본문 단어를 바꿔 보여주는데 첫소리가 바뀌면 관사가 어긋난다.
    #        실측: 'an enclosed space' → 'an ⑤ sealed space' (YBM(박) 01과 FR).
    #        학생 눈에 바로 보이는 비문이고 "여기가 바뀐 자리"라는 힌트까지 된다.
    #        paragraphs_render 가 캐시에 들어가므로 버전을 올려야 옛 것이 고쳐진다.
    #        ★ a/an 은 소리 규칙이라(an hour / a university) 코드가 다 못 맞힌다 —
    #          확실한 것만 고치고 애매하면 그대로 둔다(needs_an 이 None 을 낸다).
    #        렌더에서 고치되 validate_vocab 에도 같은 검사를 둔다 — 검사가 한 곳에만
    #        있으면 _s135·_s140 과 같은 구조가 다시 생긴다.
    # (구) _s142 = Q3어휘 목록 차단을 **관문 한 곳(vocab_q3.blocked_reason)** 으로 모았다.
    #        밑줄 단어가 정해지는 길이 셋인데(코드 픽 / LLM 픽 / 마지막 확인)
    #        같은 검사가 셋에 흩어져 있어, 한쪽을 손대면 다른 쪽이 조용히 비었다.
    #        _s109 에서 answer_pos_ok 를 안 쓰게 하자 LLM 픽 검사가 통째로 사라졌고,
    #        같은 원인으로 _s135('why?')·_s140('years' 'people')이 다섯 버전을
    #        사이에 두고 두 번 터졌다. 그때마다 한 항목씩 손으로 옮겨 붙였을 뿐이다.
    #        마지막 확인(validate_vocab)은 아직도 담화표지 하나만 보고 있었다 —
    #        옛 캐시·관대모드로 올라온 항목은 여기가 마지막인데 뚫려 있었다.
    #        이제 셋 다 blocked_reason() 을 지난다. 목록에 단어를 넣으면 세 곳에
    #        동시에 걸린다. 죽은 검사대(answer_pos_ok)는 철거했다 — 코드를 열면
    #        막고 있는 것처럼 보이는데 아무도 안 부르는 게 이 사고의 뿌리였다.
    #        ★ 의미 판단(반대말을 댈 수 있는가)은 관문에 넣지 않는다 — 닫힌 목록만.
    # (구) _s141 = Q3어휘 선지(shown)끼리도 어근 중복을 본다. original 만 비교해서
    #        ② 'relinquishing' ⑤ 'relinquished' 가 같이 나갔다 —
    #        원문어는 달랐지만(giving / claimed) 학생이 보는 건 선지다.
    #        + 구동사 동사만 바꿔 전치사가 남는 것도 프롬프트에 명시
    #        (실측 'relinquishing up' — 원문 'giving up').
    # (구) _s140 = Q3어휘에서 방향 없는 구체명사(_CONCRETE) 차단. 목록은 있었는데
    #        코드 픽(_looks_gradable)에서만 쓰고 LLM 픽은 안 봤다 —
    #        _s109 에서 answer_pos_ok 를 안 쓰게 하면서 같이 빠졌다.
    #        실측: 능률(오) 2과 4번에 'years' 'people' 이 밑줄로 나갔다.
    #        _s135(기능어)와 같은 원인이다 — 목록은 있는데 검사할 자리가 없었다.
    # (구) _s139 = 정답 어휘 중복 판정을 **어근**으로 넓혔다. 소문자 비교만으로는
    #        'ignorance'(2번 정답)와 'ignoring'(5번 정답)을 다른 단어로 봤다 —
    #        학생 눈엔 같은 말이다. 파생 접미사까지 떼어 둘 다 'ignor' 로 만든다.
    #        겹치면 재시도시킨다(관대 모드에서는 통과).
    # (구) _s138 = 같은 강에서 정답 어휘가 겹치는 것을 막는다. 실측 비상(홍) 1과에서
    #        'trivial' 이 4번·5번 두 지문의 정답으로 나왔다 — 같은 과에서 같은 단어가
    #        반복되면 학생이 눈치챈다. 강 단위로 정답 단어를 기록하고 다음 지문
    #        프롬프트에 '이미 쓴 것'으로 넘긴다(캐시 히트도 기록한다).
    #        + 방향 없는 부사(supposedly) 차단 — antonym 이 원문어와 어근이 같으면
    #        반대말이 아니라 형태만 바꾼 것이다.
    # (구) _s137 = Q3 어휘 정답 자리 쏠림 수정. 실측 비상(홍) 1과 다섯 지문이 ③③③③④.
    #        원인은 출력 예시 JSON 이 계속 3번을 정답처럼 보여준 것 —
    #        3번에만 evidence/evidence_type 이 붙어 있어 눈에 띄었다(_s96 과 같은 원인).
    #        예시 다섯 항목을 같은 모양으로 만들고 is_answer 를 <ANSWER_HERE> 로 바꿨다.
    #        ★ 코드로 자리를 옮기는 건 안 된다 — 어휘는 각 항목이 지문의 특정 자리에
    #        묶여 있어, n 만 바꾸면 본문 밑줄 순서가 어긋나고 좌표까지 바꾸면
    #        정답 단어가 엉뚱한 문맥에 놓인다. 주제·제목(_s136)과 성격이 다르다.
    # (구) _s137 = Q3 어휘 정답 자리도 코드가 옮긴다. want_n 은 프롬프트 지시일 뿐이라
    #        안 지켜졌다 — 실측 비상 1과 다섯 지문이 ③③③③④. 출력 예시 JSON 이
    #        계속 n:3 을 정답으로 보여주는 게 앵커다(_s96 에서 확인한 원인).
    #        n 만 서로 교환하므로 본문 밑줄 위치(para/idx)는 그대로다.
    # (구) _s136 = Q1 주제 / Q2 제목 정답 자리도 코드가 강 단위로 돌린다. LLM 이 정하니
    #        한 자리에 몰렸다 — 실측 16강 주제 ①에 4/6, YBM 1과 ④에 3/4.
    #        어휘(want_n)와 B Q3(summary_design)는 이미 코드가 정하는데
    #        주제·제목만 빠져 있었다. 검증을 다 통과한 뒤 저장 직전에 섞는다
    #        (앞서 섞으면 topic_correct 를 참조하는 검사들과 어긋난다).
    #        A 와 B 는 salt 를 달리해 같은 지문에서 같은 번호에 안 몰리게 한다.
    # (구) _s135 = Q3어휘에서 기능어(관사·전치사·대명사·의문사·조동사) 차단. _VOCAB_STOP 이
    #        코드 픽과 answer_pos_ok 에만 걸려 있었는데 answer_pos_ok 는 _s109 에서
    #        안 쓰게 돼 **LLM 픽을 아무도 안 막았다** — 실측 'why?' 가 선지로 나갔다.
    #        의문사의 반대말이란 없다. 닫힌 목록이라 형태 판정처럼 새지 않는다.
    # (구) _s134 = Sonnet 5 의 thinking 블록 때문에 A·B 가 전멸했다. 응답에 thinking 이
    #        먼저 오고 그게 max_tokens 를 다 먹어 정작 JSON(text)이 안 나왔다
    #        (실측: '텍스트 없음' 50건, stop_reason=max_tokens 32건).
    #        payload 에 thinking:{"type":"disabled"} 를 넣어 끄고,
    #        혹시 그 파라미터를 모르는 모델이면 빼고 재시도한다.
    #        + text 블록을 하나만 보지 않고 전부 모은다.
    # (구) _s133 = 절대어 코드 검사를 뺐다. "이 'never' 가 절대 주장인가 관용구인가"는
    #        의미 판단이라 코드가 못 한다. 단어 목록으로 재니 정상 제목이 계속 걸렸다 —
    #        One-Size-Fits-All → One Reasoning Fits All → Never Enough / No One Wants,
    #        세 판 연속 새 관용구가 나왔다. 예외를 추가하는 건 audience 때와 같은 실수다.
    #        B 는 검사가 일곱 개라 재시도 한 자리가 아까워 손해가 더 크다.
    #        프롬프트에 근거(기출 28개 오답 중 0개)와 판별법을 넣고 LLM 이 판단한다.
    # (구) _s132 = 캐시 버전만 올린다(코드 변경 없음). CLAUDE_MODEL 을 claude-sonnet-5 로
    #        바꿨는데 캐시 키에 모델명이 없어 옛 결과(Sonnet 4.5)가 계속 나왔다.
    #        버전을 올려야 새 모델로 다시 만든다.
    # (구) _s131 = 부사 차단을 아예 뺐다. 품사가 아니라 **방향이 기준**이다 —
    #        rarely↔frequently, willingly↔reluctantly 처럼 방향 있는 부사는 정답이 된다.
    #        방향 없는 부사(mentally/understandably/clearly)는 antonym 대입 검증이
    #        걸러낸다(반대말을 넣어도 정반대 주장이 안 된다).
    #        품사로 막으니 오답 자리까지 걸려 A 가 죽었다(_s129 실측: 03번 누락).
    #        + A 어휘 재시도 2 → 3회는 유지.
    # (구) _s130 = 부사 차단을 **정답 자리에만** 적용한다. 근거가 "기출 정답에 부사 0개"
    #        였는데 다섯 자리 전부에 걸어 A 가 죽었다 — 'understandably,' 'clearly' 로
    #        2회 재시도를 소진하고 03번이 누락됐다. 지문에 부사는 흔하고, 오답 자리는
    #        동의어 치환이라 부사여도 방향을 뒤집지 않는다.
    #        + A 어휘 재시도를 2 → 3회로. 같은 단어를 두 번 고르면 그걸로 죽었다.
    # (구) _s129 = Q3어휘 부사(-ly) 차단 + 반대말 대입 검증. 기출 정답에 부사는 0개고
    #        방향이 약해 빼도 문장이 성립한다(실측 'mentally' 'naturally,').
    #        + antonym 을 '적기만' 하면 통과하던 것 → 그 자리에 넣어 읽고 원문과
    #        정반대 주장이 되는지 확인하게 했다. 'role'·'face' 는 추상명사라
    #        형태로는 안 걸리지만 반대말을 넣으면 문장이 성립하지 않는다.
    #        ★ 구두점 차단은 넣었다가 뺐다 — 구두점은 본문에만 붙고 선지에는
    #        shown_clean 이 나가므로 힌트가 되지 않는다. 막을 이유가 없었다.
    # (구) _s128 = Q3어휘 세 가지 차단. (1) 구두점 붙은 단어 — 선지에 쉼표·마침표가 보이면
    #        그 자리가 문장 끝이라는 힌트가 된다(실측 'role,' 'pressured,' 'mistaken.').
    #        (2) 부사(-ly) — 기출 정답에 0개고 방향이 약하다(실측 'mentally' 'naturally,').
    #        (3) 반대말을 '적기만' 하면 통과하던 것 → 그 자리에 넣어 읽고 원문과 정반대
    #        주장이 되는지 확인하게 했다. 'role'·'face' 는 추상명사라 형태로는 안 걸리지만
    #        반대말을 넣으면 문장이 성립하지 않는다 — 그건 LLM 이 판단해야 한다.
    # (구) _s127 = 캐시 버전만 올린다(코드 변경 없음). _s120~_s126 이 실서비스에서 제대로
    #        돈 적이 없는데 옛 캐시가 그대로 걸려 '캐시 히트'만 났다.
    #        버전을 올리면 옛 캐시가 무시되고 새 로직으로 다시 만든다.
    # (구) _s126 = summary_design 도 summary_options 도 없을 때 사유를 정확히 알려준다.
    #        validator 가 둘을 필수로 요구해 '필수 필드 누락'으로만 죽으면 LLM 이
    #        무엇을 고쳐야 할지 모른다. + _s125 누적분.
    # (구) _s125 = summary_design 을 쓰면 summary_check 를 요구하지 않는다. _s123 에서
    #        출력 형식을 역할 칸으로 바꿔 LLM 이 summary_check 를 안 내는데,
    #        코드가 계속 "5개 채워라"를 요구해 매 시도마다 '자가검증 누락'으로
    #        재시도가 통째로 소진됐다(실측: A 1개 + B 2개 누락, 3문항만 나옴).
    #        design 방식은 배치가 구조로 보장되고 판정은 풀이 검증(_s121)이 한다.
    # (구) _s124 = 버그 둘. (1) Q4 요약문 재생성에서 blank_A/B 가 dict 로 오면
    #        'dict object has no attribute strip' 로 터져 **재생성이 아예 안 돌았다**
    #        (실측: 매 시도마다 예외 → Q3·Q4 겹침을 못 고치고 재시도 소진).
    #        (2) summary_design 을 쓰면 코드가 선지 순서와 정답 번호를 새로 정하는데
    #        summary_check 는 LLM 이 원래 순서로 쓴 것이라 어긋난다 —
    #        '정답 1번'인데 '복수정답 감지 [1]'이 떴다(정답 행을 복수정답으로 오인).
    #        design 을 쓴 경우 summary_check 판정은 건너뛰고 풀이 검증(_s121)에 맡긴다.
    # (구) _s123 = B Q3 출력 형식을 **역할 칸**으로 바꿨다. 다섯 쌍을 자유롭게 나열하게 하니
    #        (A)에 유의어를 다섯 개 넣는 식으로 설계가 무너졌다(실측 demonstrate/suggest/
    #        reveal/indicate/reflect). correct / syn_A_1 / syn_A_2 / syn_B / both_wrong
    #        칸에 직접 채우게 하면 그 배치로만 낼 수 있다 — 정답 말고는 반드시 한 칸이
    #        틀리므로 복수정답이 구조적으로 안 나온다. 평가원 40번 구조 그대로다
    #        (양쪽에 유의어를 두되 짝이 안 맞게). 섞기와 정답 번호는 코드가 정한다.
    # (구) _s122 = B 재시도를 3 → 6회로 늘리고, 사유를 **누적해서** 전달한다.
    #        B 는 검사가 훨씬 많다(제목 형식·절대어·본문베끼기·복수정답·요약문 겹침·
    #        역할 배치·풀이 검증). 3회로는 매번 다른 사유로 소진돼 정작 복수정답을
    #        고칠 기회가 없었고, 마지막 사유만 주니 그것만 고치고 앞서 지적받은 걸
    #        다시 어겨 수렴하지 않았다. 실측: B 하나가 통째로 누락됐다.
    #        문항이 빠지면 시험지가 안 되므로 넉넉히 준다. A 는 3회 그대로.
    # (구) _s121 = B Q3 를 **별도 호출로 실제로 풀려** 복수정답을 잡는다. 지금까지의 검사
    #        (summary_check, 역할 라벨, antonym)는 전부 만든 LLM 에게 되묻는 방식이라
    #        대충 채우면 통과했다 — 낸 사람에게 "복수정답 아니죠?" 하고 묻는 셈이다.
    #        문항만 떼어 정답을 감추고 풀린다. 둘 이상 성립한다고 하면 재시도,
    #        마지막 시도면 B 를 만들지 않는다. 정답이 다르게 나와도 재시도한다.
    #        호출이 지문당 1회 늘지만 B Q3 가 제일 자주 터지는 자리다.
    # (구) _s120 = B Q3 복수정답을 관대 모드에서도 통과시키지 않는다. 학생이 이의제기하는
    #        문제라 '일단 내보내기'가 성립하지 않는다 — 실측: 3회 재시도가 제목 형식·
    #        절대어 같은 다른 사유로 소진돼 복수정답인 채로 나갔다(감지 13건, B 3/3 관대).
    #        + 절대어 검사에 관용구 예외 — 'One Reasoning Fits All' 'Anchor of All ~' 은
    #        'all' 이 문법적으로 필요한 자리다. 이 오탐이 재시도를 두 번 잡아먹었다.
    # (구) _s119 = normalize_llm_vocab 이 반환값에 antonym 을 안 실어서 validate_vocab 이
    #        매번 "antonym 이 비었다"를 냈다. normalize 는 통과했는데 뒤에서 죽는
    #        구조라 A 3/3 이 3회씩 전부 관대 모드로 떨어졌다(실측 40건).
    #        지문 탓도 LLM 탓도 아니었다 — 넘기는 과정에서 필드를 흘린 것이다.
    #        16강이 통과했던 건 그때 이 검사가 배포 전이었기 때문이다.
    #        + 절대어 검사에서 하이픈 복합어 제외 — 'One-Size-Fits-All' 은 관용구다.
    # (구) _s118 = _BAD_EDGE 에 종속접속사·관계사·의문사를 채웠다. whether/although/though/
    #        unless/where/how/why/what/before/after/nor/yet 등 절반이 빠져 있어
    #        'conventions determine whether' 가 통과했다 — 뒤에 와야 할 절이
    #        빈칸 밖에 남아 갈린다. _s116(전치사 보강)과 같은 종류의 누락이다.
    # (구) _s117 = antonym 지시를 프롬프트 맨 앞으로 올리고 출력 예시를 실물로 바꿨다.
    #        13,800자 프롬프트 뒤쪽(93~99% 지점)에만 있고 예시가 "antonym": "..." 라
    #        LLM 이 통째로 무시했다 — A 3/3 이 3회씩 전부 antonym 누락으로 실패해
    #        관대 모드로 떨어졌다(실측 세 지문 연속).
    #        예시를 exciting→dull, convince→dissuade 처럼 실제 단어로 채우고,
    #        재시도 안내 맨 앞에도 같은 내용을 박았다.
    # (구) _s116 = _BAD_EDGE 에 전치사 30여 개를 채웠다. 'over' 'under' 'about' 'through'
    #        'between' 등 절반이 빠져 있어 'Rival claims over' 가 통과했다 —
    #        전치사로 끝나면 목적어가 빈칸 밖에 남아 갈린다.
    #        + pick_b_q4_blanks._locate 를 strict=True 로. 완화 모드는 '기출 30%가
    #        기능어로 시작'이라는 근거로 둔 것인데, 그건 LLM 이 논지를 보고 고른 자리
    #        얘기다. 코드가 요약문에서 기계적으로 잘라내는 경로엔 그 근거가 없다.
    # (구) _s116 참고 = _BAD_EDGE 에 전치사 30여 개를 채웠다. 'over' 'under' 'about' 'through'
    #        'between' 등 절반이 빠져 있어 'Rival claims over' 가 통과했다 —
    #        전치사로 끝나면 목적어가 빈칸 밖에 남아 갈린다.
    # (구) _s115 = 부정 접두사 판정을 antonym 대조로 바꿨다. 'un-/in- 이 붙었나'는 형태로
    #        보이지만 실제 판단은 '철자로 답이 새는가'라 의미 판단이고, 진짜 문제의
    #        일부만 잡았다 — 오답에 inhabitable→habitable 은 잡히는데
    #        significant→trivial 은 못 잡는다(둘 다 오답 자리에 반의어를 넣은 것).
    #        → 오답 자리의 shown 이 그 자리 antonym 과 같은지 문자열 비교한다.
    #        접두사든 어근이 다르든 전부 잡히고 오탐이 없다.
    #        '철자로 답이 새는 반의어'는 프롬프트가 진다.
    # (구) _s114 = sentence 칸을 없앴다. 어휘 문제는 단어 하나만 바꾸는 것인데 문장을 쓰게 하니
    #        '그 문장이 어디까지인가'라는 새 문제가 생겼다 — _s111 은 코드가 정규식으로
    #        문장을 추출해 대조하다 A 3/3 을 전멸시켰고(사유 43건), _s113 은 그걸 완화하니
    #        'shown 이 들어 있나'만 보는 껍데기가 됐다. 애초에 필요 없는 칸이었다.
    #        형태가 맞는지는 LLM 이 shown 을 고를 때 판단한다(프롬프트가 진다).
    #        코드는 대소문자만 본다.
    # (구) _s113 = sentence 대조를 완화했다. _s111 은 '그 문장'을 코드가 정규식으로 추출해
    #        LLM 이 쓴 것과 단어 단위로 맞췄는데, **문장 경계 판정 자체가 코드의 일이
    #        아니었다.** 쉼표로 이어진 긴 문장에서 LLM 은 뒷절만 '그 문장'이라 여기고
    #        코드는 앞부터 다 끌어온다(실측 26단어 vs 8단어). A 3/3 전멸, 사유 43건.
    #        → 코드는 sentence 안에 shown 이 들어 있는지만 본다. 그거면 "써 봤다"는
    #        증거가 되고, 비문 판단은 LLM 이 읽으면서 한다.
    #        + 출력 예시 다섯 항목 전부에 antonym·sentence 를 넣고 필수로 못 박았다
    #        (2·4·5번 예시에 sentence 가 없어 LLM 이 생략했다 — antonym 누락 4건).
    # (구) _s112 = Q3어휘의 'Q5 빈칸 자리' 금지 구역을 마커 토큰 그 자리로만 좁혔다.
    #        옛 코드 blank_token_spans 는 (min(hits)-1, max(hits)+1) 로 잡아
    #        **두 마커 사이 전체**를 막았다 — 'the brain <BLANK_A> ... the vast
    #        majority ... <BLANK_B>' 에서 중간의 'majority' 까지 금지 구역이 됐고
    #        A 01번이 그것 때문에 죽었다(15개 토큰이 통째로 막혔다).
    #        Q5 빈칸 안 단어는 애초에 지문에서 사라져 LLM 에게 안 보인다
    #        (프롬프트가 [[[여기는 Q5 빈칸]]] 으로 가린다). 막아야 할 것은
    #        마커 자체를 밑줄로 잡는 경우뿐이다.
    # (구) _s111 = Q3어휘 형태(굴절) 판정을 어미 검사에서 '문장 쓰기'로 바꿨다.
    #        'exciting'→'compelling', 'interesting.'→'boring.', 'desirable'→'coveted'
    #        전부 정상 치환인데 -ing/-ed 어미 때문에 오탐해 첫 시도를 낭비했다
    #        (A 01번은 그래서 두 번째 기회에 진짜 문제를 만나 죽었다).
    #        형태소로는 굴절형인지 어근이 그런 형용사인지 구분이 안 된다.
    #        → LLM 이 sentence 칸에 치환 문장을 써서 낸다. 비문이면 써 보는 순간 보인다.
    #        코드는 그 문장이 원문과 한 단어만 다른가만 대조한다(기계 확인).
    # (구) _s110 = Q3어휘에서 코드의 의미 판단을 더 걷어냈다. 뺀 것 둘:
    #        (1) '철자만 비슷한가' — difflib 유사도로 재다 'inhabitable/uninhabitable'(_s87),
    #        'modesty'(_s90) 같은 정상 치환을 반복 오탐했다. 의미 판단이라 코드가 못 한다.
    #        (2) '한 문장에 몰림' — '흩어라'는 품질 판단이고, 짧은 지문은 어쩔 수 없이 몰린다.
    #        둘 다 프롬프트의 '출력 직전 자가점검' 6항목으로 옮겼다.
    #        코드에 남긴 것은 기계 확인뿐 — 개수·형식·좌표·중복·형태(-s)·antonym 유무.
    # (구) _s109 = Q3어휘 '반대말을 댈 수 있는가' 판정을 코드에서 LLM 으로 넘겼다.
    #        옛 코드(answer_pos_ok)는 어미·목록으로 되짚어 판정했는데 양쪽으로 틀렸다 —
    #        'audience' 는 -ence 라 통과시키고(사람 집단이라 방향이 없는데),
    #        'dissuade' 는 -ade 라 거부했다(방향이 명백한 동사인데).
    #        목록에 단어를 넣을수록 새는 곳이 늘어난다(audience 넣으면 spectator 가 남는다).
    #        → 출력에 antonym 칸을 신설해 **다섯 자리 전부에 반대말을 적게** 하고,
    #        코드는 그 칸이 채워졌는지만 본다. 못 적는 자리는 LLM 이 스스로 버린다.
    #        _looks_gradable 은 코드 픽 폴백에서만 쓴다.
    # (구) _s109 참고 = Q3어휘 후보에서 방향 없는 명사를 더 막는다. 'audience' 가 -ence 어미라
    #        _GRADABLE_SUFFIX 를 통과해 밑줄로 나갔다 — -ence 는 보통 추상명사지만
    #        사람 집단은 방향이 없다. _CONCRETE 에 사람·집단·구체사물 90여 개 추가.
    #        + 방향 있는 동사 60여 개를 _GRADABLE_HINT 에 추가 — 'dissuade' 가 -ade 로
    #        끝나 어미 목록에 안 걸려 원래부터 거부되고 있었다.
    # (구) _s108 = 절대어 오답 차단을 코드로 (A Q1 주제 · B Q2 제목). 프롬프트에만 규칙이
    #        있고 코드 검사가 없어 새어 나갔다 — 실측 'Why One Sensory Pathway Is
    #        Never Enough'. 대소문자를 무시한다(제목 선지는 각 단어가 대문자로 시작해
    #        'Never' 가 된다).
    #        + B Q3 결정 칸을 (A)↔(B) 번갈아 지정 — 3문항 전부 (B)가 결정 칸이면
    #        학생이 "항상 B를 보면 된다"를 배운다.
    # (구) _s107 = B Q4 pick_b_q4_blanks._locate 에 경계 검사 추가. _s105 에서 지목 경로
    #        (_span_from_marks_summary)에만 넣고 이 경로를 놓쳐 'When contradictory
    #        claims prioritize a more valuable prize' 처럼 종속접속사로 시작하는
    #        빈칸이 나갔다. B Q4 는 경로가 셋(지목·LLM 구절·코드 픽)인데 셋 다
    #        같은 검사를 타야 한다.
    # (구) _s106 = 답지 보강 — A 정답 주제문·B 정답 제목의 한글 해석(topic_answer_kr /
    #        title_answer_kr), B Q3·Q4 요약문의 '정답 채운 완성 영문'
    #        (summary_template_en / blank_summary_template_en) 을 저장 직전에 만든다.
    #        영문만 있으면 답지를 보고도 맞는지 판단하기 어렵다.
    # (구) _s105 = 하이픈 복합어로 끝나는 빈칸 차단 (A Q5 · B Q4 · B Q5 공통).
    #        'twenty-second' 'well-known' 은 뒤에 꾸밀 명사가 따라오므로 거기서 끊으면
    #        잘린 것이다. 실측: 'straight line of the twenty-second' — twenty-second
    #        parallel 을 쪼갰다. (더) 미해결 목록에 오래 남아 있던 항목이다.
    # (구) _s104 = 빈칸 끝 경계 목록을 검사기와 동기화. generator._BAD_EDGE 에 'both' 'each'
    #        'such' 'more' 같은 한정사가 빠져 있어 'region forces both' 처럼 뒤 명사가
    #        잘린 채 통과했다(검사기는 잡는데 코드는 안 잡아 어긋나 있었다).
    # (구) _s103 = Q3어휘가 Q5 빈칸 자리에서 단어를 고르던 문제 수정. 원인은 프롬프트가
    #        'Q5 가 쓴 구절' 목록을 따로 보여준 것 — 피하라고 준 목록이 오히려 그
    #        단어들을 눈앞에 갖다놓았다(실측: 'relies' 'flexibility' 'available,' 전부
    #        Q5 빈칸 안 단어). 목록을 없애고 지문 안에 [[[여기는 Q5 빈칸]]] 으로 표시한다.
    #        + 접속부사·방향없는단어 검사를 normalize 로 앞당김. validate 에서만 잡으니
    #        사유가 재시도로 안 넘어가 'Similarly,' 를 두 번 고르고 A 02번이 누락됐다.
    #        _s102 누적분 포함.
    # (구) _s102 = Q3어휘 두 가지 금지 추가. (1) 정답 자리에 부정 접두사 반의어 금지 —
    #        'inhabitable'→'uninhabitable' 은 철자가 거의 같아 논지를 안 읽고 찍힌다.
    #        어근이 다른 반의어를 쓰게 한다(CRITICAL 아님 — 재시도만 시킨다).
    #        (2) 선지는 반드시 한 단어 — 'largest'→'most extensive'(두 단어)가 나갔다.
    #        밑줄 길이가 달라져 그 자리가 표가 난다. _s101 누적분 포함.
    # (구) _s101 = B Q4 에 '짝 표현 갈라짐' 검사 추가. 실측: 'demand (A) rather (B) to ensure'
    #        — 'rather' 를 밖에 두고 'than delayed revelation' 을 뚫어 짝이 갈렸다.
    #        A Q5 에는 관용구 규칙이 있었는데 B Q4 에는 없었다. 프롬프트·코드 양쪽에 넣었다.
    #        짝 목록은 앞말이 '그것만 나오면 뒷말이 따라오는' 것만 담는다 — 'to' 'not' 'as'
    #        처럼 홀로도 흔한 말을 넣으면 정상 문장까지 거부한다(실측).
    # (구) _s100 = Q3·Q4 요약문 겹침 비교를 '정답 채운 완성 문장'으로 (빈칸 자리에서 3연속이
    #        끊겨 실측 2/3을 놓쳤다). + Q3어휘 형태 검사를 -s 중심으로 재설계 —
    #        'overwhelming' 'boring' 같은 -ing 형용사를 굴절형으로 오판해 정상 치환을
    #        거부하고 A 를 누락시켰다. 수일치가 실제로 깨지는 -s 만 끝까지 막고
    #        -ing/-ed 는 첫 시도에만 본다. + original 을 지문에서 '베껴' 적으라고 명시
    #        (실측: 지문 'internal' 인데 'external' 이라고 적어 항목이 버려졌다).
    #        _s99 누적분 포함.
    # (구) _s99 = B Q3 요약빈칸을 평가원 40번 실측(8문항)으로 재작성. 복수정답이 세 번 연속
    #        났다(disclosure/revelation/presentation × engagement/curiosity/attention).
    #        원인은 (A)(B) 양쪽에 유의어를 둔 것. 기출은 유의어를 쓰되 그 행의
    #        반대쪽 칸을 틀리게 만들어 '짝'이 하나만 맞게 짠다.
    #        + summary_check 필드 신설 — 다섯 선지를 요약문에 넣어 완성 문장으로 적게
    #        하고, 정답 외에 성립한다고 적힌 행이 있으면 코드가 재시도시킨다.
    #        + Q3·Q4 요약문 분리 — 둘이 사실상 같은 문장이었다(실측 3/3).
    #        Q3 요약문을 Q4 프롬프트에 넘겨 '다른 각도로 잘라라' 고 시키고,
    #        겹치면(3연속 겹침·첫 내용어 동일) 재생성 → 검증 단계에서도 한 번 더 막는다.
    #        ★ 비교는 '정답을 채운 완성 문장'으로 한다(_s100). 빈칸을 공백으로 지우고
    #        비교하면 3연속이 그 자리에서 끊겨 뒷부분을 통째로 공유해도 통과한다(실측 2/3).
    #        2연속은 안 본다 — 같은 지문이라 'online writing' 같은 주어가 양쪽에
    #        자연스럽게 들어가 헛돈다. 어구는 달라도 같은 주장인 경우는 코드가 못 잡으므로
    #        프롬프트에 실측 사례 3종을 넣어 LLM 이 스스로 걸러내게 했다.
    #        _s98 누적분 포함.
    # (구) _s98 = 빈칸 경계 기준을 기출 실측으로 다시 잡음 (A Q5 · B Q4 공통).
    #        기출 23개 정답 빈칸 중 기능어로 '시작'하는 것이 7개(30%)다
    #        ('the less similarity...' 'a justification for...' 'is often counterproductive...').
    #        반면 기능어로 '끝'나는 것은 0개다. 시작과 끝의 기준이 다르다.
    #        옛 코드는 양쪽을 같은 목록으로 봐서 'is known as vicarious functioning'
    #        같은 정상 빈칸을 거부했다. → 시작은 접속사·관계사만, 끝은 엄격히.
    #        + 상한을 A/B 로 분리: A Q5=12(수능 115선지 실측), B Q4=9(학교 기출 실측).
    #        하나(9)로 묶여 있어 A 코드픽만 10~12단어를 거부하고 LLM픽은 통과시켰다.
    #        + 프롬프트에 '네 몫과 코드 몫'을 명시 — 길이 세기·보기 생성은 코드 일이고
    #        LLM 은 논지 판단과 '글자 그대로 옮기기'에 집중한다. _s97 누적분 포함.
    # (구) _s97 = Q3어휘 normalize 를 세 갈래로 고침. (1) 좌표 찾기를 느슨하게 — original 에
    #        구두점·대소문자가 달라도 지문 자리를 찾고, 찾은 뒤 지문 실제 형태로 덮어쓴다.
    #        (2) 형태 검사 추가 — 지문 'depends' 인데 shown 이 'rely' 면 수일치가 깨진다.
    #        코드가 어미를 고치지 않고 사유를 재시도 프롬프트로 돌려준다.
    #        (3) 실패 사유를 전부 로그·재시도에 남긴다(옛날엔 'normalize 실패'만 찍혔다).
    #        + 굴절형 중복 차단('depends'/'depend' 동시 밑줄).
    #        + 4문항 폴백 폐기 — 번호가 1·2·4·5 로 건너뛰면 학생에게 못 나간다.
    #        + 명사 정답 허용 — 2015 수능 30번 ⑤ 정답이 'concern'(명사)이다.
    #        기준은 품사가 아니라 '이 문맥에서 방향을 뒤집을 수 있는가'다.
    #        + 밑줄 자리 후반부 가중치 제거 — 기출 61% 분포는 원문 순서 지문 기준이고
    #        우리 A 는 (A)(B)(C)가 셔플돼 앞뒤 개념이 없다. 균등 분할로.
    #        _s96 누적분 포함.
    # (구) _s96 = A Q3 핵심빈칸 완전 폐기. A Q3의 유일한 유형은 어휘(수능 30번)다(_s58인데
    #        옛 핵심빈칸을 '예비'로 남겨둔 게 화근이었다 — 예비가 실패하면 CRITICAL 이 나서
    #        1·2·4·5번이 멀쩡한 A 가 통째로 죽었다. 16강 3/3 누락).
    #        예비도 어휘로 한다(같은 유형 재시도, 지문 전체에서 고르므로 성공률이 높다).
    #        2회 다 실패하면 Q3 없이 4문항으로 낸다.
    #        + 정답 자리 ③④⑤ 제약 해제 → ①~⑤. 기출이 ③④⑤인 건 원문 순서가 고정이라
    #        그런 것이고, 우리 A 는 Q2 때문에 (A)(B)(C)가 셔플돼 앞뒤 개념이 없다.
    #        + 유의어·반의어를 사전적 의미가 아니라 문맥으로 판정(평가원 30번 경향).
    #        + 어휘 수준·문체를 평가원 기준으로 재작성 — '주제어가 아니라 판단어',
    #        다섯 개 난이도 균일(정답만 어려우면 어휘로 답이 새어나간다), 학술 산문
    #        (구어·과장어·지문 밖 전문용어 금지), 굴절형 유지.
    #        _s95 누적분 포함.
    # (구) _s95 = validator.blank_has_punct 정책 통일(쉼표 허용) — generator 는 통과시키는데
    #        validator 가 CRITICAL 을 쏴서 A 가 누락됐다(03번). renderer 는 A·B 둘 다
    #        보기에 쉼표를 붙여 제시하므로 학생이 복원할 수 있다.
    #        + Q3 핵심빈칸 폴백 복구 — 어휘 실패 시 폴백도 깨져 A 가 통째로 죽던 것을
    #        재생성 1회 → 그래도 실패하면 Q3 없이 4문항으로. _s94 누적분 포함.
    # (구) _s94 = A Q5 쉼표 허용 — B Q5(_s66)와 방식 통일. 보기에 'original,' 로 붙여 제시하므로
    #        학생이 쉼표 자리를 복원할 수 있다. A만 _s65의 배제 정책에 남아 있어 LLM 픽
    #        거부 5건 중 3건이 쉼표 하나 때문이었다. + 4단어 하한을 프롬프트에서 복구
    #        ('세지 마라'가 하한 검사까지 껐다 — 'designed to' 2단어 실측).
    #        + '빈칸 밖에 남는 것이 단서'를 기출 예시로 명시. _s93 누적분 포함.
    # (구) _s93 = A Q5 위치지목(starts_with/ends_with) 폐기 → 구절을 통째로 받는다.
    #        창작은 verbatim 검사로 막고, 길이는 '한 문법 단위' 기준을 프롬프트가
    #        설명해 LLM이 스스로 판단하게 한다(기출 115선지 실측 3~12단어, 12초과 0개).
    #        지목 방식은 시작·끝만 짚어 그 사이 길이가 안 보였고 5시도 전부 폴백됐다.
    #        + Q3 어휘 정답 자리를 강 단위로 3·4·5 순환(매번 ③에 몰림). _s92 누적분 포함.
    # (구) _s92 = Q3 정답 자리 기준을 품사→반대말 유무로(modesty·majority 같은 방향 있는 명사 허용). _s91 누적분 포함.
    # (구) _s91 = Q5 지목에 word_count·span_preview 자기신고 추가 + 거부 시 사유를 알려주고 1회 재시도. 18단어·쉼표 위반이 반복됐다. _s90 누적분 포함.
    # (구) _s90 = Q3 어휘 정답 자리를 형용사·동사로 강제(명사·부사 거부). 구체명사는 반의어가 없어 반전이 불가능하다. _s89 누적분 포함.
    # (구) _s89 = 로그 표기 개선(VOCAB→Q3어휘, 정답 번호를 원숫자로). _s88 누적분 포함.
    # (구) _s88 = 어휘 shown==original을 생성 단계에서 걸러 Q3만 핵심빈칸으로 우회(지문 통째 누락 방지) + Q5 실패사례 프롬프트 명시. _s87 누적분 포함.
    # (구) _s87 = 부정 접두사 반의어(inhabitable/uninhabitable)를 철자유사 오탐에서 제외. 3회 재시도 후 관대 fallback 원인이었다. _s86 누적분 포함.
    # (구) _s86 = 오류 메시지에 유형(A/B) 표시 — Q4만 보면 A(불일치)인지 B(요약빈칸)인지 알 수 없다. _s85 누적분 포함.
    # (구) _s85 = validator를 generator와 일치(B Q4 하한 4→3, 간격 3→1). 어긋나서 정상 문항이 3회 재시도 후 관대 모드로 떨어졌다. _s84 누적분 포함.
    # (구) _s84 = 어휘 폴백 폐기(shown=original은 문항 성립 불가) + 원문단어 그대로면 CRITICAL. LLM 실패 시 핵심빈칸으로 남긴다. _s83 누적분 포함.
    # (구) _s83 = A Q3 어휘 답지 구조화(정답·근거·오답이유 한줄·나머지 선지 원문단어) + 해설 40자 제한. _s82 누적분 포함.
    # (구) _s82 = B Q4 프롬프트를 지문 논리축 기반으로 재작성(By contrast/Yet/This is because를 찾아 요약문 뼈대로 옮기고 양쪽을 뚫는다). _s81 누적분 포함.
    # (구) _s81 = B Q4 요약문 25~32단어(기출 실측 18~31) + 빈칸 위치 지침 5단계·기출 정답 예시 명시. 빈칸 3~9단어. _s80 누적분 포함.
    # (구) _s80 = B Q4에 학교 기출 뼈대 4종 반영(원인→귀결·기원→결과·대비·양보→조건), 연결어는 빈칸 밖. 3~9단어. _s79 누적분 포함.
    # (구) _s79 = B Q4 빈칸 하한을 3단어로(논리 축이 3단어로 끊기는 경우 허용). _s78 누적분 포함.
    # (구) _s78 = B Q4 빈칸을 LLM이 위치로 지목(starts_with/ends_with), 코드는 verbatim·겹침·복원만 검증. 품질 조건은 프롬프트가 진다. _s77 누적분 포함.
    # (구) _s77 = B Q1 삽입 마커를 1회독 방식(균등 5분할 + 정답자리 끼워넣기)으로 이식. 남은 문장 4개면 4선지, 3개면 3선지. _s76 누적분 포함.
    # (구) _s76 = 문장 3개뿐인 지문은 intro 없이 (A)(B)(C) 구성(기존엔 순서 문제가 통째로 빠졌다). _s75 누적분 포함.
    # (구) _s75 = intro <CORE_BLANK> 복원을 try 밖으로(예외 시 빈칸 잔존 방지) + 남으면 validator가 CRITICAL. _s74 누적분 포함.
    # (구) _s74 = Q5 프롬프트에 단락 경계를 명시(=== 구분선·para 번호·단어수) — LLM이 코드가 나눈 단락을 정확히 인식해야 겹치지 않게 고른다. _s73 누적분 포함.
    # (구) _s73 = Q5 (A)(B)를 intro·A·B·C 중 서로 다른 두 단락에서 선정(LLM 판단). _s72 누적분 포함.
    # (구) _s72 = Q5 후보에 intro 포함 + 같은 단락 허용(단락 선택은 LLM 판단) + 안 쓰이는 CORE_BLANK 복원. _s71 누적분 포함.
    # (구) _s71 = Q5 길이·구두점 조건을 프롬프트에 명시하고 코드는 뒤에서 자르지 않고 거부(어정쩡한 절단 방지). _s70 누적분 포함.
    # (구) _s70 = Q5 빈칸을 LLM이 문자열이 아닌 위치로 지목(para+starts_with+ends_with), 코드가 원문에서 잘라냄 — 창작 원천 차단. _s69 누적분 포함.
    # (구) _s69 = Q5 LLM픽 거부 사유를 로그에 남김(8단계 중 어디서 걸리는지 알아야 프롬프트를 고친다). _s68 누적분 포함.
    # (구) _s68 = 어휘 정답위치 셔플 폐기(자리 불일치로 A 누락 유발) + Q5 LLM픽 경계 완화(6/7 폴백 해소) + 단어 중간 잘림 거부. _s67 누적분 포함.
    # (구) _s67 = B Q1 삽입문을 LLM이 선정(지시어·대명사·연결어 단서로 자리가 확정되는 문장), 코드는 복원·간격만 검증. _s66 누적분 포함.
    # (구) _s66 = B 요약·주제영작 보기에 중간 구두점을 단어에 붙여 제시(signals, first,) — 학생이 쉼표 위치를 알 수 있게. _s65 누적분 포함.
    # (구) _s65 = Q5 후보를 구두점 직전까지 잘라 생성(구두점은 빈칸 밖에 남김) + 최소 4단어. _s64 누적분 포함.
    # (구) _s64 = Q5 빈칸 자리를 LLM이 논지 기준으로 선정(기출23문항: 결론39%·논지핵심17%·두괄식17%·전환점12%), 코드는 verbatim·복원만 검증. _s63 누적분 포함.
    # (구) _s63 = 어휘 정답 위치 ③④⑤ 분산(프롬프트만으론 매번 ③에 몰림, 실측 3/3). _s62 누적분 포함.
    # (구) _s62 = 어휘 선지 구두점 제거(본문은 유지) + 문두 접속부사·담화표지 출제 금지. _s61 누적분 포함.
    # (구) _s61 = 어휘 발음·철자 유사어 금지(1회독 Part A 규칙 적용) + 프롬프트 명시. _s60 누적분 포함.
    # (구) _s60 = 어휘 문장분리를 generator.split_sentences 재사용으로 교체(약어·소수점 오인 제거, 인용문만 추가 분리). _s59 누적분 포함.
    # (구) _s59 = 어휘 폴백 5자리 보장 + 문장당1개 경고가 재시도 유발하던 것 제거 + 인용문 문장분리. _s58 누적분 포함.
    # (구) _s58 = A Q3를 어휘 유형(수능 30번)으로 전환 — 원문 무손실(자리만 기록), Q5 빈칸 회피, 정답 ③④⑤ 강제, 오답 4자리도 동의어 치환. _s57 누적분 포함.
    # (구) _s57 = 정답선지 패러프레이즈 5방식(문두명사 신조·사례 상위어화·대비축 유지·품사전환·부정→긍정) + 오답은 지문어휘 유지 후 한 단어만 삽입. _s56 누적분 포함.
    # ★ _s161 = B Q4 를 요약영작 → 어법 오류 찾아 고치기로 바꿨다.
    #        옛 B 캐시(실측 831건)에는 blank_A/blank_B 만 있고 grammar_q4 가 없다.
    #        새 variation_b.html 로 렌더하면 4번 답란과 답지 어법행이 통째로 빈다 —
    #        무효화하지 않으면 빈 문항이 그대로 인쇄돼 나간다.
    #        ★ 버전을 유형별로 나눈다. 통짜로 올리면 이번 수정과 무관한 A 캐시
    #          899건까지 같이 날아가 재생성 크레딧을 태운다. A 는 _s160 그대로 둔다.
    #        ★ 다음에 A 를 고칠 때는 _A_VER 만 올린다. 두 줄이 따로 움직인다.
    _ver = _B_VER if str(variation_type).lower() == "b" else _A_VER
    return f"{prefix}{txt_hash}_var{variation_type}_{_ver}"


# ============ Supabase 캐시 ============
def load_cached(cache_key: str, step_name: str) -> Optional[dict]:
    if not _supabase_enabled():
        return None
    try:
        url = f"{SB_URL}/rest/v1/step_cache"
        params = {
            "select": "data",
            "cache_key": f"eq.{cache_key}",
            "step_name": f"eq.{step_name}",
            "limit": "1",
        }
        with httpx.Client(timeout=10.0) as client:
            r = client.get(url, headers=_sb_headers(), params=params)
            r.raise_for_status()
            rows = r.json()
            if rows and isinstance(rows, list):
                return rows[0].get("data")
    except Exception as e:
        print(f"[VAR] cache load error: {e}")
    return None


def save_cached(cache_key: str, step_name: str, data: dict) -> None:
    if not _supabase_enabled():
        return
    try:
        url = f"{SB_URL}/rest/v1/step_cache"
        params = {
            "select": "id",
            "cache_key": f"eq.{cache_key}",
            "step_name": f"eq.{step_name}",
            "limit": "1",
        }
        with httpx.Client(timeout=10.0) as client:
            r = client.get(url, headers=_sb_headers(), params=params)
            r.raise_for_status()
            existing = r.json()
            if existing:
                row_id = existing[0]["id"]
                update_url = f"{SB_URL}/rest/v1/step_cache?id=eq.{row_id}"
                client.patch(update_url, headers=_sb_headers(), json={"data": data})
            else:
                payload = {"cache_key": cache_key, "step_name": step_name, "data": data}
                client.post(url, headers=_sb_headers(), json=payload)
    except Exception as e:
        print(f"[VAR] cache save error: {e}")


# ============ 캐시 관리 (2회독 탭: 생성 여부 표시 · 선택 캐시 삭제) ============
_CACHE_PAGE = 1000       # step_cache 조회 페이지 크기
_DELETE_CHUNK = 80       # id=in.(...) 한 번에 넣을 개수 (URL 길이 안전선)


def fetch_variation_cache_rows() -> list:
    """step_cache 에서 변형문제(variation_a / variation_b) 행만 전부 가져온다.

    [{"id":..., "cache_key":..., "step_name":...}, ...]

    book 이름으로 서버쪽 필터를 걸지 않는다 — 교재명에 '(' ')' ',' 같은
    PostgREST 예약문자가 섞이면(예: '능률(김)') 필터가 깨지기 때문이다.
    전부 받아서 파이썬에서 맞춘다. 데이터는 문자열 3개짜리 행이라 가볍다.
    """
    if not _supabase_enabled():
        return []
    url = f"{SB_URL}/rest/v1/step_cache"
    step_in = "in.(" + ",".join(VARIATION_STEP_NAMES) + ")"
    rows = []
    try:
        with httpx.Client(timeout=30.0) as client:
            start = 0
            while True:
                headers = dict(_sb_headers())
                headers["Range-Unit"] = "items"
                headers["Range"] = f"{start}-{start + _CACHE_PAGE - 1}"
                r = client.get(
                    url,
                    headers=headers,
                    params={
                        "select": "id,cache_key,step_name",
                        "step_name": step_in,
                        "order": "cache_key",
                    },
                )
                r.raise_for_status()
                page = r.json()
                if not isinstance(page, list) or not page:
                    break
                rows.extend(page)
                if len(page) < _CACHE_PAGE:
                    break
                start += _CACHE_PAGE
                if start > 200000:      # 무한루프 안전장치
                    break
    except Exception as e:
        print(f"[VAR] cache list error: {e}")
    return rows


def get_variation_cache_status(passages: list) -> dict:
    """지문별 변형문제 캐시 유무.

    passages: [{"book":..,"unit":..,"id":..}, ...]
    반환: {"교재|단원|번호": {"a": bool, "b": bool}}
    """
    rows = fetch_variation_cache_rows()

    # prefix -> {"a":bool,"b":bool}
    have = {}
    for r in rows:
        prefix = cache_key_to_prefix(r.get("cache_key") or "")
        if not prefix:
            continue
        slot = have.setdefault(prefix, {"a": False, "b": False})
        if r.get("step_name") == "variation_a":
            slot["a"] = True
        elif r.get("step_name") == "variation_b":
            slot["b"] = True

    out = {}
    for p in passages:
        book, unit, pid = p.get("book", ""), p.get("unit", ""), p.get("id", "")
        prefix = make_cache_key_prefix(book, unit, pid)
        out[f"{book}|{unit}|{pid}"] = have.get(prefix, {"a": False, "b": False})
    return out


def delete_variation_cache(passages: list) -> dict:
    """선택한 지문들의 변형문제 캐시(A·B)를 지운다.

    본문이 바뀌거나 버전이 올라 남아 있는 옛 캐시까지 prefix 로 같이 지운다.
    step_name 을 variation_a/b 로 한정하므로 1회독 step 캐시는 건드리지 않는다.

    반환: {"deleted": 삭제행수, "matched": 대상행수}
    """
    if not _supabase_enabled():
        return {"deleted": 0, "matched": 0}

    wanted = {
        make_cache_key_prefix(p.get("book", ""), p.get("unit", ""), p.get("id", ""))
        for p in passages
    }
    rows = fetch_variation_cache_rows()
    target_ids = [
        r["id"] for r in rows
        if cache_key_to_prefix(r.get("cache_key") or "") in wanted and r.get("id") is not None
    ]
    if not target_ids:
        return {"deleted": 0, "matched": 0}

    url = f"{SB_URL}/rest/v1/step_cache"
    deleted = 0
    try:
        with httpx.Client(timeout=30.0) as client:
            for i in range(0, len(target_ids), _DELETE_CHUNK):
                chunk = target_ids[i:i + _DELETE_CHUNK]
                id_in = "in.(" + ",".join(str(x) for x in chunk) + ")"
                r = client.delete(url, headers=_sb_headers(), params={"id": id_in})
                r.raise_for_status()
                body = r.json() if r.content else []
                deleted += len(body) if isinstance(body, list) else len(chunk)
    except Exception as e:
        print(f"[VAR] cache delete error: {e}")

    print(f"[VAR] cache deleted {deleted}/{len(target_ids)} rows for {len(wanted)} passages")
    return {"deleted": deleted, "matched": len(target_ids)}


# ============ 지문 분리 ============
def split_passage_and_translation(passage_text: str) -> tuple:
    if "###해석###" in passage_text:
        en, kr = passage_text.split("###해석###", 1)
        return en.strip(), kr.strip()
    return passage_text.strip(), ""


# ============ Claude API 호출 (httpx) ============
def call_claude(system_prompt: str, user_message: str, max_tokens: int = 8000) -> str:
    """anthropic SDK 없이 httpx 직접 호출"""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY 환경변수가 없습니다")

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": ANTHROPIC_VERSION,
        "Content-Type": "application/json",
    }
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
        # ★★ thinking 을 끈다 (_s134).
        #   Sonnet 5 는 응답에 thinking 블록을 먼저 넣는데, 그게 max_tokens 를
        #   다 먹어 정작 JSON(text 블록)이 안 나온다.
        #   실측: '텍스트 없음' 50건, stop_reason='max_tokens' 32건 → A·B 전멸.
        #   우리는 JSON 만 필요하므로 사고 과정을 받을 이유가 없다.
        "thinking": {"type": "disabled"},
    }

    with httpx.Client(timeout=120.0) as client:
        r = client.post(url, headers=headers, json=payload)
        if r.status_code != 200:
            # ★ thinking 파라미터를 모르는 모델이면 그것만 빼고 한 번 더 (_s134)
            if "thinking" in (r.text or ""):
                payload.pop("thinking", None)
                r = client.post(url, headers=headers, json=payload)
            if r.status_code != 200:
                raise RuntimeError(f"Claude API 오류 {r.status_code}: {r.text[:500]}")
        data = r.json()
        content = data.get("content", [])
        # ★ text 블록을 전부 모은다 — thinking 이 켜져 오더라도 뒤의 text 를 놓치지 않는다
        _texts = [b.get("text", "") for b in content if b.get("type") == "text"]
        _joined = "\n".join(t for t in _texts if t).strip()
        if _joined:
            return _joined
        _stop = data.get("stop_reason")
        _kinds = [b.get("type") for b in content]
        raise RuntimeError(
            f"Claude 응답에 텍스트 없음 (stop_reason={_stop}, blocks={_kinds}) — "
            f"thinking 이 max_tokens 를 다 먹었을 수 있다")


def _ensure_kr(en_sentence: str, kr_from_llm: str = "") -> str:
    """영문 문장의 한글 해석을 확보한다.

    2차 재생성이 kr을 함께 주면 그대로 쓰고, 누락했으면 번역 전용으로 한 번 더 부른다.
    (렌더러가 `{% if ..._kr %}`로 조건부 출력하므로 빈 문자열이면 해석 자체가 사라진다.
     '틀린 해석'도 안 되지만 '해석 없음'도 답지 구실을 못 한다.)"""
    kr = str(kr_from_llm or "").strip()
    if kr:
        return kr
    en = str(en_sentence or "").strip()
    if not en:
        return ""
    try:
        raw = call_claude(TRANSLATE_SYS, build_translate_prompt(en), max_tokens=400)
        got = (extract_json_from_response(raw).get("kr") or "").strip()
        if got:
            return got
    except Exception:
        pass
    return ""


# ============ 유형 A 생성 ============
class _SkipCheck(Exception):
    """검사를 건너뛴다는 뜻의 내부 신호 (_s155). 실패가 아니다."""


def generate_variation_a(
    passage_text: str,
    pid: str = "?",
    book: str = "",
    unit: str = "",
    force_regenerate: bool = False,
    cache_only: bool = False,
) -> dict:
    en_text, _ = split_passage_and_translation(passage_text)
    cache_key = make_cache_key(book, unit, pid, en_text, "a")

    if not force_regenerate:
        cached = load_cached(cache_key, "variation_a")
        if cached:
            print(f"[VAR][A][{pid}] 캐시 히트")
            # ★ 캐시로 나온 것도 이 강의 '이미 쓴 단어'다 (_s138)
            try:
                for _vi in (cached.get("vocab_items") or []):
                    if _vi.get("is_answer"):
                        note_answer_word(book, unit, _vi.get("original"))
                        note_answer_word(book, unit, _vi.get("shown"))
            except Exception:
                pass
            return cached

    # 합치기 단계: 캐시에 없으면 생성하지 않고 None (재생성으로 인한 타임아웃 방지)
    if cache_only:
        print(f"[VAR][A][{pid}] 캐시 없음 — cache_only이므로 생략")
        return None

    # ★★ 안내문·공고문이면 문항 구성을 바꾼다 (_s152).
    #   수능 27·28번은 Q2 순서배열·Q3 어휘가 성립하지 않는다 —
    #   항목 나열이라 순서를 섞어도 논리가 안 깨지고, 방향을 가진 낱말이 없다.
    #   실측: 27번이 3회 연속 생성 실패했고 원인이 전부 Q3 어휘였다.
    #   → 주제(1) · 일치(2) · 빈칸영작(3) 세 문항으로 낸다.
    _is_notice = is_notice(en_text, pid)
    # ★ 도표도 같은 3문항 경로를 탄다 (_s163). 안내문 판정이 우선.
    _is_chart = (not _is_notice) and is_chart(en_text, pid)
    #   _short = '순서·어휘를 내지 않는 지문'. 아래 분기는 전부 이 값을 본다.
    _short = _is_notice or _is_chart

    last_errors = []
    last_data = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            user_msg = (
                f"Passage ID: {pid}\n\n"
                f"Original English passage:\n{en_text}\n\n"
                "Generate the variation problem (Type A). Return ONLY the JSON object."
            )
            if _is_chart:
                # ★ 도표도 Q2·Q3 를 내지 않는다 (_s163). 묻는 결이 안내문과 다르다.
                user_msg += (
                    "\n\n# ★ THIS PASSAGE DESCRIBES A GRAPH / CHART (CSAT #25 style).\n"
                    "  - Q1 topic options: describe WHAT THE GRAPH SHOWS "
                    "(e.g. 'online news consumption preferences across six countries'), "
                    "not an abstract thesis or a lesson.\n"
                    "  - Q4 statements: check the NUMBERS and COMPARISONS — which country is "
                    "highest/lowest, ratios ('three times as high as'), thresholds "
                    "('over 60 percent'), and which way was preferred most.\n"
                    "  - Q5 blanks: pick from FULL SENTENCES only. Do NOT pick a span that is "
                    "just a number or a bare superlative — the student must be able to restore it.\n"
                    "  - Do NOT produce Q2 (order) or Q3 (vocabulary). They are discarded.\n"
                )
            if _is_notice:
                # ★ 안내문은 Q2·Q3 를 내지 않는다 — 만들어 보내도 코드가 버린다.
                #   대신 Q1 주제와 Q4 진술의 성격을 안내문에 맞게 잡아 준다.
                user_msg += (
                    "\n\n# ★ THIS PASSAGE IS A NOTICE / ANNOUNCEMENT (CSAT #27-28 style).\n"
                    "  - Q1 topic options: describe WHAT THE NOTICE IS ANNOUNCING "
                    "(e.g. 'a one-day glass art workshop for beginners'), not an abstract thesis.\n"
                    "  - Q4 statements: check concrete facts — date, time, place, fee, eligibility, "
                    "how to register, what is included. This is the heart of the item.\n"
                    "  - Q5 blanks: pick from FULL SENTENCES only. NEVER from a header line "
                    "('Time: ...', 'Registration: ...') or from a list of figures.\n"
                    "  - Do NOT produce order paragraphs or vocabulary items — they are discarded."
                )
            if last_errors:
                user_msg += (
                    "\n\n# ⚠️ PREVIOUS ATTEMPT FAILED — FIX THESE ERRORS:\n"
                    + "\n".join(f"  ✗ {e}" for e in last_errors[:5])
                    + "\n\n# REMINDER OF CRITICAL CHECKS FOR TYPE A (sentence-order style):\n"
                    "  1. intro = the given lead (first 1-2 sentences). It must NOT reappear in (A)/(B)/(C).\n"
                    "  2. ★ (A)(B)(C) = exactly 3 paragraphs. Each must be a CONSECUTIVE run of whole sentences from the passage — "
                    "NEVER merge sentences that are far apart in the original. Cut ONLY at sentence boundaries.\n"
                    "  3. ★ RECONSTRUCTION TEST: intro + (A)(B)(C) reassembled in the order_correct sequence must EQUAL the original passage word-for-word "
                    "(no reordering inside a paragraph, no merging distant sentences, no omission, no duplication).\n"
                    "  4. order_correct = index 0-4 into FIXED choices (0=(A)-(C)-(B) 1=(B)-(A)-(C) 2=(B)-(C)-(A) 3=(C)-(A)-(B) 4=(C)-(B)-(A)); never (A)-(B)-(C).\n"
                    "  5. blank_A and blank_B are natural key phrases (~4-8 words each, do not pad), taken verbatim from INSIDE (A)/(B)/(C) (not intro), in different paragraphs.\n"
                    "  6. bogi must contain EVERY SINGLE WORD from blank_A + blank_B — count articles ('the','a','an') and prepositions carefully.\n"
                    "  7. Q3 is a VOCABULARY item (CSAT #30) — five underlined words, four replaced by synonyms and ONE by an antonym. Do NOT produce core_blank fields."
                )

            raw = call_claude(SYSTEM_PROMPT_A, user_msg)
            data = extract_json_from_response(raw)

            # ★★ 순서배열(Q2)을 코드가 원문에서 분할 — LLM 단락을 무시하고 원문 그대로 사용.
            #    원문 무손실이라 복원검증이 깨지지 않는다. LLM은 빈칸 구절만 고른다.
            ob = (build_notice_blocks_a(en_text, pid) if _short
                  else build_order_blocks_a(en_text, pid))
            print(f"[VAR][A][{pid}] DIAG 문장수={len(split_sentences(en_text))} "
                  f"ob={'None' if not ob else 'OK'} en_len={len(en_text)} en_head={en_text[:60]!r}")
            if ob:
                data["intro"] = ob["intro"]
                data["paragraphs"] = [list(p) for p in ob["paragraphs"]]
                data["order_correct"] = ob["order_correct"]
                # ★ 안내문은 순서 문항이 없다 — order_correct 는 None 이고
                #   validate_a 가 layout 을 보고 순서 검사를 건너뛴다 (_s152).
                #   ★ 도표도 layout 값은 "notice" 로 둔다 (_s163).
                #     검증기·검사기가 전부 이 값으로 순서 검사를 건너뛴다.
                #     새 값을 만들면 그 코드를 다 고쳐야 하고, 하나라도 빠지면
                #     도표가 순서 검사에 걸려 통째로 죽는다.
                #     화면에 '안내문/도표' 중 무엇으로 쓸지는 layout_kind 가 정한다.
                data["layout"] = "notice" if _short else "order"
                if _short:
                    data["layout_kind"] = "chart" if _is_chart else "notice"

                # ★★ Q3 핵심빈칸은 _s96에서 폐기했다.
                #   A Q3의 정식 유형은 어휘(수능 30번)다(_s58). 핵심빈칸은 그때 지우지 않고
                #   '예비'로 남겨둔 옛 유형인데, 예비가 실패하면 validate_a 가 CRITICAL 을
                #   쏴서 A 가 통째로 죽었다 — 쓰지도 않는 유형 때문에 1·2·4·5번이 멀쩡한
                #   지문이 버려졌다(16강 3/3 누락).
                #   게다가 핵심빈칸은 '첫 문장 안에서만' 뚫어야 해 정식보다 훨씬 좁다.
                #   'Bir Tawil is a strange place.'(6단어) 같은 첫 문장은 아예 불가능하다.
                #   → 예비도 어휘로 한다. 같은 유형을 다시 시도하는 편이 성공률이 높다
                #     (지문 전체에서 고르므로). 그래도 실패하면 Q3 없이 4문항으로 낸다.

                # ★ 따옴표·대시·구두점·하이픈·공백 차이까지 흡수하는 마킹 함수 (Q5 빈칸용)
                def _mark_phrase(text, phrase, mk):
                    if not phrase:
                        return text, False
                    # 1) 정확 매칭
                    if phrase in text:
                        return text.replace(phrase, mk, 1), True
                    # 2) 따옴표/대시/비분리공백 통일 (1:1 치환이라 길이 보존 → 인덱스 동일)
                    qmap = {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
                            "\u2013": "-", "\u2014": "-", "\u00a0": " "}
                    def nq(s):
                        for a, b in qmap.items():
                            s = s.replace(a, b)
                        return s
                    nt, npr = nq(text), nq(phrase)
                    if npr in nt:
                        i = nt.index(npr)
                        return text[:i] + mk + text[i + len(npr):], True
                    # 3) 토큰(영숫자) 시퀀스 매칭 — 구두점/하이픈/공백/대소문자 차이를 전부 흡수
                    spans = [(m.group(0).lower(), m.start(), m.end())
                             for m in re.finditer(r"[A-Za-z0-9]+", text)]
                    tw = [w for w, _, _ in spans]
                    pt = re.findall(r"[A-Za-z0-9]+", phrase.lower())
                    if pt and len(pt) <= len(tw):
                        for i in range(len(tw) - len(pt) + 1):
                            if tw[i:i + len(pt)] == pt:
                                s_char = spans[i][1]
                                e_char = spans[i + len(pt) - 1][2]
                                return text[:s_char] + mk + text[e_char:], True
                    return text, False

                # ★ SYSTEM_PROMPT_A 가 아직 core_blank_* 를 만들어 보낼 수 있다.
                #   폐기된 필드이므로 여기서 지운다 — 남아 있으면 validate_a 의
                #   핵심빈칸 검사 블록을 타서 CRITICAL 이 난다.
                for _ck in ("core_blank_target", "core_blank_options",
                            "core_blank_correct", "core_blank_explain"):
                    data.pop(_ck, None)
                #   ★ 마커를 '지우면' 그 자리 구절이 통째로 사라져 지문이 깨진다.
                #     원문에서 intro 를 다시 만들어 온전한 문장으로 되돌린다.
                if "<CORE_BLANK>" in str(data.get("intro", "")):
                    _ob_c = build_order_blocks_a(en_text, pid)
                    if _ob_c and _ob_c.get("intro"):
                        data["intro"] = _ob_c["intro"]
                        print(f"[VAR][A][{pid}] intro 에 남은 <CORE_BLANK> → 원문에서 재구성")
                    else:
                        data["intro"] = re.sub(
                            r"\s{2,}", " ",
                            str(data["intro"]).replace("<CORE_BLANK>", "")).strip()

                # Q5 영작빈칸: ★ 코드가 (A)(B)(C)에서 직접 골라 뚫는다 (B 빈칸뚫기와 같은 철학).
                #   LLM 구절이 유효하면 우선 쓰고, 아니면 코드가 깨끗한 구절을 골라 verbatim 마킹.
                #   → 빈칸 짧음/원문 미발견/경계 단어중복(예: 'questions') 원천 차단. 서로 다른 단락.
                marked = {}
                # ★★ Q5 빈칸 자리 — LLM이 '논지가 착지하는 자리'를 고르고 코드가 복원 가능성만 검증.
                #   기출 23문항 실측: 결론 39% / 논지핵심 17% / 두괄식 첫문장 17% / 전환점 12%.
                #   코드 픽은 단어 수와 위치만 보므로 'dwindle and trail off over' 같은 자리가 나온다.
                #   판단은 LLM, 검증은 코드로 나눈다. 실패하면 기존 코드 픽으로 폴백.
                _picked = None
                try:
                    # ★ intro 를 0번 단락으로 함께 넘긴다. 어느 단락에서 고를지는 LLM 판단이고,
                    #   intro(주어진 글)도 빈칸 대상이다. Q3가 어휘로 바뀌어 intro의
                    #   핵심빈칸이 더 이상 쓰이지 않으므로 자리가 비어 있다.
                    _q5paras = [["intro", data.get("intro", "")]] + [list(x) for x in data["paragraphs"]]
                    _q5raw = call_claude(Q5_BLANK_SYS,
                                         build_q5_blank_prompt(_q5paras),
                                         max_tokens=1200)
                    _q5 = extract_json_from_response(_q5raw)
                    _ra, _rb = _q5.get("blank_A"), _q5.get("blank_B")
                    del _Q5_FAIL_REASONS[:]
                    _sa = _q5_text_of(_ra, _q5paras, pid, "A")
                    _sb = _q5_text_of(_rb, _q5paras, pid, "B")
                    _picked = validate_llm_q5_spans(_q5paras, _sa, _sb, pid) \
                        if (_sa and _sb) else None
                    # ★ 누출은 '거부 사유'로 돌려준다 (_s152). 바로 코드 픽으로 넘기면
                    #   코드 후보가 어색한 단락에서 품질이 떨어진다. LLM 에게 사유를
                    #   알려 다시 고르게 하는 편이 낫다 — 재시도 기계는 이미 있다.
                    if _picked and _q5_leaks(data.get("statements"),
                                             _picked["blank_A"], _picked["blank_B"]):
                        _Q5_FAIL_REASONS.append(
                            "고른 구절이 Q4 진술과 거의 같은 문장이다 — "
                            "Q4 만 읽어도 영작 답이 보인다. Q4 진술이 다루지 않는 문장에서 고를 것")
                        print(f"[VAR][A][{pid}] Q5 LLM 픽이 Q4 진술과 겹침 → 사유 알려주고 재시도")
                        _picked = None

                    # ★ 한 번 더 기회를 준다 — 거부 사유를 알려주고 다시 고르게 한다.
                    #   실측: 18단어·쉼표 포함처럼 프롬프트에 이미 적힌 조건을 어긴다.
                    #   코드 픽으로 바로 넘기면 'twenty-second parallel'을 쪼개는 자리가 나온다.
                    if not _picked:
                        try:
                            _why = "\n".join("  - " + r for r in _Q5_FAIL_REASONS[:4]) \
                                or "  - (사유 미기록)"
                            _retry_msg = (
                                "\n\n[이전 시도가 거부됐다. 사유는 이것이다]\n" + _why + "\n\n"
                                "다시 고를 때 이렇게 하라.\n"
                                "· 네가 쓴 구절을 지문에서 찾아 눈으로 대조하라. 한 글자라도 다르면 못 쓴다.\n"
                                "· 성분을 두 개 물었으면 뒤쪽 하나만 남겨라\n"
                                "  (조건절+주절 → 주절의 술부만 / 주어부+술부 → 술부만).\n"
                                "· 쉼표는 그대로 둬도 된다 — 보기에 'original,' 처럼 붙어 나간다.\n"
                                "  ★ 쉼표만 지워서 다시 내지 마라. 지문에 없는 문자열이 되어 또 거부된다.\n"
                                "· 마침표·물음표를 넘었으면 한 문장 안으로 줄여라.\n"
                                "· 4단어 미만이면 성분을 끝까지 잡아라('designed to' → 'designed to recognize objects').\n"
                                "· 첫 단어와 마지막 단어가 관사·전치사·접속사·조동사면 다시 잡아라.\n"
                                "· blank_A.para 와 blank_B.para 는 서로 다른 숫자여야 한다.")
                            _q5raw2 = call_claude(Q5_BLANK_SYS,
                                                  build_q5_blank_prompt(_q5paras) + _retry_msg,
                                                  max_tokens=1200)
                            _q52 = extract_json_from_response(_q5raw2)
                            _ra2, _rb2 = _q52.get("blank_A"), _q52.get("blank_B")
                            del _Q5_FAIL_REASONS[:]
                            _sa2 = _q5_text_of(_ra2, _q5paras, pid, "A")
                            _sb2 = _q5_text_of(_rb2, _q5paras, pid, "B")
                            if _sa2 and _sb2:
                                _picked = validate_llm_q5_spans(_q5paras, _sa2, _sb2, pid)
                                if _picked and _q5_leaks(data.get("statements"),
                                                         _picked["blank_A"], _picked["blank_B"]):
                                    print(f"[VAR][A][{pid}] Q5 재시도도 Q4 진술과 겹침 → 코드 픽으로")
                                    _picked = None
                                if _picked:
                                    print(f"[VAR][A][{pid}] Q5 재시도 성공")
                        except Exception as _re2:
                            print(f"[VAR][A][{pid}] Q5 재시도 예외({_re2})")
                    if _picked:
                        # intro 를 다시 떼어내 원래 구조로 복원
                        data["intro"] = _picked["paragraphs"][0][1]
                        _picked["paragraphs"] = _picked["paragraphs"][1:]
                    if _picked:
                        print(f"[VAR][A][{pid}] Q5 LLM 픽 — (A)'{_picked['blank_A'][:40]}' "
                              f"(B)'{_picked['blank_B'][:40]}'")
                    else:
                        print(f"[VAR][A][{pid}] Q5 LLM 픽 검증 실패 → 코드 픽으로 폴백")
                except Exception as _qe:
                    print(f"[VAR][A][{pid}] Q5 LLM 픽 예외({_qe}) → 코드 픽으로 폴백")

                if not _picked:
                    _picked = pick_a_q5_blanks(data["paragraphs"], data.get("blank_A", ""), data.get("blank_B", ""), pid,
                                               statements=data.get("statements"))
                if _picked:
                    # ★ 빈칸 뚫기 **전** 단락을 남겨 둔다 (_s158) — Q3 정답이 정해진 뒤
                    #   겹치면 여기서 다시 뚫는다. API 호출이 필요 없다.
                    data["_paras_preblank"] = [list(_p) for _p in data["paragraphs"]]
                    data["paragraphs"] = _picked["paragraphs"]
                    data["blank_A"] = _picked["blank_A"]
                    data["blank_B"] = _picked["blank_B"]
                    marked = {"<BLANK_A>": True, "<BLANK_B>": True}
                else:
                    # fallback: 기존 방식(LLM 구절을 찾아 마킹) — 코드픽 실패 시에도 최소한 동작
                    for mk, key in (("<BLANK_A>", "blank_A"), ("<BLANK_B>", "blank_B")):
                        val = (data.get(key) or "").strip()
                        if not val:
                            marked[mk] = False
                            continue
                        done = False
                        for p in data["paragraphs"]:
                            new_txt, ok = _mark_phrase(p[1], val, mk)
                            if ok:
                                p[1] = new_txt
                                done = True
                                break
                        marked[mk] = done
                data["_blanks_marked"] = marked

                # ★★ Q3 어휘 유형 (수능 30번) — A Q3의 정식이자 유일한 유형이다(_s96).
                #   Q5 빈칸이 확정된 뒤에 호출한다. 원문(paragraphs)은 안 건드리고
                #   '자리(인덱스)'만 기록하므로 Q2 순서 복원 검증에 영향이 없다.
                #   밑줄은 Q5 빈칸이 차지한 토큰 범위를 피해서 잡는다.
                #
                #   ★ 실패하면 같은 유형으로 한 번 더 시도한다(_s96).
                #     옛 코드는 핵심빈칸이라는 다른 유형으로 갈아탔는데, 그건 첫 문장
                #     안에서만 뚫을 수 있어 정식보다 훨씬 좁았고, 실패하면 CRITICAL 이
                #     나서 A 가 통째로 죽었다. 같은 유형 재시도가 성공률이 높다 —
                #     지문 전체에서 고르므로. 두 번 다 실패하면 Q3 없이 4문항으로 낸다.
                _vfail = []          # normalize 실패 사유 — 재시도 프롬프트로 돌려준다

                def _try_vocab(_attempt: int):
                    _spans = blank_token_spans(data["paragraphs"])
                    # ★ 정답 자리를 코드가 정해 내려보낸다.
                    #   프롬프트만으론 매번 ③에 몰린다(_s63·_s92 실측 3/3).
                    #   ★ ①~⑤ 전부 쓴다(_s96). 기출이 ③④⑤인 것은 원문 지문 순서가
                    #     고정이라 '앞쪽이 논지를 확인시키고 뒤에서 뒤집는' 구조가
                    #     성립하기 때문이다. 우리 A 지문은 Q2 순서배열 때문에
                    #     (A)(B)(C)가 셔플돼 있어 앞뒤 개념 자체가 없다.
                    #   강 안에서 1~5를 돌려 써 한 강에서 번호가 겹치지 않게 한다.
                    _seq = re.findall(r"\d+", str(pid))
                    _seq = int(_seq[0]) if _seq else 0
                    _start = int(hashlib.md5(
                        (str(book) + "|" + str(unit) + "|vocabpos").encode()
                    ).hexdigest()[:8], 16) % 5
                    _want_n = 1 + ((_start + _seq + _attempt) % 5)
                    _msg = build_vocab_prompt(
                        data["paragraphs"],
                        [data.get("blank_A", ""), data.get("blank_B", "")],
                        want_n=_want_n,
                        used_words=used_answer_words(book, unit))
                    # ★ 재시도에는 앞선 실패 사유를 그대로 붙인다 (_s97).
                    #   "다시 만들어라"만 하면 같은 실수를 반복한다 — Q5 에서 사유를
                    #   돌려주니 성공률이 올랐던 것과 같은 처방이다.
                    # ★ JSON 을 아예 안 낸 적이 있으면 그 지시부터 앞세운다 (_s155).
                    #   실측(25년 고1 9월 23번): 두 번 연속 분석 서술만 내고 JSON 이
                    #   없어 파싱 실패했다. 사유 목록 맨 뒤에 묻히면 안 읽는다.
                    if _attempt and any("JSON" in x for x in _vfail):
                        _msg += ("\n\n★★★ 지난 시도에 JSON 이 아예 없었다 ★★★\n"
                                 "생각 과정·분석·설명을 쓰지 마라. 한 글자도 쓰지 마라.\n"
                                 "첫 글자가 '{' 이고 마지막 글자가 '}' 여야 한다.\n"
                                 "코드블록 표시(```)도 붙이지 마라.")
                    if _attempt and _vfail:
                        _msg += ("\n\n★★★ [이전 시도가 거부됐다] ★★★\n"
                                 "가장 흔한 이유는 **antonym 칸을 안 채운 것**이다.\n"
                                 "다섯 항목 전부에 반대말을 적어라. 하나라도 비면 버려진다.\n"
                                 "  exciting → dull    convince → dissuade    rely → disregard\n"
                                 "  adapted → unsuited internal → external    greater → lesser\n"
                                 "오답 자리 넷도 적는다(시험지에 안 나간다. 확인용이다).\n\n"
                                 "[사유]\n"
                                 + "\n".join("  - " + x for x in _vfail[-5:])
                                 + "\n\n다시 만들 때 이렇게 하라.\n"
                                   "· original 은 지문에 인쇄된 글자를 그대로 옮겨 적어라.\n"
                                   "  구두점·대소문자까지 포함해서다('uncomfortable.' 'Similarly').\n"
                                   "· ★ shown 을 그 자리에 넣었을 때 문장이 읽히는가 보라.\n"
                                   "  'the brain rely on ...' 처럼 수일치가 깨지면 안 된다.\n"
                                   "· para 와 idx 는 그 단락 안에서 0부터 공백으로 센 위치다.\n"
                                   "· 다섯 자리는 서로 다른 단어여야 한다 — 굴절형도 같은 단어다\n"
                                   "  ('depends' 와 'depend' 를 둘 다 밑줄 치지 마라).\n"
                                   "· ★ [[[여기는 Q5 빈칸 …]]] 안의 단어는 고르지 마라. 이미 사라졌다.\n"
                                   "· ★ 접속부사·담화표지는 밑줄 대상이 아니다\n"
                                   "  (Similarly, Conversely, However, Therefore, Moreover …).\n"
                                   "  논리 흐름 표지지 문맥 판단 대상이 아니다.\n"
                                   "· ★★ 다섯 자리 **전부** antonym 칸에 반대말을 적어라.\n"
                                   "  적을 수 있는 단어인지 확인하는 용도다.\n"
                                   "  못 적겠으면 그 문장을 밑줄 자리로 쓰지 마라 — 다른 문장을 고른다.\n"
                                   "  (audience / story / region / brain / process 처럼\n"
                                   "   사람·집단·구체 사물은 반대말이 없다)\n"
                                   "· 선지는 반드시 한 단어. 'most extensive' 같은 두 단어는 안 된다.\n"
                                   "· ★ 오답 자리(①②④⑤)에 반의어를 넣지 마라 — 정답이 둘이 된다.\n"
                                   "  그 자리 antonym 칸의 말을 shown 에 옮기면 안 된다.\n"
                                   "· 정답 반의어는 원문어와 철자가 확연히 달라야 한다\n"
                                   "  ('inhabitable'→'uninhabitable' 은 눈으로 찾힌다. 'barren' 처럼).")
                    _vraw = call_claude(VOCAB_SYS, _msg, max_tokens=1800)
                    _v = extract_json_from_response(_vraw)
                    #   ★ 첫 시도만 -ing/-ed 형태까지 본다(_s100). 재시도에서는
                    #     -s 불일치만 막는다 — 'vast'→'overwhelming' 같은 정상 치환을
                    #     형태소로 못 가려 지문이 통째로 누락됐다.
                    # ★ want_n 은 프롬프트 지시일 뿐 강제력이 없다 (_s137 에서 확인).
                    #   실측: 비상 1과에서 다섯 지문 정답이 ③③③③④ 로 몰렸다.
                    #   출력 예시 JSON 이 계속 n:3 을 정답으로 보여주는 게 앵커다
                    #   (_s96 에서 같은 원인을 확인했는데 예시는 못 고쳤다).
                    #   → 받은 뒤 코드가 자리를 옮긴다. 아래 _place_answer 참조.
                    _items = normalize_llm_vocab(_v.get("vocab_items"),
                                                 data["paragraphs"], _spans,
                                                 pid=pid, report=_vfail,
                                                 strict=(_attempt == 0))
                    if not _items:
                        raise ValueError(_vfail[-1] if _vfail else "normalize 실패")
                    # shown == original 이면 그 자리는 바뀐 게 없다.
                    # 정답 자리가 그러면 '틀린 단어'가 아예 없어 문항이 성립하지 않는다.
                    _same = [i for i in _items
                             if str(i.get("shown", "")).strip().lower()
                             == str(i.get("original", "")).strip().lower()]
                    if _same:
                        _ns = "".join("①②③④⑤"[(i.get('n', 0) - 1)]
                                      for i in _same if 1 <= i.get('n', 0) <= 5)
                        _as = any(i.get("is_answer") for i in _same)
                        raise ValueError(f"{_ns}번이 원문 단어 그대로"
                                         + (" (정답 자리)" if _as else ""))

                    # ★★ 정답 자리가 Q4 진술의 근거 문장이면 정답이 갈린다 (_s152).
                    #   Q4 진술은 원문 기준인데 학생이 보는 지문은 그 자리가 뒤집혀 있다.
                    #   실측(25년 고2 9월 37번 _s151): 지문엔 'much smaller' 인데
                    #   Q4 '라' 가 "…is smaller than the friction force" 이고 답지는
                    #   원문 'larger' 를 근거로 (X) 라 했다. 학생이 지문대로 읽으면 (O) 다.
                    #   ★ 이 검사는 바깥 블록에도 있지만 `not is_last` 로 묶여 있어
                    #     마지막 시도에서는 꺼진다. 그래서 37번이 그대로 나갔다.
                    #     여기서 걸면 **어휘만 3회 다시 고르면 되므로** 지문이 안 빠진다.
                    #   ★ 마지막 시도에서는 이 조건을 뺀다 (_s155).
                    #     실측(25년 고1 9월 23번): 지문이 5문장인데 Q4 진술이 5개라
                    #     **모든 문장이 어느 진술의 근거**다. 어느 낱말을 골라도 걸리므로
                    #     통과할 방법이 없었다 — 9번 시도 중 3번을 이 검사가 잡아먹고
                    #     지문이 통째로 빠졌다. 조건이 아니라 함정이 된 것이다.
                    #     1·2차에서만 걸어 좋은 자리를 유도하고, 그래도 안 되면 통과시킨다.
                    #     바깥 검증 블록의 같은 검사가 (is_last 가 아닐 때) Q4 를 다시 받게 한다.
                    try:
                        from variation.vocab_q3 import (q4_conflicts_with_answer as _q4c2,
                                                        q4_conflict_unsatisfiable as _q4u)
                        _evs = data.get("statements_evidence") or []
                        if _evs and isinstance(_evs[0], (list, tuple)):
                            _evs = [e for _, _, e in _evs]
                        # ★ 통과할 수 있는 조건인지 먼저 본다 (_s156).
                        #   Q4 근거가 모든 문장을 덮으면 어느 자리를 골라도 걸린다.
                        if _q4u(data["paragraphs"], _evs):
                            print(f"[VAR][A][{pid}] Q4 근거가 지문의 모든 문장을 덮는다 "
                                  f"— 겹침 검사 생략 (_s156)")
                            raise _SkipCheck
                        if _attempt >= 2:
                            raise _SkipCheck
                        _cf = _q4c2(_items, data["paragraphs"], _evs,
                                    statements=data.get("statements"))
                        if _cf:
                            raise ValueError(
                                f"정답 자리가 Q4 진술 {'·'.join(_cf)} 의 근거 문장이다 — "
                                f"그 문장은 학생이 보는 지문에서 뒤집혀 있어 Q4 정답이 "
                                f"갈린다. 다른 문장의 낱말을 정답 자리로 고를 것")
                    except _SkipCheck:
                        print(f"[VAR][A][{pid}] Q3어휘 마지막 시도 — Q4 근거 겹침 검사 생략 (_s155)")
                    except ValueError:
                        raise
                    except Exception as _e:
                        print(f"[VAR][A][{pid}] ⚠ 검사 건너뜀 (q4_conflicts_with_answer/vocab): {_e}")

                    return _items, _v

                # ★ 안내문은 Q3 어휘를 아예 시도하지 않는다 (_s152).
                #   _vok 를 참으로 두어 아래 '2회 실패 → raise' 도 타지 않게 한다.
                _vok = _short
                for _va in (() if _short else range(3)):   # ★ _s130: 2 → 3회
                    try:
                        _items, _v = _try_vocab(_va)
                        data["vocab_items"] = _items
                        if _v.get("vocab_explain"):
                            data["vocab_explain"] = _v["vocab_explain"]
                        _ans = next((i for i in _items if i.get("is_answer")), None)
                        _cn = "①②③④⑤"[(_ans['n'] - 1)] if (_ans and 1 <= _ans.get('n', 0) <= 5) else '?'
                        print(f"[VAR][A][{pid}] Q3어휘 ok — 정답 {_cn}번 "
                              f"원문'{_ans['original'] if _ans else ''}' "
                              f"→ 제시'{_ans['shown'] if _ans else ''}'"
                              + (f" (재시도 {_va})" if _va else ""))
                        _vok = True
                        break
                    except Exception as _ve:
                        # ★ 사유를 반드시 남긴다 (_s155). normalize 실패만 담다 보니
                        #   JSON 미출력·겹침 검사 실패는 기록이 없어 'Q3어휘 2회 실패 —
                        #   사유 미기록' 이 찍혔다(실측 23번). 재시도 프롬프트도 비어 간다.
                        if str(_ve) not in _vfail:
                            _vfail.append(str(_ve))
                        print(f"[VAR][A][{pid}] Q3어휘 시도 {_va + 1} 실패 — {_ve}")
                if not _vok:
                    # ★ 4문항 시험지는 만들지 않는다 (_s97).
                    #   번호가 1·2·4·5 로 건너뛰면 학생에게 못 나간다.
                    #   여기서 raise 하면 바깥 재시도 루프가 A 를 처음부터 다시 만든다.
                    data.pop("vocab_items", None)
                    data.pop("vocab_explain", None)
                    raise ValueError(
                        "Q3어휘 3회 실패 — "
                        + " / ".join(_vfail[-3:] or ["사유 미기록"]))

            if "mismatch_count" not in data and "statements" in data:
                data["mismatch_count"] = sum(1 for _, _, ok in data["statements"] if not ok)

            # ★★ 안전장치: intro가 (A)(B)(C)에 중복되면(order override 미적용·LLM 순서 폴백 등 어떤 경로든)
            #     코드가 원문에서 순서를 강제 재분할해 중복을 제거한다. (16강 03번류 intro 중복 실패 차단)
            try:
                def _nz_dup(t):
                    t = re.sub(r"<[^>]+>", " ", str(t or ""))
                    t = re.sub(r"[^a-z0-9 ]", " ", t.lower())
                    return re.sub(r"\s+", " ", t).strip()
                _iw = _nz_dup(data.get("intro", "")).split()
                _probe = " ".join(_iw[:8]) if len(_iw) >= 8 else " ".join(_iw)
                _paras_now = [p for p in data.get("paragraphs", []) if isinstance(p, (list, tuple)) and len(p) >= 2]
                _dup = bool(_probe) and any(_probe in _nz_dup(p[1]) for p in _paras_now)
                if _dup:
                    ob2 = (build_notice_blocks_a(en_text, pid) if _short
                           else build_order_blocks_a(en_text, pid))
                    if ob2:
                        data["intro"] = ob2["intro"]
                        data["paragraphs"] = [list(p) for p in ob2["paragraphs"]]
                        data["order_correct"] = ob2["order_correct"]
                        _pk2 = pick_a_q5_blanks(data["paragraphs"], data.get("blank_A", ""), data.get("blank_B", ""), pid,
                                                statements=data.get("statements"))
                        if _pk2:
                            data["paragraphs"] = _pk2["paragraphs"]
                            data["blank_A"] = _pk2["blank_A"]
                            data["blank_B"] = _pk2["blank_B"]
                        # ★ 여기도 단락을 통째로 갈아끼운다 — 어휘가 이미 잡혀 있으면
                        #   좌표가 어긋난다. 같은 이유로 재정렬한다 (_s162).
                        if data.get("vocab_items"):
                            _rerr2 = []
                            _ri2 = normalize_llm_vocab(
                                data["vocab_items"], data["paragraphs"],
                                blank_token_spans(data["paragraphs"]),
                                pid=pid, report=_rerr2, strict=False)
                            if _ri2:
                                data["vocab_items"] = _ri2
                            else:
                                print(f"[VAR][A][{pid}] ⚠ 재분할 후 어휘 좌표 재정렬 실패 "
                                      f"({_rerr2[-1] if _rerr2 else '사유 미기록'}) (_s162)")
                        print(f"[VAR][A][{pid}] intro 중복 감지 → 코드가 순서 강제 재분할(안전장치)")
            except Exception:
                pass

            # ════════════════════════════════════════════════════════════
            # ★ Q3 정답과 Q5 빈칸이 같은 문장이면 **빈칸을 옮긴다** (_s158)
            #   왜 여기냐: 빈칸은 어휘보다 먼저 뽑히므로 뚫을 때는 Q3 정답을
            #   알 수 없다. 어휘가 확정된 지금이 유일하게 둘 다 아는 시점이고,
            #   보기(bogi)를 만들기 **전**이라 빈칸을 바꿔도 보기가 어긋나지 않는다.
            #   ★ 재시도가 아니라 코드 재선정이다 — API 호출 0회, 크레딧 0.
            #   이 자리를 놓치면 바깥 검사(answer_in_blank_sentence)로 넘어가는데
            #   그건 `not is_last` 로 묶여 있어 **마지막 시도에서는 그냥 통과**한다.
            #   실측(업로드 22파일 58문항): 같은 문장 7건이 그렇게 나갔다.
            # ════════════════════════════════════════════════════════════
            try:
                from variation.vocab_q3 import answer_sentence_span as _asp
                _ap, _asent = _asp(data.get("vocab_items"), data.get("paragraphs"))
                if _asent and ("<BLANK_A>" in _asent or "<BLANK_B>" in _asent):
                    _orig_sent = (_asent.replace("<BLANK_A>", data.get("blank_A", ""))
                                        .replace("<BLANK_B>", data.get("blank_B", "")))
                    _pre = data.get("_paras_preblank")
                    _alt = (pick_a_q5_blanks(_pre, "", "", pid,
                                             statements=data.get("statements"),
                                             avoid=[_orig_sent])
                            if _pre else None)
                    if _alt:
                        # ★★ 빈칸을 옮겼으면 어휘 좌표를 **반드시** 다시 맞춘다 (_s162)
                        #   빈칸은 여러 낱말이 <BLANK_A> 토큰 하나로 줄어든다. 자리가
                        #   바뀌면 그 뒤 낱말의 인덱스가 통째로 밀린다. 어휘 항목의
                        #   (para, idx) 는 옛 단락 기준이라 그대로 두면 전부 어긋난다.
                        #   실측(25년 고2 9월 22번): 세 번의 시도가 전부 이 경로로 죽어
                        #   "Q3 어휘 N번 자리 불일치 — 원문[0][31]='that' vs 'growing'"
                        #   이 뜨고 지문이 통째로 빠졌다. _s158 이 심은 회귀다.
                        #   → 새 단락 기준으로 재정렬하고, 실패하면 빈칸 이동을 **되돌린다.**
                        #     (어긋난 좌표로 내보내느니 겹친 채로 바깥 검사에 맡긴다)
                        _bk = (data["paragraphs"], data.get("blank_A"), data.get("blank_B"))
                        data["paragraphs"] = _alt["paragraphs"]
                        data["blank_A"] = _alt["blank_A"]
                        data["blank_B"] = _alt["blank_B"]
                        _rerr = []
                        _re_items = normalize_llm_vocab(
                            data.get("vocab_items"), data["paragraphs"],
                            blank_token_spans(data["paragraphs"]),
                            pid=pid, report=_rerr, strict=False)
                        if _re_items:
                            data["vocab_items"] = _re_items
                            print(f"[VAR][A][{pid}] Q3 정답이 Q5 빈칸과 같은 문장 → "
                                  f"빈칸 재선정 (A)'{_alt['blank_A'][:35]}' "
                                  f"(B)'{_alt['blank_B'][:35]}' + 어휘 좌표 재정렬 (_s162)")
                        else:
                            (data["paragraphs"], data["blank_A"], data["blank_B"]) = _bk
                            print(f"[VAR][A][{pid}] ⚠ 빈칸을 옮기면 어휘 좌표가 깨진다 "
                                  f"({_rerr[-1] if _rerr else '사유 미기록'}) → 빈칸 이동 취소, "
                                  f"바깥 검사로 넘김 (_s162)")
                    else:
                        print(f"[VAR][A][{pid}] ⚠ Q3 정답이 Q5 빈칸과 같은 문장인데 "
                              f"대체 빈칸을 못 찾음 — 바깥 검사로 넘김 (_s158)")
            except Exception as _bse:
                print(f"[VAR][A][{pid}] ⚠ 검사 건너뜀 (Q3↔Q5 빈칸 재선정): {_bse}")

            # ★ Q5 보기(bogi) 자동 생성: blank_A + blank_B의 모든 단어를 셔플해서 사용.
            #   모델이 만든 bogi는 무시 → 보기 누락/변형으로 인한 불일치를 원천 차단.
            try:
                bw = (str(data.get("blank_A", "")) + " " + str(data.get("blank_B", ""))).split()
                if bw:
                    seed = int(hashlib.md5((pid + str(data.get("blank_A", ""))).encode()).hexdigest()[:8], 16)
                    shuffled = list(bw)
                    rng = random.Random(seed)
                    for _ in range(5):
                        rng.shuffle(shuffled)
                        if shuffled != bw:
                            break
                    data["bogi"] = shuffled
            except Exception:
                pass

            is_last = (attempt == MAX_RETRIES)
            # === PREVAL 진단: 검증 직전 data가 실제로 뭘 담는지 ===
            try:
                def _pv_norm(t):
                    t = re.sub(r"<[^>]+>", " ", str(t or "")); t = re.sub(r"[^a-z0-9 ]", " ", t.lower()); return re.sub(r"\s+", " ", t).strip()
                _pv_iw = _pv_norm(data.get("intro", "")).split()
                _pv_probe = " ".join(_pv_iw[:8]) if len(_pv_iw) >= 8 else " ".join(_pv_iw)
                _pv_paras = [p[1] for p in data.get("paragraphs", []) if isinstance(p, (list, tuple)) and len(p) >= 2]
                _pv_hits = [i for i, t in enumerate(_pv_paras) if _pv_probe and _pv_probe in _pv_norm(t)]
                print(f"[VAR][A][{pid}] PREVAL intro={data.get('intro','')[:45]!r} "
                      f"npara={len(_pv_paras)} probe={_pv_probe[:40]!r} dup_hits={_pv_hits} "
                      f"p0={_pv_paras[0][:45]!r}" if _pv_paras else f"[VAR][A][{pid}] PREVAL npara=0")
            except Exception as _pe:
                print(f"[VAR][A][{pid}] PREVAL 예외: {_pe}")

            # ★★ (버) 객관식 정답 위치 셔플 (A: 주제 Q1 / 핵심빈칸 Q3) — 정답이 ①에 쏠리던 문제 교정.
            #   순서(order_correct)는 위치형이라 손대지 않는다.
            #   Q3는 어휘 유형이라 번호가 지문 등장 순서로 붙는다 — 셔플 대상이 아니다.
            for _tag, _ok, _ck in (("topicA", "topic_options", "topic_correct"),):
                if isinstance(data.get(_ok), list) and isinstance(data.get(_ck), int):
                    _before = data[_ck]
                    data[_ok], data[_ck] = _shuffle_choices(
                        data[_ok], data[_ck], _choice_seed(pid, _tag, data.get(_ok)))
                    print(f"[VAR][A][{pid}] SHUF {_tag}(loop): {_before} -> {data[_ck]}")
                else:
                    print(f"[VAR][A][{pid}] SHUF {_tag}(loop): SKIP "
                          f"(opt={type(data.get(_ok)).__name__}, cor={type(data.get(_ck)).__name__})")

            # ★★ Q1 주제 선지의 관사를 통일한다 (_s153).
            #   기출 35개 선지 중 관사로 시작한 것은 0개다(프롬프트 3-0 문두 명사 목록).
            #   다섯 중 하나만 'the ~' 면 학생은 지문을 안 읽고 그것을 고른다 — 실측 5건.
            #   ★ **소수파일 때만** 뗀다. 다섯 개가 다 관사로 시작하면 그건 통일된 것이니
            #     건드리지 않는다. 거르는 게 아니라 맞추는 것이므로 항목이 빠질 일이 없다.
            try:
                _to = data.get("topic_options")
                if isinstance(_to, list) and len(_to) == 5:
                    _art = [i for i, o in enumerate(_to)
                            if re.match(r"^\s*(?:a|an|the)\s+\S", str(o), re.I)]
                    if 1 <= len(_art) <= 2:
                        for i in _art:
                            _new = re.sub(r"^\s*(?:a|an|the)\s+", "", str(_to[i]), flags=re.I)
                            print(f"[VAR][A][{pid}] Q1 선지 관사 통일: "
                                  f"'{str(_to[i])[:40]}' → '{_new[:40]}'")
                            _to[i] = _new
            except Exception as _ae:
                print(f"[VAR][A][{pid}] ⚠ 검사 건너뜀 (Q1 선지 관사 통일): {_ae}")

            # ★ 렌더링용 단락 — 원본 paragraphs는 그대로 두고 밑줄만 얹은 사본을 만든다.
            #   검증은 원본으로 하고 화면에는 이걸 쓴다. (Q2 복원 검증 무영향)
            if data.get("vocab_items"):
                try:
                    from variation.vocab_q3 import apply_vocab_items
                    data["paragraphs_render"] = apply_vocab_items(
                        data["paragraphs"], data["vocab_items"])
                except Exception:
                    data["paragraphs_render"] = None

            # ★ 답지용 한글 해석 (_s106) — 정답 주제문은 영어라 답지만 보고는
            #   맞는지 알기 어렵다. 저장 직전 한 번만 만든다(캐시에 함께 들어간다).
            try:
                _tc = data.get("topic_correct")
                _to = data.get("topic_options")
                if (isinstance(_to, list) and isinstance(_tc, int)
                        and 0 <= _tc < len(_to) and not data.get("topic_answer_kr")):
                    data["topic_answer_kr"] = _ensure_kr(str(_to[_tc]))
            except Exception as _ke:
                print(f"[VAR][A][{pid}] 주제 해석 생성 예외({_ke})")

            # ★ 절대어 코드 검사는 뺐다 (_s133).
            #   "이 'never' 가 절대 주장인가 관용구인가"는 의미 판단이라 코드가 못 한다.
            #   단어 목록으로 재니 정상 제목이 계속 걸렸다 —
            #     One-Size-Fits-All / One Reasoning Fits All / The Anchor of All ...
            #     Never Enough / No One Wants
            #   걸릴 때마다 예외 목록에 추가하는 건 audience 때와 같은 실수다.
            #   프롬프트가 지고, 코드는 개입하지 않는다.

            # ★★ 내부 가림 문자열은 검증 **전에** 코드가 되돌린다 (_s162)
            #   아래 internal_marker_leaks 검사는 `not is_last` 로 묶여 있어
            #   마지막 시도에서는 꺼진다. 실측(25년 고2 9월 22번): 3회 시도 끝에
            #   답지 근거가 “… predicted the wrong [[[빈칸]]] quite how powerful …”
            #   로 그대로 나갔다. 선생님이 보는 답지에 코드 내부 문자열이 찍힌다.
            #   → 재시도로 못 막는 것은 코드가 고친다(_s158 과 같은 방침).
            #     인용문 앞뒤를 원문에서 찾아 그 사이 문구를 도로 끼워 넣는다.
            #     복원 못 하면 지운다 — 내부 문자열을 인쇄하느니 낫다.
            try:
                from variation.vocab_q3 import repair_internal_marks as _rim
                _rf = _rim(data, en_text)
                if _rf:
                    print(f"[VAR][A][{pid}] 내부 마커 {len(_rf)}곳 복원 (_s162) — "
                          + "; ".join(f"{k}" for k, _ in _rf[:4]))
            except Exception as _rie:
                print(f"[VAR][A][{pid}] ⚠ 검사 건너뜀 (내부 마커 복원): {_rie}")

            errors = validate_a(data, en_text, pid, lenient=is_last)

            # ★★ Q3 정답 자리 문장을 Q4 진술이 근거로 삼으면 정답이 갈린다 (_s147).
            #   Q4 진술·해설은 원문 기준인데 학생이 보는 지문은 Q3 로 뒤집혀 있다.
            #   실측(능률(민) 04과 01번): 지문엔 'seems easy' 인데 Q4 '라'가
            #   "…seems easy" 이고 해설은 원문 'difficult' 를 근거로 (X) 라 했다.
            #   학생이 지문대로 읽으면 참이다. 재시도로 Q4 진술을 다시 받게 한다.
            # ★★ 내부 마커가 답지·문항으로 새어나가면 안 된다 (_s150).
            #   실측(25년 고2 9월 37·39번): 프롬프트가 Q5 자리를 가릴 때 쓰는
            #   [[[여기는 Q5 빈칸 …]]] 를 LLM 이 근거 인용문에 그대로 베껴
            #   답지에 찍혔다. 닫힌 목록이라 오탐이 없다.
            try:
                from variation.vocab_q3 import internal_marker_leaks as _leaks
                _lk = _leaks(data)
                if _lk and not is_last:
                    _w = ", ".join(f"{p2}:{m}" for p2, m in _lk[:4])
                    errors = list(errors) + [
                        f"[{pid}] [CRITICAL] 내부 마커가 산출물에 남았다 ({_w}) — "
                        f"지문을 인용할 때 [[[...]]] 같은 표시는 빼고 실제 문장만 쓸 것"]
            except Exception as _e:
                # ★ 조용히 넘기지 않는다 (_s151) — 아래 주석 참조
                print(f"[VAR][A][{pid}] ⚠ 검사 건너뜀 (evidence_not_in_passage): {_e}")

            # ★★ Q4 근거 인용문이 지문에 실제로 있어야 한다 (_s149).
            #   O/X 판정이 맞는지는 코드가 못 본다. 그 앞 단계 — 근거로 든 문장이
            #   지문에 있기는 한가 — 는 볼 수 있고, 지어낸 근거는 판정도 대개 틀린다.
            try:
                from variation.vocab_q3 import evidence_not_in_passage as _evc
                _miss = _evc(data.get("statements"),
                             data.get("statements_evidence"), en_text)
                if _miss and not is_last:
                    _t = "; ".join(f"{l} «{e}»" for l, e in _miss)
                    errors = list(errors) + [
                        f"[{pid}] [CRITICAL] Q4 근거가 지문에 없다 ({_t}) — "
                        f"지문에 있는 문장을 그대로 인용할 것"]
            except Exception as _e:
                # ★ 조용히 넘기지 않는다 (_s151) — 아래 주석 참조
                print(f"[VAR][A][{pid}] ⚠ 검사 건너뜀 (statements_leak_blanks): {_e}")

            # ★★ Q4 진술이 Q5 빈칸 정답을 그대로 말해주면 안 된다 (_s148).
            #   실측: 진술 "Ocean Alliance was founded to protect whales and the
            #   oceans" ↔ Q5(A) 'founded Ocean Alliance to protect whales and the
            #   earth's oceans'. Q4 를 먼저 읽으면 영작 답이 보인다.
            try:
                from variation.vocab_q3 import statements_leak_blanks as _leak
                _hits = _leak(data.get("statements"),
                              str(data.get("blank_A") or ""),
                              str(data.get("blank_B") or ""))
                if _hits and not is_last:
                    _txt = ", ".join(f"{l}→Q5({t})" for l, t in _hits)
                    errors = list(errors) + [
                        f"[{pid}] [CRITICAL] Q4 진술이 Q5 빈칸 정답을 그대로 담고 "
                        f"있다 ({_txt}) — 학생이 Q4 만 보고 영작 답을 안다. "
                        f"빈칸 밖 문장을 근거로 진술을 만들 것"]
            except Exception as _e:
                # ★ 조용히 넘기지 않는다 (_s151) — 아래 주석 참조
                print(f"[VAR][A][{pid}] ⚠ 검사 건너뜀 (answer_in_blank_sentence): {_e}")

            # ★★ Q3 정답 자리가 Q5 빈칸이 든 문장이면 판단 근거가 비어 있다 (_s148).
            try:
                from variation.vocab_q3 import answer_in_blank_sentence as _abs
                if _abs(data.get("vocab_items"), data.get("paragraphs")) and not is_last:
                    errors = list(errors) + [
                        f"[{pid}] [CRITICAL] Q3 어휘 정답 단어가 Q5 빈칸과 같은 "
                        f"문장에 있다 — 그 문장의 술부가 빈칸이라 정답을 판단할 "
                        f"근거가 지문에 없다. 다른 문장에서 고를 것"]
            except Exception as _e:
                # ★ 조용히 넘기지 않는다 (_s151) — 아래 주석 참조
                print(f"[VAR][A][{pid}] ⚠ 검사 건너뜀 (q4_conflicts_with_answer): {_e}")

            try:
                from variation.vocab_q3 import (q4_conflicts_with_answer as _q4c,
                                                q4_conflict_unsatisfiable as _q4u2)
                _ev = [e for _, _, e in (data.get("statements_evidence") or [])] \
                    if data.get("statements_evidence") and isinstance(
                        (data.get("statements_evidence") or [None])[0], (list, tuple)) \
                    else (data.get("statements_evidence") or [])
                # ★ 통과 불가능한 조건이면 걸지 않는다 (_s156). 어휘 루프에서 풀어 줘도
                #   여기서 다시 잡으면 함정이 한 층 위로 옮겨갈 뿐이다 — 실측으로 그랬다.
                _unsat = _q4u2(data.get("paragraphs"), _ev)
                if _unsat:
                    print(f"[VAR][A][{pid}] Q4 근거가 모든 문장을 덮는다 "
                          f"— 바깥 겹침 검사도 생략 (_s156)")
                _hit = [] if _unsat else _q4c(data.get("vocab_items"), data.get("paragraphs"), _ev,
                                              statements=data.get("statements"))
                if _hit and not is_last:
                    errors = list(errors) + [
                        f"[{pid}] [CRITICAL] Q4 진술 {'·'.join(_hit)} 이(가) Q3 어휘 "
                        f"정답 문장을 근거로 삼는다 — 학생이 보는 지문은 그 자리가 "
                        f"뒤집혀 있어 정답이 갈린다. 다른 문장으로 진술을 만들 것"]
            except Exception as _e:
                # ★ 조용히 넘기지 않는다 (_s151) — 아래 주석 참조
                print(f"[VAR][A][{pid}] ⚠ 검사 건너뜀 (q4_conflicts_with_answer): {_e}")

            if not errors:
                # ★ Q1 주제 정답 자리를 강 단위로 돌린다 (_s136).
                #   검증을 다 통과한 뒤에 섞어야 안전하다 — 앞서 섞으면
                #   topic_correct 를 참조하는 검사들과 어긋난다.
                # ★ 이미 쓴 정답과 어근이 겹치면 재시도 (_s139)
                try:
                    _clash = ""
                    for _vi in (data.get("vocab_items") or []):
                        if _vi.get("is_answer"):
                            _clash = (answer_word_clash(book, unit, _vi.get("original"))
                                      or answer_word_clash(book, unit, _vi.get("shown")))
                            break
                    if _clash and not is_last:
                        last_errors = [
                            f"[{pid}] [유형A] Q3 어휘 정답이 이 강에서 이미 쓴 "
                            f"'{_clash}' 와 어근이 같다 — 같은 과에서 같은 말이 두 번 "
                            f"정답이면 학생이 눈치챈다. 다른 자리를 고를 것."]
                        print(f"[VAR][A][{pid}] Q3 어휘 정답 어근 중복('{_clash}') → 재시도")
                        continue
                except Exception as _ce:
                    print(f"[VAR][A][{pid}] 어근 중복 확인 예외({_ce})")

                # ★ 이 강에서 쓴 정답 어휘를 기록한다 (_s138) — 다음 지문이 피하게
                try:
                    for _vi in (data.get("vocab_items") or []):
                        if _vi.get("is_answer"):
                            note_answer_word(book, unit, _vi.get("original"))
                            note_answer_word(book, unit, _vi.get("shown"))
                except Exception as _e:
                    # ★ 조용히 넘기지 않는다 (_s151). vocab_q3.py 가 옛 버전이면
                    #   import 가 실패하는데, pass 로 삼키면 검사가 통째로 안 돈 걸
                    #   아무도 모른다. 실측: generator 만 올리고 vocab_q3 를 안 올려
                    #   _s148 의 두 검사가 조용히 꺼진 채 배포됐다.
                    print(f"[VAR][A][{pid}] ⚠ 검사 건너뜀 (internal_marker_leaks): {_e}")

                try:
                    _to2, _tc2 = shuffle_correct_position(
                        data.get("topic_options"), data.get("topic_correct"),
                        book, unit, pid, "topicpos")
                    if _tc2 != data.get("topic_correct"):
                        data["topic_options"], data["topic_correct"] = _to2, _tc2
                        print(f"[VAR][A][{pid}] Q1 주제 정답 자리 → {_tc2 + 1}번")
                except Exception as _pe:
                    print(f"[VAR][A][{pid}] 주제 자리 순환 예외({_pe})")

                # ★ 관대 모드로 나가는 것에는 반드시 이유를 남긴다 (_s158)
                #   `not is_last` 로 묶인 검사가 여섯 개인데, 마지막 시도에서는
                #   전부 조용히 꺼진다. 무엇을 봐주고 내보냈는지 로그에 없으면
                #   산출물에 결함이 있어도 아무도 모른다 — 실측으로 그랬다.
                if is_last:
                    _passed = []
                    # ★ 검사마다 따로 감싼다 (_s163).
                    #   실측(부천고1 20번): `evidence_not_in_passage` 를 인자 두 개로
                    #   부르고 있어 TypeError 가 났고, 검사 셋이 한 try 에 묶여 있던 탓에
                    #   **그 뒤 검사가 전부 안 돌았다.** 로그에는 '감사 실패' 한 줄만 남아
                    #   무엇을 못 봤는지도 알 수 없었다. 감사가 감사를 못 한 셈이다.
                    def _audit(name, fn):
                        try:
                            if fn():
                                _passed.append(name)
                        except Exception as _ae:
                            _passed.append(f"{name} 감사 실패({_ae})")
                    try:
                        from variation.vocab_q3 import (
                            answer_in_blank_sentence as _c1,
                            statements_leak_blanks as _c2,
                            evidence_not_in_passage as _c3)
                        _audit("Q3 정답이 Q5 빈칸과 같은 문장",
                               lambda: _c1(data.get("vocab_items"), data.get("paragraphs")))
                        _audit("Q4 진술이 Q5 정답을 누출",
                               lambda: _c2(data.get("statements"), data.get("blank_A", ""),
                                           data.get("blank_B", "")))
                        #   ★ 인자 셋이다 — (진술, 근거, 원문). 둘만 넘기면 TypeError.
                        _audit("Q4 근거가 지문에 없음",
                               lambda: _c3(data.get("statements"),
                                           data.get("statements_evidence"), en_text))
                        # ★ 이 검사가 감사 목록에서 빠져 있었다 (_s163).
                        #   `not is_last` 로 꺼지는 여섯 검사 중 하나인데 여기 없어서,
                        #   마지막 시도로 나간 산출물에 이 결함이 있어도 로그에 흔적이
                        #   없었다. 실측(25년 고1 9월 부천고1): 8지문 중 3지문이
                        #   이 결함을 달고 나갔고(22·23·24번) 아무 기록도 없었다.
                        from variation.vocab_q3 import (
                            q4_conflicts_with_answer as _c4,
                            q4_conflict_unsatisfiable as _c4u)
                        _ev4 = [e for _, _, e in (data.get("statements_evidence") or [])] \
                            if (data.get("statements_evidence")
                                and isinstance((data.get("statements_evidence") or [None])[0],
                                               (list, tuple))) \
                            else (data.get("statements_evidence") or [])
                        def _conflict():
                            if _c4u(data.get("paragraphs"), _ev4):
                                return None
                            _h4 = _c4(data.get("vocab_items"), data.get("paragraphs"), _ev4,
                                      statements=data.get("statements"))
                            return ("Q4 진술 " + "·".join(_h4) + " 이(가) Q3 정답 자리에 기댐"
                                    ) if _h4 else None
                        _audit("Q4 진술이 Q3 정답 자리에 기댐 — 학생 판정이 답지와 갈린다",
                               _conflict)
                    except Exception as _ae:
                        _passed.append(f"감사 자체 실패({_ae})")
                    if _passed:
                        print(f"[VAR][A][{pid}] ⚠⚠ 관대 모드로 내보냄 — "
                              f"봐준 결함: {' / '.join(_passed)} (_s158)")

                save_cached(cache_key, "variation_a", data)
                mode_str = "관대 모드" if is_last else "엄격 모드"
                print(f"[VAR][A][{pid}] 생성 완료 (시도 {attempt}, {mode_str})")
                return data
            last_errors = errors
            has_critical = any("[CRITICAL]" in e for e in errors)
            if is_last and data and not has_critical:
                last_data = data
                print(f"[VAR][A][{pid}] 마지막 시도 실패했지만 경미한 오류뿐 → 데이터 보관: {len(errors)}건")
            elif is_last and has_critical:
                print(f"[VAR][A][{pid}] 마지막 시도에 치명적 오류(순서/빈칸/원문) → fallback 거부, 이 항목 생략")
            print(f"[VAR][A][{pid}] 시도 {attempt} 실패 ({len(errors)}건):")
            for err in errors[:5]:
                print(f"    - {err[:200]}")
        except Exception as e:
            traceback.print_exc()
            last_errors = [f"예외: {e}"]

    # ★ 5회 모두 실패해도 마지막 데이터가 있으면 그거라도 사용 (불완전한 A라도 없는 것보단 나음)
    if last_data is not None:
        save_cached(cache_key, "variation_a", last_data)
        print(f"[VAR][A][{pid}] 관대 fallback 사용 ({MAX_RETRIES}회 실패)")
        return last_data
    raise RuntimeError(f"유형 A 생성 실패 ({MAX_RETRIES}회). 마지막 오류:\n" + "\n".join(last_errors[:5]))


# ============ 유형 B 생성 ============
def generate_variation_b(
    passage_text: str,
    pid: str = "?",
    book: str = "",
    unit: str = "",
    force_regenerate: bool = False,
    cache_only: bool = False,
) -> dict:
    en_text, _ = split_passage_and_translation(passage_text)
    cache_key = make_cache_key(book, unit, pid, en_text, "b")

    if not force_regenerate:
        cached = load_cached(cache_key, "variation_b")
        if cached:
            print(f"[VAR][B][{pid}] 캐시 히트")
            return cached

    # 합치기 단계: 캐시에 없으면 생성하지 않고 None
    if cache_only:
        print(f"[VAR][B][{pid}] 캐시 없음 — cache_only이므로 생략")
        return None

    # ★ 같은 지문의 유형 A 결과를 캐시에서 읽어 둔다(있으면).
    #   A Q1 주제와 B Q2 주제가 같은 명제를 쓰면 한쪽 답이 다른 쪽 힌트가 된다.
    #   A·B는 독립 호출이라 호출부를 안 고치고 캐시로 연결한다. 없으면 그냥 건너뛴다.
    _a_data = None
    try:
        _a_data = load_cached(make_cache_key(book, unit, pid, en_text, "a"), "variation_a")
    except Exception:
        pass
    _a_topics = []
    if isinstance(_a_data, dict) and isinstance(_a_data.get("topic_options"), list):
        _a_topics = [str(t) for t in _a_data["topic_options"] if str(t).strip()]

    last_errors = []
    last_data = None  # 마지막 fallback용
    _err_history = []          # ★ 재시도 사유 누적 (_s122)
    for attempt in range(1, MAX_RETRIES_B + 1):
        try:
            user_msg = (
                f"Passage ID: {pid}\n\n"
                f"Original English passage:\n{en_text}\n\n"
                "Generate the variation problem (Type B). Return ONLY the JSON object."
            )
            if _a_topics:
                user_msg += (
                    "\n\n# ★ AVOID THESE PROPOSITIONS (already used as Type A topic choices for the same passage)\n"
                    + "\n".join(f"  - {t}" for t in _a_topics)
                    + "\nYour Q2 topic options — the correct one AND the distractors — must express "
                      "DIFFERENT propositions from the list above. Do not reuse their claim even in reworded form "
                      "(e.g. if one of them is about 'balancing X with Y', do not write another 'balancing' option). "
                      "Same passage, different angle."
                )
            if last_errors:
                # ★ 지금까지 나온 사유를 **누적해서** 보여준다 (_s122).
                #   마지막 사유만 주면 그건 고치고 앞서 지적받은 걸 다시 어긴다 —
                #   재시도가 제목→절대어→복수정답 식으로 돌기만 하고 수렴하지 않았다.
                for _e in last_errors:
                    if _e not in _err_history:
                        _err_history.append(_e)
                user_msg += (
                    "\n\n# ⚠️ 지금까지 지적받은 것 전부 — 하나도 다시 어기지 마라\n"
                    "#    (앞 항목을 고치면서 뒤 항목을 어기는 일이 반복되고 있다)\n"
                    + "\n".join(f"  ✗ {e}" for e in _err_history[-8:])
                    + "\n\n# REMINDER OF CRITICAL CHECKS FOR TYPE B:\n"
                    "  1. <MARK1>, <MARK2>, <MARK3>, <MARK4>, <MARK5> MUST be spread across the ENTIRE passage\n"
                    "     There MUST be AT LEAST 3 words between every adjacent pair of markers\n"
                    "     BAD example: 'word word <MARK1> word word <MARK2><MARK3> word' — MARK2/MARK3 adjacent (0 words between)\n"
                    "     GOOD example: 'word word word <MARK1> word word word word <MARK2> word word word <MARK3> ...'\n"
                    "  2. blank_A and blank_B are natural key phrases (~4-8 words each, do not pad to a number)\n"
                    "  3. topic_writing_answer: write ONE natural, fully grammatical topic sentence FIRST (don't count words or fit a word bank; the code splits it). Naturalness/grammar first, ~12-20 words. No bare-verb subject, no sentence ending in a preposition, no 'Despite + clause', no 'modal + adjective'.\n"
                    "  4. Hyphenated words (south-facing, well-known) stay as ONE token in both blank and bogi\n"
                    "  5. blank_summary_bogi must contain EVERY word from blank_A + blank_B (count articles/preps)\n"
                    "  6. ★ Q3 summary_options: EACH (A) and (B) must be EXACTLY ONE WORD (no phrases!)\n"
                    "     GOOD: [['manipulation','extension'], ['control','delay'], ...]\n"
                    "     BAD: [['south-facing garden beds', 'flat stones from beach'], ...]\n"
                    "  7. All five (A) values must be DIFFERENT words; all five (B) values must be DIFFERENT"
                )

            # ★ (누) 결정 칸을 (A)↔(B) 번갈아 지정한다 (_s108).
            #   프롬프트에 "한쪽만 흐리게"라고만 하면 LLM 이 매번 같은 쪽을 고른다
            #   (실측: 3문항 전부 (B)가 결정 칸). 그러면 학생이 "항상 B를 보면 된다"를
            #   배운다. 어휘 정답 자리와 같은 방식으로 강 안에서 번갈아 준다.
            _seqb = re.findall(r"\d+", str(pid))
            _seqb = int(_seqb[0]) if _seqb else 0
            _startb = int(hashlib.md5(
                (str(book) + "|" + str(unit) + "|decider").encode()
            ).hexdigest()[:8], 16) % 2
            _decider = "(B)" if (_startb + _seqb) % 2 == 0 else "(A)"
            _blurred = "(A)" if _decider == "(B)" else "(B)"
            user_msg += (
                f"\n\n★★ 이 지문의 Q3 결정 칸은 **{_decider}** 다.\n"
                f"  {_blurred} 는 흐리게 — 정답 + 근사 유의어 2개를 넣어 그 칸만으로는 못 고르게 한다.\n"
                f"  {_decider} 는 선명하게 — 맞는 말이 하나뿐이고 그 유의어를 다른 행에 두지 마라.\n"
                f"  ★ 매번 같은 칸이 결정 칸이면 학생이 '항상 그 칸만 보면 된다'를 배운다.\n"
                f"    이 지문은 반드시 {_decider} 로 결정되게 하라.")

            raw = call_claude(SYSTEM_PROMPT_B, user_msg)
            data = extract_json_from_response(raw)

            # ★★★ summary_design → summary_options 변환 (_s123)
            #   LLM 이 다섯 쌍을 자유롭게 나열하면 (A)에 유의어를 다섯 개 넣는 식으로
            #   설계가 무너진다(실측: demonstrate/suggest/reveal/indicate/reflect).
            #   출력 형식을 **역할 칸**으로 바꿔 그 배치로만 낼 수 있게 했다.
            #     correct / syn_A_1 / syn_A_2 / syn_B / both_wrong
            #   정답 말고는 반드시 한 칸이 틀리므로 복수정답이 구조적으로 안 나온다.
            #   평가원 40번 구조 그대로다 — 양쪽에 유의어를 두되 짝이 안 맞게.
            #   섞기와 정답 번호는 코드가 정한다.
            _dsg = data.get("summary_design")
            # ★ _s126 — design 도 options 도 없으면 형식을 못 지킨 것이다.
            #   validator 가 summary_options/summary_correct 를 필수로 요구하므로
            #   그대로 가면 '필수 필드 누락'으로 죽는다. 사유를 정확히 알려주고 재시도.
            if not isinstance(_dsg, dict) and not isinstance(data.get("summary_options"), list):
                last_errors = [
                    f"[{pid}] [유형B] Q3 summary_design 이 없다 — 출력 JSON 에 "
                    f'"summary_design": {{"correct": {{...}}, "syn_A_1": {{...}}, '
                    f'"syn_A_2": {{...}}, "syn_B": {{...}}, "both_wrong": {{...}}}} '
                    f"다섯 칸을 반드시 넣어라. summary_options 를 직접 만들지 마라 — "
                    f"선지 순서와 정답 번호는 코드가 정한다."]
                print(f"[VAR][B][{pid}] Q3 summary_design 없음 → 재시도")
                if not is_last:
                    continue
            if isinstance(_dsg, dict):
                _keys = ("correct", "syn_A_1", "syn_A_2", "syn_B", "both_wrong")
                _miss = [k for k in _keys
                         if not isinstance(_dsg.get(k), dict)
                         or not str(_dsg[k].get("A", "")).strip()
                         or not str(_dsg[k].get("B", "")).strip()]
                if _miss:
                    last_errors = [
                        f"[{pid}] [유형B] Q3 summary_design 칸이 비었다: {_miss} — "
                        f"correct / syn_A_1 / syn_A_2 / syn_B / both_wrong 다섯 칸을 "
                        f"모두 (A)(B) 와 함께 채울 것."]
                    print(f"[VAR][B][{pid}] Q3 summary_design 미완성 {_miss} → 재시도")
                    if not is_last:
                        continue
                else:
                    _nw = []
                    if not str(_dsg["syn_A_1"].get("B_why", "")).strip(): _nw.append("syn_A_1.B_why")
                    if not str(_dsg["syn_A_2"].get("B_why", "")).strip(): _nw.append("syn_A_2.B_why")
                    if not str(_dsg["syn_B"].get("A_why", "")).strip():   _nw.append("syn_B.A_why")
                    if _nw and not is_last:
                        last_errors = [
                            f"[{pid}] [유형B] Q3 {_nw} 가 비었다 — 그 칸이 왜 틀렸는지 "
                            f"못 적으면 안 틀린 것이다. 명백히 틀린 말로 갈아라."]
                        print(f"[VAR][B][{pid}] Q3 _why 누락 {_nw} → 재시도")
                        continue
                    _rows = [(_dsg[k]["A"], _dsg[k]["B"]) for k in _keys]
                    _seq3 = re.findall(r"\d+", str(pid))
                    _seq3 = int(_seq3[0]) if _seq3 else 0
                    _st3 = int(hashlib.md5(
                        (str(book) + "|" + str(unit) + "|q3pos").encode()
                    ).hexdigest()[:8], 16) % 5
                    _cpos = (_st3 + _seq3) % 5
                    _rest = _rows[1:]
                    random.Random(f"{book}|{unit}|{pid}|q3ord").shuffle(_rest)
                    _final = _rest[:_cpos] + [_rows[0]] + _rest[_cpos:]
                    data["summary_options"] = [[str(a).strip(), str(b).strip()]
                                               for a, b in _final]
                    data["summary_correct"] = _cpos
                    print(f"[VAR][B][{pid}] Q3 design → 선지 구성 완료 "
                          f"(정답 {_cpos + 1}번)")

            # ★★ Q1 삽입 정답 위치 코드 정정 (Q4 빈칸뚫기와 같은 원리)
            #   AI가 마커는 박되 "어느 자리에서 문장이 빠졌나"(position_correct)를 자주 틀린다.
            #   코드가 마커를 1번부터 끝까지 하나씩 넣어보고, 원문이 복원되는 자리를 정답으로 정정.
            #   → 정답 위치 100% 보장. AI는 마커만 적당히 박으면 됨.
            try:
                import re as _re1
                _pwm = str(data.get("passage_with_marks") or "")
                _gs = str(data.get("given_sentence") or "").strip()
                _orig = _re1.sub(r'\s+', ' ', str(en_text or "")).strip()
                def _alnum1(t):
                    return _re1.sub(r"[^a-z0-9]", "", str(t).lower())
                if _pwm and _gs and _orig:
                    _found = None
                    for _k in range(1, 6):  # MARK1..MARK5 후보 전부 시험
                        _mk = f"<MARK{_k}>"
                        if _mk not in _pwm:
                            continue
                        _recon = _pwm.replace(_mk, " " + _gs + " ")
                        _recon = _re1.sub(r"<MARK\d>", "", _recon)
                        if _alnum1(_recon) == _alnum1(_orig):
                            _found = _k - 1  # 0-based
                            break
                    if _found is not None and _found != data.get("position_correct"):
                        data["position_correct"] = _found  # 코드가 찾은 정답으로 정정
            except Exception:
                pass  # 실패하면 AI가 찍은 값 유지 (검증에서 다시 걸러짐)

            # ★★ Q1 삽입 — LLM이 '자리를 확정하는 단서가 있는 문장'을 고르고 코드가 재구성.
            #   기존엔 코드가 '가운데 문장부터' 떼어봤다. 위치만 본 것이다.
            #   삽입 문제의 본질은 뺀 문장에 지시어·대명사·연결어·정관사 같은 단서가 있어
            #   자리가 하나로 확정되는가다. 그건 의미 판단이라 코드로 못 한다.
            #   LLM 선택이 복원 검증을 통과하면 그걸 쓰고, 실패하면 아래 fallback이 받는다.
            try:
                _sents_ins = split_sentences(en_text)
                if len(_sents_ins) >= 4:
                    _insraw = call_claude(INSERT_SYS, build_insert_prompt(_sents_ins),
                                          max_tokens=800)
                    _ins = extract_json_from_response(_insraw)
                    _idx = _ins.get("index")
                    if isinstance(_idx, int) and 1 <= _idx < len(_sents_ins):
                        _ib0 = build_insert_blocks_b(en_text, pid, preferred=_idx)
                        if _ib0 and _ib0.get("given_sentence") == _sents_ins[_idx]:
                            data["given_sentence"] = _ib0["given_sentence"]
                            data["passage_with_marks"] = _ib0["passage_with_marks"]
                            data["position_correct"] = _ib0["position_correct"]
                            data["position_count"] = _ib0["position_count"]
                            print(f"[VAR][B][{pid}] Q1 삽입 LLM 픽 — [{_idx}] "
                                  f"단서({_ins.get('anchor_type','?')}) '{_ins.get('anchor','')}'")
                        else:
                            print(f"[VAR][B][{pid}] Q1 삽입 LLM 픽[{_idx}] 복원 실패 → 코드 픽")
            except Exception as _ie:
                print(f"[VAR][B][{pid}] Q1 삽입 LLM 픽 예외({_ie}) → 코드 픽")

            # ★★ Q1 삽입 fallback — 위 자동정정으로도 '복원되는 자리'가 하나도 없으면
            #   (LLM이 given/본문을 변형해 어느 자리에 넣어도 원문이 안 맞는 경우),
            #   코드가 원문에서 삽입문제를 통째로 재구성한다. 복원되는 항목은 건드리지 않음.
            try:
                _pwm_fb = str(data.get("passage_with_marks") or "")
                _gs_fb = str(data.get("given_sentence") or "").strip()
                def _alnum_fb(t):
                    return re.sub(r"[^a-z0-9]", "", str(t).lower())
                _ok_fb = False
                if _pwm_fb and _gs_fb:
                    for _k in range(1, 6):
                        _mk = f"<MARK{_k}>"
                        if _mk not in _pwm_fb:
                            continue
                        _r = _pwm_fb.replace(_mk, " " + _gs_fb + " ")
                        _r = re.sub(r"<MARK\d>", "", _r)
                        if _alnum_fb(_r) == _alnum_fb(en_text):
                            _ok_fb = True
                            break
                if not _ok_fb:
                    _ib = build_insert_blocks_b(en_text, pid)
                    if _ib:
                        data["given_sentence"] = _ib["given_sentence"]
                        data["passage_with_marks"] = _ib["passage_with_marks"]
                        data["position_correct"] = _ib["position_correct"]
                        data["position_count"] = _ib["position_count"]
                        print(f"[VAR][B][{pid}] Q1 삽입 코드 재구성 적용 (복원되는 자리 없어 fallback)")
            except Exception:
                pass

            # ★★ Q5 주제문 단독 재생성 (1회독 step4 방식)
            #   Q1~Q5를 한 번에 만들면 주제문에 집중이 안 돼 수일치 등 실수가 난다.
            #   그래서 주제문만 따로 한 번 더 — 주제문 하나에만 집중 → 1회독 품질.
            try:
                _t_raw = call_claude(TOPIC_SENTENCE_SYS, build_topic_sentence_prompt(en_text), max_tokens=500)
                _t = extract_json_from_response(_t_raw)
                _ts = (_t.get("topic_sentence") or "").strip()
                if _ts and len(_ts.split()) >= 6:
                    data["topic_writing_answer"] = _ts  # 집중 생성한 깔끔한 주제문으로 교체
                    # ★ 한글 해석도 같은 문장의 번역으로 함께 교체.
                    #   영문만 바꾸고 한글은 1차 호출 결과를 남겨두면 답지에서 서로 다른 문장이 된다.
                    data["topic_writing_kr"] = _ensure_kr(_ts, _t.get("topic_sentence_kr"))
            except Exception:
                pass  # 실패하면 기존(한번에 만든) 주제문 유지

            # ★★ Q4 요약문 단독 재생성 (영작이라 비문 잦음 → 따로 집중 생성)
            #   요약문(full_summary)만 따로 생성하고, 그 안의 두 구절을 코드가 빈칸으로 뚫는다.
            #
            #   ★★ Q3 요약문을 함께 넘긴다 (_s99). 안 넘기면 같은 지문·같은 요구라
            #   LLM 이 Q3 와 똑같은 문장을 만든다. 실측:
            #     Q3 'The brain's (A) deployment of alternative sensory pathways ensures (B) recognition…'
            #     Q4 'The brain's (A) ensures (B).'      ← 같은 문장을 빈칸만 넓힌 것
            #   Q3 를 푼 학생이 Q4 를 그냥 베껴 쓰게 된다.
            #   → Q3 문장을 보여주고 '다른 각도로 잘라라' 고 시킨 뒤, 겹치면 한 번 더 시킨다.
            _SUM_FW = {"the", "a", "an", "of", "to", "in", "on", "at", "by", "with",
                       "from", "and", "or", "but", "that", "which", "is", "are", "was",
                       "were", "be", "as", "for", "its", "their", "this", "these", "it",
                       "while", "although", "though", "when", "may", "can", "has",
                       "have", "had", "been"}

            def _sum_grams(t, n):
                """내용어를 하나라도 포함한 n연속 집합. 기능어 연쇄는 겹쳐도 무방하다."""
                w = re.sub(r"[^a-z ]", " ", str(t or "").lower()).split()
                out = set()
                for k in range(len(w) - n + 1):
                    g = w[k:k + n]
                    if any(x not in _SUM_FW for x in g):
                        out.add(" ".join(g))
                return out

            def _summary_overlap(t1, t2):
                """Q3·Q4 요약문이 같은 각도로 잘렸는지 (_s99). 겹친 근거를 집합으로 반환.

                두 신호를 본다:
                  · 3연속 겹침    01번 'deployment of alternative' (같은 문장을 재사용)
                  · 첫 내용어 동일 02번 'Conflicting …'            (같은 자리에서 출발)

                ★ 2연속 겹침은 안 본다(_s99). 같은 지문을 요약하면 'online writing'
                  'historical fiction' 같은 주어가 양쪽에 자연스럽게 들어간다.
                  그걸로 재생성을 돌리면 헛돈다. 대신 03번 유형('online writing'만
                  겹치고 각도는 같은 경우)은 코드가 못 잡는다 — 프롬프트가 진다.
                """
                out = _sum_grams(t1, 3) & _sum_grams(t2, 3)

                def _first(t):
                    w = [x for x in re.sub(r"[^a-z ]", " ", str(t or "").lower()).split()
                         if x not in _SUM_FW]
                    return w[0] if w else ""
                if _first(t1) and _first(t1) == _first(t2):
                    out.add(f"(첫 내용어 '{_first(t1)}' 동일 — 같은 자리에서 출발)")
                return out

            def _filled_q3():
                """Q3 요약문의 (A)(B)에 정답을 채워 완성 문장으로 만든다 (_s100).

                ★ 빈칸을 공백으로 지우고 비교하면 3연속이 그 자리에서 끊겨 겹침을 못 잡는다.
                  실측: Q3 'requires (A) of key findings to sustain reader (B) in an impatient
                  digital environment' 와 Q4 '(A) requires (B) to sustain reader engagement in
                  an impatient digital environment' 이 뒷부분을 통째로 공유하는데도 통과했다."""
                _t = str(data.get("summary_template") or "")
                _o = data.get("summary_options")
                _c = data.get("summary_correct")
                if isinstance(_o, list) and isinstance(_c, int) and 0 <= _c < len(_o) \
                        and isinstance(_o[_c], (list, tuple)) and len(_o[_c]) >= 2:
                    _t = _t.replace("(A)", str(_o[_c][0]), 1).replace("(B)", str(_o[_c][1]), 1)
                return _t.replace("(A)", " ").replace("(B)", " ")

            # ════════════════════════════════════════════════════════════
            # Q4 — 어법 오류 찾아 고치기 (_s161)
            #   옛 Q4(요약영작)는 Q3(요약빈칸)와 둘 다 '지문을 한 문장으로 요약'이라
            #   논지가 하나뿐인 이상 구조적으로 겹쳤다. 프롬프트로 아무리 지시해도
            #   재시도만 태우고(실측 11회) 지문이 통째로 누락됐다.
            #   → 완전히 다른 능력으로 바꾼다. 난이도는 수능 영어 29번 수준.
            #
            #   ★ 반드시 위치 정정(원문 복원 검증)이 끝난 **뒤에** 심어야 한다.
            #     먼저 심으면 passage_with_marks 가 원문과 달라져 Q1 정답 계산이 깨진다.
            # ════════════════════════════════════════════════════════════
            data.pop("grammar_q4", None)
            try:
                _pwm = str(data.get("passage_with_marks") or "")
                _gs_avoid = str(data.get("given_sentence") or "")
                _plain = re.sub(r"<MARK\d>", " ", _pwm)
                _plain = re.sub(r"\s+", " ", _plain).strip()
                for _ga in range(2):
                    _gmsg = build_grammar_error_prompt(_plain, avoid_sentence=_gs_avoid)
                    if _ga:
                        _gmsg += ("\n\n[이전 시도가 거부됐다]\n  " + _gerr +
                                  "\n지문에 글자 그대로 있는 표현을 골라 다시 답하라.")
                    _graw = call_claude(GRAMMAR_ERROR_SYS, _gmsg, max_tokens=700)
                    _gj = extract_json_from_response(_graw) or {}
                    _cor = str(_gj.get("correct") or "").strip()
                    _wrg = str(_gj.get("wrong") or "").strip()
                    _gerr = ""
                    if not _cor or not _wrg:
                        _gerr = "correct/wrong 이 비었다"
                    elif _cor == _wrg:
                        _gerr = "correct 와 wrong 이 같다"
                    elif grammar_count(_cor, _pwm) == 0:
                        _gerr = f"'{_cor}' 가 지문에 글자 그대로 없다"
                    elif grammar_count(_cor, _pwm) != 1:
                        _gerr = (f"'{_cor}' 가 지문에 {grammar_count(_cor, _pwm)}번 나온다 "
                                 f"— 한 번만 나오는 표현을 골라라")
                    elif _gs_avoid and grammar_count(_cor, _gs_avoid):
                        _gerr = "주어진 문장(Q1)에 있는 표현이다 — 다른 자리를 골라라"
                    if _gerr:
                        print(f"[VAR][B][{pid}] Q4 어법 자리 거부 — {_gerr}")
                        continue
                    # ★ 심기 전 지문을 남긴다 (_s161 회귀 수정)
                    #   validate_b 의 Q1 삽입 위치 검증은 "정답 마커 자리에 주어진 문장을
                    #   도로 넣으면 원문이 복원되는가" 를 본다. 어법 오류를 심으면 그 한
                    #   낱말 때문에 복원이 영원히 실패한다 — 실측: B 5개 지문 전부 6회
                    #   재시도 끝에 죽고 /api/variation 이 500 을 냈다.
                    #   위치 정정 뒤에 심는 것만으로는 부족했다. 검증이 그 뒤에 또 돈다.
                    #   → 검증용 사본을 따로 넘긴다. 산출물에 나가는 것은 심은 쪽이다.
                    data["_pwm_pregrammar"] = _pwm
                    # 지문에 오류를 심는다 (딱 한 곳)
                    data["passage_with_marks"] = grammar_replace_once(_pwm, _cor, _wrg)
                    data["grammar_q4"] = {
                        "correct": _cor, "wrong": _wrg,
                        "point": str(_gj.get("point") or "").strip(),
                        "why": str(_gj.get("why") or "").strip(),
                        "sentence": str(_gj.get("sentence") or "").strip(),
                    }
                    print(f"[VAR][B][{pid}] Q4 어법 — '{_wrg}' → '{_cor}' ({_gj.get('point')})")
                    break
            except Exception as _ge:
                print(f"[VAR][B][{pid}] ⚠ Q4 어법 생성 건너뜀: {_ge}")


            # ★ Q4/Q5 보기(bogi) 자동 생성: 답지 단어를 그대로 소문자·구두점제거하여 보기로 사용.
            #   모델이 만든 보기는 무시 → 누락/잉여(예: 'for')/중복오류를 원천 차단.
            def _bogi_from(text: str):
                """정답 문장을 보기 단어로 쪼갠다.

                ★ 문장 중간 구두점(쉼표·세미콜론·콜론)은 앞 단어에 붙여서 제시한다.
                  'When markets send signals, prices rise' → [..., 'signals,', 'prices', ...]
                  구두점을 떼버리면 학생이 어디에 찍어야 할지 알 수 없어 채점이 갈린다.
                  붙여주면 그 자리가 곧 단서가 되고, 배열 결과가 원문과 글자까지 일치한다.
                  단, 문장 끝 마침표·물음표는 뗀다 — 끝은 자명하고, 붙이면 마지막 단어가
                  어느 것인지 미리 알려주는 꼴이 된다."""
                t = str(text or "").strip()
                t = re.sub(r'[.!?]+\s*$', '', t)              # 문장 끝 종결부호만 제거
                t = re.sub(r'(?<=\d),(?=\d)', '\u0001', t)   # 100,000 보호
                t = re.sub(r'\b([A-Za-z](?:\.[A-Za-z])+)\.?',
                           lambda m: m.group(0).replace('.', '\u0002'), t)  # U.S. 보호
                toks = t.split()                              # 중간 구두점은 단어에 붙은 채로
                out = []
                for w in toks:
                    w = w.replace('\u0001', ',').replace('\u0002', '.')
                    w = w.strip('()')                         # 괄호만 떼어냄
                    if w:
                        out.append(w)
                return out
            try:
                # (_s161) Q4 는 어법 오류 찾기라 보기가 없다. Q5 만 만든다.
                # Q5: topic_writing_answer
                q5 = _bogi_from(data.get("topic_writing_answer", ""))
                if q5:
                    data["topic_writing_bogi"] = q5
            except Exception:
                pass

            # 마지막 시도면 strict=False (검증 풀어서라도 받아들임)
            is_last = (attempt == MAX_RETRIES_B)

            # ★★ (버) 객관식 정답 위치 셔플 (B: 주제 Q2 / 요약빈칸 Q3) — 정답이 ①에 쏠리던 문제 교정.
            #   삽입(position_correct)은 위치형이라 손대지 않는다.
            for _tag, _ok, _ck in (("topicB", "topic_options", "topic_correct"),
                                   ("summaryB", "summary_options", "summary_correct")):
                if isinstance(data.get(_ok), list) and isinstance(data.get(_ck), int):
                    data[_ok], data[_ck] = _shuffle_choices(
                        data[_ok], data[_ck], _choice_seed(pid, _tag, data.get(_ok)))

            # (_s161) Q3·Q4 요약문 겹침 검사를 걷어냈다.
            #   Q4 가 요약영작에서 어법 오류 찾기로 바뀌어 겹칠 수가 없다.
            #   옛 검사는 같은 논지를 다르게 써도 개념어가 겹쳐 계속 걸렸고,
            #   실측(Further Reading)에서 재시도 4회를 태우고 지문을 통째로 날렸다.

            # ★★ B Q3 복수정답 방지 (_s99) — 자가검증을 실제로 했는지 확인한다.
            #   프롬프트만으론 세 번 연속 복수정답이 새어나갔다
            #   (disclosure/revelation/presentation × engagement/curiosity/attention).
            #   유의어 판정은 코드가 못 하지만, '다섯 행을 문장으로 써 봤는가'는 확인할 수 있다.
            #   써 보게 만드는 것만으로도 스스로 걸러낸다 — Q5 에서 사유를 돌려주니
            #   성공률이 올랐던 것과 같은 원리다.
            try:
                _chk = data.get("summary_check")
                _sc = data.get("summary_correct")
                # ★★ _s125 — summary_design 을 쓰면 summary_check 를 요구하지 않는다.
                #   _s123 에서 출력 형식을 역할 칸(summary_design)으로 바꿔 LLM 이
                #   summary_check 를 아예 안 낸다. 그런데 코드가 계속 "5개 채워라"를
                #   요구해 매 시도마다 '자가검증 누락'으로 재시도가 소진됐다
                #   (실측: A 1개 + B 2개 누락, 3문항만 생성).
                #   design 방식에서는 배치가 구조로 보장되고, 판정은 풀이 검증(_s121)이 한다.
                _use_design = isinstance(data.get("summary_design"), dict)
                if _use_design:
                    _chk = []          # design 방식은 자가검증표를 안 낸다
                if (not _use_design) and (not isinstance(_chk, list) or len(_chk) != 5):
                    if not is_last:
                        last_errors = [
                            f"[{pid}] [유형B] Q3 summary_check 5개를 채우지 않음 — "
                            f"다섯 선지를 각각 요약문에 넣어 완성 문장으로 적고, "
                            f"오답이면 어느 칸이 왜 틀렸는지 쓸 것. "
                            f"머리로만 판단하면 복수정답이 그대로 나간다."]
                        print(f"[VAR][B][{pid}] Q3 자가검증 누락 → 재시도")
                        continue
                if True:
                    # ★★ 역할 배치 확인 (_s120) — 순수 문자열 세기라 오탐이 없다.
                    #   design 방식(_chk 가 빈 리스트)에서는 건너뛴다 — 배치가 구조로 보장된다.
                    #   설계: [정답]1 [유의어-A]2 [유의어-B]1 [둘다틀림]1
                    #   실측 실패: (A) 다섯 개를 전부 유의어로 넣어(demonstrate/suggest/
                    #   reveal/indicate/reflect) (A)로는 아무것도 못 걸렀다.
                    _roles = {}
                    for _t in _chk:
                        _m = re.match(r"\s*\[([^\]]{1,12})\]", str(_t))
                        if _m:
                            _roles[_m.group(1).strip()] = _roles.get(_m.group(1).strip(), 0) + 1
                    _want = {"정답": 1, "유의어-A": 2, "유의어-B": 1, "둘다틀림": 1}
                    if _roles and _roles != _want and not is_last:
                        last_errors = [
                            f"[{pid}] [유형B] Q3 선지 역할 배치가 어긋남 — "
                            f"지금 {_roles}, 있어야 할 것 {_want}. "
                            f"(A) 유의어는 정확히 2개, (B) 유의어는 정확히 1개다. "
                            f"유의어를 넣은 행은 반대쪽 칸을 반드시 틀리게 채울 것."]
                        print(f"[VAR][B][{pid}] Q3 역할 배치 어긋남 {_roles} → 재시도")
                        continue

                    # ★★★ 별도 호출로 실제로 풀려 본다 (_s121).
                    #   위 검사들(summary_check, 역할 라벨)은 만든 LLM 에게 되묻는
                    #   방식이라 대충 채우면 통과한다 — 낸 사람에게 "복수정답 아니죠?"
                    #   하고 묻는 셈이다. 여기서는 문항만 떼어 **다른 호출로 풀린다.**
                    #   만든 맥락을 모르므로 학생 입장에 가깝다.
                    try:
                        _so = data.get("summary_options")
                        _st2 = str(data.get("summary_template") or "")
                        if (isinstance(_so, list) and len(_so) == 5 and _st2
                                and isinstance(_sc, int)):
                            _sv = extract_json_from_response(call_claude(
                                SOLVE_SYS, build_solve_prompt(en_text, _st2, _so),
                                max_tokens=1600))
                            _valid = _sv.get("valid")
                            if isinstance(_valid, list) and _valid:
                                _v0 = [int(x) for x in _valid
                                       if str(x).strip().isdigit() and 1 <= int(x) <= 5]
                                if len(_v0) > 1:
                                    _extra = [n for n in _v0 if n != _sc + 1]
                                    _why = ""
                                    for _po in (_sv.get("per_option") or []):
                                        if _po.get("n") in _extra:
                                            _why = str(_po.get("why", ""))[:90]
                                            break
                                    last_errors = [
                                        f"[{pid}] [유형B] Q3 복수정답 — 이 문항을 따로 풀렸더니 "
                                        f"{_v0} 번이 전부 성립했다(정답은 {_sc+1}번). "
                                        f"{_extra}번의 한쪽 칸을 명백히 틀리게 갈아라. "
                                        + (f"풀이 근거: {_why}" if _why else "")]
                                    print(f"[VAR][B][{pid}] Q3 복수정답(풀이 검증) "
                                          f"{_v0} → 재시도")
                                    if is_last:
                                        raise ValueError(last_errors[0])
                                    continue
                                if _v0 and _v0[0] != _sc + 1:
                                    last_errors = [
                                        f"[{pid}] [유형B] Q3 정답 불일치 — 따로 풀렸더니 "
                                        f"{_v0[0]}번이 답이라 한다(표시된 정답은 {_sc+1}번). "
                                        f"요약문과 선지를 다시 맞춰라."]
                                    print(f"[VAR][B][{pid}] Q3 정답 불일치(풀이 검증) "
                                          f"{_v0[0]} vs {_sc+1} → 재시도")
                                    if not is_last:
                                        continue
                    except ValueError:
                        raise
                    except Exception as _se:
                        print(f"[VAR][B][{pid}] Q3 풀이 검증 예외({_se}) — 건너뜀")

                    # 정답 행 외에 '정답/성립/맞' 이라고 적힌 행이 있으면 복수정답이다
                    _alive = [k for k, t in enumerate(_chk)
                              if k != _sc and re.search(r"정답|성립|맞다|가능", str(t))
                              and not re.search(r"안 |못 |아니|틀렸|탈락|불가", str(t))]
                    if _alive:
                        # ★★ 복수정답은 관대 모드에서도 안 봐준다 (_s120).
                        #   학생이 이의제기하는 문제라 '일단 내보내기'가 성립하지 않는다.
                        #   실측: 3회 재시도가 제목 형식·절대어 같은 다른 사유로 소진돼
                        #   복수정답인 채로 관대 모드 통과했다(B 3/3, 감지만 13건).
                        #   마지막 시도에서도 걸리면 그 지문 B 는 만들지 않는다.
                        last_errors = [
                            f"[{pid}] [유형B] Q3 복수정답 — {[k+1 for k in _alive]}번도 성립한다고 "
                            f"스스로 적었다. 그 행의 한쪽 칸을 갈아라. "
                            f"유의어는 (A)(B) 중 한쪽에만 두고, 그 행의 반대쪽 칸은 반드시 틀리게 채울 것."]
                        if is_last:
                            print(f"[VAR][B][{pid}] ⚠ Q3 복수정답 {[k+1 for k in _alive]} — "
                                  f"마지막 시도라 관대 모드로 가야 하지만, 복수정답은 통과시키지 않는다")
                            raise ValueError(last_errors[0])
                        print(f"[VAR][B][{pid}] Q3 복수정답 감지 {[k+1 for k in _alive]} → 재시도")
                        continue
            except Exception as _ce:
                print(f"[VAR][B][{pid}] Q3 자가검증 확인 예외({_ce})")

            # ★ 답지용 한글 해석 + 요약문 영문 (_s106)
            try:
                _tc = data.get("topic_correct")
                _to = data.get("topic_options")
                if (isinstance(_to, list) and isinstance(_tc, int)
                        and 0 <= _tc < len(_to) and not data.get("title_answer_kr")):
                    data["title_answer_kr"] = _ensure_kr(str(_to[_tc]))

                # Q3 요약문 — 정답을 채운 완성 영문 (답지에 한글 해석 위에 넣는다)
                _st = str(data.get("summary_template") or "")
                _so = data.get("summary_options")
                _sc = data.get("summary_correct")
                if (_st and isinstance(_so, list) and isinstance(_sc, int)
                        and 0 <= _sc < len(_so) and len(_so[_sc]) >= 2):
                    data["summary_template_en"] = (
                        _st.replace("(A)", str(_so[_sc][0]), 1)
                           .replace("(B)", str(_so[_sc][1]), 1))

                # (_s161) Q4 요약문 영문은 없앴다 — Q4 가 어법 오류 찾기로 바뀌었다.
            except Exception as _ke:
                print(f"[VAR][B][{pid}] 답지 보강 예외({_ke})")

            # ★ 절대어 코드 검사는 뺐다 (_s133) — 위 A 와 같은 이유.
            #   B 는 검사가 일곱 개라 재시도 한 자리가 아깝다. 관용구 오탐으로
            #   그 자리를 까먹으면 정작 Q3 복수정답을 못 고친다.

            errors = validate_b(data, en_text, pid, strict=not is_last, a_data=_a_data)
            if not errors:
                # ★ Q2 제목 정답 자리를 강 단위로 돌린다 (_s136).
                #   A Q1 과 다른 salt 를 써서 같은 지문에서 A·B 가 같은 번호에
                #   몰리지 않게 한다.
                try:
                    _to2, _tc2 = shuffle_correct_position(
                        data.get("topic_options"), data.get("topic_correct"),
                        book, unit, pid, "titlepos")
                    if _tc2 != data.get("topic_correct"):
                        data["topic_options"], data["topic_correct"] = _to2, _tc2
                        print(f"[VAR][B][{pid}] Q2 제목 정답 자리 → {_tc2 + 1}번")
                except Exception as _pe:
                    print(f"[VAR][B][{pid}] 제목 자리 순환 예외({_pe})")

                save_cached(cache_key, "variation_b", data)
                mode_str = "관대 모드" if is_last else "엄격 모드"
                print(f"[VAR][B][{pid}] 생성 완료 (시도 {attempt}, {mode_str})")
                return data
            last_errors = errors
            has_critical = any("[CRITICAL]" in e for e in errors)
            # 마지막 시도이고 경미한 오류뿐이면 fallback용으로 보관 (치명적이면 거부)
            if is_last and data and not has_critical:
                last_data = data
                print(f"[VAR][B][{pid}] 마지막 시도 실패했지만 경미한 오류뿐 → 데이터 보관: {len(errors)}건")
            elif is_last and has_critical:
                print(f"[VAR][B][{pid}] 마지막 시도에 치명적 오류 → fallback 거부, 이 항목 생략")
            print(f"[VAR][B][{pid}] 시도 {attempt} 실패 ({len(errors)}건):")
            for err in errors[:5]:
                print(f"    - {err[:200]}")
        except Exception as e:
            traceback.print_exc()
            last_errors = [f"예외: {e}"]

    # ★ 5회 모두 실패해도 마지막 데이터가 있으면 그거라도 사용 (불완전한 B라도 없는 것보단 나음)
    if last_data is not None:
        # 필수 필드만 있으면 저장하고 반환
        required_minimum = ["given_sentence", "passage_with_marks", "grammar_q4",
                            "topic_writing_answer", "summary_options"]
        if all(k in last_data for k in required_minimum):
            save_cached(cache_key, "variation_b", last_data)
            print(f"[VAR][B][{pid}] ⚠️ 검증 실패했으나 데이터 fallback으로 저장")
            return last_data

    raise RuntimeError(f"유형 B 생성 실패 ({MAX_RETRIES_B}회). 마지막 오류:\n" + "\n".join(last_errors[:5]))


# ============ Passage 조회 ============
def fetch_passage_text(book: str, unit: str, pid: str) -> Optional[str]:
    if not _supabase_enabled():
        return None
    try:
        url = f"{SB_URL}/rest/v1/passages"
        params = {
            "select": "passage_text",
            "book": f"eq.{book}",
            "unit": f"eq.{unit}",
            "pid": f"eq.{pid}",
            "limit": "1",
        }
        with httpx.Client(timeout=10.0) as client:
            r = client.get(url, headers=_sb_headers(), params=params)
            r.raise_for_status()
            rows = r.json()
            if rows and isinstance(rows, list):
                return rows[0].get("passage_text", "")
    except Exception as e:
        print(f"[VAR] fetch_passage_text error: {e}")
    return None


# api.py 호환용 변수
sb_client = True if _supabase_enabled() else None
