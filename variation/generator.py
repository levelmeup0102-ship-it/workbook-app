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

from variation.prompts import SYSTEM_PROMPT_A, SYSTEM_PROMPT_B, extract_json_from_response, TOPIC_SENTENCE_SYS, build_topic_sentence_prompt, SUMMARY_SENTENCE_SYS, build_summary_sentence_prompt, CORE_BLANK_SYS, build_core_blank_prompt, TRANSLATE_SYS, build_translate_prompt, VOCAB_SYS, build_vocab_prompt, Q5_BLANK_SYS, build_q5_blank_prompt, INSERT_SYS, build_insert_prompt
from variation.validator import validate_a, validate_b, check_marker_positions, fill_boundary_dup, modal_no_verb
from variation.vocab_q3 import (normalize_llm_vocab, build_vocab_fallback,
                                validate_vocab, blank_token_spans,
                                shuffle_answer_position)


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


def _quote_ok(s: str) -> bool:
    """빈칸 후보 검사: 문장경계(.!?) 없고, 따옴표 '짝'이 갈리지 않음(균형).
    따옴표가 있어도 쌍이 맞으면 허용 — "good student" 통째는 OK, 여는 짝만 먹으면 제외.

    ★ 쉼표/세미콜론/콜론도 제외한다. 보기(bogi)는 구두점을 떼고 만들어지므로,
      정답 안에 쉼표가 있으면 학생이 그 쉼표를 복원할 근거가 없고 채점이 갈린다.
      (16강 03번 'empty, landlocked patch of desert,' 사례)"""
    if re.search(r'[.!?]', s):
        return False
    if re.search(r'[,;:]', s):
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
             "and","or","but","that","which","who","whose","whom","as","than","is","are",
             "was","were","be","been","being","this","these","those","their","her","his","its",
             "our","your","my","not","no","so","if","when","while","because",
             "they","we","i","he","she","it","you",
             "have","has","had","having","do","does","did",
             "may","might","can","could","will","would","shall","should","must"}
# 완화 모드에서 시작으로 절대 허용하지 않는 것 (기존 bad_start와 동일)
_BAD_START_MIN = {"and","or","but","that","which","who","of","to","for","than","as",
                  "is","are","was","were"}


def _clean_boundary_ok(phrase: str, full_text: str, strict: bool = True) -> bool:
    """빈칸 경계가 깔끔한지: 관사/전치사/접속사/관계사/주격대명사/조동사로 시작·끝나지 않고,
    괄호 짝이 맞고, 고유명사(대문자 단어)를 중간에서 쪼개지 않을 것.

    strict=True  : 시작·끝 모두 _BAD_EDGE로 판정 (깐깐)
    strict=False : 끝만 _BAD_EDGE, 시작은 _BAD_START_MIN만 (기존 동작)

    호출부가 strict로 먼저 훑고 후보가 0이면 완화 모드로 재시도하므로,
    깐깐하게 걸러도 항목이 통째로 누락되지 않는다."""
    bad_end = _BAD_EDGE
    bad_start = _BAD_EDGE if strict else _BAD_START_MIN
    ws = phrase.split()
    if not ws:
        return False
    if phrase.count("(") != phrase.count(")") or phrase.count("[") != phrase.count("]"):
        return False
    bare = lambda w: re.sub(r"[^A-Za-z'-]", "", w).lower()
    if bare(ws[0]) in bad_start or bare(ws[-1]) in bad_end:
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


def _cut_before_punct(sub: str, min_w: int = 4) -> str:
    """구절 안·끝의 구두점 직전까지 자른다. 남은 단어가 min_w 미만이면 빈 문자열.

    구두점은 빈칸 밖에 남는다 — 지문에 그대로 인쇄되고 학생은 그 앞부분만 배열한다.
      'dwindle and trail off, over the course'  →  'dwindle and trail off'
      'convince more readers for the whole story.' → 'convince more readers for the whole story'
    쉼표 위치를 학생이 알 수 없으므로 구두점을 정답에 포함시키면 채점이 갈린다."""
    t = str(sub or "").strip()
    if not t:
        return ""
    m = re.search(r'[.!?,;:]', t)
    if m:
        t = t[:m.start()].strip()
    t = t.rstrip('.,;:!?').strip()
    return t if len(t.split()) >= min_w else ""


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
                sub = _cut_before_punct(sub, min_w)
                if not sub:
                    continue
                if not _quote_ok(sub):
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


MAX_BLANK_WORDS = 9  # 영작 빈칸(A Q5 / B Q4) 최대 단어 수 — 초과 LLM 빈칸은 거부하고 코드가 5~7단어로 대체


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

    # ★ 코드는 자르지 않는다 — 조건을 지켜서 지목하는 것이 LLM의 일이다.
    #   뒤에서 잘라 맞추면 'very few of your readers would make it to your dramatic'처럼
    #   목적어(conclusion)가 빠진 어정쩡한 구절이 나온다. 조건 위반은 거부하고
    #   폴백(코드 픽)으로 넘긴다. 프롬프트가 STEP 4에서 같은 조건을 명시한다.
    ws = span.split()
    if len(ws) > 11:
        print(f"[VAR][A][{pid}] Q5 지목({lab}) {len(ws)}단어 — 11단어 초과, ends_with를 앞으로 당겨야 함")
        return None
    if len(ws) < 4:
        print(f"[VAR][A][{pid}] Q5 지목({lab}) {len(ws)}단어 — 4단어 미만")
        return None
    if re.search(r"[.!?,;:]", span):
        print(f"[VAR][A][{pid}] Q5 지목({lab}) 구두점 포함 — 구두점 앞에서 끊어야 함: '{span[:50]}'")
        return None
    return span


def validate_llm_q5_spans(paragraphs, span_a: str, span_b: str, pid: str = "?") -> Optional[dict]:
    """LLM이 고른 Q5 빈칸 두 구절을 검증한다. 통과하면 마킹된 paragraphs를 반환.

    LLM은 '논지가 착지하는 자리'를 판단하고(코드로는 불가), 코드는 복원 가능성만 본다.
    ★ 실패하면 어느 조건에서 걸렸는지 로그로 남긴다 — 로그에 '검증 실패'만 찍히면
      8단계 중 어디가 문제인지 알 수 없어 프롬프트를 고칠 수 없다.
    """
    def _fail(why):
        print(f"[VAR][A][{pid}] Q5 LLM 픽 거부: {why}")
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

    # 구두점이 붙어 오면 그 직전까지 잘라 쓴다 (구두점은 빈칸 밖)
    a = _cut_before_punct(a, 4) or a
    b = _cut_before_punct(b, 4) or b

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
        if n > 11:
            return _fail(f"({lab}) {n}단어 — 11단어 초과: '{v[:50]}'")
        if re.search(r"[.!?,;:]", v):
            return _fail(f"({lab}) 구두점 포함: '{v[:50]}'")
        if not _quote_ok(v):
            return _fail(f"({lab}) 따옴표/문장부호 문제: '{v[:50]}'")
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


def pick_a_q5_blanks(paragraphs, llm_a: str = "", llm_b: str = "", pid: str = "?") -> Optional[dict]:
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
        v = _cut_before_punct(v, 4) or v      # 구두점은 빈칸 밖에 남긴다
        if len(v.split()) < 4:
            return None
        if len(v.split()) > MAX_BLANK_WORDS:   # 너무 길면(예: 27단어) 거부 → 코드가 깔끔한 5~7단어로 대체
            return None
        if texts[idx].count(v) != 1:
            return None
        if not _quote_ok(v):
            return None
        if modal_no_verb(v):
            return None
        # ★ LLM 구절도 경계 검사를 거친다. 이게 없으면 LLM 픽이 우선 채택되면서
        #   _clean_boundary_ok가 통째로 우회돼, 코드 픽에만 걸린 경계 규칙이 무의미해진다.
        #   (거부되면 아래에서 코드 픽 후보가 대신 쓰이므로 항목 누락은 안 생긴다)
        if not _clean_boundary_ok(v, texts[idx], strict=True):
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
        lst += [c for c in cand[k] if c not in lst]
        pool.append((k, lst))

    for ai in range(len(pool)):
        ka, la = pool[ai]
        for bi in range(len(pool)):
            if bi == ai:
                continue
            kb, lb = pool[bi]
            for va in la[:6]:
                for vb in lb[:6]:
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
    """(더) 2단계: _q5_candidates와 동일. 깐깐한 경계로 먼저, 0개면 완화."""
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
    if n > 9:
        print(f"[VAR][B][{pid}] Q4 지목({lab}) {n}단어 — 9단어 초과, ends_with를 앞으로 당길 것")
        return None
    return span


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
        if len(v.split()) > MAX_BLANK_WORDS:
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
def make_cache_key(book: str, unit: str, pid: str, passage_text: str, variation_type: str) -> str:
    """캐시 키: {책}_{단원}_{번호}_{md5}_v{유형}"""
    txt_hash = hashlib.md5(passage_text.encode("utf-8")).hexdigest()[:8]
    book_safe = book[:15].replace(" ", "_").replace("/", "_")
    unit_safe = unit[:8].replace(" ", "_").replace("/", "_")
    pid_safe = pid[:6].replace(" ", "_").replace("/", "_")
    # _s83 = A Q3 어휘 답지 구조화(정답·근거·오답이유 한줄·나머지 선지 원문단어) + 해설 40자 제한. _s82 누적분 포함.
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
    return f"{book_safe}_{unit_safe}_{pid_safe}_{txt_hash}_var{variation_type}_s83"


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
    }

    with httpx.Client(timeout=120.0) as client:
        r = client.post(url, headers=headers, json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"Claude API 오류 {r.status_code}: {r.text[:500]}")
        data = r.json()
        content = data.get("content", [])
        for block in content:
            if block.get("type") == "text":
                return block.get("text", "")
        raise RuntimeError(f"Claude 응답에 텍스트 없음: {data}")


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
            return cached

    # 합치기 단계: 캐시에 없으면 생성하지 않고 None (재생성으로 인한 타임아웃 방지)
    if cache_only:
        print(f"[VAR][A][{pid}] 캐시 없음 — cache_only이므로 생략")
        return None

    last_errors = []
    last_data = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            user_msg = (
                f"Passage ID: {pid}\n\n"
                f"Original English passage:\n{en_text}\n\n"
                "Generate the variation problem (Type A). Return ONLY the JSON object."
            )
            if last_errors:
                user_msg += (
                    "\n\n# ⚠️ PREVIOUS ATTEMPT FAILED — FIX THESE ERRORS:\n"
                    + "\n".join(f"  ✗ {e}" for e in last_errors[:5])
                    + "\n\n# REMINDER OF CRITICAL CHECKS FOR TYPE A (sentence-order style):\n"
                    "  1. intro = the given lead (first 1-2 sentences, with <CORE_BLANK>). It must NOT reappear in (A)/(B)/(C).\n"
                    "  2. ★ (A)(B)(C) = exactly 3 paragraphs. Each must be a CONSECUTIVE run of whole sentences from the passage — "
                    "NEVER merge sentences that are far apart in the original. Cut ONLY at sentence boundaries.\n"
                    "  3. ★ RECONSTRUCTION TEST: intro + (A)(B)(C) reassembled in the order_correct sequence must EQUAL the original passage word-for-word "
                    "(no reordering inside a paragraph, no merging distant sentences, no omission, no duplication).\n"
                    "  4. order_correct = index 0-4 into FIXED choices (0=(A)-(C)-(B) 1=(B)-(A)-(C) 2=(B)-(C)-(A) 3=(C)-(A)-(B) 4=(C)-(B)-(A)); never (A)-(B)-(C).\n"
                    "  5. blank_A and blank_B are natural key phrases (~4-8 words each, do not pad), taken verbatim from INSIDE (A)/(B)/(C) (not intro), in different paragraphs.\n"
                    "  6. bogi must contain EVERY SINGLE WORD from blank_A + blank_B — count articles ('the','a','an') and prepositions carefully.\n"
                    "  7. core_blank_target must have AT LEAST 3 words; the Q3 correct option must be a PARAPHRASE of core_blank_target "
                    "(synonym or figurative rewording) that keeps the SAME grammatical structure as the original (clause stays a clause, noun phrase stays a noun phrase) so the sentence reads grammatically when the option fills the blank; NOT the original wording copied verbatim."
                )

            raw = call_claude(SYSTEM_PROMPT_A, user_msg)
            data = extract_json_from_response(raw)

            # ★★ 순서배열(Q2)을 코드가 원문에서 분할 — LLM 단락을 무시하고 원문 그대로 사용.
            #    원문 무손실이라 복원검증이 깨지지 않는다. LLM은 빈칸 구절만 고른다.
            ob = build_order_blocks_a(en_text, pid)
            print(f"[VAR][A][{pid}] DIAG 문장수={len(split_sentences(en_text))} "
                  f"ob={'None' if not ob else 'OK'} en_len={len(en_text)} en_head={en_text[:60]!r}")
            if ob:
                data["intro"] = ob["intro"]
                data["paragraphs"] = [list(p) for p in ob["paragraphs"]]
                data["order_correct"] = ob["order_correct"]

                # ★★ Q3 핵심빈칸 단독 재생성 (첫 문장만 주고 집중 — that절↔명사구 불일치 방지)
                #   intro(첫 문장)는 코드가 확정했으므로, 그 문장에서 핵심 구절+패러프레이즈 정답을
                #   따로 한 번 더 만든다. "원문이 절이면 정답도 절" 규칙으로 빈칸 문법 불일치를 막는다.
                try:
                    # intro에서 첫 한 문장만 추출 (마침표 기준)
                    _intro_txt = re.sub(r'\s+', ' ', str(data.get("intro", ""))).strip()
                    _sents_i = re.split(r'(?<=[.!?])\s+', _intro_txt) if _intro_txt else []
                    _first = ""
                    for _s in _sents_i:
                        if len(_s.split()) > 4:
                            _first = _s
                            break
                    if not _first and _sents_i:
                        _first = _sents_i[0]
                    if _first and len(_first.split()) >= 4:
                        _c_raw = call_claude(CORE_BLANK_SYS, build_core_blank_prompt(_first), max_tokens=700)
                        _c = extract_json_from_response(_c_raw)
                        _tg = (_c.get("core_blank_target") or "").strip()
                        _op = _c.get("core_blank_options")
                        _co = _c.get("core_blank_correct")
                        # target이 첫 문장 안에 실제로 있고 선지 5개가 정상일 때만 교체
                        if (_tg and _tg in _first and isinstance(_op, list) and len(_op) == 5
                                and isinstance(_co, int) and 0 <= _co <= 4):
                            data["core_blank_target"] = _tg
                            # ★ (버 하드닝) focused 재생성 결과를 '소스에서 즉시' 셔플한다.
                            #   바깥 루프(검증 직전)가 core_blank엔 실측상 안 먹어(같은 행 topic은 섞임)
                            #   원인 불명이라, 값이 정해지는 이 자리에서 직접 섞고 인덱스를 재계산해
                            #   ②쏠림을 원천 차단. 원소 추적이라 정답 의미 불변. 바깥 루프는 이걸 건너뜀.
                            _op, _co = _shuffle_choices(_op, _co, _choice_seed(pid, "coreA", _op))
                            data["core_blank_options"] = _op
                            data["core_blank_correct"] = _co
                            data["_core_shuffled"] = True
                            print(f"[VAR][A][{pid}] SHUF coreA(src): -> {_co}")
                            if _c.get("core_blank_explain"):
                                data["core_blank_explain"] = _c["core_blank_explain"]
                except Exception:
                    pass  # 실패하면 기존(한번에 만든) Q3 유지

                # ★ 따옴표·대시·구두점·하이픈·공백·대소문자 차이까지 흡수하는 마킹 함수 (core_blank / blank 공통)
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

                # Q3 핵심빈칸: LLM이 고른 구절을 intro(첫 문장)에서 찾아 마킹 (따옴표 흡수)
                tgt = (data.get("core_blank_target") or "").strip()
                if tgt:
                    new_intro, core_ok = _mark_phrase(data["intro"], tgt, "<CORE_BLANK>")
                    if core_ok:
                        data["intro"] = new_intro
                    data["_core_marked"] = core_ok

                # ★★ Q3가 어휘 유형으로 대체되면 intro 의 <CORE_BLANK> 는 아무 문항도
                #   참조하지 않는다. 그대로 두면 지문에 쓰이지 않는 빈칸이 남는다(실측:
                #   'you are taught to write up your results thus: "____, and so on until').
                #   원문 구절로 되돌려 intro 를 온전한 문장으로 만든다.
                #   ※ Q5 LLM 픽 try 블록 밖에서 처리한다 — 안에 두면 호출이 예외로 죽을 때
                #     빈칸이 그대로 남는다.
                if data.get("vocab_items") and "<CORE_BLANK>" in str(data.get("intro", "")):
                    _tgt_r = str(data.get("core_blank_target", "") or "").strip()
                    if _tgt_r:
                        data["intro"] = data["intro"].replace("<CORE_BLANK>", _tgt_r)
                        print(f"[VAR][A][{pid}] intro <CORE_BLANK> → 원문 복원")
                    else:
                        _ob_r = build_order_blocks_a(en_text, pid)
                        if _ob_r and _ob_r.get("intro"):
                            data["intro"] = _ob_r["intro"]
                            print(f"[VAR][A][{pid}] intro <CORE_BLANK> → 원문에서 재구성")
                        else:
                            print(f"[VAR][A][{pid}] ⚠ intro <CORE_BLANK> 복원 실패 (target 없음)")

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
                    # 새 형식: {para, starts_with, ends_with} 지목 → 코드가 원문에서 잘라냄
                    if isinstance(_ra, dict) or isinstance(_rb, dict):
                        _sa = _span_from_marks(_q5paras, _ra, pid, "A")
                        _sb = _span_from_marks(_q5paras, _rb, pid, "B")
                    else:                                   # 구 형식(문자열) 하위호환
                        _sa, _sb = str(_ra or ""), str(_rb or "")
                    _picked = validate_llm_q5_spans(_q5paras, _sa, _sb, pid) \
                        if (_sa and _sb) else None
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
                    _picked = pick_a_q5_blanks(data["paragraphs"], data.get("blank_A", ""), data.get("blank_B", ""), pid)
                if _picked:
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

                # ★★ Q3가 어휘 유형으로 대체되면 intro 의 <CORE_BLANK> 는 아무 문항도
                #   참조하지 않는다. 그대로 두면 지문에 쓰이지 않는 빈칸이 남는다(실측).
                #   원문 구절로 되돌려 intro 를 온전한 문장으로 만든다.
                #   ※ Q5 LLM 픽 try 블록 밖에서 처리한다 — 그 안에 두면 호출이 예외로
                #     죽었을 때 빈칸이 그대로 남는다.
                if "<CORE_BLANK>" in str(data.get("intro", "")):
                    _tgt = str(data.get("core_blank_target", "") or "").strip()
                    if _tgt:
                        data["intro"] = data["intro"].replace("<CORE_BLANK>", _tgt)
                        print(f"[VAR][A][{pid}] intro <CORE_BLANK> → 원문 복원")
                    else:
                        # target 이 없으면 되돌릴 말이 없다 — 원문 첫 문장으로 intro 를 재구성
                        _ob2 = build_order_blocks_a(en_text, pid)
                        if _ob2 and _ob2.get("intro"):
                            data["intro"] = _ob2["intro"]
                            print(f"[VAR][A][{pid}] intro <CORE_BLANK> → 원문에서 재구성")
                        else:
                            print(f"[VAR][A][{pid}] ⚠ intro <CORE_BLANK> 복원 실패 — target 없음")

                # ★★ Q3 어휘 유형 (수능 30번) — Q5 빈칸이 확정된 뒤에 호출한다.
                #   원문(paragraphs)은 안 건드리고 '자리(인덱스)'만 기록하므로
                #   Q2 순서 복원 검증은 지금 코드 그대로 돌아간다.
                #   밑줄은 Q5 빈칸이 차지한 토큰 범위를 피해서 잡는다.
                try:
                    _spans = blank_token_spans(data["paragraphs"])
                    _vraw = call_claude(
                        VOCAB_SYS,
                        build_vocab_prompt(data["paragraphs"],
                                           [data.get("blank_A", ""), data.get("blank_B", "")]),
                        max_tokens=1800)
                    _v = extract_json_from_response(_vraw)
                    _items = normalize_llm_vocab(_v.get("vocab_items"), data["paragraphs"], _spans)
                    if _items:
                        # ★ 정답 위치 분산 — 프롬프트만으론 매번 ③에 몰린다(실측 3/3)
                        _items = shuffle_answer_position(_items, pid)
                        data["vocab_items"] = _items
                        if _v.get("vocab_explain"):
                            data["vocab_explain"] = _v["vocab_explain"]
                        _ans = next((i for i in _items if i.get("is_answer")), None)
                        print(f"[VAR][A][{pid}] VOCAB ok — 정답 {_ans['n'] if _ans else '?'}번 "
                              f"'{_ans['original'] if _ans else ''}' → '{_ans['shown'] if _ans else ''}'")
                    else:
                        raise ValueError("normalize 실패")
                except Exception as _ve:
                    _fb = build_vocab_fallback(data["paragraphs"], blank_token_spans(data["paragraphs"]))
                    if _fb:
                        data["vocab_items"] = shuffle_answer_position(_fb, pid)
                        print(f"[VAR][A][{pid}] VOCAB 폴백 사용 ({_ve})")
                    else:
                        print(f"[VAR][A][{pid}] VOCAB 생성 실패 ({_ve}) — Q3는 핵심빈칸으로 유지")

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
                    ob2 = build_order_blocks_a(en_text, pid)
                    if ob2:
                        data["intro"] = ob2["intro"]
                        data["paragraphs"] = [list(p) for p in ob2["paragraphs"]]
                        data["order_correct"] = ob2["order_correct"]
                        _pk2 = pick_a_q5_blanks(data["paragraphs"], data.get("blank_A", ""), data.get("blank_B", ""), pid)
                        if _pk2:
                            data["paragraphs"] = _pk2["paragraphs"]
                            data["blank_A"] = _pk2["blank_A"]
                            data["blank_B"] = _pk2["blank_B"]
                        _tg2 = (data.get("core_blank_target") or "").strip()
                        if _tg2 and "<CORE_BLANK>" not in data["intro"] and _tg2 in data["intro"]:
                            data["intro"] = data["intro"].replace(_tg2, "<CORE_BLANK>", 1)
                        print(f"[VAR][A][{pid}] intro 중복 감지 → 코드가 순서 강제 재분할(안전장치)")
            except Exception:
                pass

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
            #   순서(order_correct)는 위치형이라 손대지 않는다. core_blank가 소스에서 이미 셔플됐으면
            #   건너뛴다(중복 셔플 방지). before/after를 로그로 남겨 다음 생성 때 동작을 확인.
            for _tag, _ok, _ck in (("topicA", "topic_options", "topic_correct"),
                                   ("coreA", "core_blank_options", "core_blank_correct")):
                if _tag == "coreA" and data.get("_core_shuffled"):
                    print(f"[VAR][A][{pid}] SHUF coreA(loop): 소스에서 이미 셔플 -> {data.get(_ck)}")
                    continue
                if isinstance(data.get(_ok), list) and isinstance(data.get(_ck), int):
                    _before = data[_ck]
                    data[_ok], data[_ck] = _shuffle_choices(
                        data[_ok], data[_ck], _choice_seed(pid, _tag, data.get(_ok)))
                    print(f"[VAR][A][{pid}] SHUF {_tag}(loop): {_before} -> {data[_ck]}")
                else:
                    print(f"[VAR][A][{pid}] SHUF {_tag}(loop): SKIP "
                          f"(opt={type(data.get(_ok)).__name__}, cor={type(data.get(_ck)).__name__})")

            # ★ 렌더링용 단락 — 원본 paragraphs는 그대로 두고 밑줄만 얹은 사본을 만든다.
            #   검증은 원본으로 하고 화면에는 이걸 쓴다. (Q2 복원 검증 무영향)
            if data.get("vocab_items"):
                try:
                    from variation.vocab_q3 import apply_vocab_items
                    data["paragraphs_render"] = apply_vocab_items(
                        data["paragraphs"], data["vocab_items"])
                except Exception:
                    data["paragraphs_render"] = None

            errors = validate_a(data, en_text, pid, lenient=is_last)
            if not errors:
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
    for attempt in range(1, MAX_RETRIES + 1):
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
                user_msg += (
                    "\n\n# ⚠️ PREVIOUS ATTEMPT FAILED — FIX THESE ERRORS:\n"
                    + "\n".join(f"  ✗ {e}" for e in last_errors[:5])
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

            raw = call_claude(SYSTEM_PROMPT_B, user_msg)
            data = extract_json_from_response(raw)

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
            try:
                _s_raw = call_claude(SUMMARY_SENTENCE_SYS, build_summary_sentence_prompt(en_text), max_tokens=600)
                _s = extract_json_from_response(_s_raw)
                _fs = (_s.get("full_summary") or "").strip()
                _ba = (_s.get("blank_A") or "").strip()
                _bb = (_s.get("blank_B") or "").strip()
                # 셋 다 있고 blank_A/B가 full_summary 안에 실제로 들어있을 때만 교체
                if _fs and _ba and _bb and _ba in _fs and _bb in _fs and _ba != _bb:
                    data["full_summary"] = _fs
                    data["blank_A"] = _ba
                    data["blank_B"] = _bb
                    # ★ 요약문 해석도 같은 문장의 번역으로 함께 교체 (topic과 동일 이유).
                    data["blank_summary_template_kr"] = _ensure_kr(_fs, _s.get("full_summary_kr"))
            except Exception:
                pass  # 실패하면 기존(한번에 만든) 요약문 유지

            # ★★ Q4 빈칸을 코드가 요약문에서 직접 골라 4단어+ 보장 (A Q5와 같은 철학).
            #   LLM이 짧게/비verbatim으로 뽑아도 코드가 깨끗한 4단어 구절로 대체 → 누락 차단.
            try:
                _fulls = data.get("full_summary") or data.get("summary_full") or ""
                _ra4, _rb4 = data.get("blank_A"), data.get("blank_B")
                # 새 형식: {starts_with, ends_with} 지목 → 코드가 요약문에서 잘라냄
                if isinstance(_ra4, dict) or isinstance(_rb4, dict):
                    _sa4 = _span_from_marks_summary(_fulls, _ra4, pid, "A")
                    _sb4 = _span_from_marks_summary(_fulls, _rb4, pid, "B")
                    if _sa4 and _sb4:
                        print(f"[VAR][B][{pid}] Q4 LLM 지목 — (A)'{_sa4}' (B)'{_sb4}'")
                    else:
                        print(f"[VAR][B][{pid}] Q4 지목 실패 → 코드 픽으로 폴백")
                    _sa4, _sb4 = _sa4 or "", _sb4 or ""
                else:                                    # 구 형식(문자열) 하위호환
                    _sa4, _sb4 = str(_ra4 or ""), str(_rb4 or "")
                _bp = pick_b_q4_blanks(_fulls, _sa4, _sb4)
                if _bp:
                    data["blank_A"] = _bp["blank_A"]
                    data["blank_B"] = _bp["blank_B"]
            except Exception as _q4e:
                print(f"[VAR][B][{pid}] Q4 빈칸 처리 예외({_q4e})")

            # ★★ 코드가 빈칸 뚫기 (우리가 정한 방식: 완성문장 먼저 → 코드가 뚫기)
            #   LLM이 준 full_summary(빈칸 없는 완성문장)에서 blank_A/blank_B를 찾아
            #   (A)/(B)로 코드가 직접 치환 → blank_summary_template 생성.
            #   이렇게 하면 "되넣으면 복원"이 코드로 보장 → 빈칸범위/중복/마킹 오류 원천 차단.
            def _punch_blanks(full, a, b):
                """full에서 a→(A), b→(B)로 치환. 대소문자/공백 차이 흡수. 성공 시 (template, True)."""
                import re as _re
                if not full or not a or not b:
                    return None, False
                def _find(hay, needle):
                    # 1) 그대로  2) 공백 정규화 후 토큰시퀀스 매칭
                    i = hay.find(needle)
                    if i >= 0:
                        return i, i + len(needle)
                    hn = _re.sub(r'\s+', ' ', hay)
                    nn = _re.sub(r'\s+', ' ', needle).strip()
                    j = hn.find(nn)
                    if j >= 0:
                        # 원본 인덱스 보정이 복잡하므로, 정규화 문자열에서 작업
                        return None
                    return None
                # 단순/정규화 치환을 정규화 평면에서 수행
                hn = _re.sub(r'\s+', ' ', full).strip()
                an = _re.sub(r'\s+', ' ', a).strip()
                bn = _re.sub(r'\s+', ' ', b).strip()
                if an not in hn or bn not in hn:
                    return None, False
                # A를 먼저 치환하되, B가 A의 부분문자열이면 충돌 → 더 긴 것부터
                first, fk, second, sk = (an, "(A)", bn, "(B)")
                if len(bn) > len(an):
                    first, fk, second, sk = (bn, "(B)", an, "(A)")
                t = hn.replace(first, fk, 1)
                if second not in t:  # 치환 후 두 번째 구절이 사라졌으면(겹침) 실패
                    return None, False
                t = t.replace(second, sk, 1)
                if "(A)" not in t or "(B)" not in t:
                    return None, False
                return t, True

            try:
                _full = data.get("full_summary") or data.get("summary_full") or ""
                _tmpl, _ok = _punch_blanks(_full, data.get("blank_A", ""), data.get("blank_B", ""))
                if _ok:
                    data["blank_summary_template"] = _tmpl  # 코드가 만든 빈칸 버전으로 덮어쓰기
            except Exception:
                pass

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
                # Q4: blank_A + blank_B
                q4 = _bogi_from(str(data.get("blank_A", "")) + " " + str(data.get("blank_B", "")))
                if q4:
                    data["blank_summary_bogi"] = q4
                # Q5: topic_writing_answer
                q5 = _bogi_from(data.get("topic_writing_answer", ""))
                if q5:
                    data["topic_writing_bogi"] = q5
            except Exception:
                pass

            # 마지막 시도면 strict=False (검증 풀어서라도 받아들임)
            is_last = (attempt == MAX_RETRIES)

            # ★★ (버) 객관식 정답 위치 셔플 (B: 주제 Q2 / 요약빈칸 Q3) — 정답이 ①에 쏠리던 문제 교정.
            #   삽입(position_correct)은 위치형이라 손대지 않는다.
            for _tag, _ok, _ck in (("topicB", "topic_options", "topic_correct"),
                                   ("summaryB", "summary_options", "summary_correct")):
                if isinstance(data.get(_ok), list) and isinstance(data.get(_ck), int):
                    data[_ok], data[_ck] = _shuffle_choices(
                        data[_ok], data[_ck], _choice_seed(pid, _tag, data.get(_ok)))

            errors = validate_b(data, en_text, pid, strict=not is_last, a_data=_a_data)
            if not errors:
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
        required_minimum = ["given_sentence", "passage_with_marks", "blank_A", "blank_B",
                            "topic_writing_answer", "summary_options"]
        if all(k in last_data for k in required_minimum):
            save_cached(cache_key, "variation_b", last_data)
            print(f"[VAR][B][{pid}] ⚠️ 검증 실패했으나 데이터 fallback으로 저장")
            return last_data

    raise RuntimeError(f"유형 B 생성 실패 ({MAX_RETRIES}회). 마지막 오류:\n" + "\n".join(last_errors[:5]))


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
