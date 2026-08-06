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

from variation.prompts import SYSTEM_PROMPT_A, SYSTEM_PROMPT_B, extract_json_from_response, TOPIC_SENTENCE_SYS, build_topic_sentence_prompt, SUMMARY_SENTENCE_SYS, build_summary_sentence_prompt, TRANSLATE_SYS, build_translate_prompt, VOCAB_SYS, build_vocab_prompt, Q5_BLANK_SYS, build_q5_blank_prompt, INSERT_SYS, build_insert_prompt
from variation.validator import validate_a, validate_b, check_marker_positions, fill_boundary_dup, modal_no_verb
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
    if re.search(r'[.!?]', s):
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
             "like","there","then"}
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
    bad_end = _BAD_EDGE - {"they", "we", "he", "she", "it", "you", "i"}
    # ★ 하이픈 복합어로 끝나면 잘린 것이다 (_s105).
    #   'twenty-second' 'well-known' 'long-term' 은 뒤에 꾸밀 명사가 따라온다.
    #   실측: 'straight line of the twenty-second' — twenty-second parallel 을 쪼갰다.
    #   ※ 하이픈이 있어도 홀로 쓰는 말은 예외로 둔다(self-esteem, one-of-a-kind 등 명사).
    _HYPHEN_OK_END = {"one-of-a-kind", "self-esteem", "well-being", "know-how",
                      "trade-off", "by-product", "side-effect", "vice-versa"}
    bad_start = (_BAD_START_MIN | {"they", "we", "he", "she", "it", "you", "i",
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


def _cut_before_punct(sub: str, min_w: int = 4, sentence_only: bool = False) -> str:
    """구절 안·끝의 구두점 직전까지 자른다. 남은 단어가 min_w 미만이면 빈 문자열.

    구두점은 빈칸 밖에 남는다 — 지문에 그대로 인쇄되고 학생은 그 앞부분만 배열한다.
      'dwindle and trail off, over the course'  →  'dwindle and trail off'
      'convince more readers for the whole story.' → 'convince more readers for the whole story'
    쉼표 위치를 학생이 알 수 없으므로 구두점을 정답에 포함시키면 채점이 갈린다."""
    t = str(sub or "").strip()
    if not t:
        return ""
    # ★ sentence_only=True 면 문장 경계(. ! ?)에서만 자른다 (_s94, A Q5 전용).
    #   쉼표는 빈칸 안에 남고 보기에 'original,' 처럼 붙어 제시된다.
    m = re.search(r'[.!?]' if sentence_only else r'[.!?,;:]', t)
    if m:
        t = t[:m.start()].strip()
    t = t.rstrip('.,;:!?').strip()          # 끝 구두점은 언제나 떼어낸다
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
                sub = _cut_before_punct(sub, min_w, sentence_only=True)
                if not sub:
                    continue
                if not _quote_ok(sub, allow_comma=True):
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
    """선지에 절대어가 있으면 그 단어, 없으면 빈 문자열. 대소문자 무시."""
    t = " " + re.sub(r"[^A-Za-z ]", " ", str(opt or "").lower()) + " "
    t = re.sub(r"\s+", " ", t)
    for w in _ABSOLUTE_WORDS:
        if f" {w} " in t:
            return w
    return ""


def check_absolute_words(options, correct_idx, label, pid="?"):
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
        if not _clean_boundary_ok(span, hn, strict=False):
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
    # _s109 = Q3어휘 후보에서 방향 없는 명사를 더 막는다. 'audience' 가 -ence 어미라
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
    return f"{book_safe}_{unit_safe}_{pid_safe}_{txt_hash}_var{variation_type}_s109"


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
            ob = build_order_blocks_a(en_text, pid)
            print(f"[VAR][A][{pid}] DIAG 문장수={len(split_sentences(en_text))} "
                  f"ob={'None' if not ob else 'OK'} en_len={len(en_text)} en_head={en_text[:60]!r}")
            if ob:
                data["intro"] = ob["intro"]
                data["paragraphs"] = [list(p) for p in ob["paragraphs"]]
                data["order_correct"] = ob["order_correct"]

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
                        want_n=_want_n)
                    # ★ 재시도에는 앞선 실패 사유를 그대로 붙인다 (_s97).
                    #   "다시 만들어라"만 하면 같은 실수를 반복한다 — Q5 에서 사유를
                    #   돌려주니 성공률이 올랐던 것과 같은 처방이다.
                    if _attempt and _vfail:
                        _msg += ("\n\n[이전 시도가 거부됐다. 사유는 이것이다]\n"
                                 + "\n".join("  - " + x for x in _vfail[-5:])
                                 + "\n\n다시 만들 때 이렇게 하라.\n"
                                   "· original 은 지문에 인쇄된 글자를 그대로 옮겨 적어라.\n"
                                   "  구두점·대소문자까지 포함해서다('uncomfortable.' 'Similarly').\n"
                                   "· shown 은 original 과 같은 형태여야 한다.\n"
                                   "  지문이 'depends'(3인칭 단수)면 'relies' 다. 'rely' 를 쓰면\n"
                                   "  본문이 'the brain rely on ...' 이 되어 수일치가 깨진다.\n"
                                   "· para 와 idx 는 그 단락 안에서 0부터 공백으로 센 위치다.\n"
                                   "· 다섯 자리는 서로 다른 단어여야 한다 — 굴절형도 같은 단어다\n"
                                   "  ('depends' 와 'depend' 를 둘 다 밑줄 치지 마라).\n"
                                   "· ★ [[[여기는 Q5 빈칸 …]]] 안의 단어는 고르지 마라. 이미 사라졌다.\n"
                                   "· ★ 접속부사·담화표지는 밑줄 대상이 아니다\n"
                                   "  (Similarly, Conversely, However, Therefore, Moreover …).\n"
                                   "  논리 흐름 표지지 문맥 판단 대상이 아니다.\n"
                                   "· ★ 정답 자리는 반대말을 댈 수 있는 단어여야 한다.\n"
                                   "  방향 없는 말(tasks, time, readers, story)은 못 쓴다.\n"
                                   "· 선지는 반드시 한 단어. 'most extensive' 같은 두 단어는 안 된다.\n"
                                   "· 부정 접두사만 붙이거나 떼서 만들지 마라\n"
                                   "  ('inhabitable'↔'uninhabitable'). 어근이 다른 말을 쓸 것.")
                    _vraw = call_claude(VOCAB_SYS, _msg, max_tokens=1800)
                    _v = extract_json_from_response(_vraw)
                    #   ★ 첫 시도만 -ing/-ed 형태까지 본다(_s100). 재시도에서는
                    #     -s 불일치만 막는다 — 'vast'→'overwhelming' 같은 정상 치환을
                    #     형태소로 못 가려 지문이 통째로 누락됐다.
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
                    return _items, _v

                _vok = False
                for _va in range(2):
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
                        print(f"[VAR][A][{pid}] Q3어휘 시도 {_va + 1} 실패 — {_ve}")
                if not _vok:
                    # ★ 4문항 시험지는 만들지 않는다 (_s97).
                    #   번호가 1·2·4·5 로 건너뛰면 학생에게 못 나간다.
                    #   여기서 raise 하면 바깥 재시도 루프가 A 를 처음부터 다시 만든다.
                    data.pop("vocab_items", None)
                    data.pop("vocab_explain", None)
                    raise ValueError(
                        "Q3어휘 2회 실패 — "
                        + (_vfail[-1] if _vfail else "사유 미기록"))

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

            # ★ 절대어 오답 차단 (_s108) — 코드 검사가 없어 새어 나갔다.
            _abs = check_absolute_words(data.get("topic_options"),
                                        data.get("topic_correct"), "Q1 주제", pid)
            if _abs and not is_last:
                last_errors = _abs
                print(f"[VAR][A][{pid}] 주제 오답에 절대어 → 재시도")
                continue

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

            _q3_sum = _filled_q3()
            try:
                for _sa in range(2):
                    _msg = build_summary_sentence_prompt(en_text, q3_summary=_q3_sum)
                    if _sa:
                        _msg += ("\n\n[이전 시도가 Q3 요약문과 겹쳤다]\n"
                                 "  겹친 어구: " + ", ".join(sorted(_ov)[:4]) + "\n"
                                 "논리 구조·주어·축이 되는 문장을 전부 바꿔서 다시 써라.")
                    _s_raw = call_claude(SUMMARY_SENTENCE_SYS, _msg, max_tokens=600)
                    _s = extract_json_from_response(_s_raw)
                    _fs = (_s.get("full_summary") or "").strip()
                    _ba = (_s.get("blank_A") or "").strip()
                    _bb = (_s.get("blank_B") or "").strip()
                    if not (_fs and _ba and _bb and _ba in _fs and _bb in _fs and _ba != _bb):
                        break                      # 형식 오류 — 기존(한번에 만든) 요약문 유지
                    _ov = _summary_overlap(_fs, _q3_sum)
                    if _ov and _sa == 0:
                        print(f"[VAR][B][{pid}] Q4 요약문이 Q3 와 겹침 {sorted(_ov)[:3]} → 재생성")
                        continue
                    if _ov:
                        print(f"[VAR][B][{pid}] ⚠ Q4 요약문이 Q3 와 여전히 겹침 {sorted(_ov)[:3]}")
                    data["full_summary"] = _fs
                    data["blank_A"] = _ba
                    data["blank_B"] = _bb
                    # ★ 요약문 해석도 같은 문장의 번역으로 함께 교체 (topic과 동일 이유).
                    data["blank_summary_template_kr"] = _ensure_kr(_fs, _s.get("full_summary_kr"))
                    break
            except Exception as _se:
                print(f"[VAR][B][{pid}] Q4 요약문 재생성 예외({_se})")

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

            # ★★ Q3·Q4 요약문 겹침 최종 확인 (_s99)
            #   재생성 2회가 다 실패해도 여기서 막는다. 같은 문장이면 Q3 를 푼 학생이
            #   Q4 를 그냥 베껴 쓴다 — 두 문항이 사실상 하나가 된다.
            try:
                # ★ 양쪽 다 '정답을 채운 완성 문장'으로 비교한다 (_s100).
                #   Q4 는 full_summary 가 곧 완성 문장이고, Q3 는 정답을 끼워 넣어 만든다.
                _q3t = _filled_q3()
                _q4t = str(data.get("full_summary") or "")
                if not _q4t:
                    _q4t = str(data.get("blank_summary_template") or "")
                    _ba2, _bb2 = data.get("blank_A"), data.get("blank_B")
                    if _ba2 and _bb2:
                        _q4t = _q4t.replace("(A)", str(_ba2), 1).replace("(B)", str(_bb2), 1)
                    _q4t = _q4t.replace("(A)", " ").replace("(B)", " ")
                _ov2 = _summary_overlap(_q4t, _q3t)
                if _ov2 and not is_last:
                    last_errors = [
                        f"[{pid}] [유형B] Q3 요약문과 Q4 요약문이 겹침 {sorted(_ov2)[:4]} — "
                        f"두 문항은 별개다. Q3 를 푼 학생이 Q4 를 베껴 쓰게 된다. "
                        f"논리 구조(속성→결과 / 조건→귀결 / 대비)와 주어와 축이 되는 문장을 "
                        f"전부 다르게 잡아 Q4 요약문을 새로 쓸 것."]
                    print(f"[VAR][B][{pid}] Q3·Q4 요약문 겹침 {sorted(_ov2)[:3]} → 재시도")
                    continue
            except Exception as _oe:
                print(f"[VAR][B][{pid}] Q3·Q4 겹침 확인 예외({_oe})")

            # ★★ B Q3 복수정답 방지 (_s99) — 자가검증을 실제로 했는지 확인한다.
            #   프롬프트만으론 세 번 연속 복수정답이 새어나갔다
            #   (disclosure/revelation/presentation × engagement/curiosity/attention).
            #   유의어 판정은 코드가 못 하지만, '다섯 행을 문장으로 써 봤는가'는 확인할 수 있다.
            #   써 보게 만드는 것만으로도 스스로 걸러낸다 — Q5 에서 사유를 돌려주니
            #   성공률이 올랐던 것과 같은 원리다.
            try:
                _chk = data.get("summary_check")
                _sc = data.get("summary_correct")
                if not isinstance(_chk, list) or len(_chk) != 5:
                    if not is_last:
                        last_errors = [
                            f"[{pid}] [유형B] Q3 summary_check 5개를 채우지 않음 — "
                            f"다섯 선지를 각각 요약문에 넣어 완성 문장으로 적고, "
                            f"오답이면 어느 칸이 왜 틀렸는지 쓸 것. "
                            f"머리로만 판단하면 복수정답이 그대로 나간다."]
                        print(f"[VAR][B][{pid}] Q3 자가검증 누락 → 재시도")
                        continue
                else:
                    # 정답 행 외에 '정답/성립/맞' 이라고 적힌 행이 있으면 복수정답이다
                    _alive = [k for k, t in enumerate(_chk)
                              if k != _sc and re.search(r"정답|성립|맞다|가능", str(t))
                              and not re.search(r"안 |못 |아니|틀렸|탈락|불가", str(t))]
                    if _alive and not is_last:
                        last_errors = [
                            f"[{pid}] [유형B] Q3 복수정답 — {[k+1 for k in _alive]}번도 성립한다고 "
                            f"스스로 적었다. 그 행의 한쪽 칸을 갈아라. "
                            f"유의어는 (A)(B) 중 한쪽에만 두고, 그 행의 반대쪽 칸은 반드시 틀리게 채울 것."]
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

                # Q4 요약문 — 정답 구절을 채운 완성 영문
                _bt = str(data.get("blank_summary_template") or "")
                _ba, _bb = data.get("blank_A"), data.get("blank_B")
                if _bt and _ba and _bb:
                    data["blank_summary_template_en"] = (
                        _bt.replace("(A)", str(_ba), 1).replace("(B)", str(_bb), 1))
                elif data.get("full_summary"):
                    data["blank_summary_template_en"] = str(data["full_summary"])
            except Exception as _ke:
                print(f"[VAR][B][{pid}] 답지 보강 예외({_ke})")

            # ★ 절대어 오답 차단 (_s108)
            #   실측: 'Why One Sensory Pathway Is Never Enough' — 대문자라 프롬프트 규칙을
            #   빠져나갔고 코드 검사는 아예 없었다.
            _abs = check_absolute_words(data.get("topic_options"),
                                        data.get("topic_correct"), "Q2 제목", pid)
            if _abs and not is_last:
                last_errors = _abs
                print(f"[VAR][B][{pid}] 제목 오답에 절대어 → 재시도")
                continue

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
