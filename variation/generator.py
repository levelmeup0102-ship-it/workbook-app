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

from variation.prompts import SYSTEM_PROMPT_A, SYSTEM_PROMPT_B, extract_json_from_response, TOPIC_SENTENCE_SYS, build_topic_sentence_prompt, SUMMARY_SENTENCE_SYS, build_summary_sentence_prompt, CORE_BLANK_SYS, build_core_blank_prompt
from variation.validator import validate_a, validate_b, check_marker_positions, fill_boundary_dup, modal_no_verb


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
    if len(sents) < 4:
        print(f"[VAR][A][{pid}] 문장 {len(sents)}개 — 순서배열(intro+3단락) 불가")
        return None

    # intro = 첫 1문장 (남은 문장이 3개 미만이면 순서배열 불가)
    intro_text = sents[0].strip()
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

def build_insert_blocks_b(en_text: str, pid: str = "?") -> Optional[dict]:
    """원문에서 문장 하나를 떼어 given_sentence로, 나머지에 코드가 마커를 박아
    삽입문제(Q1)를 무손실로 재구성한다. (순서배열 build_order_blocks_a와 같은 철학)
    validator.check_marker_positions를 통과하고 '정답 자리에 도로 넣으면 원문 복원'이
    성립하는 구성만 반환. 재구성 불가하면 None(→ 기존 LLM 결과 유지).
    LLM이 given/본문을 변형해 '어느 자리도 복원 안 되는' 항목을 코드가 살린다."""
    def _alnum_ib(t):
        return re.sub(r"[^a-z0-9]", "", str(t).lower())
    sents = split_sentences(en_text)
    m = len(sents)
    if m < 4:
        return None
    mid = m // 2
    order = sorted(range(1, m), key=lambda g: abs(g - mid))  # 가운데 문장부터 시도(첫 문장 제외)
    for g in order:
        given = sents[g]
        remaining = sents[:g] + sents[g + 1:]
        L = len(remaining)
        real_gap = g
        pool = list(range(1, L + 1))  # 각 갭은 앞에 최소 한 문장 → 문장경계 보장
        if real_gap not in pool or len(pool) < 3:
            continue
        target = 5 if len(pool) >= 5 else (4 if len(pool) >= 4 else 3)
        picks = {pool[0], pool[-1], real_gap}
        if target > len(picks):
            for i in range(target):
                ix = round(i * (len(pool) - 1) / (target - 1))
                picks.add(pool[ix])
        chosen = sorted(picks)
        protected = {pool[0], pool[-1], real_gap}
        while len(chosen) > 5:
            for c in chosen:
                if c not in protected:
                    chosen.remove(c)
                    break
            else:
                break
        chosen = sorted(chosen)
        rank = {gap: i + 1 for i, gap in enumerate(chosen)}  # gap → MARK번호(위치순)
        out = []
        for j in range(L + 1):
            if j in rank:
                out.append(f"<MARK{rank[j]}>")
            if j < L:
                out.append(remaining[j])
        pwm = " ".join(out)
        pos_correct = chosen.index(real_gap)
        pos_count = len(chosen)
        # 1) 정답 자리에 given 도로 넣으면 원문 복원?
        recon = pwm.replace(f"<MARK{pos_correct + 1}>", " " + given + " ")
        recon = re.sub(r"<MARK\d>", "", recon)
        if _alnum_ib(recon) != _alnum_ib(en_text):
            continue
        # 2) 배포 validator 마커 검사 통과?
        errs = check_marker_positions(pwm, pid, min_between=3,
                                      position_correct=pos_correct,
                                      position_count=pos_count, strict=True)
        if errs:
            continue
        return {"given_sentence": given, "passage_with_marks": pwm,
                "position_correct": pos_correct, "position_count": pos_count}
    return None


_Q5_MODALS = {"can", "will", "must", "should", "would", "could", "may", "might", "shall"}


def _q5_candidates(ptext: str, min_w: int = 5, max_w: int = 7) -> list:
    """단락에서 '문장 중간 연속 구절'(verbatim) 후보 생성. 가운데 우선.
    문장경계/따옴표 포함 제외, 조동사 시작 제외, 단락 내 유일 등장만."""
    spans = [(m.start(), m.end()) for m in re.finditer(r'\S+', ptext)]
    toks = [ptext[s:e] for s, e in spans]
    n = len(toks)
    cands = []
    for L in range(min_w, max_w + 1):
        for i in range(1, n - L):  # 양끝 한 토큰씩 비워 경계 확보
            j = i + L - 1
            sub = ptext[spans[i][0]:spans[j][1]]
            if re.search(r'[.!?"\u201c\u201d]', sub):
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
        if len(v.split()) < 4:
            return None
        if texts[idx].count(v) != 1:
            return None
        if re.search(r'[.!?"\u201c\u201d]', v):
            return None
        if modal_no_verb(v):
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
                    return {"paragraphs": new, "blank_A": va, "blank_B": vb}
    return None


def _b_candidates(hn: str, min_w: int = 4, max_w: int = 7) -> list:
    spans = [(m.start(), m.end()) for m in re.finditer(r'\S+', hn)]
    toks = [hn[s:e] for s, e in spans]
    n = len(toks)
    out = []
    for L in range(min_w, max_w + 1):
        for i in range(0, n - L + 1):
            j = i + L - 1
            sub = hn[spans[i][0]:spans[j][1]]
            if re.search(r'[.!?"\u201c\u201d]', sub):
                continue
            fw = re.sub(r'[^a-z]', '', toks[i].lower())
            if fw in _Q5_MODALS:
                continue
            if hn.count(sub) != 1:
                continue
            out.append((i, j, sub))
    return out


def pick_b_q4_blanks(full_summary, llm_a: str = "", llm_b: str = "", min_w: int = 4, max_w: int = 7) -> Optional[dict]:
    """B Q4 빈칸을 코드가 요약문에서 직접 골라줌 (A Q5와 같은 철학).
    각 4단어 이상, 서로 안 겹침, 경계중복 None. LLM 구절이 유효하면 우선 사용. 실패 시 None.
    요약문은 우리가 만든 한 문장이라, 짧거나 비verbatim인 LLM 빈칸을 코드가 대체해 누락을 막는다."""
    hn = re.sub(r'\s+', ' ', str(full_summary or "")).strip()
    if len(hn.split()) < (2 * min_w + 1):
        return None
    cands = _b_candidates(hn, min_w, max_w)
    if not cands:
        return None

    def span_of(v):
        v = re.sub(r'\s+', ' ', str(v or "")).strip()
        if len(v.split()) < min_w or hn.count(v) != 1:
            return None
        if re.search(r'[.!?"\u201c\u201d]', v):
            return None
        if modal_no_verb(v):
            return None
        toks = hn.split(); pw = v.split()
        for i in range(len(toks) - len(pw) + 1):
            if toks[i:i + len(pw)] == pw:
                return (i, i + len(pw) - 1, v)
        return None

    la = [span_of(llm_a)] if span_of(llm_a) else []
    lb = [span_of(llm_b)] if span_of(llm_b) else []
    poolA = la + cands
    poolB = lb + cands
    for ca in poolA:
        for cb in poolB:
            lo, hi = (ca, cb) if ca[1] < cb[0] else (cb, ca)
            if not (lo[1] < hi[0]):  # 겹치거나 인접(사이 0단어) 제외
                continue
            va, vb = ca[2], cb[2]
            if va == vb:
                continue
            tmpl = hn.replace(va, "(A)", 1)
            if "(A)" not in tmpl:
                continue
            tmpl = tmpl.replace(vb, "(B)", 1)
            if "(B)" not in tmpl:
                continue
            if fill_boundary_dup(tmpl, [("(A)", va), ("(B)", vb)]):
                continue
            return {"blank_A": va, "blank_B": vb}
    return None


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
    # _s41 = 스키마 v6 (STEP0 지문 전체 독해→논지 추출 후 요약문 생성) — 옛 캐시 무효화
    return f"{book_safe}_{unit_safe}_{pid_safe}_{txt_hash}_var{variation_type}_s41"


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
                    _first = re.split(r'(?<=[.!?])\s+', _intro_txt)[0] if _intro_txt else ""
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
                            data["core_blank_options"] = _op
                            data["core_blank_correct"] = _co
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

                # Q5 영작빈칸: ★ 코드가 (A)(B)(C)에서 직접 골라 뚫는다 (B 빈칸뚫기와 같은 철학).
                #   LLM 구절이 유효하면 우선 쓰고, 아니면 코드가 깨끗한 구절을 골라 verbatim 마킹.
                #   → 빈칸 짧음/원문 미발견/경계 단어중복(예: 'questions') 원천 차단. 서로 다른 단락.
                marked = {}
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

            if "mismatch_count" not in data and "statements" in data:
                data["mismatch_count"] = sum(1 for _, _, ok in data["statements"] if not ok)

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
    
    last_errors = []
    last_data = None  # 마지막 fallback용
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            user_msg = (
                f"Passage ID: {pid}\n\n"
                f"Original English passage:\n{en_text}\n\n"
                "Generate the variation problem (Type B). Return ONLY the JSON object."
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
            except Exception:
                pass  # 실패하면 기존(한번에 만든) 요약문 유지

            # ★★ Q4 빈칸을 코드가 요약문에서 직접 골라 4단어+ 보장 (A Q5와 같은 철학).
            #   LLM이 짧게/비verbatim으로 뽑아도 코드가 깨끗한 4단어 구절로 대체 → 누락 차단.
            try:
                _fulls = data.get("full_summary") or data.get("summary_full") or ""
                _bp = pick_b_q4_blanks(_fulls, data.get("blank_A", ""), data.get("blank_B", ""))
                if _bp:
                    data["blank_A"] = _bp["blank_A"]
                    data["blank_B"] = _bp["blank_B"]
            except Exception:
                pass

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
                s = re.sub(r'(?<=\d),(?=\d)', '\u0001', str(text or ""))  # 100,000 보호
                s = re.sub(r'\b([A-Za-z](?:\.[A-Za-z])+)\.?', lambda m: m.group(0).replace('.', '\u0002'), s)  # U.S. 보호
                toks = re.sub(r'[.,;:!?"()]', ' ', s).split()
                return [t.replace('\u0001', ',').replace('\u0002', '.').lower() for t in toks if t]
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
            errors = validate_b(data, en_text, pid, strict=not is_last)
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
