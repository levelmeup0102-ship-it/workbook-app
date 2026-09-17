"""
seosul/generator.py
서술형 종합 세트 생성 오케스트레이션.
흐름: 지문 fetch → 문장 분리 → 역할 배정(코드) → 유형별 LLM 생성 → 검증 → 실패 시 재생성.
변형문제 모듈과 동일하게 httpx로 Anthropic / Supabase REST 직접 호출 (SDK 의존성 없음).
"""
import os
import json
import re
import time
import httpx
from typing import List, Dict, Optional

from . import validator as V
from . import prompts as P

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5")
ANTHROPIC_VERSION = "2023-06-01"
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY", "")
MAX_REPAIR = 3

# 화면·산출물에 보이는 문구는 코드(SA/SC/SD/SE) 대신 문제 유형 이름으로 쓴다.
TYPE_NAME = {"SA": "본문 빈칸 영작", "SB": "대화문 빈칸", "SC": "요약문 빈칸 영작",
             "SD": "어법 틀린 곳 고치기", "SE": "어휘 품사 변형"}

def type_name(code: str) -> str:
    return TYPE_NAME.get(code, code)


# ---------- Supabase REST ----------
def _sb_get(path: str, params: dict) -> list:
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    with httpx.Client(timeout=15.0) as c:
        r = c.get(url, headers=headers, params=params)
        r.raise_for_status()
        return r.json()

def fetch_passage_text(book: str, unit: str, pid: str) -> Optional[str]:
    rows = _sb_get("passages", {"select": "passage_text", "book": f"eq.{book}",
                                "unit": f"eq.{unit}", "pid": f"eq.{pid}", "limit": "1"})
    if not rows:
        return None
    return rows[0]["passage_text"].split("###해석###")[0].strip()

def fetch_seosul_types() -> Dict[str, dict]:
    rows = _sb_get("seosul_types", {"select": "*", "active": "eq.true"})
    return {r["code"]: r for r in rows}

def fetch_grammar_points() -> Dict[int, dict]:
    rows = _sb_get("grammar_points", {"select": "*", "active": "eq.true"})
    return {r["id"]: r for r in rows}


# ---------- 캐시 (seosul_cache) ----------
def _sb_post(path: str, rows: list, params: dict = None) -> list:
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
               "Content-Type": "application/json",
               "Prefer": "resolution=merge-duplicates,return=minimal"}
    with httpx.Client(timeout=15.0) as c:
        r = c.post(url, headers=headers, params=params or {}, json=rows)
        r.raise_for_status()
        return r.json() if r.text else []

# ★ 로직(generator/validator/prompts/renderer)을 고칠 때마다 +1 하면 전체 캐시가 무효화된다.
#   _s04 = SE 라벨 점프 수정(SA 개수에 맞춰 시작). _s03 = SC 정답/보기 한글 혼입 차단. _s02(SD 문장중복 금지 + SD 0개 문항 제거
#          + SC 보기 12~18개 + 실패결과 캐시 금지) 누적분 포함
_SEOSUL_VER = "_s05"

def _cache_key(book, unit, pid, types):
    return f"{book}|{unit}|{pid}|{','.join(sorted(types))}|{_SEOSUL_VER}"

def cache_get(book: str, unit: str, pid: str, types: List[str]) -> Optional[dict]:
    if not SUPABASE_URL:
        return None
    try:
        rows = _sb_get("seosul_cache", {"select": "data",
                                        "cache_key": f"eq.{_cache_key(book, unit, pid, types)}",
                                        "limit": "1"})
        return rows[0]["data"] if rows else None
    except Exception:
        return None

def cache_set(book: str, unit: str, pid: str, types: List[str], data: dict) -> None:
    if not SUPABASE_URL:
        return
    try:
        _sb_post("seosul_cache",
                 [{"cache_key": _cache_key(book, unit, pid, types),
                   "book": book, "unit": unit, "pid": pid, "data": data}],
                 params={"on_conflict": "cache_key"})
    except Exception:
        pass


# ---------- 캐시 상태 조회 / 삭제 (사이트 버튼용) ----------
def _sb_delete(path: str, params: dict) -> int:
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
               "Prefer": "return=representation"}
    with httpx.Client(timeout=20.0) as c:
        r = c.delete(url, headers=headers, params=params)
        r.raise_for_status()
        try:
            return len(r.json())
        except Exception:
            return 0


def _key_prefix(book: str, unit: str, pid: str) -> str:
    """types 조합과 무관하게 이 지문의 캐시를 전부 가리키는 앞부분."""
    return f"{book}|{unit}|{pid}|"


def _is_unverified(data: dict) -> bool:
    """검증을 통과 못 했는데 살아서 캐시된 세트인가.

    두 군데를 다 봐야 한다 — 실측(46행 중 16행 불량)에서
    _warnings 만 보면 7행밖에 못 잡았다. 나머지 9행은 옛 코드가
    _soft_fail 을 경고로 올리지 않아 items 안에만 남아 있다.
    """
    for it in (data.get("items") or []):
        if it.get("_soft_fail"):
            return True
    for w in (data.get("_warnings") or []):
        w = str(w)
        if "검증 미통과" in w or w.lstrip().startswith("["):
            return True
    return False


_CACHE_PAGE = 1000

def get_cache_status(passages: List[dict]) -> Dict[str, dict]:
    """지문별 서술형 캐시 유무 — 사이트의 ✅ / ⚠️ / ○ 표시용.

    반환: {"교재|단원|번호": {"ok": 생성됨, "unverified": 검증 미통과 섞임}}
    """
    if not SUPABASE_URL:
        return {}
    found: Dict[str, dict] = {}
    offset = 0
    while True:
        rows = _sb_get("seosul_cache",
                       {"select": "cache_key,data",
                        "cache_key": f"like.*|{_SEOSUL_VER}",
                        "limit": str(_CACHE_PAGE),
                        "offset": str(offset)})
        for r in rows:
            parts = (r.get("cache_key") or "").split("|")
            if len(parts) < 5:
                continue
            k = "|".join(parts[:3])
            st = found.setdefault(k, {"ok": False, "unverified": False})
            st["ok"] = True
            if _is_unverified(r.get("data") or {}):
                st["unverified"] = True
        if len(rows) < _CACHE_PAGE:
            break
        offset += _CACHE_PAGE
    return {f"{p['book']}|{p['unit']}|{p['id']}":
            found.get(f"{p['book']}|{p['unit']}|{p['id']}", {"ok": False, "unverified": False})
            for p in passages}


def delete_cache(passages: List[dict]) -> int:
    """선택한 지문의 서술형 캐시만 지운다. 1회독·2회독 캐시는 다른 테이블이라 무관."""
    if not SUPABASE_URL:
        return 0
    total = 0
    for p in passages:
        pref = _key_prefix(p["book"], p["unit"], p["id"])
        total += _sb_delete("seosul_cache",
                            {"cache_key": f'like."{pref}*"'})
    return total


# ---------- Claude ----------
def _call_claude(prompt: str, max_tokens: int = 8000, _tries: int = 4) -> str:
    """Anthropic 호출. 일시적 오류(429/5xx, 또는 overloaded 류 400)는 지수 백오프로 재시도.
    이걸로 07번처럼 '일시 400 → 유형 통째 드롭'이 사라진다."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY 없음")
    last = None
    for attempt in range(_tries):
        try:
            with httpx.Client(timeout=120.0) as c:
                r = c.post("https://api.anthropic.com/v1/messages",
                           headers={"x-api-key": ANTHROPIC_API_KEY,
                                    "anthropic-version": ANTHROPIC_VERSION,
                                    "content-type": "application/json"},
                           json={"model": CLAUDE_MODEL, "max_tokens": max_tokens,
                                 "messages": [{"role": "user", "content": prompt}]})
                if r.status_code >= 400:
                    body = r.text or ""
                    transient = (r.status_code in (408, 409, 425, 429, 500, 502, 503, 504, 529)
                                 or "overloaded" in body.lower() or "rate_limit" in body.lower())
                    if transient and attempt < _tries - 1:
                        time.sleep(min(2 ** attempt * 1.5, 12)); last = body; continue
                    raise RuntimeError(f"Anthropic {r.status_code}: {body[:400]}")
                return "".join(b.get("text", "") for b in r.json().get("content", []))
        except (httpx.TransportError, httpx.TimeoutException) as e:
            last = str(e)
            if attempt < _tries - 1:
                time.sleep(min(2 ** attempt * 1.5, 12)); continue
            raise
    raise RuntimeError(f"Anthropic 재시도 실패: {last}")

def _parse_json(txt: str) -> dict:
    txt = re.sub(r"```(json)?", "", txt).strip()
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0) if m else txt)


# ---------- 문장 역할 배정 (코드, 누더기 방지) ----------
def allocate_roles(n: int) -> Dict[str, List[int]]:
    """문장 역할 배정(비겹침). 길이에 상관없이 SA(최대 2)·SE(최대 4)·SD(최소 1, 최대 3)를
    가능한 한 확보한다. 짧은 지문에서 SD가 비어 'SD 생략'되던 버그를 방지."""
    roles = {"SA": [], "SC": [], "SD": [], "SE": []}
    if n <= 0:
        return roles
    if n >= 9:  # 검증된 기존 배치 유지
        roles["SA"] = [0, 2]
        roles["SE"] = [1, 3, 5, 6]
        roles["SD"] = [4, 7, 8]
        return roles
    order = list(range(n))
    # SA: 떨어진 2곳(0,2) 우선
    sa = [0] + ([2] if n > 2 else ([1] if n > 1 else []))
    rem = [i for i in order if i not in sa]
    # SD 최소 1곳 확보를 위해 SE가 다 먹지 않게
    se_cap = min(4, max(1, len(rem) - 1))
    se = rem[:se_cap]
    sd = [i for i in rem if i not in se][:3]
    if not sd and se:           # 그래도 비면 SE 마지막 1개를 SD로 양보
        sd = [se.pop()]
    roles["SA"], roles["SE"], roles["SD"] = sa, se, sd
    return roles


# ---------- 유형별 생성 + 검증·재생성 ----------
def _gen_with_repair(prompt_fn, validate_fn, *args) -> dict:
    last_err = []
    best = None  # (오류개수, item) — 구조는 멀쩡(키 있음)하나 소프트 검증만 실패한 최선 후보
    for attempt in range(MAX_REPAIR):
        prompt = prompt_fn(*args)
        if attempt and last_err:
            prompt += f"\n\n[직전 실패 사유 — 반드시 교정]\n" + "\n".join(last_err)
        try:
            item = _parse_json(_call_claude(prompt))
        except Exception as e:
            last_err = [f"JSON 파싱 실패: {e}"]
            continue
        try:
            errs = validate_fn(item)
        except Exception as e:
            last_err = [f"필수 키 누락/형식 오류: {e} → 스키마대로 모든 키를 채워라"]
            continue
        if not errs:
            return item
        # 키는 다 있으나 소프트 규칙 미달 → 최선 후보로 보관(나중에 살려 출력)
        if best is None or len(errs) < best[0]:
            best = (len(errs), item)
        last_err = errs
    # 재생성 다 써도 완벽하진 않지만, 렌더 가능한 최선 시도가 있으면 그걸 출력(유형 드롭 금지)
    if best is not None:
        best[1]["_soft_fail"] = last_err
        return best[1]
    raise RuntimeError("재생성 한도 초과(렌더 가능한 결과 없음): " + "; ".join(last_err))


_FUNC_WORDS = {"a", "an", "the", "in", "on", "of", "to", "and", "or", "for", "with",
               "as", "that", "than", "by", "at", "from", "into", "about", "but", "so"}

def _validate_sa(it: dict) -> List[str]:
    """SA 종합 검증: 보기 토큰 + 빈칸 비겹침 + 원문 그대로(복원).
    관사·전치사 등 기능어가 정답에 있는데 보기에서 빠졌으면 자동 보충(탈락 방지)."""
    ans_tok = V.tokenize(" ".join((it.get("answers") or {}).values()))
    bset = set(V.tokenize(" ".join(it.get("bogi", []))))
    for t in ans_tok:
        if t in _FUNC_WORDS and t not in bset:
            it.setdefault("bogi", []).append(t)
            bset.add(t)
    e = V.validate_arrangement(it["bogi"], it["answers"], True, True)
    nb = len(it.get("bogi", []))
    # ★ 단어 중복 불가 — 보기 개수와 정답 단어 개수가 1:1이어야 한다.
    n_ans = len(ans_tok)
    if nb != n_ans:
        e.append(f"[개수불일치] 보기 {nb}개 ≠ 정답 단어 {n_ans}개 → 보기를 모두 한 번씩만 쓰도록 "
                 f"(A)(B)를 다시 잡아라. 같은 단어가 두 번 필요한 어구는 빈칸으로 고르지 마라")
    if nb < 8:
        e.append(f"[보기부족] 보기 단어 {nb}개 → 8~14개가 되도록 조금 더 긴 어구를 (A)(B)로 골라라")
    elif nb > 14:
        e.append(f"[보기과다] 보기 단어 {nb}개 → 14개 이하가 되도록 어구 길이를 줄여라")
    # ★ 빈칸 하나가 문장을 통째로 먹지 않도록 개별 길이도 본다(각 4~7단어)
    for _lab, _v in (it.get("answers") or {}).items():
        _n = len(V.tokenize(_v or ""))
        if _n < 4:
            e.append(f"[빈칸짧음] ({_lab}) {_n}단어 → 4~7단어가 되도록 늘려라")
        elif _n > 7:
            e.append(f"[빈칸김] ({_lab}) {_n}단어 → 4~7단어가 되도록 줄여라")
    avals = [v for v in (it.get("answers") or {}).values() if v]
    for i in range(len(avals)):
        for j in range(len(avals)):
            if i != j and avals[i] in avals[j]:
                e.append(f"[빈칸겹침] '{avals[i]}' ⊂ '{avals[j]}' → 서로 겹치지 않는 다른 어구로 고를 것")
    for lab, meta in (it.get("blanks") or {}).items():
        orig = meta.get("original", "") or ""
        ans = (it.get("answers") or {}).get(lab, "")
        if ans and orig and ans not in orig:
            e.append(f"[복원불일치] ({lab}) 정답 '{ans[:30]}…'가 원문에 그대로 없음 → 원문 어구를 그대로 떼어낼 것")
    return e


def _validate_sc(it: dict) -> List[str]:
    """SC 검증: 정답에 있는데 보기에 부족한 토큰을 자동 보충(중복 단어 누락 방지) 후 다중집합 확인.
    ★ 보기 개수 12~18개 강제 + 한글 혼입 차단."""
    ans_tok = V.tokenize(" ".join((it.get("answers") or {}).values()))
    bogi_tok = V.tokenize(" ".join(it.get("bogi", [])))
    need, have = {}, {}
    for t in ans_tok:
        need[t] = need.get(t, 0) + 1
    for t in bogi_tok:
        have[t] = have.get(t, 0) + 1
    for t, c in need.items():
        if not re.fullmatch(r"[A-Za-z'\-]+", t):   # 한글·숫자·기호는 보기에 넣지 않음
            continue
        miss = c - have.get(t, 0)
        if miss > 0:
            it.setdefault("bogi", []).extend([t] * miss)
    e = V.validate_arrangement(it["bogi"], it["answers"], False, False)
    for lab, v in (it.get("answers") or {}).items():
        if re.search(r"[가-힣]", str(v)):
            e.append(f"[한글혼입] ({lab}) 정답에 한글이 있음: '{str(v)[:40]}' → 영어만 쓸 것")
    nb = len(it.get("bogi", []))
    if nb < 12:
        e.append(f"[보기부족] 보기 {nb}개 → (A)(B) 정답 합계가 12~18단어가 되도록 더 긴 어구로 잡아라")
    elif nb > 18:
        e.append(f"[보기과다] 보기 {nb}개 → 18개 이하가 되도록 어구를 줄여라")
    return e


def generate_set(book: str, unit: str, pid: str, types: List[str],
                 gp_index: Dict[int, dict], stypes: Dict[str, dict],
                 use_cache: bool = True) -> dict:
    if use_cache:
        cached = cache_get(book, unit, pid, types)
        if cached:
            cached["_cached"] = True
            return cached

    text = fetch_passage_text(book, unit, pid)
    if not text:
        raise RuntimeError(f"지문 없음: {book} {unit} {pid}")
    sents = V.split_sentences(text)
    roles = allocate_roles(len(sents))
    roles = {t: roles[t] for t in roles if t in types}  # 선택 유형만
    blank_sents = set()   # 실제 생성 결과로 아래에서 채운다

    # 어법 화이트리스트: 블랙리스트(출제 금지) 제거
    allowed_gp = [g for g in gp_index.values()
                  if not ((g.get("prohibited_analysis") or "") and "출제 금지" in g["prohibited_analysis"])]

    items = []
    warnings = []
    used_sents = set()          # ★ 앞 단계가 점유한 문장 (뒤 단계 프롬프트로 전달)

    def _mark_used(it):
        """생성된 문항이 실제로 점유한 문장 번호를 등록."""
        t = it.get("type")
        if t == "SA":
            for _lab, m in (it.get("blanks") or {}).items():
                if isinstance(m, dict) and isinstance(m.get("sent"), int):
                    used_sents.add(m["sent"])
        # SE(제목 빈칸)는 본문을 점유하지 않으므로 여기서 아무것도 등록하지 않는다.
        elif t == "SD":
            for e in (it.get("errors") or []):
                if isinstance(e.get("sent"), int):
                    used_sents.add(e["sent"])

    def _try(typ, prompt_fn, validate_fn, *args):
        try:
            it = _gen_with_repair(prompt_fn, validate_fn, *args)
            items.append(it)
            _mark_used(it)
            # ★ 재생성 한도를 다 쓰고도 검증을 통과 못 했지만 '렌더는 되니까' 살려낸 문항.
            #   여기서 경고로 올리지 않으면 선생님도 캐시 차단도 이 사실을 모른 채
            #   틀린 문항이 그대로 인쇄된다(실제 사고: 어휘 품사 변형 cycle→recycle).
            for _e in (it.get("_soft_fail") or []):
                warnings.append(f"{type_name(typ)} 검증 미통과(그대로 출력됨): {_e}")
        except Exception as e:
            warnings.append(f"{type_name(typ)} 생략(자동 검증 미통과): {e}")

    # ══════════════════════════════════════════════════════
    #  생성 순서: 1) SA 본문빈칸 → 2) SE 어휘변형 → 3) SD 어법 → 4) SC 요약
    #  앞 단계가 문장을 점유하면 뒤 단계는 그 문장을 피한다(겹침 폐기 원천 차단).
    # ══════════════════════════════════════════════════════

    all_idx = list(range(len(sents)))

    # ---- 1) SA : 본문 빈칸 영작 (A)(B) ----
    #   ★ 문장 번호를 고정하지 않는다. 지문 전체를 후보로 주고 '주제·결론 문장과
    #     그 짝'을 모델이 직접 고르게 한다(고르는 기준은 prompts.prompt_SA 안에).
    #     예전에는 무조건 문장 0,2번이라 도입부 배경 문장을 뚫는 일이 잦았다.
    if "SA" in types:
        _try("SA", P.prompt_SA, _validate_sa,
             sents, all_idx, stypes.get("SA", {}))
        # 실제로 고른 문장으로 역할을 갱신한다. 이걸 안 하면 검증기가 보는
        # blank_sents 와 어긋나 멀쩡한 문항이 '겹침'으로 폐기된다.
        _sa_used = sorted({m["sent"] for it in items if it.get("type") == "SA"
                           for m in (it.get("blanks") or {}).values()
                           if isinstance(m.get("sent"), int)})
        if _sa_used:
            roles["SA"] = _sa_used

    # ---- 2) SE : 제목 빈칸 (본문 점유 없음) ----
    #   ★ 지문을 건드리지 않는다. 별도 제목을 쓰고 거기에 빈칸 1개를 판다.
    #     그래서 used_sents 에 아무것도 넣지 않고, 뒤 문항과 문장을 두고 다투지 않는다.
    #     규칙이 예전과 정반대다 — 정답은 본문에 '없어야' 한다(있으면 베껴 쓰면 되므로).
    if "SE" in types:
        roles["SE"] = []          # 본문 문장을 하나도 점유하지 않는다
        _try("SE", P.prompt_SE,
             lambda it: V.validate_title_blank(it, sents),
             sents, [], stypes.get("SE", {}), sorted(used_sents))

    # 여기까지가 '빈칸이 뚫린 문장'. SD는 이 문장들을 피해야 한다.
    blank_sents = set(used_sents)

    # ---- 3) SD : 어법 틀린 곳 (구문 재작성) ----
    if "SD" in types:
        _sd_targets = [i for i in roles.get("SD", []) if i not in blank_sents]
        if len(_sd_targets) < 2:   # SA가 가져가서 자리가 없으면 남은 문장에서 보충
            _sd_targets += [i for i in all_idx
                            if i not in blank_sents and i not in _sd_targets]
        _sd_targets = _sd_targets or roles.get("SD", [])
        roles["SD"] = _sd_targets
        _try("SD", P.prompt_SD,
             lambda it: V.validate_grammar_errors(it["errors"], gp_index, blank_sents,
                                                  single_passage=True, sentences=sents),
             sents, _sd_targets, allowed_gp, sorted(blank_sents))

    # ---- 4) SC : 요약문 빈칸 (본문 점유 없음, 맨 마지막) ----
    if "SC" in types:
        _prior = []
        for it in items:
            if it.get("type") == "SA":
                _prior.append("- 본문 빈칸 영작: " + " / ".join((it.get("answers") or {}).values()))
            elif it.get("type") == "SE":
                _prior.append("- 어휘 품사 변형: " + ", ".join(
                    str(b.get("answer", "")) for b in (it.get("blanks") or [])))
        _try("SC", P.prompt_SC, _validate_sc,
             sents, stypes.get("SC", {}), "\n".join(_prior))

    if not items:
        raise RuntimeError("모든 유형 생성 실패: " + "; ".join(warnings))

    # ★ SD 금지유형/중복문장/빈칸문장 개별 제거(유형은 살리되 나쁜 오류만 버림)
    for it in items:
        if it.get("type") == "SD":
            clean, seen = [], set()
            for e in (it.get("errors") or []):
                why = V.error_is_forbidden(e, gp_index, sents)
                si = e.get("sent")
                if why:
                    warnings.append(f"{type_name('SD')} 오류 제거({why}): {e.get('wrong')}→{e.get('right')}")
                elif si in blank_sents:
                    warnings.append(f"{type_name('SD')} 오류 제거(빈칸문장): 문장{si}")
                elif si in seen:
                    warnings.append(f"{type_name('SD')} 오류 제거(중복문장): 문장{si}")
                else:
                    clean.append(e); seen.add(si)
            it["errors"] = clean

    # ★ 필터 후 오류가 0개가 된 SD는 문항 자체를 제거 ("틀린 곳 0군데" 지시문 방지)
    _before_n = len(items)
    items = [it for it in items
             if not (it.get("type") == "SD" and not it.get("errors"))]
    if len(items) < _before_n:
        warnings.append(f"{type_name('SD')} 문항 제거(남은 오류 0개 → '0군데' 지시문 방지)")

    # 라벨 강제 배정 (SA=A,B / SE=C,D,E … 충돌·괄호 제거)
    _normalize_labels(items)

    # ★ SD는 '정확히 2곳'으로 확정한다. 3곳 이상 남았으면 앞의 2개만 쓴다.
    #   (합성 전에 잘라야 본문에 주입되는 오류 수와 답지 개수가 어긋나지 않는다)
    for it in items:
        if it.get("type") == "SD" and len(it.get("errors") or []) > 2:
            warnings.append(f"{type_name('SD')} 오류 {len(it['errors'])}개 → 2개로 확정(나머지 미사용)")
            it["errors"] = it["errors"][:2]

    # 지문 자리표시 합성 (빈칸/오류 주입)
    passage_sentences = _assemble_passage(sents, items, roles)

    # ★★ _assemble_passage가 SD 오류를 '한 번 더' 폐기한다(빈칸문장/원문부재).
    #    그래서 위쪽 0개 검사만으로는 부족하고, 합성 '이후'에 다시 걸러야
    #    "틀린 곳 0군데" 지시문이 나가지 않는다.
    _before_n2 = len(items)
    items = [it for it in items
             if not (it.get("type") == "SD" and len(it.get("errors") or []) < 2)]
    if len(items) < _before_n2:
        warnings.append(f"{type_name('SD')} 문항 제거(살아남은 오류가 2곳 미만 → 1군데/0군데 출제 방지)")

    s = {"passage_ref": {"book": book, "unit": unit, "pid": pid},
         "passage_sentences": passage_sentences, "roles": roles,
         "single_passage": True, "items": _attach_meta(items, stypes),
         "_warnings": warnings}
    ok, errs = V.validate_set(s, gp_index)
    if not ok:
        # 개별 유형은 통과했으므로 전체는 막지 않고 경고만 남긴다
        s["_warnings"] = warnings + errs

    # ★ 실패(유형 생략/폴백)한 결과는 캐시에 저장하지 않는다.
    #   저장해 버리면 원인을 고쳐도 낡은 실패본이 계속 나와서, 매번 _SEOSUL_VER을 올려야 한다.
    #   저장을 건너뛰면 다음 생성 때 자동으로 재시도된다.
    #   "검증 미통과"(소프트 실패)도 같이 막는다 — 통과 못 한 문항을 캐시에 박아 두면
    #   원인을 고쳐도 _SEOSUL_VER을 올리기 전까지 같은 불량이 계속 나온다.
    _bad_words = ("생략", "폴백", "검증 미통과")
    _failed = (any(any(k in w for k in _bad_words) for w in s["_warnings"])
               or not ok
               or any(it.get("_soft_fail") for it in items))
    if use_cache and not _failed:
        cache_set(book, unit, pid, types, s)
    return s


_LETTERS = "ABCDEFGH"

def _normalize_labels(items):
    """라벨을 코드가 강제로 배정: SA→A,B / SE→그 다음(C,D,E). 괄호 라벨·충돌 제거."""
    used = 0
    sa = next((it for it in items if it["type"] == "SA"), None)
    se = next((it for it in items if it["type"] == "SE"), None)
    if sa:
        new_ans, new_blk = {}, {}
        for old in list((sa.get("answers") or {}).keys()):
            L = _LETTERS[used]; used += 1
            ans = sa["answers"][old]
            blk = (sa.get("blanks") or {}).get(old, {}) or {}
            orig = blk.get("original", "") or ""
            if ans and ans in orig:
                tpl = orig.replace(ans, "{{%s}}" % L, 1)
            else:
                tpl = (blk.get("tpl", "") or "").replace("{{%s}}" % old, "{{%s}}" % L)
            new_ans[L] = ans
            new_blk[L] = {"sent": blk.get("sent"), "tpl": tpl, "original": orig}
        sa["answers"], sa["blanks"] = new_ans, new_blk
    if se:
        for bl in (se.get("blanks") or []):
            bl["label"] = _LETTERS[used]; used += 1
    return items


def _assemble_passage(sents, items, roles) -> List[str]:
    """본문에 빈칸/오류를 주입하되, 실제로 못 뚫은 빈칸·못 주입한 오류는
    정답/보기에서 '동기 제거'한다. → 유령 빈칸·빈칸문장 오류가 구조적으로 0이 된다.
    SE는 폐기 후 라벨을 다시 연속(C,D,E…)으로 재배열한다."""
    out = list(sents)
    blanked_sents = set()
    # ★ SE 라벨 시작점: SA가 실제로 몇 개 살아남았는지에 맞춘다.
    #   (SA가 (A) 하나만 남았는데 SE가 C부터 시작해 (B)가 비는 문제 방지)
    _sa_item = next((x for x in items if x.get("type") == "SA"), None)

    def _sub1(text, surface, repl):
        new, n = re.subn(rf"(?<![A-Za-z]){re.escape(surface)}(?![A-Za-z])", repl, text, count=1)
        return (new, True) if n else (text, False)

    def _base_surface(text, base):
        if not base:
            return None
        stem = re.escape(base[:-1] if len(base) > 4 and base.endswith("e") else base)
        m = re.search(rf"(?<![A-Za-z]){stem}[A-Za-z]*(?![A-Za-z])", text)
        return m.group(0) if m else None

    # 1) SA — 정답어구를 원문에 직접 천공. 실패하면 그 빈칸 폐기.
    for it in items:
        if it.get("type") != "SA":
            continue
        bysent = {}
        for lab, meta in (it.get("blanks") or {}).items():
            bysent.setdefault(meta.get("sent"), []).append(lab)
        for sent, labs in bysent.items():
            if not isinstance(sent, int) or sent >= len(out):
                for lab in labs:
                    it["blanks"].pop(lab, None); it["answers"].pop(lab, None)
                continue
            base = out[sent]
            for lab in labs:
                ans = (it.get("answers") or {}).get(lab, "")
                new, ok = _sub1(base, ans, "{{%s}}" % lab) if ans else (base, False)
                if ok:
                    base = new; blanked_sents.add(sent)
                else:
                    it["blanks"].pop(lab, None); it["answers"].pop(lab, None)
                    it.setdefault("_dropped", []).append(("SA", lab))
            out[sent] = base

    # 2) SE — 제목 빈칸이라 본문을 건드리지 않는다(예전엔 여기서 빈칸을 뚫었다).

    # 3) SD — ★ 구문 재작성 문장으로 통째 교체(rewritten). 없으면 기존 단어치환 방식.
    #    한 문장에 하나씩만 들어가므로 sent 중복은 위쪽 필터가 이미 제거했다.
    for it in items:
        if it.get("type") != "SD":
            continue
        kept = []
        for e in (it.get("errors") or []):
            sent = e.get("sent")
            if not isinstance(sent, int) or sent >= len(out) or sent in blanked_sents:
                it.setdefault("_dropped", []).append(("SD", "빈칸문장/범위")); continue
            rw = str(e.get("rewritten") or "").strip()
            wrong = str(e.get("wrong") or "").strip()
            # (a) 재작성 문장 방식 — wrong이 실제로 들어 있어야 채택
            if rw and wrong and re.search(rf"(?<![A-Za-z]){re.escape(wrong)}(?![A-Za-z])", rw):
                out[sent] = rw
                kept.append(e)
                continue
            # (b) 폴백 — 원문에서 right를 찾아 wrong으로 치환(구버전 방식)
            new, ok = _sub1(out[sent], e.get("right", ""), wrong)
            if ok:
                out[sent] = new; kept.append(e)
            else:
                it.setdefault("_dropped", []).append(("SD", e.get("right")))
        it["errors"] = kept

    return out


# (결정형 SE 폴백 제거 — 제목 빈칸은 평가원체 문장이라 코드가 지어낼 수 없다.
#  생성 실패 시 _gen_with_repair 가 최선 후보를 내보내고 경고를 남긴다.)


def _attach_meta(items, stypes) -> list:
    pts = {"SA": 6, "SC": 4, "SD": 4, "SE": 6}
    for it in items:
        t = it["type"]
        it["points"] = pts.get(t, "")
        if t == "SA":
            it["allow_inflect"] = True
            it["allow_dup"] = False      # ★ 단어 중복 사용 금지
            labs = ", ".join(f"({k})" for k in (it.get("answers") or {}))
            it["instruction"] = (f"윗글의 빈칸 {labs}에 들어갈 적절한 말을 "
                                 f"&lt;보기&gt;의 단어를 사용하여 작성하시오.")
        elif t == "SC":
            it["instruction"] = ("윗글의 내용을 아래와 같이 요약할 때 빈칸에 들어갈 말을 "
                                 "&lt;보기&gt;의 어구를 <b>변형 없이 모두 한 번씩만 배열하여</b> 완성하시오.")
        elif t == "SD":
            n = len(it.get("errors") or [])
            it["instruction"] = (f"윗글의 <b>빈칸을 제외한 부분</b>에서 어법상 틀린 곳 {n}군데를 "
                                 f"찾아 바르게 고쳐 쓰시오. (밑줄 없음)")
        elif t == "SE":
            it["instruction"] = ("윗글의 제목이다. 빈칸에 들어갈 말을 <b>윗글에서 찾아 "
                                 "알맞은 형태로 바꿔</b> 한 단어로 쓰시오.")
    return items
