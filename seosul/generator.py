"""
seosul/generator.py
서술형 종합 세트 생성 오케스트레이션.
흐름: 지문 fetch → 문장 분리 → 역할 배정(코드) → 유형별 LLM 생성 → 검증 → 실패 시 재생성.
변형문제 모듈과 동일하게 httpx로 Anthropic / Supabase REST 직접 호출 (SDK 의존성 없음).
"""
import os
import json
import re
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

def _cache_key(book: str, unit: str, pid: str, types: List[str]) -> str:
    return f"{book}|{unit}|{pid}|{','.join(sorted(types))}"

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


# ---------- Claude ----------
def _call_claude(prompt: str, max_tokens: int = 1500) -> str:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY 없음")
    with httpx.Client(timeout=120.0) as c:
        r = c.post("https://api.anthropic.com/v1/messages",
                   headers={"x-api-key": ANTHROPIC_API_KEY,
                            "anthropic-version": ANTHROPIC_VERSION,
                            "content-type": "application/json"},
                   json={"model": CLAUDE_MODEL, "max_tokens": max_tokens,
                         "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", []))

def _parse_json(txt: str) -> dict:
    txt = re.sub(r"```(json)?", "", txt).strip()
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0) if m else txt)


# ---------- 문장 역할 배정 (코드, 누더기 방지) ----------
def allocate_roles(n: int) -> Dict[str, List[int]]:
    """
    기본 배정. SA 2 / SE 4 / SD 3. 빈칸·오류 문장 비겹침.
    인용문([1] 등 따옴표 문장)은 어법 오류에서 제외하는 로직을 호출부에서 추가 가능.
    """
    idx = list(range(n))
    roles = {"SA": [], "SC": [], "SD": [], "SE": []}
    if n >= 9:
        roles["SA"] = [0, 2]
        roles["SE"] = [1, 3, 5, 6]
        roles["SD"] = [4, 7, 8]
    else:  # 짧은 지문 폴백
        roles["SA"] = idx[:1]
        rem = idx[1:]
        roles["SE"] = rem[:4]
        rem2 = [i for i in rem if i not in roles["SE"]]
        roles["SD"] = rem2[:3]
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
    if nb < 10:
        e.append(f"[보기부족] 보기 단어 {nb}개 → 10~20개가 되도록 더 긴 어구를 (A)(B)로 골라라")
    elif nb > 20:
        e.append(f"[보기과다] 보기 단어 {nb}개 → 20개 이하가 되도록 어구 길이를 줄여라")
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
    """SC 검증: 정답에 있는데 보기에 부족한 토큰을 자동 보충(중복 단어 누락 방지) 후 다중집합 확인."""
    ans_tok = V.tokenize(" ".join((it.get("answers") or {}).values()))
    bogi_tok = V.tokenize(" ".join(it.get("bogi", [])))
    need, have = {}, {}
    for t in ans_tok:
        need[t] = need.get(t, 0) + 1
    for t in bogi_tok:
        have[t] = have.get(t, 0) + 1
    for t, c in need.items():
        miss = c - have.get(t, 0)
        if miss > 0:
            it.setdefault("bogi", []).extend([t] * miss)
    return V.validate_arrangement(it["bogi"], it["answers"], False, False)


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
    blank_sents = set(roles.get("SA", [])) | set(roles.get("SE", []))

    # 어법 화이트리스트: 블랙리스트(출제 금지) 제거
    allowed_gp = [g for g in gp_index.values()
                  if not ((g.get("prohibited_analysis") or "") and "출제 금지" in g["prohibited_analysis"])]

    items = []
    warnings = []

    def _try(typ, prompt_fn, validate_fn, *args):
        try:
            items.append(_gen_with_repair(prompt_fn, validate_fn, *args))
        except Exception as e:
            warnings.append(f"{typ} 생략(자동 검증 미통과): {e}")

    if "SA" in types:
        _try("SA", P.prompt_SA, _validate_sa,
             sents, roles["SA"], stypes.get("SA", {}))
    if "SC" in types:
        _try("SC", P.prompt_SC, _validate_sc, sents, stypes.get("SC", {}))
    if "SD" in types:
        _try("SD", P.prompt_SD,
             lambda it: V.validate_grammar_errors(it["errors"], gp_index, blank_sents, single_passage=True),
             sents, roles["SD"], allowed_gp)
    if "SE" in types:
        _try("SE", P.prompt_SE,
             lambda it: V.validate_word_forms(it["blanks"], it["bogi"], set(roles["SE"])),
             sents, roles["SE"], stypes.get("SE", {}))

    if not items:
        raise RuntimeError("모든 유형 생성 실패: " + "; ".join(warnings))

    # 라벨 강제 배정 (SA=A,B / SE=C,D,E … 충돌·괄호 제거)
    _normalize_labels(items)

    # 지문 자리표시 합성 (빈칸/오류 주입)
    passage_sentences = _assemble_passage(sents, items, roles)
    s = {"passage_ref": {"book": book, "unit": unit, "pid": pid},
         "passage_sentences": passage_sentences, "roles": roles,
         "single_passage": True, "items": _attach_meta(items, stypes),
         "_warnings": warnings}
    ok, errs = V.validate_set(s, gp_index)
    if not ok:
        # 개별 유형은 통과했으므로 전체는 막지 않고 경고만 남긴다
        s["_warnings"] = warnings + errs
    if use_cache:
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
    out = list(sents)
    for it in items:
        if it["type"] == "SA":
            bysent = {}
            for lab, meta in (it.get("blanks") or {}).items():
                bysent.setdefault(meta.get("sent"), []).append((lab, it["answers"][lab]))
            for sent, blanks in bysent.items():
                if sent is None or sent >= len(out):
                    continue
                base = sents[sent]
                for lab, ans in blanks:
                    if ans and ans in base:
                        base = base.replace(ans, "{{%s}}" % lab, 1)
                out[sent] = base
        elif it["type"] == "SE":
            for bl in (it.get("blanks") or []):
                sent = bl.get("sent")
                if sent is None or sent >= len(out):
                    continue
                out[sent] = re.sub(rf"(?<![\w]){re.escape(bl['answer'])}(?![\w])",
                                   "{{%s}}" % bl["label"], out[sent], count=1)
        elif it["type"] == "SD":
            for e in (it.get("errors") or []):
                sent = e.get("sent")
                if sent is None or sent >= len(out):
                    continue
                out[sent] = re.sub(rf"(?<![\w]){re.escape(e['right'])}(?![\w])",
                                   e["wrong"], out[sent], count=1)
    return out


def _attach_meta(items, stypes) -> list:
    pts = {"SA": 6, "SC": 4, "SD": 4, "SE": 6}
    for it in items:
        t = it["type"]
        it["points"] = pts.get(t, "")
        if t == "SA":
            it["allow_inflect"] = True
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
            labs = ", ".join(f"({b['label']})" for b in (it.get("blanks") or []))
            it["instruction"] = (f"윗글의 빈칸 {labs}에 들어갈 단어를 &lt;보기&gt;에서 골라 "
                                 f"<b>흐름과 어법에 맞게 변형하시오.</b>")
    return items
