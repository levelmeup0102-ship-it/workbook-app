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
#   _s03 = SC 정답/보기 한글 혼입 차단. _s02(SD 문장중복 금지 + SD 0개 문항 제거
#          + SC 보기 12~18개 + 실패결과 캐시 금지) 누적분 포함
_SEOSUL_VER = "_s03"

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
             lambda it: V.validate_grammar_errors(it["errors"], gp_index, blank_sents,
                                                  single_passage=True, sentences=sents),
             sents, roles["SD"], allowed_gp)
    if "SE" in types:
        try:
            items.append(_gen_with_repair(
                P.prompt_SE,
                lambda it: V.validate_word_forms(it["blanks"], it["bogi"], set(roles["SE"]), sents),
                sents, roles["SE"], stypes.get("SE", {})))
        except Exception as e:
            # ★ 유형 드롭 금지: 코드 기반 결정형 SE로 채운다(07번류 통째 누락 방지)
            avoid = set(roles.get("SA", [])) | set(roles.get("SD", []))
            fb = _fallback_se(sents, roles["SE"], avoid=avoid)
            if fb:
                items.append(fb)
                # 폴백이 빌린 문장을 SE 역할로 등록(위치/겹침 검증 일관성)
                roles["SE"] = sorted(set(roles.get("SE", [])) | {b["sent"] for b in fb["blanks"]})
                blank_sents = set(roles.get("SA", [])) | set(roles.get("SE", []))
                warnings.append(f"SE 폴백(결정형 생성 사용): {e}")
            else:
                warnings.append(f"SE 생략(폴백도 불가): {e}")

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
                    warnings.append(f"SD 오류 제거({why}): {e.get('wrong')}→{e.get('right')}")
                elif si in blank_sents:
                    warnings.append(f"SD 오류 제거(빈칸문장): 문장{si}")
                elif si in seen:
                    warnings.append(f"SD 오류 제거(중복문장): 문장{si}")
                else:
                    clean.append(e); seen.add(si)
            it["errors"] = clean

    # ★ 필터 후 오류가 0개가 된 SD는 문항 자체를 제거 ("틀린 곳 0군데" 지시문 방지)
    _before_n = len(items)
    items = [it for it in items
             if not (it.get("type") == "SD" and not it.get("errors"))]
    if len(items) < _before_n:
        warnings.append("SD 문항 제거(남은 오류 0개 → '0군데' 지시문 방지)")

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

    # ★ 실패(유형 생략/폴백)한 결과는 캐시에 저장하지 않는다.
    #   저장해 버리면 원인을 고쳐도 낡은 실패본이 계속 나와서, 매번 _SEOSUL_VER을 올려야 한다.
    #   저장을 건너뛰면 다음 생성 때 자동으로 재시도된다.
    _failed = any(("생략" in w or "폴백" in w) for w in warnings)
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

    # 2) SE — answer 표면형 우선, 없으면 base의 굴절/파생형 탐색+정답보정. 실패 시 빈칸·보기 동기 폐기.
    for it in items:
        if it.get("type") != "SE":
            continue
        kept = []
        for bl in (it.get("blanks") or []):
            sent = bl.get("sent")
            if not isinstance(sent, int) or sent >= len(out):
                it.setdefault("_dropped", []).append(("SE", bl.get("label"))); continue
            new, ok = _sub1(out[sent], bl.get("answer", ""), "{{%s}}" % bl["label"])
            if not ok:
                surf = _base_surface(out[sent], bl.get("base", ""))
                if surf and surf.lower() != bl.get("base", "").lower():
                    new, ok = _sub1(out[sent], surf, "{{%s}}" % bl["label"])
                    if ok:
                        bl["answer"] = surf
            if ok:
                out[sent] = new; blanked_sents.add(sent); kept.append(bl)
            else:
                it["bogi"] = [w for w in it.get("bogi", []) if w.lower() != bl.get("base", "").lower()]
                it.setdefault("_dropped", []).append(("SE", bl.get("label")))
        # 라벨 재배열(C,D,E…) + 본문 placeholder 갱신
        relabeled = []
        for i, bl in enumerate(kept):
            newL = _LETTERS[2 + i]  # SE는 C부터
            if bl["label"] != newL:
                out[bl["sent"]] = out[bl["sent"]].replace("{{%s}}" % bl["label"], "{{%s}}" % newL)
                bl["label"] = newL
            relabeled.append(bl)
        it["blanks"] = relabeled

    # 3) SD — right가 대상 문장에 있고 그 문장이 빈칸 문장이 아닐 때만 주입. 아니면 폐기.
    for it in items:
        if it.get("type") != "SD":
            continue
        kept = []
        for e in (it.get("errors") or []):
            sent = e.get("sent")
            if not isinstance(sent, int) or sent >= len(out) or sent in blanked_sents:
                it.setdefault("_dropped", []).append(("SD", "빈칸문장/범위")); continue
            new, ok = _sub1(out[sent], e.get("right", ""), e.get("wrong", ""))
            if ok:
                out[sent] = new; kept.append(e)
            else:
                it.setdefault("_dropped", []).append(("SD", e.get("right")))
        it["errors"] = kept

    return out


# ---------- 결정형 SE 폴백 (LLM 실패 시 코드가 직접 생성) ----------
_COMMON_VERBS = {
 "use","make","take","give","find","show","tell","ask","work","call","try","need","feel",
 "become","leave","put","mean","keep","let","begin","seem","help","talk","turn","start","move",
 "like","live","believe","hold","bring","happen","write","provide","sit","stand","lose","pay",
 "meet","include","continue","set","learn","change","lead","understand","watch","follow","stop",
 "create","speak","read","spend","grow","open","walk","win","teach","offer","remember","consider",
 "appear","buy","serve","die","send","build","stay","fall","reach","remain","suggest","raise",
 "pass","sell","require","report","decide","pull","return","explain","develop","carry","break",
 "receive","agree","support","hit","produce","eat","cover","catch","draw","choose","cause","point",
 "listen","realize","press","close","wait","prepare","share","examine","compare","describe","ensure",
 "secure","reduce","differ","compromise","adhere","educate","perform","expose","react","divide",
 "consume","implement","ignore","intend","compete","express","recognize","translate","mimic","match",
 "marry","strike","interpret","fragment","weaken","function","evolve","constrain","communicate",
 "rationalize","humanize","simulate","represent","generate","play","surf","fill","name","process",
 "trigger","respond","evoke","capture","gain","watch","encounter","spend","reinforce","protect",
 "tend","prevent","overwhelm","inundate","sort","group","value","investigate","hinder","handle",
 "donate","purchase","tout","spend","empathize","facilitate","judge","increase","split","bring",
}

def _verb_base_from_ing(stem):
    """동명사/분사 어간 → '내장 빈출동사'에 있는 실단어 원형만 반환(없으면 None)."""
    if len(stem) >= 4 and stem[-1] == stem[-2] and stem[-1] in "bdgklmnprt":
        stem = stem[:-1]                       # running→run
    for cand in (stem, stem + "e"):            # fill / ensure(e)
        if cand in _COMMON_VERBS:
            return cand
    return None

_DERIV = [
    (re.compile(r"^(.{3,})ically$"), lambda m: m.group(1) + "ical", "형용사→부사"),   # radically→radical
    (re.compile(r"^(.{4,})ally$"),   lambda m: m.group(1),          "형용사→부사"),   # energetically→energetic은 별도
    (re.compile(r"^(.{4,})ly$"),      lambda m: (m.group(1)[:-1] + "y") if m.group(1).endswith("i") else m.group(1), "형용사→부사"),
    (re.compile(r"^(.{4,})ing$"),     lambda m: _verb_base_from_ing(m.group(1)), "동사→동명사"),
    (re.compile(r"^(.{4,})ed$"),      lambda m: _verb_base_from_ing(m.group(1)), "동사→과거분사"),
]
_STOP = {"being","doing","having","during","anything","something","nothing","according",
         "including","regarding","interesting","following","everything","understanding",
         "willing","ongoing","wedding","finding","really","only","family","early","likely",
         "supply","apply","reply"}

def _fallback_se(sents, target_idx, avoid=None, need=4):
    """대상 문장에서 부사(-ly)·동명사/분사(-ing/-ed, 내장 동사로 검증) 단어를 골라
    빈칸/정답/보기를 코드로 생성. answer는 항상 원문 표면형이라 빈칸이 반드시 뚫린다.
    avoid: SA/SD가 이미 점유한 문장(겹침 방지)."""
    avoid = set(avoid or [])
    cand = []  # (sent, surface, base, note)
    pool = [i for i in target_idx if i not in avoid] + \
           [i for i in range(len(sents)) if i not in target_idx and i not in avoid]
    for si in pool:
        if not (0 <= si < len(sents)):
            continue
        seen = set()
        for w in re.findall(r"[A-Za-z]+", sents[si]):
            wl = w.lower()
            if len(wl) < 6 or wl in _STOP or wl in seen:
                continue
            for rx, fn, note in _DERIV:
                m = rx.match(wl)
                if m:
                    base = fn(m)
                    if base and base != wl and len(base) >= 3 and base.isalpha():
                        cand.append((si, w, base, note)); seen.add(wl)
                    break
    picked, used_base = [], set()
    for si, surf, base, note in cand:
        if base in used_base:
            continue
        picked.append((si, surf, base, note)); used_base.add(base)
        if len(picked) >= need:
            break
    if len(picked) < 2:
        return None
    blanks, bogi = [], []
    for i, (si, surf, base, note) in enumerate(picked):
        L = _LETTERS[2 + i]
        blanks.append({"label": L, "sent": si, "base": base, "answer": surf, "note": note})
        bogi.append(base)
    extras = [b for (_, _, b, _) in cand if b not in used_base and b in _COMMON_VERBS]
    bogi += (extras[:2] if len(extras) >= 2 else (extras + ["create", "process"])[:2])
    import random as _r
    _r.shuffle(bogi)
    return {"type": "SE", "blanks": blanks, "bogi": bogi, "_fallback": True}


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
