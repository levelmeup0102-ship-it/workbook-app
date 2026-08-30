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

from core.settings import settings

ANTHROPIC_API_KEY = settings.ANTHROPIC_API_KEY or ""
CLAUDE_MODEL = settings.CLAUDE_MODEL
ANTHROPIC_VERSION = "2023-06-01"
SUPABASE_URL = settings.SUPABASE_URL or ""
SUPABASE_KEY = settings.SUPABASE_KEY or ""
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
    9문장 가정 기본 배정. SA 2 / SE 3 / SD 2~3. 빈칸·오류 문장 비겹침.
    인용문([1] 등 따옴표 문장)은 어법 오류에서 제외하는 로직을 호출부에서 추가 가능.
    """
    idx = list(range(n))
    roles = {"SA": [], "SC": [], "SD": [], "SE": []}
    if n >= 9:
        roles["SA"] = [0, 2]
        roles["SE"] = [3, 5, 6]
        roles["SD"] = [4, 8]
    else:  # 짧은 지문 폴백
        roles["SA"] = idx[:1]
        roles["SE"] = idx[2:4]
        roles["SD"] = [i for i in idx if i not in roles["SA"] + roles["SE"]][:2]
    return roles


# ---------- 유형별 생성 + 검증·재생성 ----------
def _gen_with_repair(prompt_fn, validate_fn, *args) -> dict:
    last_err = []
    for attempt in range(MAX_REPAIR):
        prompt = prompt_fn(*args)
        if attempt and last_err:
            prompt += f"\n\n[직전 실패 사유 — 반드시 교정]\n" + "\n".join(last_err)
        try:
            item = _parse_json(_call_claude(prompt))
        except Exception as e:
            last_err = [f"JSON 파싱 실패: {e}"]
            continue
        errs = validate_fn(item)
        if not errs:
            return item
        last_err = errs
    raise RuntimeError("재생성 한도 초과: " + "; ".join(last_err))


def generate_set(book: str, unit: str, pid: str, types: List[str],
                 gp_index: Dict[int, dict], stypes: Dict[str, dict]) -> dict:
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
    if "SA" in types:
        items.append(_gen_with_repair(
            P.prompt_SA,
            lambda it: V.validate_arrangement(it["bogi"], it["answers"], True, True),
            sents, roles["SA"], stypes["SA"]))
    if "SC" in types:
        items.append(_gen_with_repair(
            P.prompt_SC,
            lambda it: V.validate_arrangement(it["bogi"], it["answers"], False, False),
            sents, stypes["SC"]))
    if "SD" in types:
        items.append(_gen_with_repair(
            P.prompt_SD,
            lambda it: V.validate_grammar_errors(it["errors"], gp_index, blank_sents, single_passage=True),
            sents, roles["SD"], allowed_gp))
    if "SE" in types:
        items.append(_gen_with_repair(
            P.prompt_SE,
            lambda it: V.validate_word_forms(it["blanks"], it["bogi"], set(roles["SE"])),
            sents, roles["SE"], stypes["SE"]))

    # 지문 자리표시 합성 (빈칸/오류 주입)
    passage_sentences = _assemble_passage(sents, items, roles)
    s = {"passage_ref": {"book": book, "unit": unit, "pid": pid},
         "passage_sentences": passage_sentences, "roles": roles,
         "single_passage": True, "items": _attach_meta(items, stypes)}
    ok, errs = V.validate_set(s, gp_index)
    if not ok:
        raise RuntimeError("최종 세트 검증 실패: " + "; ".join(errs))
    return s


def _assemble_passage(sents, items, roles) -> List[str]:
    out = list(sents)
    # SA/SE 빈칸 치환
    for it in items:
        if it["type"] == "SA":
            for lab, meta in it["blanks"].items():
                out[meta["sent"]] = meta["tpl"]
        if it["type"] == "SE":
            for bl in it["blanks"]:
                out[bl["sent"]] = re.sub(rf"(?<![\w]){re.escape(bl['answer'])}(?![\w])",
                                         "{{%s}}" % bl["label"], out[bl["sent"]], count=1)
        if it["type"] == "SD":
            for e in it["errors"]:
                out[e["sent"]] = re.sub(rf"(?<![\w]){re.escape(e['right'])}(?![\w])",
                                        e["wrong"], out[e["sent"]], count=1)
    return out


def _attach_meta(items, stypes) -> list:
    pts = {"SA": 6, "SC": 4, "SD": 4, "SE": 6}
    for it in items:
        sp = stypes.get(it["type"], {})
        it["instruction"] = sp.get("instruction", "")
        it["points"] = pts.get(it["type"], "")
        if it["type"] == "SA":
            it["allow_inflect"] = True
    return items
