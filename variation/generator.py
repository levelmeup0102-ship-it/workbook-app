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

from variation.prompts import SYSTEM_PROMPT_A, SYSTEM_PROMPT_B, extract_json_from_response
from variation.validator import validate_a, validate_b

# ============ 환경 변수 ============
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5")
ANTHROPIC_VERSION = "2023-06-01"
MAX_RETRIES = 5

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
    # _s7 = 스키마 v6 (STEP0 지문 전체 독해→논지 추출 후 요약문 생성) — 옛 캐시 무효화
    return f"{book_safe}_{unit_safe}_{pid_safe}_{txt_hash}_var{variation_type}_s7"


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
    
    with httpx.Client(timeout=180.0) as client:
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
) -> dict:
    en_text, _ = split_passage_and_translation(passage_text)
    cache_key = make_cache_key(book, unit, pid, en_text, "a")
    
    if not force_regenerate:
        cached = load_cached(cache_key, "variation_a")
        if cached:
            print(f"[VAR][A][{pid}] 캐시 히트")
            return cached
    
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
                    "  5. blank_A and blank_B must EACH have AT LEAST 5 words, placed INSIDE (A)/(B)/(C) (not in intro), in different paragraphs.\n"
                    "  6. bogi must contain EVERY SINGLE WORD from blank_A + blank_B — count articles ('the','a','an') and prepositions carefully.\n"
                    "  7. core_blank_target must have AT LEAST 3 words; the Q3 correct option must equal core_blank_target exactly."
                )
            
            raw = call_claude(SYSTEM_PROMPT_A, user_msg)
            data = extract_json_from_response(raw)
            
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
) -> dict:
    en_text, _ = split_passage_and_translation(passage_text)
    cache_key = make_cache_key(book, unit, pid, en_text, "b")
    
    if not force_regenerate:
        cached = load_cached(cache_key, "variation_b")
        if cached:
            print(f"[VAR][B][{pid}] 캐시 히트")
            return cached
    
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
                    "  2. blank_A and blank_B must EACH have AT LEAST 6 words\n"
                    "  3. topic_writing_answer must have AT LEAST 10 words\n"
                    "  4. Hyphenated words (south-facing, well-known) stay as ONE token in both blank and bogi\n"
                    "  5. blank_summary_bogi must contain EVERY word from blank_A + blank_B (count articles/preps)\n"
                    "  6. ★ Q3 summary_options: EACH (A) and (B) must be EXACTLY ONE WORD (no phrases!)\n"
                    "     GOOD: [['manipulation','extension'], ['control','delay'], ...]\n"
                    "     BAD: [['south-facing garden beds', 'flat stones from beach'], ...]\n"
                    "  7. All five (A) values must be DIFFERENT words; all five (B) values must be DIFFERENT"
                )
            
            raw = call_claude(SYSTEM_PROMPT_B, user_msg)
            data = extract_json_from_response(raw)

            # ★ Q4/Q5 보기(bogi) 자동 생성: 답지 단어를 그대로 소문자·구두점제거하여 보기로 사용.
            #   모델이 만든 보기는 무시 → 누락/잉여(예: 'for')/중복오류를 원천 차단.
            def _bogi_from(text: str):
                toks = re.sub(r'[.,;:!?"()]', ' ', str(text or "")).split()
                return [t.lower() for t in toks if t]
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
