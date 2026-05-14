"""
variation/generator.py
변형문제 데이터 생성 - Claude API 호출 + 자동 검증 + 재시도 + 캐싱
"""
import os
import json
import hashlib
import random
import traceback
from typing import Optional

import anthropic

from variation.prompts import SYSTEM_PROMPT_A, SYSTEM_PROMPT_B, extract_json_from_response
from variation.validator import validate_a, validate_b

# ============ 환경 설정 ============
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20251022")
MAX_RETRIES = 3

# Supabase 클라이언트 (선택적)
try:
    from supabase import create_client
    SB_URL = os.environ.get("SUPABASE_URL", "")
    SB_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY", "")
    sb_client = create_client(SB_URL, SB_KEY) if (SB_URL and SB_KEY) else None
except ImportError:
    sb_client = None

claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY) if CLAUDE_API_KEY else None


# ============ 캐시 키 생성 (pipeline.py 호환) ============
def make_cache_key(book: str, unit: str, pid: str, passage_text: str, variation_type: str) -> str:
    """캐시 키: {책}_{단원}_{번호}_{md5}_v{유형}"""
    txt_hash = hashlib.md5(passage_text.encode("utf-8")).hexdigest()[:8]
    book_safe = book[:15].replace(" ", "_")
    unit_safe = unit[:8].replace(" ", "_")
    pid_safe = pid[:6].replace(" ", "_")
    return f"{book_safe}_{unit_safe}_{pid_safe}_{txt_hash}_var{variation_type}"


# ============ Supabase 캐시 ============
def load_cached(cache_key: str, step_name: str) -> Optional[dict]:
    """Supabase step_cache에서 캐시 로드"""
    if not sb_client:
        return None
    try:
        res = sb_client.table("step_cache") \
            .select("data") \
            .eq("cache_key", cache_key) \
            .eq("step_name", step_name) \
            .limit(1) \
            .execute()
        if res.data and len(res.data) > 0:
            return res.data[0]["data"]
    except Exception as e:
        print(f"[VAR] cache load error: {e}")
    return None


def save_cached(cache_key: str, step_name: str, data: dict) -> None:
    """Supabase step_cache에 캐시 저장"""
    if not sb_client:
        return
    try:
        sb_client.table("step_cache") \
            .upsert({
                "cache_key": cache_key,
                "step_name": step_name,
                "data": data,
            }, on_conflict="cache_key,step_name") \
            .execute()
    except Exception as e:
        print(f"[VAR] cache save error: {e}")


# ============ 지문 텍스트 처리 ============
def split_passage_and_translation(passage_text: str) -> tuple:
    """passage_text에서 영문/한국어 해석 분리"""
    if "###해석###" in passage_text:
        en, kr = passage_text.split("###해석###", 1)
        return en.strip(), kr.strip()
    return passage_text.strip(), ""


# ============ Claude API 호출 ============
def call_claude(system_prompt: str, user_message: str, max_tokens: int = 8000) -> str:
    """Claude API 호출"""
    if not claude_client:
        raise RuntimeError("ANTHROPIC_API_KEY 환경변수가 없습니다")
    
    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


# ============ 유형 A 생성 ============
def generate_variation_a(
    passage_text: str,
    pid: str = "?",
    book: str = "",
    unit: str = "",
    force_regenerate: bool = False,
) -> dict:
    """유형 A 변형문제 생성 (검증 통과 시까지 재시도)"""
    en_text, _ = split_passage_and_translation(passage_text)
    cache_key = make_cache_key(book, unit, pid, en_text, "a")
    
    # 캐시 확인
    if not force_regenerate:
        cached = load_cached(cache_key, "variation_a")
        if cached:
            return cached
    
    last_errors = []
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            user_msg = (
                f"Passage ID: {pid}\n\n"
                f"Original English passage:\n{en_text}\n\n"
                "Generate the variation problem (Type A) for this passage. "
                "Return ONLY the JSON object."
            )
            if last_errors:
                user_msg += (
                    "\n\n# Previous attempt had errors. Fix these specifically:\n"
                    + "\n".join(f"- {e}" for e in last_errors[:5])
                )
            
            raw = call_claude(SYSTEM_PROMPT_A, user_msg)
            data = extract_json_from_response(raw)
            
            # mismatch_count 자동 보정
            if "mismatch_count" not in data and "statements" in data:
                data["mismatch_count"] = sum(1 for _, _, ok in data["statements"] if not ok)
            
            # 검증
            errors = validate_a(data, en_text, pid)
            if not errors:
                save_cached(cache_key, "variation_a", data)
                return data
            last_errors = errors
            print(f"[VAR][A][{pid}] 시도 {attempt} 실패: {len(errors)}건 - {errors[0][:100]}")
        except Exception as e:
            traceback.print_exc()
            last_errors = [f"예외: {e}"]
            print(f"[VAR][A][{pid}] 시도 {attempt} 예외: {e}")
    
    raise RuntimeError(
        f"유형 A 생성 실패 (3회 시도). 마지막 오류:\n" + "\n".join(last_errors[:3])
    )


# ============ 유형 B 생성 ============
def generate_variation_b(
    passage_text: str,
    pid: str = "?",
    book: str = "",
    unit: str = "",
    force_regenerate: bool = False,
) -> dict:
    """유형 B 변형문제 생성"""
    en_text, _ = split_passage_and_translation(passage_text)
    cache_key = make_cache_key(book, unit, pid, en_text, "b")
    
    if not force_regenerate:
        cached = load_cached(cache_key, "variation_b")
        if cached:
            return cached
    
    last_errors = []
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            user_msg = (
                f"Passage ID: {pid}\n\n"
                f"Original English passage:\n{en_text}\n\n"
                "Generate the variation problem (Type B) for this passage. "
                "Return ONLY the JSON object."
            )
            if last_errors:
                user_msg += (
                    "\n\n# Previous attempt had errors. Fix these specifically:\n"
                    + "\n".join(f"- {e}" for e in last_errors[:5])
                )
            
            raw = call_claude(SYSTEM_PROMPT_B, user_msg)
            data = extract_json_from_response(raw)
            
            errors = validate_b(data, en_text, pid)
            if not errors:
                save_cached(cache_key, "variation_b", data)
                return data
            last_errors = errors
            print(f"[VAR][B][{pid}] 시도 {attempt} 실패: {len(errors)}건 - {errors[0][:100]}")
        except Exception as e:
            traceback.print_exc()
            last_errors = [f"예외: {e}"]
            print(f"[VAR][B][{pid}] 시도 {attempt} 예외: {e}")
    
    raise RuntimeError(
        f"유형 B 생성 실패 (3회 시도). 마지막 오류:\n" + "\n".join(last_errors[:3])
    )
