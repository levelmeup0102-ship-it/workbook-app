"""Supabase helper - passages + step cache storage via httpx (async)

v2 변경점
  ① _request 가 에러를 '조용히 리턴'하지 않고 raise 할 수 있게 (raise_on_error)
     → main.py 의 배치 재시도 / try-except 가 실제로 동작함
  ② get_all_passages() Range 헤더 페이지네이션 → PostgREST max-rows 상한 무력화
  ③ 정렬키에 book 추가 (unit,pid 는 교재 간 유일하지 않아 페이지네이션이 깨짐)
  ④ upsert_passages_bulk 가 배치 내 (book,unit,pid) 중복 제거
     → "ON CONFLICT DO UPDATE command cannot affect row a second time" 방지
  ⑤ count_passages() 추가 — Content-Range 헤더로 진짜 행 수 조회(진단용)
"""
import json, os
from urllib.parse import quote

import httpx

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# 한 번에 가져올 행 수 (PostgREST Range 페이지 크기)
PAGE_SIZE = 500
# 무한루프 안전장치
MAX_ROWS_HARD = 50000


class SupaError(RuntimeError):
    """Supabase REST 호출 실패 — 호출측에서 재시도 판단용"""
    pass


def _enabled():
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _headers(extra: dict | None = None) -> dict:
    h = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    if extra:
        h.update(extra)
    return h


async def _request(
    method: str,
    endpoint: str,
    body=None,
    extra_headers: dict | None = None,
    raise_on_error: bool = False,
    return_response: bool = False,
):
    """Async HTTP call to Supabase REST API

    raise_on_error=True 이면 실패 시 SupaError 를 던진다.
      - 쓰기(upsert/delete) 경로에서 반드시 True 로 쓸 것.
      - 기존처럼 조용히 None 을 리턴하면 상위의 재시도 로직이 무의미해진다.
    return_response=True 이면 (parsed, httpx.Response) 튜플 반환 (헤더 필요할 때).
    """
    if not _enabled():
        if raise_on_error:
            raise SupaError("Supabase not configured (SUPABASE_URL/KEY 없음)")
        return (None, None) if return_response else None

    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    headers = _headers(extra_headers)

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.request(
                method,
                url,
                headers=headers,
                content=json.dumps(body, ensure_ascii=False) if body is not None else None,
            )
    except Exception as e:
        msg = f"{method} {endpoint[:80]} 네트워크 실패: {type(e).__name__}: {str(e)[:200]}"
        print(f"[supa] {msg}")
        if raise_on_error:
            raise SupaError(msg) from e
        return (None, None) if return_response else None

    # ★ HTTP 상태 코드 먼저 확인 (기존 코드엔 이 검사가 아예 없었음)
    if resp.status_code >= 400:
        msg = f"{method} {endpoint[:80]} HTTP {resp.status_code}: {resp.text[:300]}"
        print(f"[supa] {msg}")
        if raise_on_error:
            raise SupaError(msg)
        return (None, resp) if return_response else None

    raw = resp.text.strip()
    if not raw:
        # 204 No Content / return=minimal 등은 정상
        if raise_on_error and resp.status_code not in (200, 201, 204):
            raise SupaError(f"{method} {endpoint[:80]} 빈 응답 (status={resp.status_code})")
        return (None, resp) if return_response else None

    try:
        parsed = json.loads(raw)
    except Exception as e:
        msg = f"{method} {endpoint[:80]} JSON 파싱 실패: {raw[:200]}"
        print(f"[supa] {msg}")
        if raise_on_error:
            raise SupaError(msg) from e
        return (None, resp) if return_response else None

    if isinstance(parsed, dict) and parsed.get("message"):
        msg = (f"API error: {parsed.get('message','')[:200]} "
               f"| details={str(parsed.get('details',''))[:150]} "
               f"| hint={str(parsed.get('hint',''))[:100]}")
        print(f"[supa] {msg}")
        if raise_on_error:
            raise SupaError(msg)

    return (parsed, resp) if return_response else parsed


# ========================
# Passages
# ========================
async def count_passages() -> int:
    """passages 테이블의 '진짜' 행 수. Content-Range 헤더 기반이라 max-rows 영향 안 받음.

    진단용: get_all_passages() 결과 길이와 이 값이 다르면
            → PostgREST max-rows 에 걸려 잘리고 있다는 뜻.
    실패 시 -1.
    """
    if not _enabled():
        return -1
    parsed, resp = await _request(
        "GET", "passages?select=pid",
        extra_headers={"Range-Unit": "items", "Range": "0-0", "Prefer": "count=exact"},
        return_response=True,
    )
    if resp is None:
        return -1
    cr = resp.headers.get("content-range", "")   # 예: "0-0/28"
    if "/" in cr:
        tail = cr.rsplit("/", 1)[-1].strip()
        if tail.isdigit():
            return int(tail)
    return -1


async def get_all_passages():
    """전체 지문 조회 — Range 페이지네이션으로 끝까지 긁어온다.

    ★ 기존: "passages?select=*&order=unit,pid" 단발 호출.
      - PostgREST 의 max-rows 설정에 걸리면 조용히 잘린 목록이 돌아왔고,
        main.py 는 그걸 '전체'로 믿었다.
      - order 도 (unit,pid) 라 교재가 다르면 값이 겹쳐 정렬이 불안정.
    ★ 변경: order=book,unit,pid (유일) + Range 로 페이지 단위 수집.
    """
    out = []
    start = 0
    page = 0

    while True:
        end = start + PAGE_SIZE - 1
        result = await _request(
            "GET",
            "passages?select=*&order=book,unit,pid",
            extra_headers={"Range-Unit": "items", "Range": f"{start}-{end}"},
        )

        if not isinstance(result, list):
            if page == 0:
                print("[supa] get_all_passages: 첫 페이지 조회 실패 (None/에러)")
            break

        out.extend(result)
        page += 1

        if len(result) < PAGE_SIZE:
            break

        start += PAGE_SIZE
        if start >= MAX_ROWS_HARD:
            print(f"[supa] get_all_passages: {MAX_ROWS_HARD}행 안전 상한 도달, 중단")
            break

    if page > 1:
        print(f"[supa] get_all_passages: {page}페이지 / 총 {len(out)}행")
    return out


async def get_passage(book, unit, pid):
    q = (
        "passages?"
        f"book=eq.{quote(book, safe='')}&"
        f"unit=eq.{quote(unit, safe='')}&"
        f"pid=eq.{quote(pid, safe='')}&select=*"
    )
    result = await _request("GET", q)
    if isinstance(result, list) and len(result) > 0:
        return result[0]
    return None


async def upsert_passage(book, unit, pid, title, text):
    body = {"book": book, "unit": unit, "pid": pid, "title": title, "passage_text": text}
    return await _request(
        "POST",
        "passages?on_conflict=book,unit,pid",
        body=body,
        extra_headers={"Prefer": "resolution=merge-duplicates, return=representation"},
        raise_on_error=True,
    )


def _dedupe_rows(rows):
    """배치 내 (book, unit, pid) 중복 제거 — 뒤에 온 것이 이김.

    ★ 이게 없으면 Postgres 가 배치 전체를 거부한다:
      "ON CONFLICT DO UPDATE command cannot affect row a second time"
      모의고사는 전 지문이 unit='etc' 로 몰려서 pid 하나만 겹쳐도 터짐.
    """
    seen = {}
    dupes = []
    for r in rows:
        key = (r.get("book"), r.get("unit"), r.get("pid"))
        if key in seen:
            dupes.append(key)
        seen[key] = r
    if dupes:
        print(f"[supa] ⚠ 배치 내 중복 키 {len(dupes)}건 제거: "
              f"{[f'{b}/{u}/{p}' for b, u, p in dupes[:5]]}")
    return list(seen.values())


async def upsert_passages_bulk(rows):
    """Bulk upsert: [{book, unit, pid, title, passage_text}, ...]

    실패하면 SupaError 를 던진다 (기존엔 조용히 에러 dict 를 리턴해서
    main.py 의 재시도 로직이 한 번도 발동하지 않았음).
    """
    if not rows:
        return None

    rows = _dedupe_rows(rows)

    # 필수 키 누락 행 제거 (빈 pid 등이 섞이면 배치 전체가 죽음)
    clean = [r for r in rows if r.get("book") and r.get("unit") and r.get("pid")]
    if len(clean) != len(rows):
        print(f"[supa] ⚠ book/unit/pid 누락 행 {len(rows) - len(clean)}건 제외")
    if not clean:
        return None

    print(f"[supa] upserting {len(clean)} passages...")
    result = await _request(
        "POST",
        "passages?on_conflict=book,unit,pid",
        body=clean,
        extra_headers={"Prefer": "resolution=merge-duplicates, return=representation"},
        raise_on_error=True,
    )

    if isinstance(result, list):
        print(f"[supa] upsert success: {len(result)} rows")
        if len(result) != len(clean):
            print(f"[supa] ⚠ 요청 {len(clean)}행 / 반영 {len(result)}행 — 불일치")
    else:
        # raise_on_error=True 이므로 여기 도달 = 응답 본문이 비었지만 2xx
        print(f"[supa] upsert result(non-list): {str(result)[:200]}")

    return result


# ========================
# Step Cache (table: step_cache)
# ========================
async def get_step(cache_key, step_name):
    q = (
        "step_cache?"
        f"cache_key=eq.{quote(cache_key, safe='')}&"
        f"step_name=eq.{quote(step_name, safe='')}&select=data"
    )
    result = await _request("GET", q)
    if isinstance(result, list) and len(result) > 0:
        return result[0].get("data")
    return None


async def save_step_supa(cache_key, step_name, data):
    body = {"cache_key": cache_key, "step_name": step_name, "data": data}
    return await _request(
        "POST",
        "step_cache?on_conflict=cache_key,step_name",
        body=body,
        extra_headers={"Prefer": "resolution=merge-duplicates, return=representation"},
        raise_on_error=True,
    )


async def count_steps(cache_key):
    q = f"step_cache?cache_key=eq.{quote(cache_key, safe='')}&select=step_name"
    result = await _request("GET", q)
    if isinstance(result, list):
        return len(result)
    return 0


async def delete_steps_by_cache_key(cache_key):
    """Delete all cached steps for a cache_key from step_cache"""
    q = f"step_cache?cache_key=eq.{quote(cache_key, safe='')}"
    return await _request("DELETE", q, raise_on_error=True)


async def delete_step(cache_key, step_name):
    """Delete a single step cache entry"""
    q = (
        "step_cache?"
        f"cache_key=eq.{quote(cache_key, safe='')}&"
        f"step_name=eq.{quote(step_name, safe='')}"
    )
    return await _request("DELETE", q, raise_on_error=True)


# pipeline.py에서 호출하는 이름
async def delete_all_steps(cache_key):
    """Alias for delete_steps_by_cache_key (pipeline.py 호환)"""
    return await delete_steps_by_cache_key(cache_key)


# ========================
# Delete passages
# ========================
async def delete_passage(book, unit, pid):
    """Delete a single passage"""
    q = (
        "passages?"
        f"book=eq.{quote(book, safe='')}&"
        f"unit=eq.{quote(unit, safe='')}&"
        f"pid=eq.{quote(pid, safe='')}"
    )
    return await _request("DELETE", q, raise_on_error=True)


async def delete_book(book):
    """Delete all passages of a book"""
    q = f"passages?book=eq.{quote(book, safe='')}"
    return await _request("DELETE", q, raise_on_error=True)
