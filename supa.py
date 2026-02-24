"""Supabase helper - passages + step cache storage via httpx (async)"""
import json, os
from urllib.parse import quote

import httpx

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

_client: httpx.AsyncClient | None = None

def _enabled():
    return bool(SUPABASE_URL and SUPABASE_KEY)

def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=60)
    return _client

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

async def _request(method: str, endpoint: str, body=None, extra_headers: dict | None = None):
    """Async HTTP call to Supabase REST API"""
    if not _enabled():
        return None
    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    headers = _headers(extra_headers)
    client = _get_client()
    try:
        resp = await client.request(method, url, headers=headers,
                                    content=json.dumps(body, ensure_ascii=False) if body is not None else None)
        raw = resp.text.strip()
        if not raw:
            print(f"[supa] empty response for {method} {endpoint}")
            return None
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "message" in parsed:
            print(f"[supa] API error: {parsed.get('message','')[:200]}")
        return parsed
    except Exception as e:
        print(f"[supa] request exception: {str(e)[:200]}")
        return None

# ========================
# Passages
# ========================
async def get_all_passages():
    result = await _request("GET", "passages?select=*&order=unit,pid")
    return result if isinstance(result, list) else []

async def get_passage(book, unit, pid):
    q = f"passages?book=eq.{quote(book, safe='')}&unit=eq.{quote(unit, safe='')}&pid=eq.{quote(pid, safe='')}&select=*"
    result = await _request("GET", q)
    if isinstance(result, list) and len(result) > 0:
        return result[0]
    return None

async def upsert_passage(book, unit, pid, title, text):
    body = {"book": book, "unit": unit, "pid": pid, "title": title, "passage_text": text}
    return await _request("POST", "passages?on_conflict=book,unit,pid",
                          body=body,
                          extra_headers={"Prefer": "resolution=merge-duplicates, return=representation"})

async def upsert_passages_bulk(rows):
    """Bulk upsert: [{book, unit, pid, title, passage_text}, ...]"""
    if not rows:
        return None
    print(f"[supa] upserting {len(rows)} passages...")
    result = await _request("POST", "passages?on_conflict=book,unit,pid",
                            body=rows,
                            extra_headers={"Prefer": "resolution=merge-duplicates, return=representation"})
    if isinstance(result, list):
        print(f"[supa] upsert success: {len(result)} rows")
    else:
        print(f"[supa] upsert result: {str(result)[:200]}")
    return result

# ========================
# Step Cache
# ========================
async def get_step(cache_key, step_name):
    q = f"step_cache?cache_key=eq.{quote(cache_key, safe='')}&step_name=eq.{quote(step_name, safe='')}&select=data"
    result = await _request("GET", q)
    if isinstance(result, list) and len(result) > 0:
        return result[0].get("data")
    return None

async def save_step_supa(cache_key, step_name, data):
    body = {"cache_key": cache_key, "step_name": step_name, "data": data}
    return await _request("POST", "step_cache?on_conflict=cache_key,step_name",
                          body=body,
                          extra_headers={"Prefer": "resolution=merge-duplicates, return=representation"})

async def count_steps(cache_key):
    q = f"step_cache?cache_key=eq.{quote(cache_key, safe='')}&select=step_name"
    result = await _request("GET", q)
    if isinstance(result, list):
        return len(result)
    return 0

# ========================
# Delete
# ========================
async def delete_passage(book, unit, pid):
    """Delete a single passage"""
    q = f"passages?book=eq.{quote(book, safe='')}&unit=eq.{quote(unit, safe='')}&pid=eq.{quote(pid, safe='')}"
    return await _request("DELETE", q)

async def delete_book(book):
    """Delete all passages of a book"""
    q = f"passages?book=eq.{quote(book, safe='')}"
    return await _request("DELETE", q)
