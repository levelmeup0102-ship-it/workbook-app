"""Supabase helper - passages + step cache storage via curl"""
import json, os, subprocess, tempfile
from urllib.parse import quote

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

def _enabled():
    return bool(SUPABASE_URL and SUPABASE_KEY)

def _curl(method, endpoint, body=None, extra_headers=None):
    """Raw curl call to Supabase REST API"""
    if not _enabled():
        return None
    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    
    # Build headers - extra_headers can override defaults
    headers = {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': 'application/json',
        'Prefer': 'return=representation'
    }
    if extra_headers:
        for h in extra_headers:
            if ':' in h:
                k, v = h.split(':', 1)
                headers[k.strip()] = v.strip()
    
    cmd = ['curl', '-s', '-X', method, url]
    for k, v in headers.items():
        cmd.extend(['-H', f'{k}: {v}'])
    
    tmp_path = None
    if body is not None:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8', delete=False) as tmp:
            tmp.write(json.dumps(body, ensure_ascii=False))
            tmp_path = tmp.name
        cmd.extend(['-d', f'@{tmp_path}'])
    
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if tmp_path:
            os.unlink(tmp_path)
        raw = result.stdout.decode('utf-8').strip()
        stderr = result.stderr.decode('utf-8').strip()
        if stderr:
            print(f"[supa] curl stderr: {stderr[:200]}")
        if not raw:
            print(f"[supa] curl empty response for {method} {endpoint}")
            return None
        parsed = json.loads(raw)
        # Check for Supabase error
        if isinstance(parsed, dict) and "message" in parsed:
            print(f"[supa] API error: {parsed.get('message','')[:200]}")
        return parsed
    except Exception as e:
        print(f"[supa] curl exception: {str(e)[:200]}")
        if tmp_path:
            try: os.unlink(tmp_path)
            except: pass
        return None

# ========================
# Passages
# ========================
def get_all_passages():
    result = _curl("GET", "passages?select=*&order=unit,pid")
    return result if isinstance(result, list) else []

def get_passage(book, unit, pid):
    q = f"passages?book=eq.{quote(book, safe='')}&unit=eq.{quote(unit, safe='')}&pid=eq.{quote(pid, safe='')}&select=*"
    result = _curl("GET", q)
    if isinstance(result, list) and len(result) > 0:
        return result[0]
    return None

def upsert_passage(book, unit, pid, title, text):
    body = {"book": book, "unit": unit, "pid": pid, "title": title, "passage_text": text}
    return _curl("POST", "passages?on_conflict=book,unit,pid",
                 body=body,
                 extra_headers=["Prefer: resolution=merge-duplicates, return=representation"])

def upsert_passages_bulk(rows):
    """Bulk upsert: [{book, unit, pid, title, passage_text}, ...]"""
    if not rows:
        return None
    print(f"[supa] upserting {len(rows)} passages...")
    result = _curl("POST", "passages?on_conflict=book,unit,pid",
                 body=rows,
                 extra_headers=["Prefer: resolution=merge-duplicates, return=representation"])
    if isinstance(result, list):
        print(f"[supa] upsert success: {len(result)} rows")
    else:
        print(f"[supa] upsert result: {str(result)[:200]}")
    return result

# ========================
# Step Cache
# ========================
def get_step(cache_key, step_name):
    q = f"step_cache?cache_key=eq.{quote(cache_key, safe='')}&step_name=eq.{quote(step_name, safe='')}&select=data"
    result = _curl("GET", q)
    if isinstance(result, list) and len(result) > 0:
        return result[0].get("data")
    return None

def save_step_supa(cache_key, step_name, data):
    body = {"cache_key": cache_key, "step_name": step_name, "data": data}
    return _curl("POST", "step_cache?on_conflict=cache_key,step_name",
                 body=body,
                 extra_headers=["Prefer: resolution=merge-duplicates, return=representation"])

def count_steps(cache_key):
    q = f"step_cache?cache_key=eq.{quote(cache_key, safe='')}&select=step_name"
    result = _curl("GET", q)
    if isinstance(result, list):
        return len(result)
    return 0
