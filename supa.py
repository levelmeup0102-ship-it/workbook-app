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
    cmd = ['curl', '-s', '-X', method, url,
           '-H', f'apikey: {SUPABASE_KEY}',
           '-H', f'Authorization: Bearer {SUPABASE_KEY}',
           '-H', 'Content-Type: application/json',
           '-H', 'Prefer: return=representation']
    if extra_headers:
        for h in extra_headers:
            cmd.extend(['-H', h])
    
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
        if not raw:
            return None
        return json.loads(raw)
    except Exception as e:
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
    return _curl("POST", "passages",
                 body=body,
                 extra_headers=["Prefer: resolution=merge-duplicates,return=representation"])

def upsert_passages_bulk(rows):
    """Bulk upsert: [{book, unit, pid, title, passage_text}, ...]"""
    if not rows:
        return None
    return _curl("POST", "passages",
                 body=rows,
                 extra_headers=["Prefer: resolution=merge-duplicates,return=representation"])

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
    return _curl("POST", "step_cache",
                 body=body,
                 extra_headers=["Prefer: resolution=merge-duplicates,return=representation"])

def count_steps(cache_key):
    q = f"step_cache?cache_key=eq.{quote(cache_key, safe='')}&select=step_name"
    result = _curl("GET", q)
    if isinstance(result, list):
        return len(result)
    return 0
