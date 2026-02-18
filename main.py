#!/usr/bin/env python3
"""Workbook webapp server v7"""
import os, json, hashlib, re, sys, io
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

APP_VERSION = "v11b-supa-fix"

# Clear bytecode cache on startup (prevent stale .pyc from old deploys)
import shutil
for p in Path(".").glob("__pycache__"):
    shutil.rmtree(p, ignore_errors=True)

APP_PASSWORD = os.getenv("APP_PASSWORD", "levelmeup2026")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
PASSAGES_FILE = DATA_DIR / "passages.json"  # data/ 안에 저장 → 볼륨으로 영속

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/api/version")
async def version():
    key = os.getenv("ANTHROPIC_API_KEY", "NOT_SET")
    pf_exists = PASSAGES_FILE.exists()
    passage_count = 0
    supa_count = 0
    supa_ok = False
    
    # Count from _load_db (Supabase or local)
    try:
        db = _load_db()
        for bk in db.get("books", {}).values():
            for ud in bk.get("units", {}).values():
                passage_count += len(ud.get("passages", {}))
    except: pass
    
    # Check Supabase directly
    try:
        import supa
        if supa._enabled():
            rows = supa.get_all_passages()
            supa_count = len(rows)
            supa_ok = True
    except: pass
    
    cache_dirs = len(list(DATA_DIR.glob("*_*"))) if DATA_DIR.exists() else 0
    return {
        "version": APP_VERSION,
        "key_ok": len(key) > 50,
        "passages_file": str(PASSAGES_FILE),
        "passages_exist": pf_exists,
        "passage_count": passage_count,
        "supa_ok": supa_ok,
        "supa_count": supa_count,
        "cache_dirs": cache_dirs,
    }

def _token(pw): return hashlib.sha256(f"{pw}_wb2026".encode()).hexdigest()[:32]
def _verify(r: Request):
    if r.headers.get("Authorization","").replace("Bearer ","") != _token(APP_PASSWORD):
        raise HTTPException(401)

def _load_db():
    """Load passages - Supabase first, local fallback"""
    try:
        import supa
        if supa._enabled():
            rows = supa.get_all_passages()
            if rows:
                db = {"books": {}}
                for r in rows:
                    bk = r["book"]
                    unit = r["unit"]
                    pid = r["pid"]
                    if bk not in db["books"]:
                        db["books"][bk] = {"units": {}}
                    if unit not in db["books"][bk]["units"]:
                        db["books"][bk]["units"][unit] = {"passages": {}}
                    db["books"][bk]["units"][unit]["passages"][pid] = {
                        "title": r["title"], "text": r["passage_text"]
                    }
                return db
    except Exception as e:
        print(f"[supa] load error: {e}")
    # Local fallback
    if PASSAGES_FILE.exists():
        return json.loads(PASSAGES_FILE.read_text(encoding="utf-8"))
    return {"books": {}}

def _save_db(d):
    """Save passages - local + Supabase"""
    # Count total passages
    total = sum(
        len(ud.get("passages", {}))
        for bk, bd in d.get("books", {}).items()
        for unit, ud in bd.get("units", {}).items()
    )
    print(f"[save_db] saving {total} passages...")
    
    PASSAGES_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save_db] local file written OK")
    
    # Also save to Supabase
    try:
        import supa
        if supa._enabled():
            rows = []
            for bk, bd in d.get("books", {}).items():
                for unit, ud in bd.get("units", {}).items():
                    for pid, pi in ud.get("passages", {}).items():
                        rows.append({
                            "book": bk, "unit": unit, "pid": pid,
                            "title": pi.get("title", pid),
                            "passage_text": pi.get("text", "")
                        })
            if rows:
                print(f"[save_db] sending {len(rows)} rows to Supabase...")
                supa.upsert_passages_bulk(rows)
        else:
            print("[save_db] Supabase not enabled")
    except Exception as e:
        print(f"[supa] save error: {e}")

def _ck(book, unit, pid):
    """캐시 키: 한국어 → ASCII 해시로 변환"""
    raw = f"{book}_{unit}_{pid}"
    h = hashlib.md5(raw.encode('utf-8')).hexdigest()[:12]
    # 숫자만 추출해서 읽기 쉽게
    nums = re.findall(r'\d+', raw)
    prefix = "_".join(nums) if nums else "p"
    return f"{prefix}_{h}"

def _is_cached(ck):
    """Check cache - local first, then Supabase"""
    d = DATA_DIR / ck
    if d.exists() and sum(1 for f in d.glob("step*.json")) >= 8:
        return True
    # Check Supabase
    try:
        import supa
        if supa._enabled() and supa.count_steps(ck) >= 8:
            return True
    except: pass
    return False

@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("static/index.html").read_text(encoding="utf-8")

@app.post("/api/auth")
async def auth(request: Request):
    body = await request.json()
    if body.get("password") == APP_PASSWORD:
        return {"ok": True, "token": _token(APP_PASSWORD)}
    raise HTTPException(401, "wrong password")

@app.get("/api/passages")
async def list_passages(request: Request):
    _verify(request)
    db = _load_db()
    result = []
    for bk, bd in db.get("books",{}).items():
        for unit, ud in bd.get("units",{}).items():
            for pid, pi in ud.get("passages",{}).items():
                result.append({
                    "book": bk, "unit": unit, "id": pid,
                    "title": pi.get("title", pid),
                    "cache_status": "ready" if _is_cached(_ck(bk,unit,pid)) else "not_ready"
                })
    return result

@app.post("/api/passages/upload")
async def upload_passages(request: Request):
    _verify(request)
    body = await request.json()
    book = body.get("book", "26 suteuk")
    text = body.get("text", "")
    parts = re.split(r'###(.+?)###', text)
    db = _load_db()
    if book not in db["books"]:
        db["books"][book] = {"units": {}}
    count = 0
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        passage = parts[i+1].strip() if i+1 < len(parts) else ""
        if not passage: continue
        m = re.match(r'(\d+강|SL|L\d+)\s*(.*)', title)
        unit_name = m.group(1) if m else "etc"
        pid = m.group(2) if m and m.group(2) else title
        if unit_name not in db["books"][book]["units"]:
            db["books"][book]["units"][unit_name] = {"passages": {}}
        db["books"][book]["units"][unit_name]["passages"][pid] = {"title": title, "text": passage}
        count += 1
    _save_db(db)
    return {"ok": True, "count": count}

@app.post("/api/generate")
async def generate(request: Request):
    _verify(request)
    body = await request.json()
    book, unit, pid = body.get("book"), body.get("unit"), body.get("passage_id")
    levels = body.get("levels")
    
    # Debug logging
    print(f"[generate] book={book}, unit={unit}, pid={pid}")
    print(f"[generate] passages.json exists: {PASSAGES_FILE.exists()}")
    
    db = _load_db()
    
    # Debug: show what's in the DB
    books_list = list(db.get("books", {}).keys())
    print(f"[generate] books in db: {books_list}")
    if book in db.get("books", {}):
        units_list = list(db["books"][book].get("units", {}).keys())
        print(f"[generate] units in '{book}': {units_list}")
        if unit in db["books"][book].get("units", {}):
            pids_list = list(db["books"][book]["units"][unit].get("passages", {}).keys())
            print(f"[generate] passages in '{unit}': {pids_list}")
    
    try:
        pinfo = db["books"][book]["units"][unit]["passages"][pid]
    except (KeyError, TypeError) as e:
        print(f"[generate] PASSAGE NOT FOUND: {e}")
        raise HTTPException(404, f"passage not found: book={book}, unit={unit}, pid={pid}")

    passage_text = pinfo["text"]
    title = pinfo["title"]
    m = re.match(r'(\d+)', unit)
    lesson_num = m.group(1) if m else "00"
    ck = _ck(book, unit, pid)

    try:
        import pipeline as pl
        pl.DATA_DIR = DATA_DIR
        pl.TEMPLATE_DIR = Path(".")
        pl.OUTPUT_DIR = Path("output")
        pl.OUTPUT_DIR.mkdir(exist_ok=True)

        meta = {"lesson_num": lesson_num, "challenge_title": title}
        result_path = pl.process_passage(passage=passage_text, meta=meta, passage_id=ck, levels=levels)

        if result_path:
            hp = result_path.with_suffix('.html') if result_path.suffix != '.html' else result_path
            if hp.exists():
                return {"ok": True, "html": hp.read_text(encoding="utf-8"), "filename": hp.name}
        raise HTTPException(500, "generation failed")
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
