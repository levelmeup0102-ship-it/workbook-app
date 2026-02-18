#!/usr/bin/env python3
"""Workbook webapp server v7"""
import os, json, hashlib, re, sys, io
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

APP_VERSION = "v9-curl-final"

# Clear bytecode cache on startup (prevent stale .pyc from old deploys)
import shutil
for p in Path(".").glob("__pycache__"):
    shutil.rmtree(p, ignore_errors=True)

APP_PASSWORD = os.getenv("APP_PASSWORD", "levelmeup2026")
DATA_DIR = Path("data")
PASSAGES_FILE = Path("passages.json")
DATA_DIR.mkdir(exist_ok=True)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/api/version")
async def version():
    return {"version": APP_VERSION, "python": sys.version, "encoding": sys.getdefaultencoding()}

def _token(pw): return hashlib.sha256(f"{pw}_wb2026".encode()).hexdigest()[:32]
def _verify(r: Request):
    if r.headers.get("Authorization","").replace("Bearer ","") != _token(APP_PASSWORD):
        raise HTTPException(401)

def _load_db():
    if PASSAGES_FILE.exists():
        return json.loads(PASSAGES_FILE.read_text(encoding="utf-8"))
    return {"books": {}}
def _save_db(d):
    PASSAGES_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def _ck(book, unit, pid):
    """캐시 키: 한국어 → ASCII 해시로 변환"""
    raw = f"{book}_{unit}_{pid}"
    h = hashlib.md5(raw.encode('utf-8')).hexdigest()[:12]
    # 숫자만 추출해서 읽기 쉽게
    nums = re.findall(r'\d+', raw)
    prefix = "_".join(nums) if nums else "p"
    return f"{prefix}_{h}"

def _is_cached(ck):
    d = DATA_DIR / ck
    if not d.exists(): return False
    return sum(1 for f in d.glob("step*.json")) >= 8

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
        m = re.match(r'(\d+강)\s*(.*)', title)
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
    db = _load_db()
    try:
        pinfo = db["books"][book]["units"][unit]["passages"][pid]
    except (KeyError, TypeError):
        raise HTTPException(404, "passage not found")

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
