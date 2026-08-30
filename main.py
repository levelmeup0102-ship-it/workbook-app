#!/usr/bin/env python3
"""Workbook webapp server v12 - stable local + supabase passages, cache status"""
from core.settings import settings  # noqa: F401 — env/테이블명 로딩 최우선. 다른 import보다 먼저.
import os, json, hashlib, re, shutil
from pathlib import Path
from contextlib import asynccontextmanager
import uvicorn
import logging
import supa

from database.client import init as supabase_init, close as supabase_close

from fastapi.exceptions import RequestValidationError
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException as StarletteHTTPException

from first_round_workbook.router import router as first_round_router
from core.exceptions import AppError
from core.exception_handler import (
    app_error_handler,
    request_validation_exception_handler,
    http_exception_handler,
    unhandled_exception_handler
)
from core.logging_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

APP_VERSION = "v12-main-replace"

# Clear bytecode cache on startup (prevent stale .pyc from old deploys)
for p in Path(".").glob("__pycache__"):
    shutil.rmtree(p, ignore_errors=True)

APP_PASSWORD = settings.APP_PASSWORD

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

PASSAGES_FILE = DATA_DIR / "passages.json"  # data/ 안에 저장 → 볼륨으로 영속

# ============================================================
# Supabase 연결 - lifespan
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: env 있으면 async client 생성 → app.state 에 보관 (없으면 None=로컬 모드)
    app.state.supabase = await supabase_init()
    yield
    # shutdown: postgrest 세션 정리(DB 연결 해제)
    await supabase_close(app.state.supabase)

app = FastAPI(lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================
# ★ 1회독 교재 생성 라우터 등록
#   passages/books/generation 3개 도메인 전체 등록 (집계 라우터)
# ============================================================
app.include_router(first_round_router)

app.add_exception_handler(AppError, app_error_handler) # type checker로 인한 오류
app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)



# ============================================================
# ★ 변형문제 (2회독) 라우터 등록 - 2026-05-13 추가
# ============================================================
try:
    from variation.api import router as variation_router, download_router
    app.include_router(variation_router)
    app.include_router(download_router)
    print("[variation] 변형문제 라우터 등록 완료")
except Exception as e:
    print(f"[variation] 라우터 등록 실패 (변형문제 기능 비활성): {e}")


# ============================================================
# ★ 서술형 종합 (2회독) 라우터 등록 - seosul 모듈
# ============================================================
try:
    from seosul.api import router as seosul_router, download_router as seosul_dl
    app.include_router(seosul_router)
    app.include_router(seosul_dl)
    print("[seosul] 서술형 종합 라우터 등록 완료")
except Exception as e:
    print(f"[seosul] 라우터 등록 실패 (서술형 기능 비활성): {e}")


# ============================================================
# Auth
# ============================================================
# 인증 유틸 → util.security 로 이전 (main/패키지 공용, 순환 import 방지)
from core.security import _token, _verify


# ============================================================
# DB Load/Save (Supabase first, local fallback)
# ============================================================
async def _load_db():
    """Load passages - Supabase first, local fallback"""
    # Supabase
    try:
        if supa._enabled():
            rows = await supa.get_all_passages()
            if isinstance(rows, list) and rows:
                db = {"books": {}}
                for r in rows:
                    bk = r.get("book", "")
                    unit = r.get("unit", "")
                    pid = r.get("pid", "")
                    if not (bk and unit and pid):
                        continue
                    db["books"].setdefault(bk, {"units": {}})
                    db["books"][bk]["units"].setdefault(unit, {"passages": {}})
                    db["books"][bk]["units"][unit]["passages"][pid] = {
                        "title": r.get("title", pid),
                        "text": r.get("passage_text", ""),
                    }
                return db
    except Exception as e:
        logger.error(f"[supa] load error (NO SUPABASE_URL OR SUPABASE_KEY): {e}")

    # Local fallback
    if PASSAGES_FILE.exists():
        try:
            return json.loads(PASSAGES_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[local] passages.json parse error: {e}")

    return {"books": {}}

async def _save_db(d):
    """Save passages - local + Supabase (batch)"""
    # local
    PASSAGES_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[save_db] local file written OK")

    # supabase passages sync (best-effort)
    try:
        if not supa._enabled():
            print("[save_db] Supabase not enabled")
            return

        rows = []
        for bk, bd in d.get("books", {}).items():
            for unit, ud in bd.get("units", {}).items():
                for pid, pi in ud.get("passages", {}).items():
                    rows.append({
                        "book": bk,
                        "unit": unit,
                        "pid": pid,
                        "title": pi.get("title", pid),
                        "passage_text": pi.get("text", ""),
                    })

        if not rows:
            return

        batch_size = 50
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            print(f"[save_db] Supabase upsert batch {start//batch_size + 1} ({len(batch)} rows)")
            await supa.upsert_passages_bulk(batch)

        print(f"[save_db] Supabase sync done: {len(rows)} rows total")
    except Exception as e:
        print(f"[supa] save error: {e}")


# ============================================================
# Cache Key / Cache Check
# ============================================================
# 캐시 키 유틸 → util.cache 로 이전
from utils.cache_key import _ck

async def _is_cached(ck: str) -> bool:
    """Check cache - local first, then Supabase (count only)"""
    # local cache: step*.json 8개 이상이면 ready로 간주
    d = DATA_DIR / ck
    if d.exists():
        try:
            if sum(1 for _ in d.glob("step*.json")) >= 8:
                return True
        except Exception:
            pass

    # supabase cache count (best-effort)
    try:
        if supa._enabled():
            n = await supa.count_steps(ck)
            if isinstance(n, int) and n >= 8:
                return True
    except Exception:
        pass

    return False


# ============================================================
# Routes
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("static/index.html").read_text(encoding="utf-8")


@app.get("/api/version")
async def version():
    key = settings.ANTHROPIC_API_KEY or "NOT_SET"

    pf_exists = PASSAGES_FILE.exists()
    passage_count = 0
    supa_count = 0
    supa_ok = False

    try:
        db = await _load_db()
        for bk in db.get("books", {}).values():
            for ud in bk.get("units", {}).values():
                passage_count += len(ud.get("passages", {}))
    except Exception:
        pass

    try:
        if supa._enabled():
            rows = await supa.get_all_passages()
            supa_count = len(rows) if isinstance(rows, list) else 0
            supa_ok = True
    except Exception:
        pass

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


@app.post("/api/auth")
async def auth(request: Request):
    body = await request.json()
    if body.get("password") == APP_PASSWORD:
        return {"ok": True, "token": _token(APP_PASSWORD)}
    raise HTTPException(401, "wrong password")


# ===== [리팩토링 중복 코드] first_round_workbook.passages / books 로 이전됨 — 비활성 보존 (아래 블록 전체, 다음 라운드에 삭제) =====
r'''
# @app.get("/api/passages")  # → first_round_workbook.passages 라우터로 이전 (기존 코드 보존, 나중 삭제)
async def list_passages(request: Request):
    _verify(request)
    db = await _load_db()
    result = []
    for bk, bd in db.get("books", {}).items():
        for unit, ud in bd.get("units", {}).items():
            for pid, pi in ud.get("passages", {}).items():
                ck = _ck(bk, unit, pid)
                result.append({
                    "book": bk,
                    "unit": unit,
                    "id": pid,  # 프론트에서 p.id 로 씀
                    "title": pi.get("title", pid),
                    "passage_text": pi.get("text", ""),  # ★ 원문 추출용 추가
                    "cache_status": "ready" if await _is_cached(ck) else "not_ready",
                })
    # unit(숫자 기준), pid(숫자 기준) 정렬
    def _sort_key(p):
        unit_num = int(re.search(r'\d+', p["unit"]).group()) if re.search(r'\d+', p["unit"]) else 0
        pid_num = int(re.search(r'\d+', p["id"]).group()) if re.search(r'\d+', p["id"]) else 999
        return (unit_num, pid_num)
    result.sort(key=_sort_key)
    return result


# @app.post("/api/passages/upload")  # → first_round_workbook.passages 라우터로 이전 (기존 코드 보존, 나중 삭제)
async def upload_passages(request: Request):
    _verify(request)
    body = await request.json()
    book = (body.get("book") or "").strip()
    text = body.get("text") or ""

    if not book:
        raise HTTPException(400, "book 필요")
    if not text.strip():
        raise HTTPException(400, "text 필요")

    parts = re.split(r"###(.+?)###", text)
    db = await _load_db()
    db.setdefault("books", {})
    db["books"].setdefault(book, {"units": {}})

    count = 0
    last_unit = None
    last_pid = None

    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        passage = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if not passage:
            continue

        # ★ ###해석### → 이전 지문의 passage_text에 합침
        if title == '해석' and last_unit and last_pid:
            prev = db["books"][book]["units"][last_unit]["passages"].get(last_pid)
            if prev:
                prev["text"] = prev["text"] + "\n###해석###\n" + passage
                print(f"[upload] 해석 → {last_unit}/{last_pid} 에 합침")
            continue

        # 다양한 교재 형식 매칭
        m = re.match(
            r"(\d+강|\d+과|Lesson\s*\d+|L\d+|Chapter\s*\d+|Unit\s*\d+|\d+단원|SL)\s*(.*)",
            title,
            re.IGNORECASE,
        )
        unit_name = m.group(1).strip() if m else "etc"
        pid = m.group(2).strip() if (m and m.group(2).strip()) else title

        db["books"][book]["units"].setdefault(unit_name, {"passages": {}})
        db["books"][book]["units"][unit_name]["passages"][pid] = {"title": title, "text": passage}
        last_unit = unit_name
        last_pid = pid
        count += 1

    await _save_db(db)
    print(f"[upload] saved ({count} passages) book='{book}'")
    return {"ok": True, "count": count}


# @app.delete("/api/passages")  # → first_round_workbook.passages 라우터로 이전 (기존 코드 보존, 나중 삭제)
async def delete_passage_api(request: Request):
    """개별 지문 삭제"""
    _verify(request)
    body = await request.json()

    # 프론트 deletePassage()는 {book, unit, pid}로 보냄
    book = body.get("book")
    unit = body.get("unit")
    pid = body.get("pid")
    if not all([book, unit, pid]):
        raise HTTPException(400, "book, unit, pid 필요")

    db = await _load_db()
    try:
        del db["books"][book]["units"][unit]["passages"][pid]
        # 빈 단원/교재 정리
        if not db["books"][book]["units"][unit]["passages"]:
            del db["books"][book]["units"][unit]
        if not db["books"][book]["units"]:
            del db["books"][book]
    except Exception:
        raise HTTPException(404, "passage not found")

    await _save_db(db)

    # 로컬 캐시도 삭제
    ck = _ck(book, unit, pid)
    cache_dir = DATA_DIR / ck
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"[cache] deleted local cache dir {ck}")

    # Supabase passage row 삭제 (best-effort)
    try:
        if supa._enabled():
            await supa.delete_passage(book, unit, pid)
    except Exception as e:
        print(f"[supa] delete passage error: {e}")

    return {"ok": True}


# @app.delete("/api/books")  # → first_round_workbook.books 라우터로 이전 (기존 코드 보존, 나중 삭제)
async def delete_book_api(request: Request):
    """교재 전체 삭제"""
    _verify(request)
    body = await request.json()
    book = body.get("book")
    if not book:
        raise HTTPException(400, "book 필요")

    db = await _load_db()
    if book not in db.get("books", {}):
        raise HTTPException(404, "book not found")

    # 로컬 캐시도 삭제
    for unit, ud in db["books"][book].get("units", {}).items():
        for pid in ud.get("passages", {}).keys():
            ck = _ck(book, unit, pid)
            cache_dir = DATA_DIR / ck
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"[cache] deleted all local cache for book '{book}'")

    del db["books"][book]
    await _save_db(db)

    # Supabase에서도 삭제 (best-effort)
    try:
        if supa._enabled():
            await supa.delete_book(book)
    except Exception as e:
        print(f"[supa] delete book error: {e}")

    return {"ok": True}
'''
# ===== [리팩토링 중복 코드] 끝 (passages / books) =====


@app.post("/api/sync-supabase")
async def sync_supabase(request: Request):
    """로컬 DB를 수파베이스에 강제 동기화"""
    _verify(request)
    try:
        if not supa._enabled():
            return {"ok": False, "error": "Supabase not enabled"}

        db = await _load_db()

        rows = []
        for bk, bd in db.get("books", {}).items():
            for unit, ud in bd.get("units", {}).items():
                for pid, pi in ud.get("passages", {}).items():
                    rows.append({
                        "book": bk,
                        "unit": unit,
                        "pid": pid,
                        "title": pi.get("title", pid),
                        "passage_text": pi.get("text", ""),
                    })

        if not rows:
            return {"ok": True, "count": 0, "total": 0}

        batch_size = 50
        success = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            result = await supa.upsert_passages_bulk(batch)
            if isinstance(result, list):
                success += len(result)

        return {"ok": True, "count": success, "total": len(rows)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/clear-cache")
async def clear_cache(request: Request):
    """특정 교재/지문의 step 캐시 삭제 (로컬 + Supabase step_cache 같이 삭제)"""
    _verify(request)
    body = await request.json()

    book = body.get("book")
    unit = body.get("unit")
    pid = body.get("passage_id")
    scope = body.get("scope", "all")  # "all" = 교재 전체, "passage" = 특정 지문

    deleted_local = 0
    deleted_supa_targets = 0  # 몇 개 cache_key를 대상으로 supa delete 요청했는지(카운트용)

    # supabase helper (없어도 서버가 죽지 않게)
    try:
        pass
    except Exception:
        supa = None

    if scope == "passage" and all([book, unit, pid]):
        ck = _ck(book, unit, pid)

        # 로컬 step*.json 삭제
        cache_dir = DATA_DIR / ck
        if cache_dir.exists():
            for f in cache_dir.glob("step*.json"):
                try:
                    f.unlink()
                    deleted_local += 1
                except Exception:
                    pass
            print(f"[cache] deleted {deleted_local} local cache files for {ck}")

        # Supabase step_cache 삭제
        try:
            if supa and supa._enabled():
                await supa.delete_steps_by_cache_key(ck)
                deleted_supa_targets += 1
                print(f"[cache] deleted supabase step_cache for {ck}")
        except Exception as e:
            print(f"[cache] supabase delete error: {e}")

    elif scope == "all" and book:
        db = await _load_db()
        if book in db.get("books", {}):
            for u, ud in db["books"][book].get("units", {}).items():
                for p in ud.get("passages", {}).keys():
                    ck = _ck(book, u, p)

                    # 로컬 삭제
                    cache_dir = DATA_DIR / ck
                    if cache_dir.exists():
                        for f in cache_dir.glob("step*.json"):
                            try:
                                f.unlink()
                                deleted_local += 1
                            except Exception:
                                pass

                    # Supabase 삭제
                    try:
                        if supa and supa._enabled():
                            await supa.delete_steps_by_cache_key(ck)
                            deleted_supa_targets += 1
                    except Exception as e:
                        print(f"[cache] supabase delete error: {e}")

        print(f"[cache] deleted {deleted_local} local cache files for book '{book}'")
        if deleted_supa_targets:
            print(f"[cache] supabase step_cache delete targets: {deleted_supa_targets}")

    else:
        raise HTTPException(400, "book 필요")

    return {
        "ok": True,
        "deleted": deleted_local,
        "supa_targets": deleted_supa_targets,
    }


# ===== [리팩토링 중복 코드] first_round_workbook.generation 로 이전됨 — 비활성 보존 (아래 블록 전체, 다음 라운드에 삭제) =====
r'''
# @app.post("/api/generate")  # → first_round_workbook.generation 라우터로 이전 (기존 코드 보존, 나중 삭제)
async def generate(request: Request):
    _verify(request)
    body = await request.json()

    book = body.get("book")
    unit = body.get("unit")
    pid = body.get("passage_id")
    levels = body.get("levels")

    if not all([book, unit, pid]):
        raise HTTPException(400, "book, unit, passage_id 필요")

    db = await _load_db()

    try:
        pinfo = db["books"][book]["units"][unit]["passages"][pid]
    except Exception as e:
        print(f"[generate] passage not found: {e}")
        raise HTTPException(404, f"passage not found: book={book}, unit={unit}, pid={pid}")

    passage_text = pinfo.get("text", "")
    title = pinfo.get("title", pid)

    m = re.match(r"(\d+)", unit or "")
    lesson_num = m.group(1) if m else "00"

    ck = _ck(book, unit, pid)

    try:
        import pipeline as pl

        pl.DATA_DIR = DATA_DIR
        pl.TEMPLATE_DIR = Path(".")
        pl.OUTPUT_DIR = Path("output")
        pl.OUTPUT_DIR.mkdir(exist_ok=True)

        meta = {
            "lesson_num": lesson_num,
            "lesson_n": lesson_num,
            "challenge_title": title,
            "subject": book,
        }

        result_path = pl.process_passage(
            passage=passage_text,
            meta=meta,
            passage_id=ck,
            levels=levels,
        )

        if result_path:
            hp = result_path.with_suffix(".html") if result_path.suffix != ".html" else result_path
            if hp.exists():
                return {"ok": True, "html": hp.read_text(encoding="utf-8"), "filename": hp.name}

        raise HTTPException(500, "generation failed")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
'''
# ===== [리팩토링 중복 코드] 끝 (generation) =====



@app.get("/api/notice")
async def get_notice():
    """공지사항 조회"""
    import json
    notice_file = DATA_DIR / "notice.json"
    if notice_file.exists():
        return json.loads(notice_file.read_text(encoding="utf-8"))
    return {"text": "", "updated_at": ""}

@app.post("/api/notice")
async def set_notice(request: Request):
    """공지사항 저장"""
    _verify(request)
    import json
    from datetime import datetime
    body = await request.json()
    text = body.get("text", "").strip()
    data = {"text": text, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
    (DATA_DIR / "notice.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return {"ok": True}

# ============================================================
# ★ 비밀노트 엔드포인트 - 추가 코드
# ============================================================
@app.post("/api/secret-note")
async def secret_note(request: Request):
    """비밀노트 / 0회독 생성: type A (한국어 종합) / B (영어 중심) / C (어휘+분석) / D (0회독 수업 전 4페이지 분석)
    
    ★ 에러 시 JSON 응답으로 반환 (Internal Server Error HTML 방지) — 프론트에 명확한 에러 메시지 표시
    """
    _verify(request)
    try:
        body = await request.json()
        return await _secret_note_impl(body)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        # 서버 로그 — 어디서 깨졌는지 정확히 파악용
        print(f"[/api/secret-note ERROR] {type(e).__name__}: {e}\n{tb}", flush=True)
        # 프론트 — 한 줄 요약만
        last_line = tb.strip().splitlines()[-1] if tb else ""
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]} ({last_line[:150]})"}


async def _secret_note_impl(body: dict):
    """secret-note 실제 처리 로직 (예외 처리 분리)"""
    note_type    = (body.get("type") or "B").upper()          # "A", "B", "C", or "D"
    school_name  = (body.get("school_name") or "레벨미업학원").strip()
    teacher_name = (body.get("teacher_name") or "").strip()   # 강사명 (선택)
    passages_in  = body.get("passages") or []
    # ★ 0회독(D) 전용 옵션: "preclass"(기본·0회독만) / "both"(0회독+순서배열) / "sentence_order"(순서배열만)
    pc_mode      = (body.get("mode") or "preclass").lower()
    # passages_in: [{"book":"...", "unit":"...", "id":"..."}]

    if not passages_in:
        raise HTTPException(400, "passages 필요")

    db = await _load_db()

    import pipeline as pl
    pl.DATA_DIR = DATA_DIR
    pl.TEMPLATE_DIR = Path(".")
    import topic_background as tbk   # ★ 수업배경자료 렌더러
    import tb_generate as tbg          # ★ 수업배경자료 자동 생성(web_search)

    passages_data = []
    for p in passages_in:
        book, unit, pid = p.get("book"), p.get("unit"), p.get("id")
        if not all([book, unit, pid]):
            continue
        try:
            pinfo = db["books"][book]["units"][unit]["passages"][pid]
        except Exception:
            continue

        raw_text = pinfo.get("text", "")
        label = f"{unit} {pid}"
        ck = _ck(book, unit, pid)
        passage_dir = DATA_DIR / ck

        # ###해석### 구분자로 영어만 추출 (변형 없이 그대로)
        if "###해석###" in raw_text:
            parts = raw_text.split("###해석###", 1)
            passage_text = parts[0].strip()
            translation = "\n".join(l.strip() for l in parts[1].strip().splitlines() if l.strip())
        else:
            passage_text = raw_text.strip()
            translation = ""
            # step1 캐시에서 번역 가져오기 (유형A 프롬프트용)
            try:
                s1 = pl.load_step(passage_dir, "step1_basic")
                if s1:
                    translation = s1.get("translation", "")
            except Exception:
                pass

        if note_type == "A":
            note_data = pl.generate_secret_note_a(passage_text, translation, passage_dir)
        elif note_type == "C":
            note_data = pl.generate_secret_note_c(passage_text, passage_dir, translation)
        elif note_type == "D":
            if pc_mode == "topic":
                note_data = None   # ★ 수업배경자료는 LLM 생성 안 함 (조각 조립)
            else:
                # 유형 D — 0회독 (수업 전 4페이지 완전 분석)
                note_data = pl.generate_preclass_analysis(passage_text, passage_dir, translation)
        else:  # B가 기본
            note_data = pl.generate_secret_note_b(passage_text, passage_dir, translation)

        item = {
            "label":       label,
            "passage":     passage_text,
            "translation": translation,
            "data":        note_data,
        }
        if note_type == "D" and pc_mode == "topic":
            # 캐시 우선, 없으면 web_search 기반 자동 생성
            try:
                item["topic"] = tbg.generate_topic_background(
                    passage_text, passage_dir, label=label,
                    save_step_fn=pl.save_step, load_step_fn=pl.load_step,
                    step_name=tbk.TOPIC_STEP_NAME, max_uses=5,
                )
            except Exception as e:
                item["topic"] = None
                item["topic_error"] = str(e)[:200]
        passages_data.append(item)

    if not passages_data:
        raise HTTPException(404, "처리 가능한 지문 없음")

    # 유형 D는 별도 템플릿(preclass_analysis.html) + 다른 파일명
    if note_type == "D":
        # ★ pc_mode에 따라 render 옵션 다르게 호출
        if pc_mode == "topic":   # ★ 수업배경자료 (캐시 or 자동 생성, web_search)
            html = tbk.render_topic_background(passages_data, school_name)
            return {"ok": True, "html": html, "filename": "수업배경자료.html"}
        if pc_mode == "sentence_order":
            html = pl.render_preclass_analysis(passages_data, school_name,
                                                sentence_order_only=True)
            filename = "한문장순서배열.html"
        elif pc_mode == "both":
            html = pl.render_preclass_analysis(passages_data, school_name,
                                                include_sentence_order=True)
            filename = "0회독_수업전분석_순서배열포함.html"
        else:  # "preclass" (기본 — 0회독만)
            html = pl.render_preclass_analysis(passages_data, school_name)
            filename = "0회독_수업전분석.html"
        return {"ok": True, "html": html, "filename": filename}

    html = pl.render_secret_note(passages_data, note_type, school_name, teacher_name)
    return {"ok": True, "html": html, "filename": f"비밀노트_유형{note_type}.html"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
