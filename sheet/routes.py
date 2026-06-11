#!/usr/bin/env python3
"""
sheet_routes.py  ―  분석지(three_modes) 라우트.
main.py 에 그대로 붙이거나, import 해서 라우터로 등록.

[중요] three_modes 의 HTML/CSS/폰트는 한 글자도 안 바꾼다.
       이 파일은 "데이터 주입 + 저장/공유" 만 담당.

[엔드포인트]
  GET  /sheet?book=&unit=&pid=     → 분석지 저작 화면 (three_modes HTML + 0회독 데이터 주입)
  POST /api/sheet/save             → sheet_cache 에 저장 (sheet_save RPC)
  POST /api/sheet/publish          → 공유 토큰 발급 (sheet_publish RPC)
  GET  /s/{token}                  → 학생 게시물 (읽기 전용, 저장된 분석지)

[연결 전제]
  - sheet_cache 테이블 + sheet_save/sheet_publish RPC 는 이미 Supabase 에 있음.
  - three_modes 템플릿 파일은 sheet/sheet_three_modes.html 로 배치
    (원본 그대로. <!--SHEET_DATA_INJECT--> 주석 한 줄만 추가해 두면 됨)
  - 0회독 데이터는 step_cache(preclass_analysis_v24) 에서 읽음.

[main.py 에 추가하는 법]
  from sheet_routes import router as sheet_router
  app.include_router(sheet_router)
  ↑ 기존 variation/seosul 등록과 동일 패턴. _ck/_verify/_load_db 는
    main.py 의 것을 그대로 재사용하도록 di(아래 set_deps)로 주입.
"""
import os
import json
import hashlib
import re
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse

from . import convert as sc

router = APIRouter()

# ── main.py 의 함수/상수를 주입받는다 (순환 import 방지) ──
_DEPS = {
    "ck": None,          # _ck(book,unit,pid) -> cache_key
    "verify": None,      # _verify(request)
    "load_db": None,     # async _load_db() -> {"books":{...}}
    "data_dir": None,    # Path("data")
    "template_dir": Path("sheet"),
}


def set_deps(*, ck, verify, load_db, data_dir, template_dir=None):
    _DEPS["ck"] = ck
    _DEPS["verify"] = verify
    _DEPS["load_db"] = load_db
    _DEPS["data_dir"] = data_dir
    if template_dir:
        _DEPS["template_dir"] = Path(template_dir)


# ════════════════════════════════════════════════════════════════
# Supabase REST 헬퍼 (httpx) — supa 모듈 있으면 그걸 쓰고, 없으면 직접 호출
# ════════════════════════════════════════════════════════════════
def _supa_headers():
    key = os.environ.get("SUPABASE_KEY", "") or os.environ.get("SUPABASE_SERVICE_KEY", "")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def _supa_url():
    return os.environ.get("SUPABASE_URL", "").rstrip("/")


def _load_preclass(cache_key: str) -> dict | None:
    """step_cache(preclass_analysis_v24) 의 data 읽기. 로컬 우선, 없으면 Supabase REST."""
    # 1) 로컬 파일
    dd = _DEPS["data_dir"] or Path("data")
    local = Path(dd) / cache_key / "preclass_analysis_v24.json"
    if local.exists():
        try:
            return json.loads(local.read_text(encoding="utf-8"))
        except Exception:
            pass
    # 2) Supabase REST
    url, base = _supa_url(), None
    if url:
        try:
            import httpx
            ep = (f"{url}/rest/v1/step_cache"
                  f"?cache_key=eq.{cache_key}&step_name=eq.preclass_analysis_v24"
                  f"&select=data&limit=1")
            with httpx.Client(timeout=10) as c:
                r = c.get(ep, headers=_supa_headers())
            if r.status_code == 200:
                rows = r.json()
                if rows and isinstance(rows, list):
                    return rows[0].get("data")
        except Exception as e:
            print(f"[sheet] preclass load error: {str(e)[:120]}")
    return None


def _load_sheet(cache_key: str = None, token: str = None) -> dict | None:
    """sheet_cache 에서 저장된 분석지 읽기."""
    url = _supa_url()
    if not url:
        return None
    try:
        import httpx
        if token:
            ep = (f"{url}/rest/v1/sheet_cache"
                  f"?share_token=eq.{token}&select=sheet,cache_key,book,unit,pid&limit=1")
        else:
            ep = (f"{url}/rest/v1/sheet_cache"
                  f"?cache_key=eq.{cache_key}&select=sheet,cache_key,book,unit,pid&limit=1")
        with httpx.Client(timeout=10) as c:
            r = c.get(ep, headers=_supa_headers())
        if r.status_code == 200:
            rows = r.json()
            if rows and isinstance(rows, list):
                return rows[0]
    except Exception as e:
        print(f"[sheet] sheet load error: {str(e)[:120]}")
    return None


def _rpc(fn: str, payload: dict) -> dict | None:
    url = _supa_url()
    if not url:
        raise HTTPException(500, "SUPABASE_URL 미설정")
    try:
        import httpx
        ep = f"{url}/rest/v1/rpc/{fn}"
        with httpx.Client(timeout=15) as c:
            r = c.post(ep, headers=_supa_headers(), json=payload)
        if r.status_code in (200, 201, 204):
            try:
                return r.json()
            except Exception:
                return {"ok": True}
        raise HTTPException(500, f"{fn} 실패: {r.status_code} {r.text[:150]}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"{fn} 예외: {str(e)[:150]}")


# ════════════════════════════════════════════════════════════════
# 지문 원문/번역 가져오기
# ════════════════════════════════════════════════════════════════
async def _get_passage_text(book, unit, pid):
    """passages 의 passage_text → (english, translation)."""
    load_db = _DEPS["load_db"]
    db = await load_db()
    try:
        pinfo = db["books"][book]["units"][unit]["passages"][pid]
    except Exception:
        return "", ""
    raw = pinfo.get("text", "") or ""
    if "###해석###" in raw:
        eng, kr = raw.split("###해석###", 1)
        kr = "\n".join(l.strip() for l in kr.strip().splitlines() if l.strip())
        return eng.strip(), kr
    return raw.strip(), ""


# ════════════════════════════════════════════════════════════════
# HTML 주입 — three_modes 원본에 SHEET_DATA 만 끼워넣기
# ════════════════════════════════════════════════════════════════
def _inject(html: str, sheet_data: dict, mode: str = "auth", readonly: bool = False) -> str:
    """three_modes HTML 에 전역 데이터 주입.
    원본 HTML 은 <!--SHEET_DATA_INJECT--> 한 줄만 있으면 됨.
    없으면 </head> 앞에 자동 삽입.
    """
    payload = {
        "S": sheet_data.get("S", []),
        "FLOW": sheet_data.get("FLOW", []),
        "TITLES": sheet_data.get("TITLES", []),
        "TOPICS": sheet_data.get("TOPICS", []),
        "DIFF": sheet_data.get("DIFF", 3),
        "VPOOL": sheet_data.get("VPOOL", {}),
        "POOL": sheet_data.get("POOL", {}),
        "SEL": sheet_data.get("_SEL", {}),
        "MODE": mode,
        "READONLY": readonly,
    }
    snippet = (
        "<script>\n"
        "window.SHEET_DATA = " + json.dumps(payload, ensure_ascii=False) + ";\n"
        "</script>\n"
    )
    if "<!--SHEET_DATA_INJECT-->" in html:
        return html.replace("<!--SHEET_DATA_INJECT-->", snippet)
    if "</head>" in html:
        return html.replace("</head>", snippet + "</head>", 1)
    return snippet + html


def _read_template() -> str:
    tdir = _DEPS["template_dir"] or Path("sheet")
    p = Path(tdir) / "sheet_three_modes.html"
    if not p.exists():
        raise HTTPException(500, f"분석지 템플릿 없음: {p}")
    return p.read_text(encoding="utf-8")


# ════════════════════════════════════════════════════════════════
# 라우트
# ════════════════════════════════════════════════════════════════
@router.get("/sheet", response_class=HTMLResponse)
async def sheet_page(request: Request, book: str = "", unit: str = "", pid: str = ""):
    """분석지 저작 화면. 0회독 데이터 + 저장본을 three_modes 에 주입해 반환."""
    # /sheet 자체는 로그인 화면 안에서 fetch 로 부르는 게 아니라
    # 새 탭으로 열리므로 토큰 검증을 쿼리/세션 대신 그대로 둠.
    # (민감정보 없음 — 어차피 같은 지문 분석 데이터. 필요하면 _verify 추가)
    if not all([book, unit, pid]):
        raise HTTPException(400, "book, unit, pid 필요")

    ck = _DEPS["ck"](book, unit, pid)
    pre = _load_preclass(ck)
    if not pre:
        raise HTTPException(404, f"0회독 분석이 아직 없습니다. 먼저 0회독을 생성하세요. (key={ck})")

    eng, kr = await _get_passage_text(book, unit, pid)

    saved_row = _load_sheet(cache_key=ck)
    saved_sheet = (saved_row or {}).get("sheet") if saved_row else None

    data = sc.build_sheet_data(pre, translation=kr, saved_sheet=saved_sheet)
    html = _read_template()
    return _inject(html, data, mode="auth", readonly=False)


@router.post("/api/sheet/save")
async def sheet_save(request: Request):
    """분석지 편집본 저장 → sheet_cache upsert."""
    _DEPS["verify"](request)
    body = await request.json()
    book = body.get("book"); unit = body.get("unit"); pid = body.get("pid")
    sheet = body.get("sheet")
    if not all([book, unit, pid]):
        raise HTTPException(400, "book, unit, pid 필요")
    if sheet is None:
        raise HTTPException(400, "sheet 필요")
    ck = _DEPS["ck"](book, unit, pid)
    _rpc("sheet_save", {
        "p_cache_key": ck, "p_book": book, "p_unit": unit,
        "p_pid": pid, "p_sheet": sheet,
    })
    return {"ok": True, "cache_key": ck}


@router.post("/api/sheet/publish")
async def sheet_publish(request: Request):
    """공유 토큰 발급 → /s/{token} 으로 학생 배포."""
    _DEPS["verify"](request)
    body = await request.json()
    book = body.get("book"); unit = body.get("unit"); pid = body.get("pid")
    if not all([book, unit, pid]):
        raise HTTPException(400, "book, unit, pid 필요")
    ck = _DEPS["ck"](book, unit, pid)
    res = _rpc("sheet_publish", {"p_cache_key": ck})
    # res 는 share_token (문자열 or [{share_token}] 형태)
    token = None
    if isinstance(res, str):
        token = res
    elif isinstance(res, list) and res:
        token = res[0].get("share_token") if isinstance(res[0], dict) else res[0]
    elif isinstance(res, dict):
        token = res.get("share_token") or res.get("sheet_publish")
    return {"ok": True, "token": token, "url": f"/s/{token}" if token else None}


@router.get("/s/{token}", response_class=HTMLResponse)
async def sheet_student(token: str):
    """학생 게시물 (읽기 전용). 저장된 분석지를 학생 모드로 렌더."""
    row = _load_sheet(token=token)
    if not row:
        raise HTTPException(404, "공유된 분석지를 찾을 수 없습니다.")
    ck = row.get("cache_key")
    book, unit, pid = row.get("book"), row.get("unit"), row.get("pid")
    pre = _load_preclass(ck)
    if not pre:
        raise HTTPException(404, "원본 분석 데이터가 없습니다.")

    # 원문/번역
    eng, kr = "", ""
    try:
        eng, kr = await _get_passage_text(book, unit, pid)
    except Exception:
        pass

    data = sc.build_sheet_data(pre, translation=kr, saved_sheet=row.get("sheet"))
    html = _read_template()
    # 학생 모드 + 읽기전용
    return _inject(html, data, mode="student", readonly=True)
