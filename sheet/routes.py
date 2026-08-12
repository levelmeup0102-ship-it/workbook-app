#!/usr/bin/env python3
"""
sheet_routes.py  ―  분석지(three_modes) 라우트.
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

_DEPS = {
    "ck": None,
    "verify": None,
    "load_db": None,
    "data_dir": None,
    "template_dir": Path("sheet"),
}


def set_deps(*, ck, verify, load_db, data_dir, template_dir=None):
    _DEPS["ck"] = ck
    _DEPS["verify"] = verify
    _DEPS["load_db"] = load_db
    _DEPS["data_dir"] = data_dir
    if template_dir:
        _DEPS["template_dir"] = Path(template_dir)


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
    dd = _DEPS["data_dir"] or Path("data")
    local = Path(dd) / cache_key / "preclass_analysis_v24.json"
    if local.exists():
        try:
            return json.loads(local.read_text(encoding="utf-8"))
        except Exception:
            pass
    url = _supa_url()
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


def _load_sheet(cache_key: str = None, token: str = None, teacher: str = "") -> dict | None:
    url = _supa_url()
    if not url:
        return None
    try:
        import httpx
        from urllib.parse import quote
        if token:
            ep = (f"{url}/rest/v1/sheet_cache"
                  f"?share_token=eq.{token}&select=sheet,cache_key,book,unit,pid,teacher&limit=1")
        else:
            ep = (f"{url}/rest/v1/sheet_cache"
                  f"?cache_key=eq.{cache_key}&teacher=eq.{quote(teacher or '', safe='')}"
                  f"&select=sheet,cache_key,book,unit,pid,teacher&limit=1")
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


async def _get_passage_text(book, unit, pid):
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


def _inject(html: str, sheet_data: dict, mode: str = "auth", readonly: bool = False, teacher: str = "") -> str:
    payload = {
        "S": sheet_data.get("S", []),
        "FLOW": sheet_data.get("FLOW", []),
        "TITLES": sheet_data.get("TITLES", []),
        "TOPICS": sheet_data.get("TOPICS", []),
        "DIFF": sheet_data.get("DIFF", 3),
        "VPOOL": sheet_data.get("VPOOL", {}),
        "POOL": sheet_data.get("POOL", {}),
        "SEL": sheet_data.get("_SEL", {}),
        "MARKS": sheet_data.get("MARKS", []),
        "MODE": mode,
        "READONLY": readonly,
        "TEACHER": teacher,
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


@router.get("/sheet", response_class=HTMLResponse)
async def sheet_page(request: Request, book: str = "", unit: str = "", pid: str = "", teacher: str = ""):
    if not all([book, unit, pid]):
        raise HTTPException(400, "book, unit, pid 필요")

    ck = _DEPS["ck"](book, unit, pid)
    pre = _load_preclass(ck)
    eng, kr = await _get_passage_text(book, unit, pid)

    teacher = (teacher or "").strip()
    saved_row = _load_sheet(cache_key=ck, teacher=teacher)
    saved_sheet = (saved_row or {}).get("sheet") if saved_row else None

    if pre:
        # 0회독(자동 분석)이 있으면 그걸 소스로
        data = sc.build_sheet_data(pre, translation=kr, saved_sheet=saved_sheet)
    elif (eng or "").strip():
        # ★ 0회독이 없어도 원문만으로 '빈 분석지'를 연다 (선생님이 직접 강조/편집)
        data = sc.build_blank_sheet_data(eng, translation=kr, saved_sheet=saved_sheet)
    else:
        # 원문조차 없는 경우에만 막는다
        raise HTTPException(404, f"이 지문의 원문이 없습니다. 먼저 지문을 업로드하세요. (key={ck})")

    html = _read_template()
    return _inject(html, data, mode="auth", readonly=False, teacher=teacher)


@router.post("/api/sheet/save")
async def sheet_save(request: Request):
    _DEPS["verify"](request)
    body = await request.json()
    book = body.get("book"); unit = body.get("unit"); pid = body.get("pid")
    sheet = body.get("sheet")
    teacher = (body.get("teacher") or "").strip()
    if not all([book, unit, pid]):
        raise HTTPException(400, "book, unit, pid 필요")
    if sheet is None:
        raise HTTPException(400, "sheet 필요")
    ck = _DEPS["ck"](book, unit, pid)
    _rpc("sheet_save", {
        "p_cache_key": ck, "p_book": book, "p_unit": unit,
        "p_pid": pid, "p_sheet": sheet, "p_teacher": teacher,
    })
    return {"ok": True, "cache_key": ck, "teacher": teacher}


@router.post("/api/sheet/publish")
async def sheet_publish(request: Request):
    _DEPS["verify"](request)
    body = await request.json()
    book = body.get("book"); unit = body.get("unit"); pid = body.get("pid")
    teacher = (body.get("teacher") or "").strip()
    if not all([book, unit, pid]):
        raise HTTPException(400, "book, unit, pid 필요")
    ck = _DEPS["ck"](book, unit, pid)
    res = _rpc("sheet_publish", {"p_cache_key": ck, "p_teacher": teacher})
    token = None
    if isinstance(res, str):
        token = res
    elif isinstance(res, list) and res:
        token = res[0].get("share_token") if isinstance(res[0], dict) else res[0]
    elif isinstance(res, dict):
        token = res.get("share_token") or res.get("sheet_publish")
    return {"ok": True, "token": token, "teacher": teacher,
            "url": f"/s/{token}" if token else None}


@router.get("/api/sheet/teachers")
async def sheet_teachers(book: str = "", unit: str = "", pid: str = ""):
    """이 지문에 저장본이 있는 강사명 목록 (저작모드 드롭다운용)."""
    if not all([book, unit, pid]):
        raise HTTPException(400, "book, unit, pid 필요")
    ck = _DEPS["ck"](book, unit, pid)
    res = _rpc("sheet_teachers", {"p_cache_key": ck})
    rows = res if isinstance(res, list) else []
    return {"ok": True, "cache_key": ck, "teachers": rows}


# ════════════════════════════════════════════════════════════════
# ★ AI 동의어/반의어 — 서버에서 Anthropic 호출 (브라우저 직접 호출 금지)
#   three_modes 의 "+ AI로 동의어 받기" 버튼이 이 엔드포인트를 부른다.
#   0회독에 없는 단어만 여기로 옴 (있는 단어는 VPOOL 에서 자동으로 뜸).
# ════════════════════════════════════════════════════════════════
@router.post("/api/sheet/ai-synonym")
async def sheet_ai_synonym(request: Request):
    _DEPS["verify"](request)
    body = await request.json()
    word = (body.get("word") or "").strip()
    sentence = (body.get("sentence") or "").strip()
    if not word:
        raise HTTPException(400, "word 필요")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(500, "ANTHROPIC_API_KEY 미설정")

    prompt = (
        f'영어 단어 "{word}"'
        + (f' (문맥 문장: "{sentence}")' if sentence else "")
        + '에 대해 한국 고교 영어 어휘 학습용 동의어와 반의어를 제시하라. '
        '반드시 JSON만, 마크다운/설명 없이: '
        '{"syn":[["english","한국어뜻"]...최대6],"ant":[["english","한국어뜻"]...최대3]}'
    )

    try:
        import httpx
        with httpx.Client(timeout=30) as c:
            r = c.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
        if r.status_code != 200:
            raise HTTPException(502, f"AI 호출 실패: {r.status_code} {r.text[:150]}")
        d = r.json()
        text = "".join(
            blk.get("text", "")
            for blk in (d.get("content") or [])
            if blk.get("type") == "text"
        )
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        # 형태 정규화
        syn = [[str(x[0]), str(x[1])] for x in (parsed.get("syn") or [])
               if isinstance(x, (list, tuple)) and len(x) >= 2]
        ant = [[str(x[0]), str(x[1])] for x in (parsed.get("ant") or [])
               if isinstance(x, (list, tuple)) and len(x) >= 2]
        return {"ok": True, "syn": syn, "ant": ant}
    except HTTPException:
        raise
    except json.JSONDecodeError:
        raise HTTPException(502, "AI 응답 파싱 실패")
    except Exception as e:
        raise HTTPException(502, f"AI 예외: {str(e)[:150]}")


@router.get("/s/{token}", response_class=HTMLResponse)
async def sheet_student(token: str):
    row = _load_sheet(token=token)
    if not row:
        raise HTTPException(404, "공유된 분석지를 찾을 수 없습니다.")
    ck = row.get("cache_key")
    book, unit, pid = row.get("book"), row.get("unit"), row.get("pid")
    pre = _load_preclass(ck)

    eng, kr = "", ""
    try:
        eng, kr = await _get_passage_text(book, unit, pid)
    except Exception:
        pass

    if pre:
        data = sc.build_sheet_data(pre, translation=kr, saved_sheet=row.get("sheet"))
    elif (eng or "").strip():
        # ★ 0회독 없이 만든 분석지도 학생 공유 가능 (원문 기반 + 강사 저장본 반영)
        data = sc.build_blank_sheet_data(eng, translation=kr, saved_sheet=row.get("sheet"))
    else:
        raise HTTPException(404, "원본 지문이 없습니다.")
    html = _read_template()
    return _inject(html, data, mode="student", readonly=True)
