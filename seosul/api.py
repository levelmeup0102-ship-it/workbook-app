"""
seosul/api.py
서술형 종합 FastAPI 라우터. variation 모듈과 동일한 인증·HTML 서빙·반환 형식.

main.py에 추가:
    from seosul.api import router as seosul_router, download_router as seosul_dl
    app.include_router(seosul_router)
    app.include_router(seosul_dl)
"""
import os
import uuid
import hashlib
import traceback
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from .generator import (
    generate_set, fetch_grammar_points, fetch_seosul_types,
)
from .renderer import render_fragments, wrap_document

OUTPUT_DIR = os.environ.get("SEOSUL_OUTPUT_DIR", "/tmp/seosul_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

APP_PASSWORD = os.environ.get("APP_PASSWORD", "levelmeup2026")

def _token(pw: str) -> str:
    return hashlib.sha256(f"{pw}_wb2026".encode()).hexdigest()[:32]

def verify_token(authorization: Optional[str] = Header(None)):
    if not authorization or authorization.replace("Bearer ", "") != _token(APP_PASSWORD):
        raise HTTPException(status_code=401, detail="Invalid token")
    return True


class PassageRef(BaseModel):
    book: str
    unit: str
    id: str  # pid

class SeosulRequest(BaseModel):
    passages: List[PassageRef]
    types: List[str] = ["SA", "SC", "SD", "SE"]   # 대화문(SB) 제외 기본
    school_name: str = "레벨미업학원"
    answers_at_back: bool = False   # True면 모든 문제 먼저, 정답은 맨 뒤로 모음


router = APIRouter(prefix="/api", tags=["seosul"])
download_router = APIRouter(prefix="/api", tags=["seosul-dl"])


@router.post("/seosul")
def create_seosul(req: SeosulRequest, _=Depends(verify_token)):
    gp = fetch_grammar_points()
    stypes = fetch_seosul_types()

    probs, anss, warnings, n_ok = [], [], [], 0
    for p in req.passages:
        try:
            s = generate_set(p.book, p.unit, p.id, req.types, gp, stypes)
            pr, an = render_fragments(s, teacher=False, school_name=req.school_name)
            probs.append(pr); anss.append(an); n_ok += 1
            for w in s.get("_warnings", []):
                warnings.append(f"{p.book} {p.unit} {p.id}: {w}")
        except Exception as e:
            warnings.append(f"{p.book} {p.unit} {p.id}: {e}")
            traceback.print_exc()

    if n_ok == 0:
        raise HTTPException(status_code=500, detail="생성 실패: " + "; ".join(warnings))

    PB = '<div style="page-break-after:always"></div>'
    if req.answers_at_back:
        # 모든 문제 먼저 → (페이지 분리) → 모든 정답 모음
        body = PB.join(probs) + '<div class="ans-start"></div>' + PB.join(anss)
    else:
        # 지문별로 [문제 → 정답] 묶음을 이어붙임
        sections = [pr + '<div class="ans-start"></div>' + an for pr, an in zip(probs, anss)]
        body = PB.join(sections)
    html = wrap_document(body, req.school_name)

    uid = uuid.uuid4().hex[:12]
    with open(os.path.join(OUTPUT_DIR, f"{uid}.html"), "w", encoding="utf-8") as f:
        f.write(html)

    return {
        "ok": True,
        "passages_generated": n_ok,
        "html_url": f"/api/seosul/view/{uid}",
        "download_url": f"/api/seosul/view/{uid}?download=1",
        "warnings": warnings,
    }


@download_router.get("/seosul/view/{uid}")
def view_seosul(uid: str, download: int = 0):
    path = os.path.join(OUTPUT_DIR, f"{uid}.html")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="not found")
    if download:
        return FileResponse(path, media_type="text/html",
                            filename=f"seosul_{uid}.html")
    return HTMLResponse(open(path, encoding="utf-8").read())


def _merge(sections: List[str]) -> str:
    bodies = []
    for sec in sections:
        bodies.append(sec.split("<body>", 1)[1].rsplit("</body>", 1)[0])
    head = sections[0].split("<body>", 1)[0]
    return head + "<body>" + '<div style="page-break-after:always"></div>'.join(bodies) + "</body></html>"
