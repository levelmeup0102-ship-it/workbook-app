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
from .renderer import render_seosul_set

from core.settings import settings

OUTPUT_DIR = settings.SEOSUL_OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

APP_PASSWORD = settings.APP_PASSWORD

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


router = APIRouter(prefix="/api", tags=["seosul"])
download_router = APIRouter(prefix="/api", tags=["seosul-dl"])


@router.post("/seosul")
def create_seosul(req: SeosulRequest, _=Depends(verify_token)):
    gp = fetch_grammar_points()
    stypes = fetch_seosul_types()

    sections, warnings = [], []
    for p in req.passages:
        try:
            s = generate_set(p.book, p.unit, p.id, req.types, gp, stypes)
            sections.append(render_seosul_set(s, teacher=False, school_name=req.school_name))
        except Exception as e:
            warnings.append(f"{p.book} {p.unit} {p.id}: {e}")
            traceback.print_exc()

    if not sections:
        raise HTTPException(status_code=500, detail="생성 실패: " + "; ".join(warnings))

    html = sections[0] if len(sections) == 1 else _merge(sections)
    uid = uuid.uuid4().hex[:12]
    with open(os.path.join(OUTPUT_DIR, f"{uid}.html"), "w", encoding="utf-8") as f:
        f.write(html)

    return {
        "ok": True,
        "passages_generated": len(sections),
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
