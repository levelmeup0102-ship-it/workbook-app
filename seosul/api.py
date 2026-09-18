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
    get_cache_status, delete_cache, type_name,
)
from .renderer import render_fragments, wrap_document, ANS_HDR

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
    force: bool = False             # True면 캐시 무시하고 새로 생성
    cache_only: bool = False        # True면 LLM 호출 없이 캐시만 모아 합본(끊김 방지)


router = APIRouter(prefix="/api", tags=["seosul"])
download_router = APIRouter(prefix="/api", tags=["seosul-dl"])


class PassageListRequest(BaseModel):
    passages: List[PassageRef]


@router.post("/seosul/cache-status")
def seosul_cache_status(req: PassageListRequest, _=Depends(verify_token)):
    """지문별 서술형 캐시 유무 — 탭의 생성됨 / 검증 미통과 / 미생성 표시용."""
    try:
        return {"ok": True,
                "status": get_cache_status([p.model_dump() for p in req.passages])}
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "status": {}, "error": str(e)}


@router.post("/seosul/clear-cache")
def seosul_clear_cache(req: PassageListRequest, _=Depends(verify_token)):
    """선택한 지문의 서술형 캐시만 삭제. 1회독·2회독 캐시는 다른 테이블이라 무관."""
    if not req.passages:
        raise HTTPException(status_code=400, detail="지문이 선택되지 않음")
    try:
        n = delete_cache([p.model_dump() for p in req.passages])
        return {"ok": True, "passages": len(req.passages), "deleted": n}
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "deleted": 0, "error": str(e)}


class SeosulItemRequest(BaseModel):
    book: str
    unit: str
    id: str
    types: List[str] = ["SA", "SC", "SD", "SE"]
    only: Optional[str] = None     # 이번 요청이 맡을 유형 하나 (없으면 지문 통째)
    force: bool = False


@router.post("/seosul/item")
def create_seosul_item(req: SeosulItemRequest, _=Depends(verify_token)):
    """지문 하나 × 유형 하나만 만들어 캐시에 넣는다. HTML 은 만들지 않는다.

    ★ 왜 이렇게까지 잘게 나누나
      지문 전부를 한 요청에 넣으면 20분~1시간이 걸려 프록시가 끊는다('업스트림 에러').
      지문 하나로 줄여도 유형 4개 × 재작성 3회 = LLM 12번이라 10분을 넘겨 또 끊겼다.
      유형 하나면 길어야 3번, 1~3분이면 끝난다.
      2회독 변형문제가 /api/variation/item 을 '지문 하나 × 유형 하나'로 부르는 것과 같다.

    ★ 순서를 지켜야 한다
      본문 빈칸 → 제목 빈칸 → 어법 → 요약. 앞 유형의 결과를 캐시에서 읽어
      같은 문장을 두 번 쓰지 않게 피하기 때문이다. 순서를 섞으면 겹친다.
    """
    gp = fetch_grammar_points()
    stypes = fetch_seosul_types()
    try:
        s = generate_set(req.book, req.unit, req.id, req.types, gp, stypes,
                         use_cache=not req.force, only=req.only)
        if req.only:
            return {"ok": True, "type": type_name(req.only),
                    "done": bool(s.get("_done")),
                    "warnings": s.get("_warnings", [])}
        made = [type_name(it.get("type")) for it in s.get("items", [])]
        return {"ok": True, "made": made,
                "missing": [type_name(t) for t in s.get("_missing", [])],
                "warnings": s.get("_warnings", [])}
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "type": type_name(req.only) if req.only else None,
                "error": str(e)}


@router.post("/seosul")
def create_seosul(req: SeosulRequest, _=Depends(verify_token)):
    gp = fetch_grammar_points()
    stypes = fetch_seosul_types()

    probs, anss, warnings, n_ok = [], [], [], 0
    for p in req.passages:
        try:
            s = generate_set(p.book, p.unit, p.id, req.types, gp, stypes,
                             use_cache=not req.force, cache_only=req.cache_only)
            pr, an = render_fragments(s, teacher=False, school_name=req.school_name)
            probs.append(pr); anss.append(an); n_ok += 1
            # 빠진 유형은 시험지에 '직접 채우는 빈칸 틀'로 들어간다 — 어느 지문 어느 유형인지 알려준다.
            if s.get("_missing"):
                names = ", ".join(type_name(t) for t in s["_missing"])
                warnings.append(f"{p.book} {p.unit} {p.id}: 빈칸으로 남김 → {names} (직접 채워 주세요)")
            for w in s.get("_warnings", []):
                warnings.append(f"{p.book} {p.unit} {p.id}: {w}")
        except Exception as e:
            warnings.append(f"{p.book} {p.unit} {p.id}: {e}")
            traceback.print_exc()

    if n_ok == 0:
        raise HTTPException(status_code=500, detail="생성 실패: " + "; ".join(warnings))

    PB = '<div style="page-break-after:always"></div>'
    if req.answers_at_back:
        # 모든 문제 먼저 → (페이지 분리) → '정답 및 해설' 1회 → 모든 정답 연속
        body = PB.join(probs) + '<div class="ans-start"></div>' + ANS_HDR + "".join(anss)
    else:
        # 지문별로 [문제 → 정답(헤더 1회)] 묶음을 이어붙임
        sections = [pr + '<div class="ans-start"></div>' + ANS_HDR + an for pr, an in zip(probs, anss)]
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
