"""
variation/api.py
변형문제 FastAPI 라우터 - HTML 출력 버전

사용법 (main.py에 추가):
    from variation.api import router as variation_router, download_router
    app.include_router(variation_router)
    app.include_router(download_router)
"""
import os
import uuid
import hashlib
import traceback
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, Header, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from variation.generator import (
    generate_variation_a,
    generate_variation_b,
    fetch_passage_text,
    sb_client,
)
from variation.renderer import (
    render_variation_html,
    prepare_a_passage,
    prepare_b_passage,
)

# ============ 출력 디렉토리 ============
OUTPUT_DIR = os.environ.get("VARIATION_OUTPUT_DIR", "/tmp/variation_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============ 인증 (main.py와 정확히 동일) ============
APP_PASSWORD = os.environ.get("APP_PASSWORD", "levelmeup2026")


def _token(pw: str) -> str:
    """main.py의 _token 함수와 동일"""
    return hashlib.sha256(f"{pw}_wb2026".encode()).hexdigest()[:32]


def verify_token(authorization: Optional[str] = Header(None)):
    """main.py의 _verify와 동일한 로직"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization 헤더 필요")
    got = authorization.replace("Bearer ", "")
    if got != _token(APP_PASSWORD):
        raise HTTPException(status_code=401, detail="Invalid token")
    return got


# ============ 요청/응답 모델 ============
class PassageRef(BaseModel):
    book: str
    unit: str
    id: str  # pid (예: "01번")


class VariationRequest(BaseModel):
    passages: List[PassageRef]
    types: List[str]  # ['A'] / ['B'] / ['A','B']
    mode: str = "by-type"  # "by-type" or "by-passage"
    school_name: str = "레벨미업학원"


class VariationItemRequest(BaseModel):
    book: str
    unit: str
    id: str
    type: str  # 'A' or 'B'


# ============ 라우터 ============
router = APIRouter(prefix="/api", tags=["variation"])


@router.post("/variation/item")
def create_variation_item(
    req: VariationItemRequest,
    _token_val: str = Depends(verify_token),
):
    """
    변형문제 1개 항목(지문 1개 × 유형 1개)만 생성해서 캐시에 적재.
    프런트가 항목을 하나씩 순차 호출 → 각 요청이 짧게 끝나 타임아웃 회피.
    결과 데이터는 step_cache에 저장되고, 최종 /variation 호출 때 캐시 히트로 즉시 합쳐짐.
    """
    if not sb_client:
        raise HTTPException(status_code=500, detail="Supabase 환경변수 미설정")
    if req.type not in ("A", "B"):
        raise HTTPException(status_code=400, detail=f"알 수 없는 유형: {req.type}")

    text = fetch_passage_text(req.book, req.unit, req.id)
    if text is None:
        raise HTTPException(status_code=404, detail=f"지문을 찾을 수 없음: {req.book} / {req.unit} / {req.id}")

    try:
        if req.type == "A":
            generate_variation_a(passage_text=text, pid=req.id, book=req.book, unit=req.unit)
        else:
            generate_variation_b(passage_text=text, pid=req.id, book=req.book, unit=req.unit)
        return {"ok": True, "id": req.id, "type": req.type}
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "id": req.id, "type": req.type, "error": str(e)}


@router.post("/variation")
def create_variation(
    req: VariationRequest,
    _token_val: str = Depends(verify_token),
):
    """
    변형문제 HTML 생성
    
    1. 각 지문 텍스트 조회 (passages 테이블)
    2. 선택된 유형(A/B)별로 데이터 생성 (캐시 활용 + 자동 검증)
    3. HTML 페이지 빌드 → 임시 저장
    4. HTML URL 반환 (브라우저에서 새 창으로 열고 Ctrl+P)
    """
    if not sb_client:
        raise HTTPException(status_code=500, detail="Supabase 환경변수 미설정 (SUPABASE_URL, SUPABASE_SERVICE_KEY)")
    
    if not req.passages:
        raise HTTPException(status_code=400, detail="지문이 선택되지 않음")
    if not req.types:
        raise HTTPException(status_code=400, detail="유형이 선택되지 않음")
    for t in req.types:
        if t not in ("A", "B"):
            raise HTTPException(status_code=400, detail=f"알 수 없는 유형: {t}")
    if req.mode not in ("by-type", "by-passage"):
        raise HTTPException(status_code=400, detail=f"알 수 없는 mode: {req.mode}")
    
    # ----- 1. 지문 원문 -----
    passage_texts = {}
    for p in req.passages:
        text = fetch_passage_text(p.book, p.unit, p.id)
        if text is None:
            raise HTTPException(
                status_code=404,
                detail=f"지문을 찾을 수 없음: {p.book} / {p.unit} / {p.id}"
            )
        passage_texts[(p.book, p.unit, p.id)] = text
    
    # ----- 2. 데이터 생성 -----
    a_items = []
    b_items = []
    errors_per_passage = []
    
    for p in req.passages:
        label = f"{p.book} · {p.unit} {p.id}"
        passage_text = passage_texts[(p.book, p.unit, p.id)]
        
        if "A" in req.types:
            try:
                data_a = generate_variation_a(
                    passage_text=passage_text,
                    pid=p.id, book=p.book, unit=p.unit,
                    cache_only=True,
                )
                if data_a is None:
                    errors_per_passage.append(f"[{p.id}] A 제외 (생성 단계에서 검증 미통과)")
                else:
                    a_items.append(prepare_a_passage(data_a, label))
            except Exception as e:
                traceback.print_exc()
                errors_per_passage.append(f"[{p.id}] A 생성 실패: {e}")
        
        if "B" in req.types:
            try:
                data_b = generate_variation_b(
                    passage_text=passage_text,
                    pid=p.id, book=p.book, unit=p.unit,
                    cache_only=True,
                )
                if data_b is None:
                    errors_per_passage.append(f"[{p.id}] B 제외 (생성 단계에서 검증 미통과)")
                else:
                    b_items.append(prepare_b_passage(data_b, label))
            except Exception as e:
                traceback.print_exc()
                errors_per_passage.append(f"[{p.id}] B 생성 실패: {e}")
    
    if not a_items and not b_items:
        raise HTTPException(
            status_code=500,
            detail="생성된 변형문제가 없습니다.\n" + "\n".join(errors_per_passage)
        )
    
    # ----- 3. HTML 빌드 -----
    try:
        html_text = render_variation_html(
            a_items=a_items, b_items=b_items,
            mode=req.mode, school_name=req.school_name,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"HTML 렌더링 오류: {e}")
    
    # ----- 4. 저장 + URL -----
    file_id = uuid.uuid4().hex[:12]
    html_path = os.path.join(OUTPUT_DIR, f"variation_{file_id}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    
    return {
        "questions_pdf": f"/variation-pdf/{file_id}",  # 호환 위해 기존 키 유지
        "answers_pdf": None,
        "html_url": f"/variation-pdf/{file_id}",
        "download_url": f"/variation-pdf/{file_id}?download=1",  # 다운로드 전용
        "passages_generated": len(req.passages),
        "types_generated": req.types,
        "mode": req.mode,
        "a_count": len(a_items),  # 디버그: A 유형 생성된 개수
        "b_count": len(b_items),  # 디버그: B 유형 생성된 개수
        "warnings": errors_per_passage if errors_per_passage else None,
    }


# ============ HTML 다운로드/조회 (prefix 없음) ============
download_router = APIRouter(tags=["variation-download"])


@download_router.get("/variation-pdf/{file_id}", include_in_schema=False)
def download_variation_html(file_id: str, download: int = 0):
    """변형문제 HTML 페이지 - download=1이면 다운로드, 아니면 새 창 표시"""
    if not file_id.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid file_id")
    html_path = os.path.join(OUTPUT_DIR, f"variation_{file_id}.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="변형문제 파일이 만료되었거나 없습니다")
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="변형문제_{file_id}.html"'
    return HTMLResponse(content=content, headers=headers)
