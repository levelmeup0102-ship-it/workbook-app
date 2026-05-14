"""
variation/api.py
변형문제 FastAPI 라우터

사용법 (main.py에 추가):
    from variation.api import router as variation_router
    app.include_router(variation_router)
"""
import os
import uuid
import traceback
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel

from variation.generator import (
    generate_variation_a,
    generate_variation_b,
    sb_client,
)
from variation.renderer import (
    render_variation_pdf,
    prepare_a_passage,
    prepare_b_passage,
)

# ============ 출력 디렉토리 ============
OUTPUT_DIR = os.environ.get("VARIATION_OUTPUT_DIR", "/tmp/variation_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============ 인증 (pipeline.py의 토큰 검증과 호환) ============
APP_PASSWORD = os.environ.get("APP_PASSWORD", "levelmeup2026")

def verify_token(authorization: Optional[str] = Header(None)):
    """간단한 Bearer 토큰 인증 (기존 main.py 패턴과 호환)"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization 헤더 필요")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer 토큰 형식 오류")
    token = authorization[7:]
    # 토큰 = APP_PASSWORD (기존 시스템과 동일 패턴)
    if token != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid token")
    return token


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


# ============ 라우터 ============
router = APIRouter(prefix="/api", tags=["variation"])


@router.post("/variation")
def create_variation(
    req: VariationRequest,
    _token: str = Depends(verify_token),
):
    """
    변형문제 PDF 생성
    
    1. 각 지문 텍스트를 Supabase passages 테이블에서 조회
    2. 선택된 유형(A/B)별로 데이터 생성 (캐시 활용 + 자동 검증)
    3. mode에 따라 PDF 빌드
    4. 다운로드 URL 반환
    """
    if not sb_client:
        raise HTTPException(status_code=500, detail="Supabase client 미설정")
    
    if not req.passages:
        raise HTTPException(status_code=400, detail="지문이 선택되지 않음")
    if not req.types:
        raise HTTPException(status_code=400, detail="유형이 선택되지 않음")
    for t in req.types:
        if t not in ("A", "B"):
            raise HTTPException(status_code=400, detail=f"알 수 없는 유형: {t}")
    if req.mode not in ("by-type", "by-passage"):
        raise HTTPException(status_code=400, detail=f"알 수 없는 mode: {req.mode}")
    
    # ----- 1. 지문 원문 가져오기 (사용자가 클릭한 순서 유지) -----
    passage_texts = {}
    for p in req.passages:
        try:
            res = sb_client.table("passages") \
                .select("passage_text") \
                .eq("book", p.book) \
                .eq("unit", p.unit) \
                .eq("pid", p.id) \
                .limit(1) \
                .execute()
            if not res.data:
                raise HTTPException(
                    status_code=404,
                    detail=f"지문을 찾을 수 없음: {p.book} / {p.unit} / {p.id}"
                )
            passage_texts[(p.book, p.unit, p.id)] = res.data[0]["passage_text"]
        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"DB 조회 오류: {e}")
    
    # ----- 2. 데이터 생성 (선택된 유형별) -----
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
                    pid=p.id,
                    book=p.book,
                    unit=p.unit,
                )
                a_items.append(prepare_a_passage(data_a, label))
            except Exception as e:
                traceback.print_exc()
                errors_per_passage.append(f"[{p.id}] A 생성 실패: {e}")
        
        if "B" in req.types:
            try:
                data_b = generate_variation_b(
                    passage_text=passage_text,
                    pid=p.id,
                    book=p.book,
                    unit=p.unit,
                )
                b_items.append(prepare_b_passage(data_b, label))
            except Exception as e:
                traceback.print_exc()
                errors_per_passage.append(f"[{p.id}] B 생성 실패: {e}")
    
    if not a_items and not b_items:
        raise HTTPException(
            status_code=500,
            detail="생성된 변형문제가 없습니다.\n" + "\n".join(errors_per_passage)
        )
    
    # ----- 3. PDF 빌드 -----
    try:
        pdf_bytes = render_variation_pdf(
            a_items=a_items,
            b_items=b_items,
            mode=req.mode,
            school_name=req.school_name,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF 렌더링 오류: {e}")
    
    # ----- 4. 저장 + URL 반환 -----
    file_id = uuid.uuid4().hex[:12]
    pdf_path = os.path.join(OUTPUT_DIR, f"variation_{file_id}.pdf")
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    
    return {
        "questions_pdf": f"/variation-pdf/{file_id}",
        "answers_pdf": None,  # 현재 한 PDF에 문제+답지 통합
        "passages_generated": len(req.passages),
        "types_generated": req.types,
        "mode": req.mode,
        "warnings": errors_per_passage if errors_per_passage else None,
    }


@router.get("/variation-pdf/{file_id}", include_in_schema=False)
def download_variation_pdf(file_id: str):
    """변형문제 PDF 다운로드"""
    # file_id 검증 (영숫자만)
    if not file_id.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid file_id")
    pdf_path = os.path.join(OUTPUT_DIR, f"variation_{file_id}.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"변형문제_{file_id}.pdf")


# main.py에서 라우터의 GET 라우트도 인식하도록 root 경로에 별도 등록
# (FastAPI는 prefix를 모든 라우트에 적용함. /variation-pdf/ 는 /api/variation-pdf/ 가 됨)


# ============ /variation-pdf 경로를 위한 별도 라우터 (prefix 없음) ============
download_router = APIRouter(tags=["variation-download"])


@download_router.get("/variation-pdf/{file_id}", include_in_schema=False)
def download_variation_pdf_root(file_id: str):
    if not file_id.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid file_id")
    pdf_path = os.path.join(OUTPUT_DIR, f"variation_{file_id}.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"변형문제_{file_id}.pdf")
