"""passages 라우터 — GET/POST(upload)/DELETE /api/passages.

prefix="/api/passages" + dependencies=[Depends(verify)].
경로는 기존과 100% 동일 (프론트 수정 불필요).
"""
from typing import List

from fastapi import APIRouter, Depends, Request

from core.security import _verify

from . import service
from .schemas import DeletePassageIn, PassageOut, UploadIn

router = APIRouter(prefix="/api/passages", tags=["passages"], dependencies=[Depends(_verify)])


@router.get(
        "",
        response_model=List[PassageOut],
        summary="",
        description=""
        )
async def list_passages(request: Request):
        """GET /api/passages : 지문 List를 가져오는 API입니다."""
        return await service.list_passages(request.app.state.supabase)


@router.post(
        "/upload",
        summary="지문 등록",
        description=""
        )
async def upload_passages(payload: UploadIn, request: Request):
        """POST /api/passages/upload : DB에 지문을 등록하는 API입니다."""
        return await service.upload_passages(payload, request.app.state.supabase)


@router.delete(
        "",
        summary="단일 혹은 선택한 지문 삭제",
        description=""
        )
async def delete_passage(payload: DeletePassageIn, request: Request):
        """DELETE /api/passages : 하나의 지문 혹은 선택한 지문을 삭제하는 API입니다."""
        return await service.delete_passage(payload, request.app.state.supabase)
