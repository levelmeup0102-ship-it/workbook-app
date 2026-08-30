"""passages 도메인 요청/응답 모델."""
from pydantic import BaseModel


class UploadIn(BaseModel):
    """POST /api/passages/upload 요청 (main.py:264-265)."""
    book: str
    text: str


class DeletePassageIn(BaseModel):
    """DELETE /api/passages 요청 (main.py:322-324)."""
    book: str
    unit: str
    pid: str


class PassageOut(BaseModel):
    """GET /api/passages 응답 항목 (main.py:243-250)."""
    book: str
    unit: str
    id: str               # 프론트에서 p.id 로 씀
    title: str
    passage_text: str     # 원문 추출용
    cache_status: str     # "ready" | "not_ready"
