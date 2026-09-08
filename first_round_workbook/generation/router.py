"""generation 라우터 — POST /api/generate.

prefix="/api" + dependencies=[Depends(verify)] 로
엔드포인트마다 _verify(request) 호출하던 것을 대체.
"""

from fastapi import APIRouter, Depends, Request, Query

from core.security import _verify
from core.settings import settings
from core.exceptions import BadRequestError, NotFoundError
from . import service
from .job_models import JobStatus
from .schemas import (
    GenerateIn,
    GenerateResponseOut,
    GenerateJobStatusOut,
    GenerateJobResultsOut,
)

router = APIRouter(prefix="/api", tags=["generation"], dependencies=[Depends(_verify)])


@router.post("/generate", response_model=GenerateResponseOut, summary="영어 교재 생성")
async def generate(
    payload: GenerateIn,
    request: Request,
    force_job: bool = Query(False, description="true면 지문 수가 적어도 job으로 처리"),
):
    """지문 수에 따라 분기: 소량은 즉시 생성(sync), 다량은 대기열 등록(job)."""
    client = request.app.state.supabase
    job_manager = request.app.state.job_manager

    target_count = service.count_targets(payload)
    if target_count > settings.MAX_JOB_TARGETS:
        raise BadRequestError(f"한 번에 최대 {settings.MAX_JOB_TARGETS}개 지문까지 요청할 수 있습니다.")

    # 소량 + force_job 아님 → 즉시 생성 후 결과 반환
    if not force_job and target_count <= settings.SYNC_TARGET_LIMIT:
        out = await service.generate(payload, client)
        return GenerateResponseOut(mode="sync", done=True, results=out.results)

    # 다량 → 대기열 job 등록 후 job_id 즉시 반환(워커가 백그라운드로 생성)
    job = await service.create_generation_job(payload, client, job_manager)
    status, progress = await job_manager.get_status(job.job_id)
    return GenerateResponseOut(mode="job", done=False, job_id=job.job_id, progress=progress)


@router.get("/generate/status/{job_id}", response_model=GenerateJobStatusOut, summary="job 진행 상황 조회")
async def generate_status(job_id: str, request: Request):
    """job 의 상태 + 진행상황(polling 용)."""
    result = await request.app.state.job_manager.get_status(job_id)
    if result is None:
        raise NotFoundError("job을 찾을 수 없습니다(만료되었거나 서버가 재시작되었을 수 있습니다).")
    status, progress = result
    return GenerateJobStatusOut(
        job_id=job_id,
        status=status.value,
        done=(status == JobStatus.DONE),
        progress=progress,
    )


@router.get("/generate/results/{job_id}", response_model=GenerateJobResultsOut, summary="job 결과 조회")
async def generate_results(job_id: str, request: Request):
    """job 완료 시 결과(task 순서). 미완이면 done=False, results=null."""
    job_manager = request.app.state.job_manager
    done, results = await job_manager.get_results(job_id)
    if done is None:
        raise NotFoundError("job을 찾을 수 없습니다.")
    status_result = await job_manager.get_status(job_id)
    status = status_result[0] if status_result else JobStatus.DONE
    return GenerateJobResultsOut(
        job_id=job_id,
        status=status.value,
        done=bool(done),
        results=results,
    )
