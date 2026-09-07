"""대기열(Job) 상태 관리 — 인메모리 jobs + asyncio.Queue + asyncio.Lock.

역할: job/task 생성, 큐 적재/소비, task 상태 변경(+재시도 판단), 진행상황/결과 조회, TTL 정리.
생성 로직(LLM/Supabase)은 여기 없다 — worker.py 가 큐를 소비해 실제 생성을 수행한다.

주의:
- Lock 은 '상태 접근'에만 쓴다. generate_workbook() 같은 무거운 작업은 Lock 밖(worker)에서 실행.
- queue 에는 (job_id, task_index) 만 넣는다. 실제 task/result 는 jobs 에 보관.
- prompts/grammar 는 job 생성 시 1회 로드해 job 에 저장한다.
- 재시도 판단(retry_count 증가·재적재·최종 FAILED)은 mark_failed_or_retry 한 곳에서만 한다.

⚠️ 전제: 단일 프로세스(uvicorn --workers 1). 재시작 시 진행 중 job 은 소실된다.
"""
import asyncio
import time
import uuid
import logging
from typing import Dict, List, Optional, Tuple

from .schemas import GenerateTarget, GenerateItemOut, GenerateProgressOut
from .job_models import (
    TaskStatus, JobStatus, GenerationTask, GenerationJob, ClaimedTask,
)

logger = logging.getLogger(__name__)


class JobManager:
    """여러 워커가 공유하는 job 저장소 + 작업 큐 (단일 loop, Lock 으로 상태 보호)."""

    def __init__(self, max_task_retries: int) -> None:
        self.jobs: Dict[str, GenerationJob] = {}
        self.queue: "asyncio.Queue[Tuple[str, int]]" = asyncio.Queue()   # (job_id, task_index)
        self.lock: asyncio.Lock = asyncio.Lock()
        self.max_task_retries: int = max_task_retries

    # ============================================================
    # 생성
    # ============================================================
    async def create_job(self, targets: List[GenerateTarget], prompts: dict,
                         grammar_addendum: str) -> GenerationJob:
        """job + task N개 생성 → jobs 등록 → 큐에 (job_id, index) 적재 → job 반환."""
        job_id = uuid.uuid4().hex
        tasks = [GenerationTask(index=i, target=target) for i, target in enumerate(targets)]
        job = GenerationJob(
            job_id=job_id,
            tasks=tasks,
            prompts=prompts,
            grammar_addendum=grammar_addendum,
            created_at=time.time(),
        )
        async with self.lock:
            self.jobs[job_id] = job

        for task in tasks:
            self.queue.put_nowait((job_id, task.index))   # 무제한 큐 → 논블로킹

        logger.info("[job] 생성 job_id=%s 지문수=%d", job_id, len(tasks))
        return job

    # ============================================================
    # 워커용 — 큐 소비 / task claim / 결과 반영
    # ============================================================
    async def get_next_task_reference(self) -> Tuple[str, int]:
        """큐에서 다음 (job_id, index) 를 꺼낸다(없으면 대기)."""
        return await self.queue.get()

    def queue_task_done(self) -> None:
        """get_next_task_reference() 로 꺼낸 항목의 처리 완료 표시."""
        self.queue.task_done()

    async def claim_task(self, *, job_id: str, index: int) -> Optional[ClaimedTask]:
        """task 를 PROCESSING 으로 잡고, 생성에 필요한 입력 스냅샷을 반환.
        (실제 생성은 이 값으로 Lock 밖에서 수행). job 이 없으면 None.
        """
        async with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            task = job.tasks[index]
            task.status = TaskStatus.PROCESSING
            task.started_at = time.time()
            return ClaimedTask(
                job_id=job_id,
                index=index,
                target=task.target,
                prompts=job.prompts,
                grammar_addendum=job.grammar_addendum,
            )

    async def mark_completed(self, *, job_id: str, index: int, result: GenerateItemOut) -> None:
        """생성 성공 → COMPLETED + 결과 저장."""
        async with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return
            task = job.tasks[index]
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.finished_at = time.time()
            self._finish_job_if_done(job)

    async def mark_failed_or_retry(
            self,
            *,
            job_id: str,
            index: int,
            error_code: str,
            error_message: str
            ) -> None:
        """실패 처리 — retry_count 가 최대 미만이면 재시도(PENDING 복귀 + 큐 재적재),
        최대까지 실패하면 최종 FAILED(result ok=False)."""
        async with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return
            task = job.tasks[index]
            task.error_code = error_code
            task.error_message = error_message

            if task.retry_count < self.max_task_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                self.queue.put_nowait((job_id, index))   # 재적재 → 다시 처리
                return

            task.status = TaskStatus.FAILED
            task.result = GenerateItemOut(ok=False)
            task.finished_at = time.time()
            self._finish_job_if_done(job)

    # ============================================================
    # 조회 (status API / results API) — 전부 Lock
    # ============================================================
    async def get_status(self, job_id: str) -> Optional[Tuple[JobStatus, GenerateProgressOut]]:
        """job 상태 + 진행상황. job 없으면 None. (done 여부는 status == JobStatus.DONE 로 판단)"""
        async with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            return (self._job_status(job), self._build_progress(job))

    async def get_results(self, job_id: str) -> Tuple[Optional[bool], Optional[List[GenerateItemOut]]]:
        """반환: (done, results). job 없으면 (None, None). 미완이면 (False, None).
        done이면 (True, task_index 순 결과)."""
        async with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return (None, None)
            if not self._is_done(job):
                return (False, None)
            results = [task.result or GenerateItemOut(ok=False) for task in job.tasks]
            return (True, results)

    # ============================================================
    # 정리 (TTL)
    # ============================================================
    async def cleanup(self, ttl_seconds: int) -> int:
        """finished_at + ttl 지난 완료 job 제거(진행 중 job 은 유지). 제거 수 반환."""
        now = time.time()
        async with self.lock:
            expired = [
                job_id for job_id, job in self.jobs.items()
                if job.finished_at is not None and (now - job.finished_at) >= ttl_seconds
            ]
            for job_id in expired:
                del self.jobs[job_id]
        if expired:
            logger.info("[job] cleanup: 만료 job %d개 제거", len(expired))
        return len(expired)

    # ============================================================
    # 내부 헬퍼 (반드시 Lock 안에서만 호출)
    # ============================================================
    def _is_done(self, job: GenerationJob) -> bool:
        """모든 task 가 completed 또는 failed 로 종료됐는가."""
        return all(t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED) for t in job.tasks)

    def _job_status(self, job: GenerationJob) -> JobStatus:
        if self._is_done(job):
            return JobStatus.DONE
        if any(t.status == TaskStatus.PROCESSING for t in job.tasks):
            return JobStatus.PROCESSING
        return JobStatus.PENDING

    def _build_progress(self, job: GenerationJob) -> GenerateProgressOut:
        counts = {status: 0 for status in TaskStatus}
        for task in job.tasks:
            counts[task.status] += 1
        return GenerateProgressOut(
            total=len(job.tasks),
            pending=counts[TaskStatus.PENDING],
            processing=counts[TaskStatus.PROCESSING],
            completed=counts[TaskStatus.COMPLETED],
            failed=counts[TaskStatus.FAILED],
        )

    def _finish_job_if_done(self, job: GenerationJob) -> None:
        """모든 task 종료 시 job.finished_at 기록(TTL 시작점)."""
        if job.finished_at is None and self._is_done(job):
            job.finished_at = time.time()
            logger.info("[job] 완료 job_id=%s", job.job_id)
