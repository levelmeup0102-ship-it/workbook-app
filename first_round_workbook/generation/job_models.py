"""대기열(Job) 방식 워크북 생성 — Job/Task 데이터 모델.

JobManager(job_manager.py)가 이 모델들을 인메모리로 보관한다. LLM 호출/로직 없음(순수 상태).
응답용 Pydantic 스키마(GenerateResponseOut 등)는 schemas.py 에 별도로 있다.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .schemas import GenerateTarget, GenerateItemOut


class TaskStatus(str, Enum):
    """지문 1건(task)의 처리 상태."""
    PENDING = "pending"        # 큐 대기
    PROCESSING = "processing"  # 워커가 생성 중
    COMPLETED = "completed"    # 생성 성공
    FAILED = "failed"          # 재시도까지 모두 실패


class JobStatus(str, Enum):
    """job 전체의 처리 상태."""
    PENDING = "pending"        # 아직 처리 시작 전(모두 대기)
    PROCESSING = "processing"  # 일부 task 처리 중
    DONE = "done"              # 모든 task 가 completed 또는 failed 로 종료


@dataclass
class GenerationTask:
    """지문 1건에 대한 생성 작업 단위."""
    index: int                                   # job 내 순서(결과 정렬 기준)
    target: GenerateTarget
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    result: Optional[GenerateItemOut] = None
    # ── 디버깅용 메타 ──
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[float] = None           # 생성 시작 시각
    finished_at: Optional[float] = None           # 생성 종료 시각


@dataclass
class GenerationJob:
    """한 요청(여러 지문) 단위 job. prompts/grammar 는 생성 시 1회 로드해 보관."""
    job_id: str
    tasks: List[GenerationTask]
    prompts: dict
    grammar_addendum: str
    created_at: float
    finished_at: Optional[float] = None           # 모든 task 종료 시각(TTL 시작점)


@dataclass
class ClaimedTask:
    """claim_task() 반환값 — 워커가 Lock 밖에서 지문을 생성할 때 필요한 입력 스냅샷."""
    job_id: str
    index: int
    target: GenerateTarget
    prompts: dict
    grammar_addendum: str
