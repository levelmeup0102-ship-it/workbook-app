"""generation worker.

asyncio.Queue 에서 (job_id, index) 를 꺼내 지문 1개를 생성하고,
결과를 JobManager 에 반영한다. (생성 대상 지문은 프론트 요청으로 만들어진 target 이다.)

주의:
- generate_workbook / execute_generation_target 은 Lock 밖에서 실행한다.
- worker_loop 는 서버 lifespan 에서 create_task 로 실행된다.
- shutdown 시 asyncio.CancelledError 는 반드시 다시 raise 한다.
- 재시도 판단(retry_count)은 여기서 하지 않는다 — JobManager.mark_failed_or_retry 가 담당.
"""
import asyncio
import logging

from supabase import AsyncClient

from .job_manager import JobManager
from .service import execute_generation_target

logger = logging.getLogger(__name__)


async def worker_loop(
    *,
    worker_name: str,
    job_manager: JobManager,
    client: AsyncClient,
) -> None:
    logger.info("[worker.started] worker=%s", worker_name)

    while True:
        job_id: str | None = None
        index: int | None = None
        got_task = False

        try:
            job_id, index = await job_manager.get_next_task_reference()
            got_task = True

            claimed = await job_manager.claim_task(job_id=job_id, index=index)
            if claimed is None:
                # job 이 cleanup 등으로 사라짐 → 이 task 는 건너뜀
                logger.warning(
                    "[worker.skip_task] worker=%s job_id=%s index=%s reason=not_claimed",
                    worker_name, job_id, index,
                )
                continue

            try:
                item = await execute_generation_target(
                    target=claimed.target,
                    client=client,
                    prompts=claimed.prompts,
                    grammar_addendum=claimed.grammar_addendum,
                )
            except asyncio.CancelledError:
                logger.info(
                    "[worker.cancelled_during_task] worker=%s job_id=%s index=%s",
                    worker_name, job_id, index,
                )
                raise
            except Exception as exception:
                logger.exception(
                    "[worker.task_exception] worker=%s job_id=%s index=%s",
                    worker_name, job_id, index,
                )
                await job_manager.mark_failed_or_retry(
                    job_id=job_id,
                    index=index,
                    error_code=exception.__class__.__name__,
                    error_message=str(exception),
                )
                continue

            if item.ok:
                await job_manager.mark_completed(
                    job_id=job_id, index=index, result=item,
                )
            else:
                await job_manager.mark_failed_or_retry(
                    job_id=job_id,
                    index=index,
                    error_code="GENERATION_FAILED",
                    error_message="워크북 생성 실패",
                )

        except asyncio.CancelledError:
            logger.info("[worker.cancelled] worker=%s", worker_name)
            raise

        except Exception:
            # 방어: 예기치 못한 예외로 워커 루프가 죽지 않도록 로깅 후 계속
            logger.exception(
                "[worker.loop_error] worker=%s job_id=%s index=%s",
                worker_name, job_id, index,
            )
            await asyncio.sleep(1)

        finally:
            if got_task:
                job_manager.queue_task_done()
