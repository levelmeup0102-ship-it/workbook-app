"""1회독 생성 오케스트레이션 (pipeline.py에서 이전 예정).

process_passage: step1~8 를 순서대로 호출하고 render_pdf 로 최종 산출물 생성.
/api/generate 에서만 사용.
"""
from pathlib import Path


def _split_sentences_chunks(sentences: list, max_per_page: int = 8) -> list:
    """TODO: pipeline.py:2476 (_split_sentences_chunks) 이전 — 엔진 전용 헬퍼."""
    raise NotImplementedError


def merge_to_template_data(passage: str, meta: dict, all_steps: dict) -> dict:
    """TODO: pipeline.py:2499 (merge_to_template_data) 이전 — 엔진 전용 헬퍼."""
    raise NotImplementedError


def _unique_path(directory: Path, base_name: str, ext: str) -> Path:
    """TODO: pipeline.py:2564 (_unique_path) 이전 — 엔진 전용 헬퍼."""
    raise NotImplementedError


def process_passage(passage: str, meta: dict, passage_id: str,
                    force: bool = False, levels=None) -> Path:
    """TODO: pipeline.py:2607 (process_passage) 이전.

    내부 호출: steps.step1~8 + render.render_pdf
              + _split_sentences_chunks / merge_to_template_data / _unique_path,
    공유 저수준(루트 pipeline.py): save_step, load_step, call_claude 등.
    """
    raise NotImplementedError
