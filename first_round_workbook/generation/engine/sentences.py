"""문장 분리/병합 — 1회독 엔진 전용 (pipeline.py에서 이전 예정).

step1·step2·step5·process_passage 에서만 사용 (secret-note 미사용 — 확인됨).
"""


def split_sentences(text: str) -> list:
    """TODO: pipeline.py:66 (split_sentences) 이전."""
    raise NotImplementedError


def _is_dialogue(sentences: list) -> bool:
    """TODO: pipeline.py:161 (_is_dialogue) 이전."""
    raise NotImplementedError


def _merge_short_dialogue(sentences: list, min_words: int = 6) -> list:
    """TODO: pipeline.py:169 (_merge_short_dialogue) 이전."""
    raise NotImplementedError
