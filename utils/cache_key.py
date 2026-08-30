"""캐시 키 유틸 — 캐시 조회 열쇠(_ck) + 지문 내용 지문(_hash).

    from utils.cache_key import _ck, _hash
"""
import hashlib


def _hash(passage_text: str) -> str:
    """
    - 설명: 지문 내용 fingerprint (md5 32자). 앞뒤 공백 정규화 후 계산.
    - 사용 목적: _hash 는 '지문의 내용이 바뀌었나'를 검증
        같은 내용→같은 해시, 다른 내용→다른 해시.
    - 저장 위치: step_cache 테이블 > _passage_hash 컬럼으로 저장
    """
    return hashlib.md5(passage_text.strip().encode("utf-8")).hexdigest()


def _ck(book: str, unit: str, pid: str) -> str:
    """
    - 설명: 책이름_과_번호_hash 생성(한국어 포함, 가독성 우선)
        예: 공통영어1비상홈_1과_01번_26c81a4e
    - 사용 목적: _ck 는 DB 조회 용도.
    """
    raw = f"{book}_{unit}_{pid}"
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    BAD = set(' ()[]/\\:*?"<>|')
    def safe(s: str, maxlen: int = 20) -> str:
        return "".join(ch for ch in s if ch not in BAD)[:maxlen]
    return f"{safe(book,15)}_{safe(unit,8)}_{safe(pid,6)}_{h}"
