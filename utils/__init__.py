"""공통 헬퍼 모듈 (util) — 관심사별 분리.

main.py 와 first_round_workbook 패키지 양쪽이 import 하는 중립 위치.
→ 패키지가 main 을 import 할 필요가 없어져 순환 import 가 사라진다.

파일:
    security.py  : 인증 (토큰 생성/검증)
    cache.py     : 캐시 키 생성
    (이후 logger / json / asyncio 등 → README.md 후보 목록 참고)
"""
