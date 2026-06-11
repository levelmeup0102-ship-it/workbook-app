"""
seosul 패키지 — 레벨미업학원 워크북 2회독 서술형 종합문제 모듈

main.py 등록:
    from seosul.api import router as seosul_router, download_router as seosul_dl
    app.include_router(seosul_router); app.include_router(seosul_dl)

* 시작 시 무거운/순환 import를 피하기 위해 __init__ 에서는 아무것도 끌어오지 않는다.
  필요한 객체는 각 서브모듈에서 직접 import 한다 (예: from seosul.api import router).
"""
__version__ = "1.1.0"
