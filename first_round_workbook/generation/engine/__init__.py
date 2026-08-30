"""1회독 전용 생성 엔진 (pipeline.py에서 이전).

여기에는 /api/generate 에서만 쓰이는 코드만 둔다:
    - steps.py    : step1~8 생성기
    - pipeline.py : process_passage (step1~8 오케스트레이션)
    - render.py   : render_pdf

공유 저수준(save_step/load_step·LLM호출·문장분리)과
secret-note 전용 생성기는 루트 pipeline.py에 남는다.
"""
