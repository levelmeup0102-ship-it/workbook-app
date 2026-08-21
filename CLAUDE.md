CLAUDE.md
레벨미업학원 워크북 앱 — FastAPI + Supabase 기반 영어 지문 워크북/문제 자동 생성 시스템.
브랜치
`feature/backend-from-design` 에서만 작업한다. (main보다 최신)
작업 규칙
코드·템플릿 커밋은 수지님이 한다. Claude는 열람·분석·검증·수정안 제시까지.
수정본을 보여주고 판단을 맡긴다. 올리는 건 수지님이다.
예외 — 이 두 문서만 Claude가 직접 커밋해도 된다(상시 허용):
`CLAUDE.md`, `docs/2회독_변형문제.md`
`git push`는 이 환경에서 막혀 있다(403). 문서 커밋은 GitHub API로 한다.
지금 하는 작업
2회독 변형문제(유형 A/B) 수정 중이다.
👉 2회독 관련 작업이면 `docs/2회독_변형문제.md` 를 먼저 읽을 것.
문항 구성, 파일 지도, 캐시 버전 규칙, 알려진 함정이 전부 거기 있다.
시스템 구성
모듈	기능	회독
`pipeline.py` + `template.html`	워크북 8단계 (Lv.1~Lv.10)	1회독
`variation/` + `variation.html` / `variation_b.html`	변형문제 유형 A·B	2회독
`seosul/`	서술형 종합	2회독
`sheet/`	선생님 분석지	0회독
`main.py`	지문 관리·인증·라우터 등록	공통
환경변수
`SUPABASE_URL`, `SUPABASE_KEY`(또는 `SUPABASE_SERVICE_KEY`), `ANTHROPIC_API_KEY`,
`APP_PASSWORD`(기본 `levelmeup2026`), `CLAUDE_MODEL`, `VARIATION_OUTPUT_DIR`
없으면 Supabase는 `data/passages.json` 로컬 폴백으로 돌고 생성 기능은 멈춘다.
배포는 Dockerfile(uvicorn). 로그는 Railway 대시보드.
로컬 확인
```bash
pip install -r requirements.txt
python3 -c "import main"          # 라우터 3개(variation/seosul/sheet) 다 떠야 정상
python3 tests/test_vocab_gate.py  # 어휘 게이트 테스트
```
미해결 이슈
`passages` 테이블에 `order_index` 컬럼이 없는데 `main.py`가 upsert에 실어 보낸다
→ `/api/passages/reorder`로 바꾼 순서가 조용히 사라진다. (컬럼 추가 또는 코드 제거 필요)
`notice`, `seosul_types` 테이블은 RLS 비활성 + 정책 0개 — anon 키로 읽기·쓰기 전부 열림.
